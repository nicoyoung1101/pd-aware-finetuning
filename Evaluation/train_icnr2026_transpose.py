"""Fine-tune TransPose on complete CARE-PD/BMCLab walks for ICNR 2026.

The script is deliberately conservative: it trains on the frozen nested split,
uses one variable-length walk at a time, balances subjects exactly within each
epoch, and only exposes the pre-registered S1/S1.5/S2 pose_s3 capacities.

Formal runs require a passing sanity-gate report produced by
``run_icnr2026_sanity_gates.py``. Outer-test subjects are never loaded here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from icnr2026_finetune_core import (
    CAPACITY_PARAMETER_COUNTS,
    CONDITIONS,
    DATASET_PATH,
    FROZEN_JOINT_POSITION_WEIGHT,
    FROZEN_L2SP_WEIGHT,
    FROZEN_POSE_WEIGHT,
    FROZEN_SWING_WEIGHT,
    FROZEN_VELOCITY_WEIGHT,
    JOINT_POSITION_HUBER_BETA_M,
    MANIFEST_PATH,
    PROJECT_ROOT,
    PRETRAINED_WEIGHTS_PATH,
    SPLIT_PATH,
    FineTuneLoss,
    NoiseSpec,
    SubjectBalancedEpochSampler,
    WholeWalkDataset,
    chunked,
    condition_spec,
    configure_capacity,
    fold_subjects,
    forward_pose_6d,
    indices_for_subjects,
    load_cycle_counts,
    load_dataset_and_manifest,
    load_nested_splits,
    load_pretrained_net,
    make_body_model,
    prepare_imu_input,
    resolve_device,
    seed_everything,
    set_finetune_mode,
    sha256_file,
    snapshot_trainable_parameters,
)


DEFAULT_OUT_ROOT = (
    PROJECT_ROOT / "TransPose/data/results/CAREPD_BMCLab_ICNR2026_finetune"
)
DEFAULT_L2SP_PILOT_OUT_ROOT = (
    PROJECT_ROOT / "TransPose/data/results/CAREPD_BMCLab_ICNR2026_L2SP_pilot"
)
DEFAULT_GATE_REPORT = (
    PROJECT_ROOT
    / "ModelComparison/data/results/CAREPD_BMCLab_ICNR2026_sanity/sanity_gates.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(6))
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--capacity", required=True, choices=sorted(CAPACITY_PARAMETER_COUNTS))
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--splits", type=Path, default=SPLIT_PATH)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--l2sp-pilot",
        action="store_true",
        help=(
            "Run the frozen fold-0/C6/S1/no-noise pilot with lambda_SP=0 and "
            "calibrate lambda_SP from drift and complete-epoch task gradients."
        ),
    )

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--sequences-per-step", type=int, default=4)
    parser.add_argument(
        "--samples-per-subject",
        type=int,
        default=0,
        help="0 uses ceil(number of training walks / number of training subjects).",
    )
    parser.add_argument("--max-train-samples", type=int, default=0, help="Smoke-test cap; 0 disables it")
    parser.add_argument("--enable-pose-s3-dropout", action="store_true")

    parser.add_argument("--pose-weight", type=float, default=FROZEN_POSE_WEIGHT)
    parser.add_argument(
        "--velocity-weight",
        type=float,
        default=FROZEN_VELOCITY_WEIGHT,
        help="Frozen protocol value; changing it is rejected.",
    )
    parser.add_argument(
        "--swing-weight",
        type=float,
        default=FROZEN_SWING_WEIGHT,
        help="Frozen protocol value; changing it is rejected.",
    )
    parser.add_argument("--l2sp-weight", type=float, default=FROZEN_L2SP_WEIGHT)

    parser.add_argument("--acc-white-std-mps2", type=float, default=0.0)
    parser.add_argument("--acc-bias-std-mps2", type=float, default=0.0)
    parser.add_argument("--ori-jitter-std-deg", type=float, default=0.0)
    parser.add_argument("--ori-mount-std-deg", type=float, default=0.0)
    parser.add_argument("--yaw-drift-std-deg-per-min", type=float, default=0.0)
    parser.add_argument("--pelvis-noise-multiplier", type=float, default=1.0)
    parser.add_argument("--other-noise-multiplier", type=float, default=1.0)
    return parser.parse_args()


def require_passing_gates(
    path: Path,
    allow_unverified: bool,
    *,
    dataset: Path,
    manifest: Path,
    splits: Path,
) -> None:
    if allow_unverified:
        return
    if not path.exists():
        raise RuntimeError(
            f"Missing sanity-gate report: {path}. Run Evaluation/run_icnr2026_sanity_gates.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("overall_status") != "PASS":
        raise RuntimeError(f"Sanity gates are not passing: {path}")
    gate_weights = payload.get("gates", {}).get("gate_3_gradient_isolation", {}).get(
        "frozen_loss_weights", {}
    )
    expected_weights = {
        "pose": FROZEN_POSE_WEIGHT,
        "velocity": FROZEN_VELOCITY_WEIGHT,
        "swing": FROZEN_SWING_WEIGHT,
        "l2sp": FROZEN_L2SP_WEIGHT,
    }
    if gate_weights != expected_weights:
        raise RuntimeError(
            f"Sanity report uses stale loss weights: {gate_weights}; expected {expected_weights}. "
            "Rerun the gates."
        )
    expected_hashes = {
        "pretrained_checkpoint": sha256_file(PRETRAINED_WEIGHTS_PATH),
        "dataset": sha256_file(dataset),
        "manifest": sha256_file(manifest),
        "splits": sha256_file(splits),
    }
    if payload.get("sha256") != expected_hashes:
        raise RuntimeError("Sanity report hashes are stale for the requested checkpoint/data/splits")


def as_float_dict(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def aggregate_subject_equal(rows: list[dict]) -> tuple[dict[str, float], list[dict]]:
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)
    metric_names = [
        key
        for key in rows[0]
        if key not in {"index", "subject", "updrs", "medication"}
    ]
    subject_rows = []
    for subject, items in sorted(by_subject.items()):
        subject_row = {
            "subject": subject,
            "n_walks": len(items),
            "updrs_max": max(int(item["updrs"]) for item in items),
        }
        for metric in metric_names:
            values = [float(item[metric]) for item in items if math.isfinite(float(item[metric]))]
            subject_row[metric] = sum(values) / len(values) if values else math.nan
        subject_rows.append(subject_row)
    summary = {"n_subjects": len(subject_rows), "n_walks": len(rows)}
    for metric in metric_names:
        values = [float(row[metric]) for row in subject_rows if math.isfinite(float(row[metric]))]
        summary[metric] = sum(values) / len(values) if values else math.nan
    return summary, subject_rows


def evaluate_validation(
    net,
    body_model,
    loss_fn,
    pretrained,
    dataset,
    device,
    sensor_config,
    clinical,
    args,
):
    net.eval()
    rows = []
    velocity_weight = args.velocity_weight if clinical else 0.0
    swing_weight = args.swing_weight if clinical else 0.0
    with torch.no_grad():
        for position in range(len(dataset)):
            record = dataset[position]
            imu = prepare_imu_input(
                record["acc"], record["ori"], sensor_config, c5_policy="root_copy"
            ).to(device)
            predicted = forward_pose_6d(net, imu)
            losses = loss_fn(
                predicted,
                imu,
                record["pose"].to(device),
                record["cycles"],
                pose_weight=args.pose_weight,
                velocity_weight=velocity_weight,
                swing_weight=swing_weight,
                l2sp_weight=args.l2sp_weight,
                pretrained=pretrained,
            )
            row = {
                "index": record["index"],
                "subject": record["subject"],
                "updrs": record["updrs"],
                "medication": record["medication"],
                **as_float_dict(losses),
            }
            row["own_objective_score"] = (
                args.pose_weight * row["pose_total"]
                + velocity_weight * row["velocity_log_ratio"]
                + swing_weight * row["swing_surrogate_weighted"]
            )
            rows.append(row)
    return (*aggregate_subject_equal(rows), rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cpu_state_dict(net) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}


def calibrate_l2sp_from_pilot(
    net,
    body_model,
    loss_fn,
    pretrained,
    trainable,
    train_dataset,
    sampler,
    device,
    sensor_config,
    clinical,
    args,
    selected_epoch: int,
) -> dict:
    if not clinical or sensor_config != "Full6":
        raise AssertionError("L2-SP pilot calibration must use the clinical Full6 task")
    n_parameters = sum(parameter.numel() for parameter in trainable)
    squared_drift = sum(
        float(torch.sum((parameter.detach() - pretrained[name]) ** 2))
        for name, parameter in net.named_parameters()
        if parameter.requires_grad
    )
    rms_drift = math.sqrt(squared_drift / n_parameters)
    if not math.isfinite(rms_drift) or rms_drift <= 0:
        raise RuntimeError(f"Pilot produced invalid RMS parameter drift: {rms_drift}")

    # Measure the unregularized task gradient on every optimizer-sized group in
    # the same complete, subject-balanced epoch that supplied the selected state.
    # Parameters stay fixed during this diagnostic; the median limits domination
    # by one unusually long or difficult group.
    positions = sampler.epoch_positions(selected_epoch)
    step_gradient_rms = []
    set_finetune_mode(net, enable_pose_s3_dropout=False)
    for group in chunked(positions, args.sequences_per_step):
        net.zero_grad(set_to_none=True)
        for position in group:
            record = train_dataset[position]
            imu = prepare_imu_input(
                record["acc"], record["ori"], sensor_config, c5_policy="root_copy"
            ).to(device)
            predicted = forward_pose_6d(net, imu)
            losses = loss_fn(
                predicted,
                imu,
                record["pose"].to(device),
                record["cycles"],
                pose_weight=args.pose_weight,
                velocity_weight=args.velocity_weight,
                swing_weight=args.swing_weight,
                l2sp_weight=0.0,
                pretrained=pretrained,
            )
            (losses["total"] / len(group)).backward()
        squared_gradient_norm = sum(
            float(torch.sum(parameter.grad.detach() ** 2))
            for parameter in trainable
            if parameter.grad is not None
        )
        step_gradient_rms.append(math.sqrt(squared_gradient_norm / n_parameters))
    net.zero_grad(set_to_none=True)
    if not step_gradient_rms or not all(math.isfinite(value) for value in step_gradient_rms):
        raise RuntimeError("Pilot task-gradient diagnostic is empty or non-finite")

    gradient_tensor = torch.tensor(step_gradient_rms, dtype=torch.float64)
    median_task_gradient_rms = float(torch.quantile(gradient_tensor, 0.5))
    lambda_sp = 0.1 * median_task_gradient_rms / (2.0 * rms_drift)
    return {
        "formula": "lambda_sp = 0.1 * median(task_gradient_l2/sqrt(N)) / (2 * rms_drift)",
        "fold": args.fold,
        "condition": args.condition,
        "capacity": args.capacity,
        "selected_by": "minimum subject-equal validation clinical_distortion",
        "selected_epoch": selected_epoch,
        "trainable_parameters": n_parameters,
        "squared_parameter_drift": squared_drift,
        "rms_parameter_drift": rms_drift,
        "gradient_epoch": selected_epoch,
        "gradient_epoch_subject_balanced": True,
        "gradient_epoch_walks": len(positions),
        "gradient_optimizer_groups": len(step_gradient_rms),
        "task_gradient_rms_per_parameter": {
            "minimum": float(gradient_tensor.min()),
            "p25": float(torch.quantile(gradient_tensor, 0.25)),
            "median": median_task_gradient_rms,
            "mean": float(gradient_tensor.mean()),
            "p75": float(torch.quantile(gradient_tensor, 0.75)),
            "maximum": float(gradient_tensor.max()),
        },
        "restoring_gradient_fraction": 0.1,
        "calibrated_lambda_sp": lambda_sp,
    }


def main() -> None:
    args = parse_args()
    require_passing_gates(
        args.gate_report,
        args.allow_unverified,
        dataset=args.dataset,
        manifest=args.manifest,
        splits=args.splits,
    )
    if args.epochs <= 0 or args.patience <= 0 or args.sequences_per_step <= 0:
        raise ValueError("epochs, patience, and sequences-per-step must be positive")
    if args.l2sp_pilot:
        if (args.fold, args.condition, args.capacity) != (0, "C6", "S1"):
            raise ValueError("--l2sp-pilot is locked to --fold 0 --condition C6 --capacity S1")
        noise_values = (
            args.acc_white_std_mps2,
            args.acc_bias_std_mps2,
            args.ori_jitter_std_deg,
            args.ori_mount_std_deg,
            args.yaw_drift_std_deg_per_min,
        )
        if any(value != 0.0 for value in noise_values):
            raise ValueError("--l2sp-pilot requires all augmentation noise to be zero")
        if args.max_train_samples != 0 or args.enable_pose_s3_dropout:
            raise ValueError("--l2sp-pilot requires the complete epoch and disabled pose_s3 dropout")
        if args.out_root == DEFAULT_OUT_ROOT:
            args.out_root = DEFAULT_L2SP_PILOT_OUT_ROOT
    seed_everything(args.seed)
    device = resolve_device(args.device)

    spec = condition_spec(args.condition)
    clinical = bool(spec["clinical"])
    sensor_config = str(spec["sensor_config"])
    frozen_cli_weights = {
        "pose_weight": FROZEN_POSE_WEIGHT,
        "velocity_weight": FROZEN_VELOCITY_WEIGHT,
        "swing_weight": FROZEN_SWING_WEIGHT,
        "l2sp_weight": FROZEN_L2SP_WEIGHT,
    }
    for name, expected in frozen_cli_weights.items():
        observed = float(getattr(args, name))
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{name} is frozen at {expected}; received {observed}. "
                "Formal folds must not search or retune loss weights."
            )
    if args.l2sp_pilot:
        # This is the single preregistration pilot, not a formal fold. The CLI
        # still has to carry the frozen value; only the effective training term
        # is set to zero here.
        args.l2sp_weight = 0.0
    run_dir = (
        args.out_root
        / f"fold_{args.fold}"
        / f"{args.condition}_{args.capacity}_seed{args.seed}"
    )
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Run directory is not empty: {run_dir}; pass --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    data, manifest = load_dataset_and_manifest(args.dataset, args.manifest)
    split_payload = load_nested_splits(args.splits)
    train_subjects = fold_subjects(split_payload, args.fold, "inner_train")
    val_subjects = fold_subjects(split_payload, args.fold, "inner_val")
    test_subjects = fold_subjects(split_payload, args.fold, "outer_test")
    if set(train_subjects) & set(val_subjects) or set(train_subjects) & set(test_subjects):
        raise AssertionError("Frozen nested split has overlapping roles")
    cycles = load_cycle_counts()
    train_dataset = WholeWalkDataset(
        data, manifest, indices_for_subjects(manifest, train_subjects), cycles
    )
    val_dataset = WholeWalkDataset(
        data, manifest, indices_for_subjects(manifest, val_subjects), cycles
    )
    sampler = SubjectBalancedEpochSampler(
        train_dataset,
        seed=args.seed + 10_000 * args.fold,
        samples_per_subject=args.samples_per_subject or None,
    )

    net = load_pretrained_net(device)
    trainable = configure_capacity(net, args.capacity)
    pretrained = snapshot_trainable_parameters(net)
    body_model = make_body_model(device)
    loss_fn = FineTuneLoss(net, body_model)
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    noise = NoiseSpec(
        acc_white_std_mps2=args.acc_white_std_mps2,
        acc_bias_std_mps2=args.acc_bias_std_mps2,
        ori_jitter_std_deg=args.ori_jitter_std_deg,
        ori_mount_std_deg=args.ori_mount_std_deg,
        yaw_drift_std_deg_per_min=args.yaw_drift_std_deg_per_min,
        pelvis_multiplier=args.pelvis_noise_multiplier,
        other_multiplier=args.other_noise_multiplier,
    )

    metadata = {
        "purpose": "ICNR2026 CARE-PD PD-aware TransPose fine-tuning",
        "l2sp_calibration_pilot": args.l2sp_pilot,
        "fold": args.fold,
        "condition": args.condition,
        "capacity": args.capacity,
        "trainable_parameters": CAPACITY_PARAMETER_COUNTS[args.capacity],
        "sensor_config": sensor_config,
        "c5_policy": "root_copy" if sensor_config == "C5NoHead" else None,
        "clinical_loss_enabled": clinical,
        "seed": args.seed,
        "device": str(device),
        "whole_walk_training": True,
        "subject_balanced_sampling": True,
        "samples_per_subject_per_epoch": sampler.samples_per_subject,
        "sequences_per_optimizer_step": args.sequences_per_step,
        "train_subjects": train_subjects,
        "validation_subjects": val_subjects,
        "outer_test_subjects_not_used_for_training_or_validation": test_subjects,
        "n_train_walks": len(train_dataset),
        "n_validation_walks": len(val_dataset),
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "gradient_clip": args.gradient_clip,
        "maximum_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "early_stopping_min_delta": args.min_delta,
        "pose_s3_dropout_enabled": args.enable_pose_s3_dropout,
        "main_checkpoint_selection": "minimum subject-equal validation clinical_distortion for every arm",
        "sensitivity_checkpoint_selection": "minimum condition-specific own_objective_score",
        "early_stopping_monitor": (
            "dual-monitor: stop only after both clinical_distortion and own_objective_score "
            "have each been stale for patience epochs"
        ),
        "loss_weights": {
            "pose": args.pose_weight,
            "velocity": args.velocity_weight if clinical else 0.0,
            "swing": args.swing_weight if clinical else 0.0,
            "joint_position_within_pose": FROZEN_JOINT_POSITION_WEIGHT,
            "l2sp": args.l2sp_weight,
        },
        "loss_protocol": {
            "rotation": "mean chordal/Frobenius matrix loss on 15 predicted global rotations",
            "joint_position": "root-relative predicted-joint Smooth L1 distance",
            "joint_position_huber_beta_m": JOINT_POSITION_HUBER_BETA_M,
            "swing_training": "bilateral top/bottom 5% mean sagittal range; minimum 3 frames per tail",
            "swing_validation_endpoint": "bilateral hard p95-p5 sagittal range",
            "l2sp": "unnormalized squared displacement sum over trainable parameters",
            "l2sp_capacity_invariance": (
                "per-parameter restoring gradient is 2*lambda_sp*(theta-theta_pre)"
            ),
            "clinical_weights_frozen_from": "fold-0 inner-train pretrained-gradient diagnostic",
            "l2sp_weight_frozen_from": (
                "fold-0 inner-train C6/S1 lambda_sp=0 pilot; 10% of median complete-group "
                "task-gradient RMS at the clinical-selected checkpoint"
            ),
            "clinical_weights_no_retuning_after_null": True,
        },
        "noise": noise.to_dict(),
        "gate_report": str(args.gate_report),
        "pretrained_checkpoint": str(PRETRAINED_WEIGHTS_PATH),
        "pretrained_checkpoint_sha256": sha256_file(PRETRAINED_WEIGHTS_PATH),
        "dataset_sha256": sha256_file(args.dataset),
        "manifest_sha256": sha256_file(args.manifest),
        "split_sha256": sha256_file(args.splits),
        "split_file": str(args.splits),
        "smoke_max_train_samples": args.max_train_samples,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    history = []
    best_clinical_score = math.inf
    best_clinical_epoch = -1
    best_clinical_state = None
    best_objective_score = math.inf
    best_objective_epoch = -1
    best_objective_state = None
    stale_clinical_epochs = 0
    stale_objective_epochs = 0
    velocity_weight = args.velocity_weight if clinical else 0.0
    swing_weight = args.swing_weight if clinical else 0.0

    for epoch in range(args.epochs):
        positions = sampler.epoch_positions(epoch)
        if args.max_train_samples > 0:
            positions = positions[: args.max_train_samples]
        set_finetune_mode(net, enable_pose_s3_dropout=args.enable_pose_s3_dropout)
        train_sums: dict[str, float] = defaultdict(float)
        n_train = 0

        for group in chunked(positions, args.sequences_per_step):
            optimizer.zero_grad(set_to_none=True)
            for position in group:
                record = train_dataset[position]
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    args.seed + epoch * 1_000_003 + int(record["index"]) * 101
                )
                imu = prepare_imu_input(
                    record["acc"],
                    record["ori"],
                    sensor_config,
                    c5_policy="root_copy",
                    noise=noise,
                    generator=generator,
                ).to(device)
                predicted = forward_pose_6d(net, imu)
                losses = loss_fn(
                    predicted,
                    imu,
                    record["pose"].to(device),
                    record["cycles"],
                    pose_weight=args.pose_weight,
                    velocity_weight=velocity_weight,
                    swing_weight=swing_weight,
                    l2sp_weight=args.l2sp_weight,
                    pretrained=pretrained,
                )
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch}, sequence={record['index']}"
                    )
                (losses["total"] / len(group)).backward()
                for name, value in as_float_dict(losses).items():
                    train_sums[name] += value
                n_train += 1
            torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            optimizer.step()

        train_mean = {name: value / n_train for name, value in train_sums.items()}
        val_summary, _, _ = evaluate_validation(
            net,
            body_model,
            loss_fn,
            pretrained,
            val_dataset,
            device,
            sensor_config,
            clinical,
            args,
        )
        row = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_mean.items()},
            **{f"val_{name}": value for name, value in val_summary.items()},
        }
        history.append(row)
        clinical_score = float(val_summary["clinical_distortion"])
        objective_score = float(val_summary["own_objective_score"])
        clinical_improved = clinical_score < best_clinical_score - args.min_delta
        if clinical_improved:
            best_clinical_score = clinical_score
            best_clinical_epoch = epoch
            best_clinical_state = cpu_state_dict(net)
            stale_clinical_epochs = 0
        else:
            stale_clinical_epochs += 1
        objective_improved = objective_score < best_objective_score - args.min_delta
        if objective_improved:
            best_objective_score = objective_score
            best_objective_epoch = epoch
            best_objective_state = cpu_state_dict(net)
            stale_objective_epochs = 0
        else:
            stale_objective_epochs += 1
        print(
            f"epoch={epoch:03d} train_total={train_mean['total']:.6f} "
            f"val_clinical={clinical_score:.6f} best_clinical={best_clinical_score:.6f} "
            f"val_own={objective_score:.6f} best_own={best_objective_score:.6f} "
            f"stale_clinical={stale_clinical_epochs} stale_own={stale_objective_epochs}"
        )
        if stale_clinical_epochs >= args.patience and stale_objective_epochs >= args.patience:
            break

    if best_clinical_state is None or best_objective_state is None:
        raise RuntimeError("Training ended without both finite validation checkpoints")

    # Main result: both pose-only and clinical-aware arms get their strongest
    # validation Clinical Distortion checkpoint.
    net.load_state_dict(best_clinical_state)
    main_val_summary, main_val_subject_rows, main_val_walk_rows = evaluate_validation(
        net,
        body_model,
        loss_fn,
        pretrained,
        val_dataset,
        device,
        sensor_config,
        clinical,
        args,
    )
    main_checkpoint = {
        "model_state_dict": best_clinical_state,
        "selection_role": "main",
        "selection_metric": "subject-equal validation clinical_distortion",
        "best_epoch": best_clinical_epoch,
        "best_validation_clinical_distortion": best_clinical_score,
        "run_config": metadata,
        "validation_subject_equal": main_val_summary,
    }
    torch.save(main_checkpoint, run_dir / "best_checkpoint.pt")
    write_csv(run_dir / "history.csv", history)
    write_csv(run_dir / "best_validation_per_subject.csv", main_val_subject_rows)
    write_csv(run_dir / "best_validation_per_walk.csv", main_val_walk_rows)

    l2sp_pilot_calibration = None
    if args.l2sp_pilot:
        l2sp_pilot_calibration = calibrate_l2sp_from_pilot(
            net,
            body_model,
            loss_fn,
            pretrained,
            trainable,
            train_dataset,
            sampler,
            device,
            sensor_config,
            clinical,
            args,
            best_clinical_epoch,
        )
        (run_dir / "l2sp_pilot_calibration.json").write_text(
            json.dumps(l2sp_pilot_calibration, indent=2), encoding="utf-8"
        )

    # Sensitivity: retain the original condition-specific objective selection.
    net.load_state_dict(best_objective_state)
    objective_val_summary, objective_val_subject_rows, objective_val_walk_rows = evaluate_validation(
        net,
        body_model,
        loss_fn,
        pretrained,
        val_dataset,
        device,
        sensor_config,
        clinical,
        args,
    )
    objective_checkpoint = {
        "model_state_dict": best_objective_state,
        "selection_role": "sensitivity",
        "selection_metric": "condition-specific validation own_objective_score",
        "best_epoch": best_objective_epoch,
        "best_validation_own_objective_score": best_objective_score,
        "run_config": metadata,
        "validation_subject_equal": objective_val_summary,
    }
    torch.save(objective_checkpoint, run_dir / "best_own_objective_checkpoint.pt")
    write_csv(
        run_dir / "best_own_objective_validation_per_subject.csv", objective_val_subject_rows
    )
    write_csv(
        run_dir / "best_own_objective_validation_per_walk.csv", objective_val_walk_rows
    )
    summary = {
        **metadata,
        "epochs_completed": len(history),
        "main": {
            "best_epoch": best_clinical_epoch,
            "best_validation_clinical_distortion": best_clinical_score,
            "best_validation_subject_equal": main_val_summary,
        },
        "own_objective_sensitivity": {
            "best_epoch": best_objective_epoch,
            "best_validation_own_objective_score": best_objective_score,
            "best_validation_subject_equal": objective_val_summary,
        },
        "l2sp_pilot_calibration": l2sp_pilot_calibration,
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"Saved main clinical epoch {best_clinical_epoch} and own-objective sensitivity "
        f"epoch {best_objective_epoch} to {run_dir}"
    )


if __name__ == "__main__":
    # Avoid an OpenMP runtime collision seen in the local macOS data environment.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
