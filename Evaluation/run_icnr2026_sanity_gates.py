"""Run the three mandatory sanity gates before ICNR 2026 fine-tuning.

Gate 1: data/split/normalization contract, including C5 root-copy == (0, I).
Gate 2: current healthy-pretrained checkpoint reproduces the locked U0/U1/U2
        no-root MPJPE values on all 781 walks.
Gate 3: a full-walk clinical backward/update is finite and changes only the
        authorized S1 output head.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from icnr2026_finetune_core import (
    CAPACITY_PARAMETER_COUNTS,
    DATASET_PATH,
    FROZEN_L2SP_WEIGHT,
    FROZEN_POSE_WEIGHT,
    FROZEN_SWING_WEIGHT,
    FROZEN_VELOCITY_WEIGHT,
    HEAD_INDEX,
    MANIFEST_PATH,
    NoiseSpec,
    PROJECT_ROOT,
    PRETRAINED_WEIGHTS_PATH,
    ROOT_INDEX,
    SPLIT_PATH,
    FineTuneLoss,
    SubjectBalancedEpochSampler,
    WholeWalkDataset,
    configure_capacity,
    fold_subjects,
    forward_pose_6d,
    indices_for_subjects,
    load_cycle_counts,
    load_dataset_and_manifest,
    load_nested_splits,
    load_pretrained_net,
    l2sp_loss,
    make_body_model,
    no_root_mpjpe_mm,
    pose_axis_angle_to_matrix,
    prepare_imu_input,
    reduced_global_6d_to_local_pose,
    resolve_device,
    seed_everything,
    set_finetune_mode,
    sha256_file,
    snapshot_trainable_parameters,
    top_bottom_mean_range,
)
from utils import normalize_and_concat


DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "ModelComparison/data/results/CAREPD_BMCLab_ICNR2026_sanity"
)
LOCKED_MPJPE_MM = {
    0: 16.740611473486222,
    1: 18.48227025114972,
    2: 26.96144716914107,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--splits", type=Path, default=SPLIT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--mpjpe-tolerance-mm", type=float, default=0.005)
    return parser.parse_args()


def assert_finite_tensor(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value).all():
        raise AssertionError(f"Non-finite values in {label}")


def gate_1_data_contract(data: dict, manifest: list[dict], splits: dict) -> dict:
    if len(manifest) != 781:
        raise AssertionError(f"Expected 781 walks, found {len(manifest)}")
    lengths = []
    max_full6_delta = 0.0
    max_c5_head_acc = 0.0
    max_c5_head_ori = 0.0
    eye = torch.eye(3).flatten()
    for index in range(len(manifest)):
        seq_lengths = [len(data[key][index]) for key in ("acc", "ori", "pose", "tran")]
        if len(set(seq_lengths)) != 1:
            raise AssertionError(f"Sequence {index} length mismatch: {seq_lengths}")
        lengths.append(seq_lengths[0])
        for key in ("acc", "ori", "pose", "tran"):
            assert_finite_tensor(data[key][index], f"{key}[{index}]")

        direct = normalize_and_concat(data["acc"][index], data["ori"][index])
        prepared = prepare_imu_input(
            data["acc"][index], data["ori"][index], "Full6"
        )
        max_full6_delta = max(max_full6_delta, float((direct - prepared).abs().max()))

        c5 = prepare_imu_input(
            data["acc"][index], data["ori"][index], "C5NoHead", c5_policy="root_copy"
        )
        head_acc = c5[:, HEAD_INDEX * 3 : (HEAD_INDEX + 1) * 3]
        ori_start = 18 + HEAD_INDEX * 9
        head_ori = c5[:, ori_start : ori_start + 9]
        max_c5_head_acc = max(max_c5_head_acc, float(head_acc.abs().max()))
        max_c5_head_ori = max(max_c5_head_ori, float((head_ori - eye).abs().max()))

    if max_full6_delta != 0.0:
        raise AssertionError(f"Full6 preprocessing differs from canonical normalization: {max_full6_delta}")
    if max_c5_head_acc > 1e-6 or max_c5_head_ori > 1e-6:
        raise AssertionError(
            f"C5 root-copy is not normalized (0,I): acc={max_c5_head_acc}, ori={max_c5_head_ori}"
        )

    all_subjects = {str(item["subject_id"]) for item in manifest}
    test_counts: dict[str, int] = defaultdict(int)
    for fold in range(6):
        roles = {
            role: set(fold_subjects(splits, fold, role))
            for role in ("inner_train", "inner_val", "outer_test")
        }
        if roles["inner_train"] & roles["inner_val"] or roles["inner_val"] & roles["outer_test"]:
            raise AssertionError(f"Fold {fold} role overlap")
        if set.union(*roles.values()) != all_subjects:
            raise AssertionError(f"Fold {fold} does not cover all subjects")
        if len(roles["inner_val"]) != 4:
            raise AssertionError(f"Fold {fold} inner validation size is not four")
        for subject in roles["outer_test"]:
            test_counts[subject] += 1
    if set(test_counts) != all_subjects or set(test_counts.values()) != {1}:
        raise AssertionError("Each subject must occur in exactly one outer test fold")

    fold0_train = fold_subjects(splits, 0, "inner_train")
    fold0_dataset = WholeWalkDataset(
        data, manifest, indices_for_subjects(manifest, fold0_train)
    )
    sampler = SubjectBalancedEpochSampler(fold0_dataset, seed=20260802)
    counts = sampler.subject_counts(0)
    if len(set(counts.values())) != 1:
        raise AssertionError(f"Subject-balanced sampler is not exact: {counts}")

    noise = NoiseSpec(
        acc_white_std_mps2=0.05,
        acc_bias_std_mps2=0.02,
        ori_jitter_std_deg=0.5,
        ori_mount_std_deg=1.0,
        yaw_drift_std_deg_per_min=2.0,
        pelvis_multiplier=1.0,
        other_multiplier=0.0,
    )
    generator_a = torch.Generator(device="cpu").manual_seed(20260802)
    generator_b = torch.Generator(device="cpu").manual_seed(20260802)
    noisy_a = prepare_imu_input(
        data["acc"][0], data["ori"][0], "C5NoHead", noise=noise, generator=generator_a
    )
    noisy_b = prepare_imu_input(
        data["acc"][0], data["ori"][0], "C5NoHead", noise=noise, generator=generator_b
    )
    assert_finite_tensor(noisy_a, "deterministic noisy C5 input")
    noise_repeat_delta = float((noisy_a - noisy_b).abs().max())
    if noise_repeat_delta != 0.0:
        raise AssertionError(f"Seeded noise augmentation is not deterministic: {noise_repeat_delta}")
    noisy_head_acc = noisy_a[:, HEAD_INDEX * 3 : (HEAD_INDEX + 1) * 3]
    noisy_ori_start = 18 + HEAD_INDEX * 9
    noisy_head_ori = noisy_a[:, noisy_ori_start : noisy_ori_start + 9]
    if float(noisy_head_acc.abs().max()) > 1e-6 or float((noisy_head_ori - eye).abs().max()) > 1e-6:
        raise AssertionError("C5 root-copy semantics were not preserved after raw pelvis-only noise")

    return {
        "status": "PASS",
        "n_walks": len(manifest),
        "n_subjects": len(all_subjects),
        "min_frames": min(lengths),
        "max_frames": max(lengths),
        "n_unique_lengths": len(set(lengths)),
        "full6_preprocessing_max_abs_delta": max_full6_delta,
        "c5_normalized_head_acc_max_abs": max_c5_head_acc,
        "c5_normalized_head_orientation_identity_max_abs_delta": max_c5_head_ori,
        "root_sensor_index": ROOT_INDEX,
        "fold0_samples_per_subject": next(iter(counts.values())),
        "seeded_noise_repeat_max_abs_delta": noise_repeat_delta,
        "noise_smoke_spec": noise.to_dict(),
    }


@torch.no_grad()
def gate_2_checkpoint_regression(
    data: dict,
    manifest: list[dict],
    device: torch.device,
    tolerance_mm: float,
) -> dict:
    net = load_pretrained_net(device)
    net.eval()
    body_model = make_body_model(device)
    grouped: dict[int, list[float]] = defaultdict(list)
    first_helper_delta = None
    for index, item in enumerate(manifest):
        imu = prepare_imu_input(data["acc"][index], data["ori"][index], "Full6").to(device)
        predicted_6d = forward_pose_6d(net, imu)
        predicted_local = reduced_global_6d_to_local_pose(
            net, imu[:, -9:].view(-1, 3, 3), predicted_6d
        )
        if index == 0:
            offline_pose, _ = net.forward_offline(imu)
            first_helper_delta = float((predicted_local.cpu() - offline_pose).abs().max())
        value = no_root_mpjpe_mm(
            body_model, predicted_local, data["pose"][index].to(device)
        )
        grouped[int(item["UPDRS_GAIT"])].append(value)
        if (index + 1) % 100 == 0 or index + 1 == len(manifest):
            print(f"Gate 2 forward: {index + 1}/{len(manifest)} walks")

    observed = {level: float(sum(values) / len(values)) for level, values in grouped.items()}
    differences = {level: observed[level] - LOCKED_MPJPE_MM[level] for level in LOCKED_MPJPE_MM}
    if set(observed) != set(LOCKED_MPJPE_MM):
        raise AssertionError(f"Unexpected UPDRS levels: {sorted(observed)}")
    if max(abs(value) for value in differences.values()) > tolerance_mm:
        raise AssertionError(
            f"Checkpoint regression exceeded {tolerance_mm} mm: observed={observed}, delta={differences}"
        )
    if first_helper_delta is None or first_helper_delta > 1e-6:
        raise AssertionError(f"Training pose helper differs from forward_offline: {first_helper_delta}")
    return {
        "status": "PASS",
        "n_walks": len(manifest),
        "locked_no_root_mpjpe_mm": {str(k): v for k, v in LOCKED_MPJPE_MM.items()},
        "observed_no_root_mpjpe_mm": {str(k): v for k, v in observed.items()},
        "delta_mm": {str(k): v for k, v in differences.items()},
        "tolerance_mm": tolerance_mm,
        "training_helper_vs_forward_offline_max_abs_delta": first_helper_delta,
    }


def gate_3_gradient_isolation(
    data: dict,
    manifest: list[dict],
    splits: dict,
    device: torch.device,
    seed: int,
) -> dict:
    train_subjects = fold_subjects(splits, 0, "inner_train")
    indices = indices_for_subjects(manifest, train_subjects)
    index = min(indices, key=lambda i: len(data["acc"][i]))
    cycles = load_cycle_counts()
    record = WholeWalkDataset(data, manifest, [index], cycles)[0]

    capacity_counts = {}
    l2sp_uniform_shift_loss = {}
    l2sp_uniform_shift_gradient = {}
    for capacity in CAPACITY_PARAMETER_COUNTS:
        probe = load_pretrained_net(device)
        params = configure_capacity(probe, capacity)
        capacity_counts[capacity] = sum(p.numel() for p in params)
        probe_reference = snapshot_trainable_parameters(probe)
        with torch.no_grad():
            for parameter in params:
                parameter.add_(1e-3)
        shifted_loss = l2sp_loss(probe, probe_reference)
        gradients = torch.autograd.grad(shifted_loss, params)
        flat_gradients = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
        l2sp_uniform_shift_loss[capacity] = float(shifted_loss.detach())
        l2sp_uniform_shift_gradient[capacity] = {
            "mean": float(flat_gradients.mean()),
            "min": float(flat_gradients.min()),
            "max": float(flat_gradients.max()),
        }
        del probe
    expected_shift_gradient = 2e-3
    maximum_gradient_error = max(
        abs(item[key] - expected_shift_gradient)
        for item in l2sp_uniform_shift_gradient.values()
        for key in ("mean", "min", "max")
    )
    if maximum_gradient_error > 1e-7:
        raise AssertionError(
            "Summed L2-SP does not have capacity-invariant per-parameter restoring "
            f"gradient: {l2sp_uniform_shift_gradient}"
        )

    surrogate_probe = torch.linspace(-1.0, 1.0, 180, device=device, requires_grad=True)
    top_bottom_mean_range(surrogate_probe).backward()
    surrogate_gradient_support = int((surrogate_probe.grad.abs() > 0).sum())
    if surrogate_gradient_support != 18:
        raise AssertionError(
            f"Expected 18/180 top-bottom surrogate frames with gradient, got {surrogate_gradient_support}"
        )

    net = load_pretrained_net(device)
    trainable = configure_capacity(net, "S1")
    set_finetune_mode(net, enable_pose_s3_dropout=False)
    body_model = make_body_model(device)
    loss_fn = FineTuneLoss(net, body_model)
    pretrained = snapshot_trainable_parameters(net)
    before = {name: parameter.detach().clone() for name, parameter in net.named_parameters()}

    imu = prepare_imu_input(
        record["acc"], record["ori"], "Full6"
    ).to(device)
    predicted = forward_pose_6d(net, imu)
    losses = loss_fn(
        predicted,
        imu,
        record["pose"].to(device),
        record["cycles"],
        pose_weight=FROZEN_POSE_WEIGHT,
        velocity_weight=FROZEN_VELOCITY_WEIGHT,
        swing_weight=FROZEN_SWING_WEIGHT,
        l2sp_weight=FROZEN_L2SP_WEIGHT,
        pretrained=pretrained,
    )
    for name, value in losses.items():
        if not torch.isfinite(value):
            raise AssertionError(f"Non-finite Gate 3 loss: {name}={value}")
    if float(losses["l2sp"].detach()) != 0.0:
        raise AssertionError(f"L2-SP must be zero at initialization, got {losses['l2sp']}")

    optimizer = torch.optim.Adam(trainable, lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    component_gradient_norms = {}
    for component in (
        "rotation_chordal",
        "joint_position_huber",
        "pose_total",
        "velocity_log_ratio",
        "swing_surrogate_weighted",
    ):
        gradients = torch.autograd.grad(
            losses[component], trainable, retain_graph=True, allow_unused=True
        )
        norm = math.sqrt(
            sum(float(torch.sum(gradient.detach() ** 2)) for gradient in gradients if gradient is not None)
        )
        if not math.isfinite(norm) or norm <= 0:
            raise AssertionError(f"{component} gradient is invalid: {norm}")
        component_gradient_norms[component] = norm
    losses["total"].backward()
    trainable_grad_norm = math.sqrt(
        sum(float(torch.sum(p.grad.detach() ** 2)) for p in trainable if p.grad is not None)
    )
    if not math.isfinite(trainable_grad_norm) or trainable_grad_norm <= 0:
        raise AssertionError(f"Clinical gradient is invalid: {trainable_grad_norm}")
    unauthorized_grads = [
        name
        for name, parameter in net.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if unauthorized_grads:
        raise AssertionError(f"Frozen parameters received gradients: {unauthorized_grads}")
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()

    changed_trainable = []
    changed_frozen = []
    max_trainable_change = 0.0
    for name, parameter in net.named_parameters():
        change = float((parameter.detach() - before[name]).abs().max())
        if parameter.requires_grad and change > 0:
            changed_trainable.append(name)
            max_trainable_change = max(max_trainable_change, change)
        if not parameter.requires_grad and change != 0:
            changed_frozen.append(name)
    if not changed_trainable:
        raise AssertionError("Optimizer step did not change an authorized parameter")
    if changed_frozen:
        raise AssertionError(f"Optimizer step changed frozen parameters: {changed_frozen}")
    if any(not name.startswith("pose_s3.linear2.") for name in changed_trainable):
        raise AssertionError(f"S1 changed an unauthorized parameter: {changed_trainable}")

    balanced_clinical_to_pose = (
        FROZEN_VELOCITY_WEIGHT * component_gradient_norms["velocity_log_ratio"]
        + FROZEN_SWING_WEIGHT * component_gradient_norms["swing_surrogate_weighted"]
    ) / component_gradient_norms["pose_total"]
    if abs(balanced_clinical_to_pose - 1.0) > 0.02:
        raise AssertionError(
            f"Frozen clinical-gradient budget is not 1:1 with pose: {balanced_clinical_to_pose}"
        )

    return {
        "status": "PASS",
        "sequence_index": index,
        "sequence_subject": record["subject"],
        "sequence_frames": len(record["acc"]),
        "sequence_cycles": record["cycles"],
        "capacity_parameter_counts": capacity_counts,
        "losses_before_update": {
            name: float(value.detach().cpu()) for name, value in losses.items()
        },
        "protocol_total_gradient_l2_norm": trainable_grad_norm,
        "component_gradient_l2_norms_unweighted": component_gradient_norms,
        "frozen_loss_weights": {
            "pose": FROZEN_POSE_WEIGHT,
            "velocity": FROZEN_VELOCITY_WEIGHT,
            "swing": FROZEN_SWING_WEIGHT,
            "l2sp": FROZEN_L2SP_WEIGHT,
        },
        "sum_weighted_clinical_norms_over_pose_norm": balanced_clinical_to_pose,
        "l2sp_sum_uniform_1e-3_loss_by_capacity": l2sp_uniform_shift_loss,
        "l2sp_uniform_1e-3_per_parameter_gradient_by_capacity": l2sp_uniform_shift_gradient,
        "l2sp_uniform_1e-3_expected_per_parameter_gradient": expected_shift_gradient,
        "l2sp_uniform_1e-3_max_gradient_error": maximum_gradient_error,
        "swing_surrogate_gradient_support_180_frames": surrogate_gradient_support,
        "changed_trainable_parameters": changed_trainable,
        "max_trainable_parameter_change": max_trainable_change,
        "changed_frozen_parameters": changed_frozen,
    }


def run_gate(name: str, fn) -> dict:
    print(f"Running {name}...")
    try:
        result = fn()
        print(f"{name}: PASS")
        return result
    except Exception as exc:
        print(f"{name}: FAIL: {exc}")
        return {
            "status": "FAIL",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    data, manifest = load_dataset_and_manifest(args.dataset, args.manifest)
    splits = load_nested_splits(args.splits)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gates = {
        "gate_1_data_contract": run_gate(
            "Gate 1 (data contract)", lambda: gate_1_data_contract(data, manifest, splits)
        ),
        "gate_2_checkpoint_regression": run_gate(
            "Gate 2 (checkpoint regression)",
            lambda: gate_2_checkpoint_regression(
                data, manifest, device, args.mpjpe_tolerance_mm
            ),
        ),
        "gate_3_gradient_isolation": run_gate(
            "Gate 3 (gradient isolation)",
            lambda: gate_3_gradient_isolation(data, manifest, splits, device, args.seed),
        ),
    }
    overall = "PASS" if all(item["status"] == "PASS" for item in gates.values()) else "FAIL"
    payload = {
        "overall_status": overall,
        "device": str(device),
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "splits": str(args.splits),
        "pretrained_checkpoint": str(PRETRAINED_WEIGHTS_PATH),
        "sha256": {
            "pretrained_checkpoint": sha256_file(PRETRAINED_WEIGHTS_PATH),
            "dataset": sha256_file(args.dataset),
            "manifest": sha256_file(args.manifest),
            "splits": sha256_file(args.splits),
        },
        "seed": args.seed,
        "gates": gates,
    }
    report_path = args.out_dir / "sanity_gates.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Overall: {overall}; report={report_path}")
    if overall != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
