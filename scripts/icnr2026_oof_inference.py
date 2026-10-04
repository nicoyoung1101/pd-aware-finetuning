#!/usr/bin/env python3
"""Leakage-guarded ICNR 2026 inner-validation/outer-test inference.

The script implements the frozen nine-condition main matrix plus the four
own-objective checkpoint sensitivities.  Inner validation is always allowed.
Outer-test access requires an explicit unlock manifest whose recorded script
SHA-256 matches this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE
EVALUATION = PROJECT / "Evaluation"
TRANSPOSE = PROJECT / "TransPose"
for _path in (str(EVALUATION), str(TRANSPOSE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from analyze_gait_feature_preservation import (  # noqa: E402
    arm_swing_features,
    foot_lift_features,
    leg_swing_features,
)
from icnr2026_finetune_core import (  # noqa: E402
    PREDICTED_JOINTS,
    evaluation_leg_swing_amplitude,
    fold_subjects,
    forward_pose_6d,
    load_dataset_and_manifest,
    load_nested_splits,
    load_pretrained_net,
    make_body_model,
    pose_axis_angle_to_matrix,
    reduced_global_6d_to_local_pose,
    safe_torch_load,
)
from config import joint_set  # noqa: E402
from utils import normalize_and_concat  # noqa: E402


FPS = 60.0
HEAD_INDEX = 4
ROOT_INDEX = 5
LOWER_ROM_JOINTS = (1, 2, 4, 5)
MAIN_CONDITIONS = (
    "B0-C6",
    "P6",
    "C6",
    "B0-C5-M1",
    "P5",
    "C5",
    "B0-C5-M2",
    "P5-M2",
    "C5-M2",
)
OWN_CONDITIONS = ("P6", "C6", "P5", "C5")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("inner_val", "outer_test"), default="inner_val")
    parser.add_argument("--outer-unlock", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--full6-root",
        type=Path,
        default=WORKSPACE
        / "runs/Full6",
    )
    parser.add_argument(
        "--c5-root",
        type=Path,
        default=WORKSPACE
        / "runs/C5",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fold", type=int, action="append", choices=range(6))
    parser.add_argument("--limit-walks-per-fold", type=int, default=0)
    return parser.parse_args()


def enforce_partition_guard(args: argparse.Namespace) -> None:
    if args.role != "outer_test":
        return
    if args.outer_unlock is None or not args.outer_unlock.exists():
        raise PermissionError("outer_test is locked: provide a valid --outer-unlock manifest")
    payload = json.loads(args.outer_unlock.read_text(encoding="utf-8"))
    observed = sha256_file(Path(__file__).resolve())
    if payload.get("approved") is not True:
        raise PermissionError("outer_test unlock manifest is not approved")
    if payload.get("inference_script_sha256") != observed:
        raise PermissionError("outer_test unlock hash does not match this inference script")


def project_to_so3(matrix: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(matrix.double())
    rotation = u @ vh
    if torch.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation.float()


def m1_input(acc: torch.Tensor, ori: torch.Tensor) -> torch.Tensor:
    acc = acc.float().clone()
    ori = ori.float().clone()
    acc[:, HEAD_INDEX] = acc[:, ROOT_INDEX]
    ori[:, HEAD_INDEX] = ori[:, ROOT_INDEX]
    return normalize_and_concat(acc, ori)


def full6_input(acc: torch.Tensor, ori: torch.Tensor) -> torch.Tensor:
    return normalize_and_concat(acc.float(), ori.float())


def estimate_m2(
    data: dict,
    manifest: list[dict],
    calibration_subjects: list[str],
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Subject-equal constant normalized-head fill estimated without leakage."""
    subject_acc: list[torch.Tensor] = []
    subject_ori: list[torch.Tensor] = []
    per_subject_frames: dict[str, int] = {}
    for subject in sorted(calibration_subjects):
        indices = [
            i for i, item in enumerate(manifest) if str(item["subject_id"]) == subject
        ]
        if not indices:
            raise ValueError(f"No calibration walks for {subject}")
        acc_sum = torch.zeros(3, dtype=torch.float64)
        ori_sum = torch.zeros(3, 3, dtype=torch.float64)
        n_frames = 0
        for index in indices:
            imu = full6_input(data["acc"][index], data["ori"][index]).double()
            acc_sum += imu[:, HEAD_INDEX * 3 : (HEAD_INDEX + 1) * 3].sum(dim=0)
            start = 18 + HEAD_INDEX * 9
            ori_sum += imu[:, start : start + 9].reshape(-1, 3, 3).sum(dim=0)
            n_frames += int(imu.shape[0])
        subject_acc.append((acc_sum / n_frames).float())
        subject_ori.append((ori_sum / n_frames).float())
        per_subject_frames[subject] = n_frames
    acc_fill = torch.stack(subject_acc).mean(dim=0)
    euclidean_ori_mean = torch.stack(subject_ori).mean(dim=0)
    ori_fill = project_to_so3(euclidean_ori_mean)
    audit = {
        "calibration_subjects": sorted(calibration_subjects),
        "n_subjects": len(calibration_subjects),
        "per_subject_frames": per_subject_frames,
        "head_acc_normalized": acc_fill.tolist(),
        "head_ori_relative_so3": ori_fill.tolist(),
        "orientation_det": float(torch.linalg.det(ori_fill)),
        "orientation_orthogonality_max_abs": float(
            (ori_fill.T @ ori_fill - torch.eye(3)).abs().max()
        ),
        "distance_from_m1_acc_l2": float(torch.linalg.vector_norm(acc_fill)),
        "distance_from_m1_ori_fro": float(
            torch.linalg.matrix_norm(ori_fill - torch.eye(3))
        ),
        "weighting": "frames within subject, then subjects equally",
    }
    return acc_fill, ori_fill, audit


def m2_input(
    acc: torch.Tensor,
    ori: torch.Tensor,
    acc_fill: torch.Tensor,
    ori_fill: torch.Tensor,
) -> torch.Tensor:
    imu = full6_input(acc, ori).clone()
    imu[:, HEAD_INDEX * 3 : (HEAD_INDEX + 1) * 3] = acc_fill
    start = 18 + HEAD_INDEX * 9
    imu[:, start : start + 9] = ori_fill.reshape(1, 9)
    return imu


def checkpoint_path(
    full6_root: Path,
    c5_root: Path,
    fold: int,
    condition: str,
    family: str,
) -> Path | None:
    if condition.startswith("B0"):
        return None
    root = full6_root if condition in {"P6", "C6"} else c5_root
    base = condition.split("-")[0]
    filename = "best_checkpoint.pt" if family == "main" else "best_own_objective_checkpoint.pt"
    path = root / f"fold_{fold}" / f"{base}_S1_seed20260802" / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_net(checkpoint: Path | None) -> tuple[torch.nn.Module, dict]:
    net = load_pretrained_net(torch.device("cpu"))
    meta = {"best_epoch": None, "selection_metric": "healthy pretrained"}
    if checkpoint is not None:
        payload = safe_torch_load(checkpoint, map_location="cpu")
        net.load_state_dict(payload["model_state_dict"])
        meta = {
            "best_epoch": int(payload["best_epoch"]),
            "selection_metric": str(payload["selection_metric"]),
        }
    net.eval()
    return net, meta


def robust_range(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 5:
        return math.nan
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def lower_body_rom_deg(local_pose: torch.Tensor) -> tuple[float, float]:
    matrices = local_pose.detach().cpu().double().numpy()
    per_joint = []
    validity = []
    for joint in LOWER_ROM_JOINTS:
        trace = np.trace(matrices[:, joint], axis1=-2, axis2=-1)
        angle = np.rad2deg(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
        validity.append(float(np.mean(np.isfinite(angle))))
        per_joint.append(robust_range(angle))
    return float(np.nanmean(per_joint)), float(min(validity))


def velocity_mms(joints: np.ndarray) -> tuple[float, float]:
    aligned = joints - joints[:, :1]
    velocity = np.diff(aligned, axis=0) * FPS
    speed = np.linalg.norm(velocity[:, 1:], axis=-1)
    return float(np.nanmean(speed) * 1000.0), float(np.mean(np.isfinite(speed)))


def pose_features(joints: np.ndarray, local_pose: torch.Tensor) -> dict:
    leg = leg_swing_features(joints)
    arm = arm_swing_features(joints)
    foot = foot_lift_features(joints)
    rom, rom_valid = lower_body_rom_deg(local_pose)
    velocity, velocity_valid = velocity_mms(joints)
    foot_values = joints[:, (10, 11), 1] - joints[:, :1, 1]
    return {
        "joint_velocity_mean_mms": velocity,
        "joint_velocity_valid_frac": velocity_valid,
        "leg_swing_deg": float(leg["leg_amp_mean_p95p5_deg"]),
        "leg_swing_valid_frac": float(
            min(leg["leg_amp_left_valid_frac"], leg["leg_amp_right_valid_frac"])
        ),
        "arm_swing_deg": float(arm["arm_amp_mean_p95p5_deg"]),
        "arm_swing_asymmetry": float(arm["arm_amp_asym_p95p5"]),
        "arm_swing_valid_frac": float(
            min(
                arm["torso_frame_valid_frac"],
                arm["arm_amp_left_valid_frac"],
                arm["arm_amp_right_valid_frac"],
            )
        ),
        "foot_lift_m": float(foot["foot_lift_mean_p95p5_m"]),
        "foot_lift_valid_frac": float(np.mean(np.isfinite(foot_values))),
        "lower_body_rom_deg": rom,
        "lower_body_rom_valid_frac": rom_valid,
    }


def absolute_log_ratio(pred: float, gt: float, eps: float) -> float:
    return float(abs(math.log((pred + eps) / (gt + eps))))


def make_metric_row(
    gt_local: torch.Tensor,
    gt_joints_t: torch.Tensor,
    pred_local: torch.Tensor,
    pred_joints_t: torch.Tensor,
) -> tuple[dict, dict]:
    gt_joints = gt_joints_t.detach().cpu().numpy()
    pred_joints = pred_joints_t.detach().cpu().numpy()
    gt = pose_features(gt_joints, gt_local)
    pred = pose_features(pred_joints, pred_local)

    gt_aligned = gt_joints - gt_joints[:, :1]
    pred_aligned = pred_joints - pred_joints[:, :1]
    per_joint = np.linalg.norm(pred_aligned - gt_aligned, axis=-1) * 1000.0
    mpjpe = float(np.nanmean(per_joint[:, PREDICTED_JOINTS]))

    velocity_dist = absolute_log_ratio(
        pred["joint_velocity_mean_mms"], gt["joint_velocity_mean_mms"], 1e-3
    )
    degree_eps = math.degrees(1e-6)
    leg_dist = absolute_log_ratio(pred["leg_swing_deg"], gt["leg_swing_deg"], degree_eps)
    arm_dist = absolute_log_ratio(pred["arm_swing_deg"], gt["arm_swing_deg"], 1e-8)
    foot_dist = absolute_log_ratio(pred["foot_lift_m"], gt["foot_lift_m"], 1e-8)
    rom_dist = absolute_log_ratio(
        pred["lower_body_rom_deg"], gt["lower_body_rom_deg"], 1e-8
    )

    torch_gt_swing = float(evaluation_leg_swing_amplitude(gt_joints_t))
    torch_pred_swing = float(evaluation_leg_swing_amplitude(pred_joints_t))
    torch_gt_velocity = float(
        torch.linalg.vector_norm(
            torch.diff(gt_joints_t - gt_joints_t[:, :1], dim=0) * FPS, dim=-1
        ).mean()
    )
    torch_pred_velocity = float(
        torch.linalg.vector_norm(
            torch.diff(pred_joints_t - pred_joints_t[:, :1], dim=0) * FPS, dim=-1
        ).mean()
    )
    trainer_cd = 0.5 * (
        absolute_log_ratio(torch_pred_velocity, torch_gt_velocity, 1e-6)
        + absolute_log_ratio(torch_pred_swing, torch_gt_swing, 1e-6)
    )
    benchmark_cd = 0.5 * (velocity_dist + leg_dist)

    row = {
        **{f"gt_{key}": value for key, value in gt.items()},
        **{f"pred_{key}": value for key, value in pred.items()},
        "velocity_distortion": velocity_dist,
        "leg_swing_distortion": leg_dist,
        "clinical_distortion": benchmark_cd,
        "trainer_clinical_distortion": trainer_cd,
        "arm_swing_distortion": arm_dist,
        "foot_lift_distortion": foot_dist,
        "lower_body_rom_distortion": rom_dist,
        "arm_asymmetry_absolute_error": float(
            abs(pred["arm_swing_asymmetry"] - gt["arm_swing_asymmetry"])
        ),
        "mpjpe_predicted_no_root_mm": mpjpe,
    }
    alignment = {
        "gt_leg_swing_abs_diff_deg": abs(gt["leg_swing_deg"] - math.degrees(torch_gt_swing)),
        "pred_leg_swing_abs_diff_deg": abs(
            pred["leg_swing_deg"] - math.degrees(torch_pred_swing)
        ),
        "clinical_distortion_abs_diff": abs(benchmark_cd - trainer_cd),
    }
    return row, alignment


def condition_plan(family: str) -> list[tuple[str, str, str]]:
    if family == "own":
        return [
            ("P6", "P6", "Full6"),
            ("C6", "C6", "Full6"),
            ("P5", "P5", "M1"),
            ("C5", "C5", "M1"),
        ]
    return [
        ("B0-C6", "B0-C6", "Full6"),
        ("P6", "P6", "Full6"),
        ("C6", "C6", "Full6"),
        ("B0-C5-M1", "B0-C5-M1", "M1"),
        ("P5", "P5", "M1"),
        ("C5", "C5", "M1"),
        ("B0-C5-M2", "B0-C5-M2", "M2"),
        ("P5-M2", "P5", "M2"),
        ("C5-M2", "C5", "M2"),
    ]


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    enforce_partition_guard(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    data, manifest = load_dataset_and_manifest()
    splits = load_nested_splits()
    body_model = make_body_model(torch.device("cpu"))
    folds = args.fold if args.fold else list(range(6))
    rows: list[dict] = []
    alignment_max = {
        "gt_leg_swing_abs_diff_deg": 0.0,
        "pred_leg_swing_abs_diff_deg": 0.0,
        "clinical_distortion_abs_diff": 0.0,
    }
    m2_audit: dict[str, dict] = {}
    m1_audit_max = {"acc": 0.0, "ori": 0.0}

    for fold in folds:
        eval_subjects = fold_subjects(splits, fold, args.role)
        if args.role == "inner_val":
            calibration_subjects = fold_subjects(splits, fold, "inner_train")
        else:
            calibration_subjects = sorted(
                set(fold_subjects(splits, fold, "inner_train"))
                | set(fold_subjects(splits, fold, "inner_val"))
            )
        eval_indices = [
            i for i, item in enumerate(manifest) if str(item["subject_id"]) in eval_subjects
        ]
        if args.limit_walks_per_fold:
            eval_indices = eval_indices[: args.limit_walks_per_fold]
        acc_fill, ori_fill, fold_m2_audit = estimate_m2(
            data, manifest, calibration_subjects
        )
        m2_audit[str(fold)] = fold_m2_audit

        input_cache: dict[tuple[int, str], torch.Tensor] = {}
        gt_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for index in eval_indices:
            acc, ori = data["acc"][index], data["ori"][index]
            input_cache[(index, "Full6")] = full6_input(acc, ori)
            input_cache[(index, "M1")] = m1_input(acc, ori)
            input_cache[(index, "M2")] = m2_input(acc, ori, acc_fill, ori_fill)
            head_acc = input_cache[(index, "M1")][:, HEAD_INDEX * 3 : (HEAD_INDEX + 1) * 3]
            head_ori = input_cache[(index, "M1")][
                :, 18 + HEAD_INDEX * 9 : 18 + (HEAD_INDEX + 1) * 9
            ].reshape(-1, 3, 3)
            m1_audit_max["acc"] = max(m1_audit_max["acc"], float(head_acc.abs().max()))
            m1_audit_max["ori"] = max(
                m1_audit_max["ori"],
                float((head_ori - torch.eye(3)).abs().max()),
            )
            gt_local = pose_axis_angle_to_matrix(data["pose"][index])
            _, gt_joints = body_model.forward_kinematics(
                gt_local, shape=None, tran=None, calc_mesh=False
            )
            gt_cache[index] = (gt_local, gt_joints)

        for family in ("main", "own"):
            for label, checkpoint_condition, input_variant in condition_plan(family):
                checkpoint = checkpoint_path(
                    args.full6_root,
                    args.c5_root,
                    fold,
                    checkpoint_condition,
                    family,
                )
                net, net_meta = load_net(checkpoint)
                for number, index in enumerate(eval_indices, start=1):
                    imu = input_cache[(index, input_variant)]
                    predicted_6d = forward_pose_6d(net, imu)
                    predicted_local = reduced_global_6d_to_local_pose(
                        net, imu[:, -9:].reshape(-1, 3, 3), predicted_6d
                    )
                    _, predicted_joints = body_model.forward_kinematics(
                        predicted_local, shape=None, tran=None, calc_mesh=False
                    )
                    gt_local, gt_joints = gt_cache[index]
                    metric_row, alignment = make_metric_row(
                        gt_local, gt_joints, predicted_local, predicted_joints
                    )
                    for key, value in alignment.items():
                        alignment_max[key] = max(alignment_max[key], float(value))
                    item = manifest[index]
                    rows.append(
                        {
                            "fold": fold,
                            "role": args.role,
                            "index": index,
                            "subject": str(item["subject_id"]),
                            "walk_id": str(item["walk_id"]),
                            "medication": str(item["medication"]),
                            "updrs": int(item["UPDRS_GAIT"]),
                            "condition": label,
                            "checkpoint_family": family,
                            "input_variant": input_variant,
                            "sensor_count": 6 if input_variant == "Full6" else 5,
                            "trained": not label.startswith("B0"),
                            "best_epoch": net_meta["best_epoch"],
                            "selection_metric": net_meta["selection_metric"],
                            "checkpoint_path": str(checkpoint) if checkpoint else str(
                                PROJECT / "TransPose/data/weights.pt"
                            ),
                            "n_frames": int(data["pose"][index].shape[0]),
                            "duration_s": float(data["pose"][index].shape[0] / FPS),
                            **metric_row,
                        }
                    )
                print(
                    f"fold={fold} role={args.role} family={family} condition={label} "
                    f"walks={len(eval_indices)}",
                    flush=True,
                )
                del net

    frame = pd.DataFrame(rows).sort_values(
        ["checkpoint_family", "fold", "condition", "subject", "index"]
    )
    frame.to_csv(args.out_dir / "per_walk_metrics.csv", index=False)
    descriptor_alignment = {
        "scope": f"{args.role} only",
        "max_absolute_differences": alignment_max,
        "velocity_note": (
            "trainer all-24-joint m/s and benchmark 23-no-root mm/s differ only by "
            "23/24 and 1000 constants, which cancel in the log ratio"
        ),
        "leg_swing_note": (
            "NumPy benchmark unwrap/percentile and Torch evaluation unwrap/quantile "
            "were applied to the same joint trajectories"
        ),
    }
    (args.out_dir / "descriptor_alignment.json").write_text(
        json.dumps(descriptor_alignment, indent=2), encoding="utf-8"
    )
    (args.out_dir / "m2_fold_calibration.json").write_text(
        json.dumps(m2_audit, indent=2), encoding="utf-8"
    )
    run_meta = {
        "role": args.role,
        "folds": folds,
        "main_conditions": list(MAIN_CONDITIONS),
        "own_checkpoint_conditions": list(OWN_CONDITIONS),
        "n_rows": len(frame),
        "n_unique_subjects": int(frame["subject"].nunique()),
        "n_fold_subject_appearances": int(
            frame[["fold", "subject"]].drop_duplicates().shape[0]
        ),
        "m1_constant_channel_max_abs_residual": m1_audit_max,
        "m2_calibration_rule": (
            "inner_val: inner_train only; outer_test: all non-test subjects; "
            "frames within subject then subjects equally; orientation projected to SO(3)"
        ),
        "mpjpe_joint_indices": list(PREDICTED_JOINTS),
        "lower_body_rom_joint_indices": list(LOWER_ROM_JOINTS),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    (args.out_dir / "inference_run.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    run(parse_args())
