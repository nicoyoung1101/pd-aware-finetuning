#!/usr/bin/env python3
"""Teacher-requested post-hoc inference for the missing benchmark descriptor.

This script evaluates lower-body velocity for the already frozen main P/C
checkpoints.  It does not train, select, or alter a model.  The descriptor was
defined before the ICNR fine-tuning study in the CARE-PD cross-model benchmark
as the mean root-aligned velocity of the predicted hip/knee joints [1, 2, 4, 5].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
WORK = WORKSPACE / "scripts"
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

from icnr2026_oof_inference import (  # noqa: E402
    FPS,
    checkpoint_path,
    fold_subjects,
    forward_pose_6d,
    full6_input,
    load_dataset_and_manifest,
    load_nested_splits,
    load_net,
    m1_input,
    make_body_model,
    pose_axis_angle_to_matrix,
    reduced_global_6d_to_local_pose,
)


LOWER_BODY_JOINTS = (1, 2, 4, 5)
CONDITIONS = (
    ("B0-C6", "B0-C6", "Full6"),
    ("P6", "P6", "Full6"),
    ("C6", "C6", "Full6"),
    ("B0-C5-M1", "B0-C5-M1", "M1"),
    ("P5", "P5", "M1"),
    ("C5", "C5", "M1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lower_body_velocity_mms(joints: torch.Tensor) -> tuple[float, float]:
    aligned = joints - joints[:, :1]
    velocity = torch.diff(aligned, dim=0) * FPS
    speed = torch.linalg.vector_norm(velocity[:, LOWER_BODY_JOINTS], dim=-1)
    finite = torch.isfinite(speed)
    return float(speed[finite].mean() * 1000.0), float(finite.float().mean())


def absolute_log_ratio(predicted: float, ground_truth: float) -> float:
    return float(abs(math.log((predicted + 1e-3) / (ground_truth + 1e-3))))


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    data, manifest = load_dataset_and_manifest()
    splits = load_nested_splits()
    body_model = make_body_model(torch.device("cpu"))
    folds = args.fold if args.fold else list(range(6))
    rows: list[dict] = []

    for fold in folds:
        subjects = fold_subjects(splits, fold, "outer_test")
        indices = [
            i for i, item in enumerate(manifest)
            if str(item["subject_id"]) in subjects
        ]
        inputs: dict[tuple[int, str], torch.Tensor] = {}
        ground_truth: dict[int, float] = {}
        for index in indices:
            inputs[(index, "Full6")] = full6_input(
                data["acc"][index], data["ori"][index]
            )
            inputs[(index, "M1")] = m1_input(
                data["acc"][index], data["ori"][index]
            )
            local = pose_axis_angle_to_matrix(data["pose"][index])
            _, joints = body_model.forward_kinematics(
                local, shape=None, tran=None, calc_mesh=False
            )
            value, valid = lower_body_velocity_mms(joints)
            if not math.isclose(valid, 1.0, abs_tol=1e-7):
                raise ValueError(f"Invalid GT lower-body velocity at index {index}")
            ground_truth[index] = value

        for label, checkpoint_condition, input_variant in CONDITIONS:
            checkpoint = checkpoint_path(
                args.full6_root,
                args.c5_root,
                fold,
                checkpoint_condition,
                "main",
            )
            net, meta = load_net(checkpoint)
            for index in indices:
                imu = inputs[(index, input_variant)]
                predicted_6d = forward_pose_6d(net, imu)
                predicted_local = reduced_global_6d_to_local_pose(
                    net, imu[:, -9:].reshape(-1, 3, 3), predicted_6d
                )
                _, predicted_joints = body_model.forward_kinematics(
                    predicted_local, shape=None, tran=None, calc_mesh=False
                )
                predicted, valid = lower_body_velocity_mms(predicted_joints)
                if not math.isclose(valid, 1.0, abs_tol=1e-7):
                    raise ValueError(
                        f"Invalid predicted lower-body velocity at index {index}"
                    )
                item = manifest[index]
                gt = ground_truth[index]
                rows.append(
                    {
                        "fold": fold,
                        "index": index,
                        "subject": str(item["subject_id"]),
                        "walk_id": str(item["walk_id"]),
                        "medication": str(item["medication"]),
                        "updrs": int(item["UPDRS_GAIT"]),
                        "condition": label,
                        "checkpoint_family": "main",
                        "checkpoint_path": str(checkpoint),
                        "best_epoch": meta["best_epoch"],
                        "gt_lower_body_velocity_mms": gt,
                        "pred_lower_body_velocity_mms": predicted,
                        "lower_body_velocity_distortion": absolute_log_ratio(
                            predicted, gt
                        ),
                    }
                )
            print(
                f"fold={fold} condition={label} walks={len(indices)}",
                flush=True,
            )
            del net

    frame = pd.DataFrame(rows).sort_values(
        ["fold", "condition", "subject", "index"]
    )
    output = args.out_dir / "lower_body_velocity_per_walk.csv"
    frame.to_csv(output, index=False)
    metadata = {
        "analysis_status": "teacher-requested post-hoc benchmark-aligned analysis",
        "training_or_checkpoint_selection_performed": False,
        "conditions": [item[0] for item in CONDITIONS],
        "folds": folds,
        "n_rows": int(frame.shape[0]),
        "n_unique_walks": int(frame[["fold", "index"]].drop_duplicates().shape[0]),
        "n_unique_subjects": int(frame["subject"].nunique()),
        "descriptor": {
            "name": "lower-body velocity",
            "joint_indices": list(LOWER_BODY_JOINTS),
            "definition": (
                "mean norm of first-difference velocity at 60 Hz after root "
                "alignment, converted to mm/s"
            ),
            "comparison": "absolute log(predicted/ground-truth) distortion",
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "output_sha256": sha256_file(output),
    }
    (args.out_dir / "inference_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    run(parse_args())
