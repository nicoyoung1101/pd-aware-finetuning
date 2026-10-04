#!/usr/bin/env python3
"""Analyze the five benchmark-selected PD descriptors for frozen P/C models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPLICATES = 20_000
SEED = 20260815
METRICS = {
    "Lower-body velocity": {
        "gt": "gt_lower_body_velocity_mms",
        "pred": "pred_lower_body_velocity_mms",
        "dist": "lower_body_velocity_distortion",
        "benchmark_gt_rho": -0.7132481051746247,
    },
    "Foot lift": {
        "gt": "gt_foot_lift_m",
        "pred": "pred_foot_lift_m",
        "dist": "foot_lift_distortion",
        "benchmark_gt_rho": -0.6972746812028464,
    },
    "Lower-body RoM": {
        "gt": "gt_lower_body_rom_deg",
        "pred": "pred_lower_body_rom_deg",
        "dist": "lower_body_rom_distortion",
        "benchmark_gt_rho": -0.6736617935923914,
    },
    "Joint velocity": {
        "gt": "gt_joint_velocity_mean_mms",
        "pred": "pred_joint_velocity_mean_mms",
        "dist": "velocity_distortion",
        "benchmark_gt_rho": -0.6681058200369903,
    },
    "Leg swing": {
        "gt": "gt_leg_swing_deg",
        "pred": "pred_leg_swing_deg",
        "dist": "leg_swing_distortion",
        "benchmark_gt_rho": -0.6535213894540622,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-per-walk", type=Path, required=True)
    parser.add_argument("--lower-body-velocity", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("subject", sort=True)[column].mean()


def bootstrap_mean(
    values: pd.Series,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float]:
    array = values.to_numpy(dtype=np.float64)
    point = float(array.mean())
    draws = rng.choice(array, size=(REPLICATES, len(array)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    # Five descriptor comparisons per sensor configuration: Bonferroni
    # family-wise 95% coverage corresponds to two-sided 99% intervals.
    fwer_low, fwer_high = np.percentile(draws, [0.5, 99.5])
    return point, float(low), float(high), float(fwer_low), float(fwer_high)


def state_rho(frame: pd.DataFrame, value: str) -> float:
    state = frame.groupby(
        ["subject", "medication", "updrs"], as_index=False
    )[value].mean()
    return float(spearmanr(state[value], state["updrs"]).statistic)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(args.main_per_walk)
    main = main[
        main["checkpoint_family"].eq("main")
        & main["condition"].isin(["P6", "C6", "P5", "C5"])
    ].copy()
    lower = pd.read_csv(args.lower_body_velocity)
    keys = [
        "fold", "index", "subject", "walk_id", "medication", "updrs",
        "condition", "checkpoint_family",
    ]
    main = main.merge(
        lower[
            keys
            + [
                "gt_lower_body_velocity_mms",
                "pred_lower_body_velocity_mms",
                "lower_body_velocity_distortion",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    expected_fold_walks = {0: 108, 1: 122, 2: 151, 3: 139, 4: 145, 5: 116}
    audit = {}
    for condition in ["P6", "C6", "P5", "C5"]:
        subset = main[main["condition"].eq(condition)]
        fold_counts = subset.groupby("fold")["index"].nunique().to_dict()
        if fold_counts != expected_fold_walks:
            raise AssertionError(f"Unexpected fold counts for {condition}: {fold_counts}")
        if subset["index"].nunique() != 781 or subset["subject"].nunique() != 23:
            raise AssertionError(f"Incomplete OOF data for {condition}")
        audit[condition] = {
            "walks": int(subset["index"].nunique()),
            "subjects": int(subset["subject"].nunique()),
            "fold_walks": {str(k): int(v) for k, v in fold_counts.items()},
        }

    rows = []
    correlation_rows = []
    rng = np.random.default_rng(SEED)
    for descriptor, columns in METRICS.items():
        for sensors, pose_condition, clinical_condition in [
            ("Six", "P6", "C6"),
            ("Five", "P5", "C5"),
        ]:
            pose = main[main["condition"].eq(pose_condition)]
            clinical = main[main["condition"].eq(clinical_condition)]
            pair_keys = ["fold", "index", "subject", "walk_id", "medication", "updrs"]
            paired = clinical[pair_keys + [columns["dist"]]].merge(
                pose[pair_keys + [columns["dist"]]],
                on=pair_keys,
                validate="one_to_one",
                suffixes=("_clinical", "_pose"),
            )
            paired["difference"] = (
                paired[f'{columns["dist"]}_clinical']
                - paired[f'{columns["dist"]}_pose']
            )
            differences = subject_mean(paired, "difference")
            point, low, high, fwer_low, fwer_high = bootstrap_mean(differences, rng)
            pose_values = subject_mean(pose, columns["dist"])
            clinical_values = subject_mean(clinical, columns["dist"])
            rows.append(
                {
                    "descriptor": descriptor,
                    "benchmark_gt_updrs_rho": columns["benchmark_gt_rho"],
                    "sensors": sensors,
                    "pose_mean_distortion": float(pose_values.mean()),
                    "clinical_mean_distortion": float(clinical_values.mean()),
                    "clinical_minus_pose": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bonferroni_99_ci_low": fwer_low,
                    "bonferroni_99_ci_high": fwer_high,
                    "n_subjects": int(differences.size),
                    "n_walk_pairs": int(paired.shape[0]),
                    "interpretation": (
                        "nominal improvement" if high < 0
                        else "nominal worsening" if low > 0
                        else "uncertain"
                    ),
                    "familywise_interpretation": (
                        "improvement" if fwer_high < 0
                        else "worsening" if fwer_low > 0
                        else "uncertain"
                    ),
                }
            )

            gt_rho = state_rho(pose, columns["gt"])
            if not np.isclose(
                gt_rho, columns["benchmark_gt_rho"], atol=1e-12, rtol=0.0
            ):
                raise AssertionError(
                    f"{descriptor} GT rho {gt_rho} does not reproduce benchmark "
                    f"{columns['benchmark_gt_rho']}"
                )
            pose_rho = state_rho(pose, columns["pred"])
            clinical_rho = state_rho(clinical, columns["pred"])
            correlation_rows.append(
                {
                    "descriptor": descriptor,
                    "sensors": sensors,
                    "benchmark_gt_updrs_rho": columns["benchmark_gt_rho"],
                    "observed_gt_updrs_rho": gt_rho,
                    "pose_pred_updrs_rho": pose_rho,
                    "clinical_pred_updrs_rho": clinical_rho,
                    "pose_absolute_rho_gap_from_gt": abs(pose_rho - gt_rho),
                    "clinical_absolute_rho_gap_from_gt": abs(clinical_rho - gt_rho),
                    "closer_to_gt": (
                        "C" if abs(clinical_rho - gt_rho) < abs(pose_rho - gt_rho)
                        else "P" if abs(clinical_rho - gt_rho) > abs(pose_rho - gt_rho)
                        else "tie"
                    ),
                    "inference": "descriptive only; no delta-rho significance test",
                }
            )

    result = pd.DataFrame(rows)
    correlations = pd.DataFrame(correlation_rows)
    result.to_csv(args.out_dir / "five_descriptor_paired_effects.csv", index=False)
    correlations.to_csv(
        args.out_dir / "five_descriptor_updrs_correlations.csv", index=False
    )
    (args.out_dir / "analysis_audit.json").write_text(
        json.dumps(
            {
                "analysis_status": "teacher-requested post-hoc benchmark-aligned analysis",
                "selection_rule": (
                    "five largest absolute GT--MDS-UPDRS gait Spearman correlations "
                    "in the pre-existing CARE-PD benchmark"
                ),
                "selection_uses_P_or_C_results": False,
                "comparison": (
                    "subject-equal paired C-minus-P difference in per-walk absolute "
                    "log(predicted/GT) distortion"
                ),
                "bootstrap": {
                    "replicates": REPLICATES,
                    "seed": SEED,
                    "cluster": "subject",
                    "intervals": (
                        "unadjusted nominal 95% CIs plus Bonferroni 99% CIs "
                        "for five descriptors within each sensor configuration; exploratory"
                    ),
                },
                "outer_test_audit": audit,
                "input_sha256": {
                    str(args.main_per_walk): sha256_file(args.main_per_walk),
                    str(args.lower_body_velocity): sha256_file(args.lower_body_velocity),
                },
                "script_sha256": sha256_file(Path(__file__).resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run(parse_args())
