#!/usr/bin/env python3
"""Frozen ICNR 2026 statistical analysis and preview figure generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / "work/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(WORKSPACE / "work/cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260808
CORE_CONDITIONS = ("B0-C6", "P6", "C6", "B0-C5-M1", "P5", "C5")
M2_CONDITIONS = ("B0-C5-M2", "P5-M2", "C5-M2")
ENDPOINTS = {
    "clinical_distortion": "Clinical Distortion",
    "arm_swing_distortion": "Arm-swing distortion",
    "foot_lift_distortion": "Foot-lift distortion",
    "lower_body_rom_distortion": "Lower-body RoM distortion",
    "mpjpe_predicted_no_root_mm": "No-root MPJPE (mm)",
    "arm_asymmetry_absolute_error": "Arm-asymmetry absolute error",
}
HELD_OUT = (
    "arm_swing_distortion",
    "foot_lift_distortion",
    "lower_body_rom_distortion",
)
PRIMARY_CONTRASTS = {
    "Full6_CminusP": ("C6", "P6", "main"),
    "C5_M1_CminusP": ("C5", "P5", "main"),
    "C5_M2_CminusP": ("C5-M2", "P5-M2", "main"),
    "Full6_CminusP_own": ("C6", "P6", "own"),
    "C5_M1_CminusP_own": ("C5", "P5", "own"),
    "C5_CminusB0": ("C5", "B0-C5-M1", "main"),
    "P5_PminusB0": ("P5", "B0-C5-M1", "main"),
    "B0_M1minusFull6": ("B0-C5-M1", "B0-C6", "main"),
    "B0_M2minusFull6": ("B0-C5-M2", "B0-C6", "main"),
}
RATIO_METRICS = {
    "velocity": ("pred_joint_velocity_mean_mms", "gt_joint_velocity_mean_mms"),
    "leg_swing": ("pred_leg_swing_deg", "gt_leg_swing_deg"),
    "arm_swing": ("pred_arm_swing_deg", "gt_arm_swing_deg"),
    "foot_lift": ("pred_foot_lift_m", "gt_foot_lift_m"),
    "lower_body_rom": ("pred_lower_body_rom_deg", "gt_lower_body_rom_deg"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-walk", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--outer-unlock", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def enforce_analysis_guard(frame: pd.DataFrame, unlock_path: Path | None) -> str:
    roles = set(frame["role"].unique())
    if len(roles) != 1 or roles not in ({"inner_val"}, {"outer_test"}):
        raise PermissionError(f"Expected exactly one valid role, observed {sorted(roles)}")
    role = next(iter(roles))
    if role == "inner_val":
        return role
    if unlock_path is None or not unlock_path.exists():
        raise PermissionError("outer_test analysis is locked: provide --outer-unlock")
    payload = json.loads(unlock_path.read_text(encoding="utf-8"))
    if payload.get("approved") is not True:
        raise PermissionError("outer_test unlock manifest is not approved")
    observed = sha256_file(Path(__file__).resolve())
    if payload.get("analysis_script_sha256") != observed:
        raise PermissionError("outer_test unlock hash does not match this analysis script")
    return role


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def endpoint_valid(row: pd.Series, endpoint: str) -> bool:
    if endpoint == "clinical_distortion":
        columns = (
            "gt_joint_velocity_valid_frac",
            "pred_joint_velocity_valid_frac",
            "gt_leg_swing_valid_frac",
            "pred_leg_swing_valid_frac",
        )
    elif endpoint in {"arm_swing_distortion", "arm_asymmetry_absolute_error"}:
        columns = ("gt_arm_swing_valid_frac", "pred_arm_swing_valid_frac")
    elif endpoint == "foot_lift_distortion":
        columns = ("gt_foot_lift_valid_frac", "pred_foot_lift_valid_frac")
    elif endpoint == "lower_body_rom_distortion":
        columns = ("gt_lower_body_rom_valid_frac", "pred_lower_body_rom_valid_frac")
    else:
        columns = ()
    if not np.isfinite(row[endpoint]):
        return False
    return all(np.isclose(float(row[column]), 1.0, atol=1e-7) for column in columns)


def subject_values(frame: pd.DataFrame, value_col: str) -> pd.Series:
    return frame.groupby("subject", sort=True)[value_col].mean()


def bootstrap_mean(
    values: pd.Series,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return math.nan, math.nan, math.nan
    point = float(np.mean(array))
    draws = rng.choice(array, size=(replicates, array.size), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return point, float(low), float(high)


def condition_summaries(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    subject_rows = []
    for family in ("main", "own"):
        subset_family = frame[frame["checkpoint_family"] == family]
        for condition in sorted(subset_family["condition"].unique()):
            subset = subset_family[subset_family["condition"] == condition]
            for endpoint, label in ENDPOINTS.items():
                valid_mask = subset.apply(lambda row: endpoint_valid(row, endpoint), axis=1)
                valid = subset.loc[valid_mask]
                values = subject_values(valid, endpoint)
                point, low, high = bootstrap_mean(values, rng, replicates)
                rows.append(
                    {
                        "checkpoint_family": family,
                        "condition": condition,
                        "endpoint": endpoint,
                        "endpoint_label": label,
                        "mean": point,
                        "ci_low": low,
                        "ci_high": high,
                        "n_subjects": int(values.size),
                        "n_walk_rows_valid": int(valid.shape[0]),
                        "n_walk_rows_total": int(subset.shape[0]),
                    }
                )
                for subject, value in values.items():
                    subject_rows.append(
                        {
                            "checkpoint_family": family,
                            "condition": condition,
                            "endpoint": endpoint,
                            "subject": subject,
                            "subject_mean": float(value),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(subject_rows)


def paired_walks(
    frame: pd.DataFrame,
    clinical: str,
    pose: str,
    family: str,
    endpoint: str,
) -> tuple[pd.DataFrame, dict]:
    keys = ["fold", "index", "subject", "walk_id", "medication", "updrs"]
    left = frame[
        (frame["checkpoint_family"] == family) & (frame["condition"] == clinical)
    ].copy()
    right = frame[
        (frame["checkpoint_family"] == family) & (frame["condition"] == pose)
    ].copy()
    left["valid_clinical"] = left.apply(lambda row: endpoint_valid(row, endpoint), axis=1)
    right["valid_pose"] = right.apply(lambda row: endpoint_valid(row, endpoint), axis=1)
    merged = left[keys + [endpoint, "valid_clinical"]].merge(
        right[keys + [endpoint, "valid_pose"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_clinical", "_pose"),
        indicator=True,
    )
    complete = merged["_merge"] == "both"
    valid = complete & merged["valid_clinical"].fillna(False) & merged["valid_pose"].fillna(False)
    audit = {
        "n_union_walk_rows": int(merged.shape[0]),
        "n_complete_pairs": int(complete.sum()),
        "n_valid_pairs": int(valid.sum()),
        "n_missing_pair_member": int((~complete).sum()),
        "n_invalid_clinical": int((complete & ~merged["valid_clinical"].fillna(False)).sum()),
        "n_invalid_pose": int((complete & ~merged["valid_pose"].fillna(False)).sum()),
    }
    valid_frame = merged.loc[valid, keys].copy()
    valid_frame["difference"] = (
        merged.loc[valid, f"{endpoint}_clinical"].to_numpy(dtype=np.float64)
        - merged.loc[valid, f"{endpoint}_pose"].to_numpy(dtype=np.float64)
    )
    return valid_frame, audit


def paired_contrasts(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contrast_rows = []
    missing_rows = []
    subject_rows = []
    paired_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for contrast, (clinical, pose, family) in PRIMARY_CONTRASTS.items():
        for endpoint, label in ENDPOINTS.items():
            paired, audit = paired_walks(frame, clinical, pose, family, endpoint)
            values = subject_values(paired, "difference")
            point, low, high = bootstrap_mean(values, rng, replicates)
            paired_cache[(contrast, endpoint)] = paired
            contrast_rows.append(
                {
                    "contrast": contrast,
                    "checkpoint_family": family,
                    "clinical_or_first": clinical,
                    "pose_or_second": pose,
                    "endpoint": endpoint,
                    "endpoint_label": label,
                    "difference_direction": "first minus second; negative is better for all endpoints",
                    "mean_difference": point,
                    "ci_low": low,
                    "ci_high": high,
                    "n_subjects": int(values.size),
                    "n_valid_walk_pairs": int(paired.shape[0]),
                }
            )
            missing_rows.append({"contrast": contrast, "endpoint": endpoint, **audit})
            for subject, value in values.items():
                subject_rows.append(
                    {
                        "contrast": contrast,
                        "endpoint": endpoint,
                        "subject": subject,
                        "subject_mean_difference": float(value),
                    }
                )

    # Difference of treatment effects: (C5-P5) - (C6-P6), paired per fold/walk.
    for endpoint, label in ENDPOINTS.items():
        c5 = paired_cache[("C5_M1_CminusP", endpoint)]
        c6 = paired_cache[("Full6_CminusP", endpoint)]
        keys = ["fold", "index", "subject", "walk_id", "medication", "updrs"]
        merged = c5.merge(c6, on=keys, how="inner", suffixes=("_c5", "_c6"), validate="one_to_one")
        merged["difference"] = merged["difference_c5"] - merged["difference_c6"]
        values = subject_values(merged, "difference")
        point, low, high = bootstrap_mean(values, rng, replicates)
        contrast_rows.append(
            {
                "contrast": "C5minusFull6_treatment_effect",
                "checkpoint_family": "main",
                "clinical_or_first": "C5-P5",
                "pose_or_second": "C6-P6",
                "endpoint": endpoint,
                "endpoint_label": label,
                "difference_direction": "(C5-P5) minus (C6-P6); negative means larger C5 improvement",
                "mean_difference": point,
                "ci_low": low,
                "ci_high": high,
                "n_subjects": int(values.size),
                "n_valid_walk_pairs": int(merged.shape[0]),
            }
        )
        missing_rows.append(
            {
                "contrast": "C5minusFull6_treatment_effect",
                "endpoint": endpoint,
                "n_union_walk_rows": int(max(c5.shape[0], c6.shape[0])),
                "n_complete_pairs": int(merged.shape[0]),
                "n_valid_pairs": int(merged.shape[0]),
                "n_missing_pair_member": int(max(c5.shape[0], c6.shape[0]) - merged.shape[0]),
                "n_invalid_clinical": 0,
                "n_invalid_pose": 0,
            }
        )
        for subject, value in values.items():
            subject_rows.append(
                {
                    "contrast": "C5minusFull6_treatment_effect",
                    "endpoint": endpoint,
                    "subject": subject,
                    "subject_mean_difference": float(value),
                }
            )
    return pd.DataFrame(contrast_rows), pd.DataFrame(missing_rows), pd.DataFrame(subject_rows)


def bootstrap_spearman(
    state_frame: pd.DataFrame,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float, float, int, int]:
    subjects = sorted(state_frame["subject"].unique())
    if len(subjects) < 3:
        return math.nan, math.nan, math.nan, len(subjects), len(state_frame)
    point = float(spearmanr(state_frame["ratio"], state_frame["updrs"]).statistic)
    # A subject-cluster bootstrap is a multinomial set of subject copy counts.
    # Convert those counts to state-row weights and compute the exact Spearman
    # correlation of the expanded sample via weighted midranks.  This is
    # algebraically equivalent to concatenating duplicate subject rows but is
    # fast enough for 20,000 replicates across all conditions/descriptors.
    subject_lookup = {subject: i for i, subject in enumerate(subjects)}
    row_subject = np.asarray(
        [subject_lookup[value] for value in state_frame["subject"]], dtype=np.int64
    )
    counts = rng.multinomial(
        len(subjects),
        np.full(len(subjects), 1.0 / len(subjects)),
        size=replicates,
    ).astype(np.float64)
    weights = counts[:, row_subject]

    def weighted_midranks(values: np.ndarray) -> np.ndarray:
        _, inverse = np.unique(values, return_inverse=True)
        membership = np.eye(int(inverse.max()) + 1, dtype=np.float64)[inverse]
        group_weights = weights @ membership
        group_ranks = np.cumsum(group_weights, axis=1) - 0.5 * group_weights + 0.5
        return group_ranks[:, inverse]

    rank_x = weighted_midranks(state_frame["ratio"].to_numpy(dtype=np.float64))
    rank_y = weighted_midranks(state_frame["updrs"].to_numpy(dtype=np.float64))
    total_weight = weights.sum(axis=1)
    mean_x = (weights * rank_x).sum(axis=1) / total_weight
    mean_y = (weights * rank_y).sum(axis=1) / total_weight
    centered_x = rank_x - mean_x[:, None]
    centered_y = rank_y - mean_y[:, None]
    covariance = (weights * centered_x * centered_y).sum(axis=1)
    variance_x = (weights * centered_x**2).sum(axis=1)
    variance_y = (weights * centered_y**2).sum(axis=1)
    denominator = np.sqrt(variance_x * variance_y)
    draws = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 0,
    )
    finite = draws[np.isfinite(draws)]
    low, high = np.percentile(finite, [2.5, 97.5]) if finite.size else (math.nan, math.nan)
    return point, float(low), float(high), len(subjects), len(state_frame)


def severity_trends(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    state_outputs = []
    for (family, condition), subset in frame.groupby(["checkpoint_family", "condition"]):
        for metric, (pred_col, gt_col) in RATIO_METRICS.items():
            local = subset[["subject", "medication", "updrs", pred_col, gt_col]].copy()
            local["ratio"] = local[pred_col] / local[gt_col]
            local = local.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio", "updrs"])
            state = (
                local.groupby(["subject", "medication", "updrs"], as_index=False)["ratio"]
                .mean()
                .sort_values(["subject", "medication"])
            )
            rho, low, high, n_subjects, n_states = bootstrap_spearman(
                state, rng, replicates
            )
            rows.append(
                {
                    "checkpoint_family": family,
                    "condition": condition,
                    "metric": metric,
                    "spearman_rho": rho,
                    "ci_low": low,
                    "ci_high": high,
                    "n_subjects": n_subjects,
                    "n_subject_medication_states": n_states,
                    "inference": "descriptive cross-validated trend; no analytic p-value or delta-rho test",
                }
            )
            state.insert(0, "metric", metric)
            state.insert(0, "condition", condition)
            state.insert(0, "checkpoint_family", family)
            state_outputs.append(state)
    return pd.DataFrame(rows), pd.concat(state_outputs, ignore_index=True)


def checkpoint_reproduction_audit(frame: pd.DataFrame) -> dict:
    max_difference = 0.0
    compared = 0
    missing_files = []
    for (family, condition, checkpoint), subset in frame.groupby(
        ["checkpoint_family", "condition", "checkpoint_path"]
    ):
        # The training-time validation reference always used the frozen M1
        # input.  M2 rows deliberately test a different inference fill and
        # therefore are not checkpoint-reproduction candidates.
        if condition.startswith("B0") or condition.endswith("-M2"):
            continue
        checkpoint_path = Path(checkpoint)
        name = (
            "best_validation_per_walk.csv"
            if family == "main"
            else "best_own_objective_validation_per_walk.csv"
        )
        reference_path = checkpoint_path.parent / name
        if not reference_path.exists():
            missing_files.append(str(reference_path))
            continue
        reference = pd.read_csv(reference_path)[["index", "clinical_distortion"]]
        observed = subset[["index", "trainer_clinical_distortion"]].merge(
            reference, on="index", validate="one_to_one"
        )
        differences = (
            observed["trainer_clinical_distortion"] - observed["clinical_distortion"]
        ).abs()
        if len(differences):
            max_difference = max(max_difference, float(differences.max()))
            compared += len(differences)
    return {
        "n_rows_compared": compared,
        "max_abs_clinical_distortion_difference": max_difference,
        "missing_reference_files": missing_files,
    }


def branch_for_primary(row: pd.Series) -> str:
    if row["ci_high"] < 0:
        return "improvement_supported"
    if row["ci_low"] > 0:
        return "worsening_supported"
    return "inconclusive_ci_includes_zero"


def headline_branches(contrasts: pd.DataFrame) -> dict:
    result = {}
    for config, contrast in (("Full6", "Full6_CminusP"), ("C5", "C5_M1_CminusP")):
        subset = contrasts[contrasts["contrast"] == contrast].set_index("endpoint")
        primary = subset.loc["clinical_distortion"]
        primary_branch = branch_for_primary(primary)
        held = subset.loc[list(HELD_OUT)]
        if bool((held["ci_high"] < 0).all()):
            held_branch = "all_three_improved"
        elif bool((held["ci_low"] > 0).any()):
            harmed = held.index[held["ci_low"] > 0].tolist()
            held_branch = "nominal_potential_decline:" + ",".join(harmed)
        else:
            held_branch = "mixed_or_uncertain"
        headline_permission = (
            held_branch if primary_branch == "improvement_supported" else "none_primary_not_supported"
        )
        result[config] = {
            "primary_branch": primary_branch,
            "held_out_branch": held_branch,
            "headline_permission": headline_permission,
            "rules": (
                "All improvement claims require the relevant subject-clustered 95% CI to exclude zero. "
                "Potential declines from nominal 95% CIs are labelled potential, not confirmatory."
            ),
        }
    effect = contrasts[
        (contrasts["contrast"] == "C5minusFull6_treatment_effect")
        & (contrasts["endpoint"] == "clinical_distortion")
    ].iloc[0]
    if effect["ci_low"] <= 0 <= effect["ci_high"]:
        c5_vs_c6 = "no_evidence_that_treatment_effects_differ"
    elif effect["ci_high"] < 0:
        c5_vs_c6 = "larger_improvement_estimated_with_five_imus"
    else:
        c5_vs_c6 = "smaller_improvement_estimated_with_five_imus"
    result["C5_vs_Full6"] = {
        "branch": c5_vs_c6,
        "forbidden_words": ["comparable", "equivalent", "non-inferior", "retained X%", "same effect"],
    }
    return result


def make_figure(condition_summary: pd.DataFrame, out_dir: Path, role: str) -> None:
    summary = condition_summary[
        (condition_summary["checkpoint_family"] == "main")
    ].set_index(["condition", "endpoint"])
    core_colors = {
        "B0-C6": "#8C8C8C",
        "P6": "#4C78A8",
        "C6": "#E45756",
        "B0-C5-M1": "#BAB0AC",
        "P5": "#72B7B2",
        "C5": "#F58518",
    }
    labels = {
        "B0-C6": "B0-6",
        "P6": "P6",
        "C6": "C6",
        "B0-C5-M1": "B0-5\n(M1)",
        "P5": "P5",
        "C5": "C5",
        "B0-C5-M2": "B0-5\n(M2)",
        "P5-M2": "P5-M2",
        "C5-M2": "C5-M2",
    }
    ordered = list(CORE_CONDITIONS) + list(M2_CONDITIONS)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.25), constrained_layout=True)

    ax = axes[0]
    for x, condition in enumerate(ordered):
        row = summary.loc[(condition, "clinical_distortion")]
        color = core_colors.get(condition, "#6F6F6F")
        marker = "o" if condition in CORE_CONDITIONS else "D"
        face = color if condition in CORE_CONDITIONS else "white"
        ax.errorbar(
            x,
            row["mean"],
            yerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
            fmt=marker,
            color=color,
            markerfacecolor=face,
            markeredgewidth=1.5,
            capsize=3,
            markersize=6,
        )
    ax.set_xticks(range(len(ordered)), [labels[x] for x in ordered], fontsize=8)
    ax.set_ylabel("Clinical Distortion, |log ratio| (mean)")
    ax.set_title("A  Clinical feature distortion")
    ax.grid(axis="y", alpha=0.25)
    ax.axvline(5.5, color="#CCCCCC", linewidth=1, linestyle="--")

    ax = axes[1]
    for condition in ordered:
        cd = summary.loc[(condition, "clinical_distortion")]
        mpjpe = summary.loc[(condition, "mpjpe_predicted_no_root_mm")]
        color = core_colors.get(condition, "#6F6F6F")
        marker = "o" if condition in CORE_CONDITIONS else "D"
        face = color if condition in CORE_CONDITIONS else "white"
        ax.errorbar(
            mpjpe["mean"],
            cd["mean"],
            xerr=[[mpjpe["mean"] - mpjpe["ci_low"]], [mpjpe["ci_high"] - mpjpe["mean"]]],
            yerr=[[cd["mean"] - cd["ci_low"]], [cd["ci_high"] - cd["mean"]]],
            fmt=marker,
            color=color,
            markerfacecolor=face,
            markeredgewidth=1.5,
            capsize=2,
            markersize=6,
        )
        ax.annotate(labels[condition].replace("\n", " "), (mpjpe["mean"], cd["mean"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Predicted-joint no-root MPJPE (mm)")
    ax.set_ylabel("Clinical Distortion, |log ratio| (mean)")
    ax.set_title("B  Pose–clinical trade-off")
    ax.grid(alpha=0.25)

    title = (
        "ICNR 2026 analysis-chain preview (inner validation only)"
        if role == "inner_val"
        else "ICNR 2026 outer-test results"
    )
    stem = "main_figure_preview" if role == "inner_val" else "main_figure_outer_test"
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_dir / f"{stem}.png", dpi=240)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.per_walk)
    role = enforce_analysis_guard(frame, args.outer_unlock)
    expected_main = set(CORE_CONDITIONS) | set(M2_CONDITIONS)
    observed_main = set(
        frame.loc[frame["checkpoint_family"] == "main", "condition"].unique()
    )
    if observed_main != expected_main:
        raise ValueError(f"Main condition mismatch: expected {expected_main}, observed {observed_main}")

    rng = np.random.default_rng(args.seed)
    condition_summary, condition_subjects = condition_summaries(
        frame, rng, args.bootstrap_replicates
    )
    contrasts, missing, contrast_subjects = paired_contrasts(
        frame, rng, args.bootstrap_replicates
    )
    severity, severity_states = severity_trends(frame, rng, args.bootstrap_replicates)
    checkpoint_audit = checkpoint_reproduction_audit(frame)
    branches = headline_branches(contrasts)

    condition_summary.to_csv(args.out_dir / "condition_endpoint_summary.csv", index=False)
    condition_subjects.to_csv(args.out_dir / "condition_subject_values.csv", index=False)
    contrasts.to_csv(args.out_dir / "paired_contrasts.csv", index=False)
    contrast_subjects.to_csv(args.out_dir / "paired_contrast_subject_values.csv", index=False)
    missing.to_csv(args.out_dir / "paired_missingness_audit.csv", index=False)
    severity.to_csv(args.out_dir / "severity_trends.csv", index=False)
    severity_states.to_csv(args.out_dir / "severity_subject_state_values.csv", index=False)
    (args.out_dir / "checkpoint_reproduction_audit.json").write_text(
        json.dumps(checkpoint_audit, indent=2), encoding="utf-8"
    )
    (args.out_dir / "headline_branch_preview.json").write_text(
        json.dumps(branches, indent=2), encoding="utf-8"
    )
    make_figure(condition_summary, args.out_dir, role)

    metadata = {
        "analysis_role": (
            "inner-validation preview only; not a paper result"
            if role == "inner_val"
            else "one-time outer-test analysis"
        ),
        "bootstrap": {
            "unit": "subject cluster; all fold appearances and ON/OFF states move together",
            "n_unique_subjects": int(frame["subject"].nunique()),
            "n_fold_subject_appearances": int(
                frame[["fold", "subject"]].drop_duplicates().shape[0]
            ),
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "final_outer_test_expected_subjects": 23,
        },
        "paired_validity_rule": "GT AND clinical AND pose valid_frac == 1.0 per descriptor",
        "mpjpe_inference": (
            "descriptive paired difference with subject-clustered 95% CI; "
            "no non-inferiority claim"
        ),
        "severity_inference": (
            "descriptive per-arm Spearman rho with subject-clustered 95% CI; "
            "no delta-rho significance test and no analytic p-value"
        ),
        "held_out_multiple_comparisons": (
            "nominal unadjusted 95% CIs; all-three-improvement headline is an "
            "intersection-union rule; any decline is labelled potential, not confirmatory"
        ),
        "per_walk_sha256": sha256_file(args.per_walk),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    (args.out_dir / "analysis_run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main(parse_args())
