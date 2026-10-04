#!/usr/bin/env python3
"""
Medication on/off paired analysis for CARE-PD model comparison.

Main question:
    When the same subject changes state between medication off/on, does the
    model preserve the GT kinematic change, or flatten it?

Primary statistic:
    Across paired subjects, regress delta_pred ~ delta_gt.
    beta ~= 1 means state change is preserved.
    beta < 1 means the model suppresses the within-subject state change.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PER_WALK = (
    PROJECT_ROOT
    / "ModelComparison/data/results/CAREPD_BMCLab_extended_metrics_with_DIP/"
    / "multimodel_extended_per_walk.csv"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "ModelComparison/data/results/CAREPD_BMCLab_medication_onoff"
)


MAIN_MODELS = ["TransPose_default", "PIP_raw", "DynaIP_default"]
QA_MODELS = ["PIP_physics", "DIP_fine_tuning"]


@dataclass(frozen=True)
class StateMetric:
    key: str
    label: str
    gt_col: str
    pred_col: str
    unit: str
    tier: str
    role: str
    note: str


STATE_METRICS = [
    StateMetric(
        "joint_velocity",
        "Joint velocity",
        "gt_joint_velocity_mean_mms",
        "pred_joint_velocity_mean_mms",
        "mm/s",
        "motion_dynamics",
        "primary",
        "Continuous GT state has sufficient on/off dynamic range.",
    ),
    StateMetric(
        "leg_amp",
        "Leg swing amplitude",
        "gt_pose_leg_amp_mean_p95p5_deg",
        "pred_pose_leg_amp_mean_p95p5_deg",
        "deg",
        "gait_biomechanics",
        "primary",
        "Continuous GT state has sufficient on/off dynamic range.",
    ),
    StateMetric(
        "lower_rom",
        "Lower-body ROM",
        "gt_rom_lower_body_deg",
        "pred_rom_lower_body_deg",
        "deg",
        "motion_dynamics",
        "secondary",
        "Weaker within-subject GT dynamic range; interpret cautiously.",
    ),
    StateMetric(
        "arm_amp",
        "Arm swing amplitude",
        "gt_pose_arm_amp_mean_p95p5_deg",
        "pred_pose_arm_amp_mean_p95p5_deg",
        "deg",
        "gait_biomechanics",
        "secondary",
        "Upper-limb PD-related exploratory metric.",
    ),
    StateMetric(
        "trunk_lean",
        "Trunk lean",
        "gt_pose_trunk_lean_mean_deg",
        "pred_pose_trunk_lean_mean_deg",
        "deg",
        "gait_biomechanics",
        "secondary",
        "Signed posture measure; model-specific in earlier analyses.",
    ),
    StateMetric(
        "cadence_pose",
        "Pose cadence",
        "gt_pose_cadence_pose_spm",
        "pred_pose_cadence_pose_spm",
        "steps/min",
        "gait_biomechanics",
        "exploratory",
        "Most subjects have little GT on/off cadence change.",
    ),
    StateMetric(
        "phase_antiphase_error",
        "Leg antiphase error",
        "gt_phase_leg_phase_antiphase_error_deg",
        "pred_phase_leg_phase_antiphase_error_deg",
        "deg",
        "motion_dynamics",
        "exploratory",
        "Phase timing metric; useful but not a primary endpoint.",
    ),
    StateMetric(
        "phase_locking",
        "Leg phase locking",
        "gt_phase_leg_phase_locking_value",
        "pred_phase_leg_phase_locking_value",
        "unitless",
        "motion_dynamics",
        "exploratory",
        "Bounded phase metric; use signed difference rather than ratio.",
    ),
]


@dataclass(frozen=True)
class DistortionMetric:
    key: str
    label: str
    value_col: str
    kind: str
    role: str
    note: str


DISTORTION_METRICS = [
    DistortionMetric(
        "joint_velocity",
        "Joint velocity distortion",
        "joint_velocity_ratio",
        "ratio_log_abs",
        "primary",
        "Auxiliary paired distortion: abs(log(pred/GT)).",
    ),
    DistortionMetric(
        "leg_amp",
        "Leg swing distortion",
        "pose_ratio_leg_amp_mean_p95p5_deg",
        "ratio_log_abs",
        "primary",
        "Auxiliary paired distortion: abs(log(pred/GT)).",
    ),
    DistortionMetric(
        "lower_rom",
        "Lower-body ROM distortion",
        "rom_ratio_lower_body",
        "ratio_log_abs",
        "secondary",
        "Weaker GT on/off dynamic range.",
    ),
    DistortionMetric(
        "arm_amp",
        "Arm swing distortion",
        "pose_ratio_arm_amp_mean_p95p5_deg",
        "ratio_log_abs",
        "secondary",
        "Upper-limb exploratory metric.",
    ),
    DistortionMetric(
        "trunk_lean",
        "Trunk lean distortion",
        "pose_diff_trunk_lean_mean_deg",
        "diff_abs",
        "secondary",
        "Signed posture error summarized as abs(pred-GT).",
    ),
    DistortionMetric(
        "leg_asym",
        "Leg swing asymmetry distortion",
        "pose_diff_leg_amp_asym_p95p5",
        "diff_abs",
        "exploratory",
        "Bounded asymmetry metric; abs difference avoids small-denominator ratios.",
    ),
    DistortionMetric(
        "arm_asym",
        "Arm swing asymmetry distortion",
        "pose_diff_arm_amp_asym_p95p5",
        "diff_abs",
        "exploratory",
        "Bounded asymmetry metric; abs difference avoids small-denominator ratios.",
    ),
    DistortionMetric(
        "foot_lift_asym",
        "Foot-lift asymmetry distortion",
        "pose_diff_foot_lift_asym_p95p5",
        "diff_abs",
        "exploratory",
        "Bounded asymmetry metric; abs difference avoids small-denominator ratios.",
    ),
    DistortionMetric(
        "cadence_pose",
        "Pose cadence distortion",
        "pose_ratio_cadence_pose_spm",
        "ratio_log_abs",
        "exploratory",
        "Most subjects have little GT on/off cadence change.",
    ),
    DistortionMetric(
        "phase_antiphase_error",
        "Leg antiphase error distortion",
        "phase_diff_leg_phase_antiphase_error_deg",
        "diff_abs",
        "exploratory",
        "Phase metric; exploratory.",
    ),
    DistortionMetric(
        "phase_locking",
        "Leg phase-locking distortion",
        "phase_diff_leg_phase_locking_value",
        "diff_abs",
        "exploratory",
        "Bounded phase metric; abs difference.",
    ),
]


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted q-values, preserving NaNs."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return q.tolist()
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.clip(adj, 0.0, 1.0)
    return q.tolist()


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    if np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return np.nan, np.nan
    r = stats.spearmanr(x[mask], y[mask])
    return float(r.statistic), float(r.pvalue)


def bootstrap_slope_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 4 or np.nanstd(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    slopes = []
    n = len(x)
    for _ in range(n_boot):
        ii = rng.integers(0, n, n)
        xb = x[ii]
        yb = y[ii]
        if np.nanstd(xb) == 0:
            continue
        slopes.append(stats.linregress(xb, yb).slope)
    if len(slopes) < 10:
        return np.nan, np.nan
    return tuple(np.nanpercentile(slopes, [2.5, 97.5]).astype(float))


def slope_summary(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.nanstd(x) == 0:
        return {
            "beta": np.nan,
            "beta_ci_low": np.nan,
            "beta_ci_high": np.nan,
            "intercept": np.nan,
            "linear_r": np.nan,
            "linear_p": np.nan,
        }
    lr = stats.linregress(x, y)
    lo, hi = bootstrap_slope_ci(x, y, n_boot=n_boot, seed=seed)
    return {
        "beta": float(lr.slope),
        "beta_ci_low": lo,
        "beta_ci_high": hi,
        "intercept": float(lr.intercept),
        "linear_r": float(lr.rvalue),
        "linear_p": float(lr.pvalue),
    }


def wilcoxon_signed(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.allclose(x, 0):
        return np.nan, np.nan
    try:
        res = stats.wilcoxon(x, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return np.nan, np.nan


def distortion_value(series: pd.Series, kind: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if kind == "ratio_log_abs":
        values = values.where(values > 0)
        return np.log(values).abs()
    if kind == "diff_abs":
        return values.abs()
    raise ValueError(f"Unknown distortion kind: {kind}")


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError("Missing required columns:\n" + "\n".join(missing))


def build_subject_med_means(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    use = df[df["model_variant"].isin(models)].copy()
    use["medication"] = use["medication"].astype(str).str.lower()
    use = use[use["medication"].isin(["on", "off"])]
    numeric = use.select_dtypes(include=[np.number]).columns.tolist()
    keep = ["model", "variant", "model_variant", "subject", "medication"]
    agg = (
        use[keep + numeric]
        .groupby(["model", "variant", "model_variant", "subject", "medication"], as_index=False)
        .mean(numeric_only=True)
    )
    return agg


def paired_subjects(subject_med: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_variant, g in subject_med.groupby("model_variant"):
        med_counts = g.pivot_table(
            index="subject",
            columns="medication",
            values="UPDRS_GAIT",
            aggfunc="size",
            fill_value=0,
        )
        paired = med_counts[(med_counts.get("off", 0) > 0) & (med_counts.get("on", 0) > 0)].index
        for subject in paired:
            off = g[(g["subject"] == subject) & (g["medication"] == "off")].iloc[0]
            on = g[(g["subject"] == subject) & (g["medication"] == "on")].iloc[0]
            rows.append(
                {
                    "model_variant": model_variant,
                    "model": off.get("model", ""),
                    "variant": off.get("variant", ""),
                    "subject": subject,
                    "UPDRS_off": off["UPDRS_GAIT"],
                    "UPDRS_on": on["UPDRS_GAIT"],
                    "delta_UPDRS_off_minus_on": off["UPDRS_GAIT"] - on["UPDRS_GAIT"],
                    "updrs_responder": off["UPDRS_GAIT"] > on["UPDRS_GAIT"],
                    "updrs_nonresponder": off["UPDRS_GAIT"] == on["UPDRS_GAIT"],
                    "updrs_reverse": off["UPDRS_GAIT"] < on["UPDRS_GAIT"],
                }
            )
    return pd.DataFrame(rows)


def make_state_pairs(subject_med: pd.DataFrame, metrics: list[StateMetric]) -> pd.DataFrame:
    rows = []
    for model_variant, g in subject_med.groupby("model_variant"):
        subjects = sorted(set(g.loc[g["medication"] == "off", "subject"]) & set(g.loc[g["medication"] == "on", "subject"]))
        for subject in subjects:
            off = g[(g["subject"] == subject) & (g["medication"] == "off")].iloc[0]
            on = g[(g["subject"] == subject) & (g["medication"] == "on")].iloc[0]
            for m in metrics:
                rows.append(
                    {
                        "model_variant": model_variant,
                        "model": off.get("model", ""),
                        "variant": off.get("variant", ""),
                        "subject": subject,
                        "metric": m.key,
                        "label": m.label,
                        "unit": m.unit,
                        "tier": m.tier,
                        "role": m.role,
                        "note": m.note,
                        "UPDRS_off": off["UPDRS_GAIT"],
                        "UPDRS_on": on["UPDRS_GAIT"],
                        "delta_UPDRS_off_minus_on": off["UPDRS_GAIT"] - on["UPDRS_GAIT"],
                        "updrs_responder": off["UPDRS_GAIT"] > on["UPDRS_GAIT"],
                        "gt_off": off[m.gt_col],
                        "gt_on": on[m.gt_col],
                        "pred_off": off[m.pred_col],
                        "pred_on": on[m.pred_col],
                        "delta_gt_off_minus_on": off[m.gt_col] - on[m.gt_col],
                        "delta_pred_off_minus_on": off[m.pred_col] - on[m.pred_col],
                        "signed_state_error": (off[m.pred_col] - on[m.pred_col]) - (off[m.gt_col] - on[m.gt_col]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_state_pairs(pairs: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for (model_variant, metric), g in pairs.groupby(["model_variant", "metric"]):
        x = g["delta_gt_off_minus_on"].to_numpy(dtype=float)
        y = g["delta_pred_off_minus_on"].to_numpy(dtype=float)
        err = g["signed_state_error"].to_numpy(dtype=float)
        sp_r, sp_p = safe_spearman(x, y)
        slope = slope_summary(x, y, n_boot=n_boot, seed=seed + abs(hash((model_variant, metric))) % 100000)
        rows.append(
            {
                "model_variant": model_variant,
                "metric": metric,
                "label": g["label"].iloc[0],
                "unit": g["unit"].iloc[0],
                "tier": g["tier"].iloc[0],
                "role": g["role"].iloc[0],
                "n_subject_pairs": int(len(g)),
                "n_updrs_responders": int(g["updrs_responder"].sum()),
                "delta_gt_mean": float(np.nanmean(x)),
                "delta_gt_median": float(np.nanmedian(x)),
                "delta_pred_mean": float(np.nanmean(y)),
                "delta_pred_median": float(np.nanmedian(y)),
                "signed_state_error_mean": float(np.nanmean(err)),
                "signed_state_error_median": float(np.nanmedian(err)),
                "abs_state_error_median": float(np.nanmedian(np.abs(err))),
                "spearman_delta_gt_pred_rho": sp_r,
                "spearman_delta_gt_pred_p": sp_p,
                **slope,
                "interpretation_hint": "beta≈1 preserved; beta<1 flattened; beta>1 amplified",
                "note": g["note"].iloc[0],
            }
        )
    out = pd.DataFrame(rows)
    out["spearman_delta_gt_pred_q_secondary"] = np.nan
    mask = out["role"] != "primary"
    out.loc[mask, "spearman_delta_gt_pred_q_secondary"] = bh_fdr(out.loc[mask, "spearman_delta_gt_pred_p"].tolist())
    out["linear_q_secondary"] = np.nan
    out.loc[mask, "linear_q_secondary"] = bh_fdr(out.loc[mask, "linear_p"].tolist())
    return out


def make_distortion_pairs(subject_med: pd.DataFrame, metrics: list[DistortionMetric]) -> pd.DataFrame:
    rows = []
    for model_variant, g in subject_med.groupby("model_variant"):
        subjects = sorted(set(g.loc[g["medication"] == "off", "subject"]) & set(g.loc[g["medication"] == "on", "subject"]))
        for subject in subjects:
            off = g[(g["subject"] == subject) & (g["medication"] == "off")].iloc[0]
            on = g[(g["subject"] == subject) & (g["medication"] == "on")].iloc[0]
            for m in metrics:
                off_dist = float(distortion_value(pd.Series([off[m.value_col]]), m.kind).iloc[0])
                on_dist = float(distortion_value(pd.Series([on[m.value_col]]), m.kind).iloc[0])
                rows.append(
                    {
                        "model_variant": model_variant,
                        "model": off.get("model", ""),
                        "variant": off.get("variant", ""),
                        "subject": subject,
                        "metric": m.key,
                        "label": m.label,
                        "role": m.role,
                        "kind": m.kind,
                        "note": m.note,
                        "UPDRS_off": off["UPDRS_GAIT"],
                        "UPDRS_on": on["UPDRS_GAIT"],
                        "delta_UPDRS_off_minus_on": off["UPDRS_GAIT"] - on["UPDRS_GAIT"],
                        "updrs_responder": off["UPDRS_GAIT"] > on["UPDRS_GAIT"],
                        "distortion_off": off_dist,
                        "distortion_on": on_dist,
                        "delta_distortion_off_minus_on": off_dist - on_dist,
                    }
                )
    return pd.DataFrame(rows)


def summarize_distortion_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all", pairs), ("updrs_responder", pairs[pairs["updrs_responder"]])]
    for group_name, frame in groups:
        for (model_variant, metric), g in frame.groupby(["model_variant", "metric"]):
            delta = g["delta_distortion_off_minus_on"].to_numpy(dtype=float)
            w_stat, w_p = wilcoxon_signed(delta)
            rows.append(
                {
                    "group": group_name,
                    "model_variant": model_variant,
                    "metric": metric,
                    "label": g["label"].iloc[0],
                    "role": g["role"].iloc[0],
                    "kind": g["kind"].iloc[0],
                    "n_subject_pairs": int(len(g)),
                    "delta_distortion_mean": float(np.nanmean(delta)),
                    "delta_distortion_median": float(np.nanmedian(delta)),
                    "off_distortion_median": float(np.nanmedian(g["distortion_off"].to_numpy(dtype=float))),
                    "on_distortion_median": float(np.nanmedian(g["distortion_on"].to_numpy(dtype=float))),
                    "wilcoxon_stat": w_stat,
                    "wilcoxon_p": w_p,
                    "note": g["note"].iloc[0],
                }
            )
    out = pd.DataFrame(rows)
    out["wilcoxon_q_secondary"] = np.nan
    mask = out["role"] != "primary"
    out.loc[mask, "wilcoxon_q_secondary"] = bh_fdr(out.loc[mask, "wilcoxon_p"].tolist())
    return out


def write_plots(state_pairs: pd.DataFrame, out_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plotting skipped: {exc}")
        return []

    paths = []
    colors = {
        "TransPose_default": "#1f77b4",
        "PIP_raw": "#ff7f0e",
        "DynaIP_default": "#2ca02c",
    }
    for metric in ["joint_velocity", "leg_amp"]:
        fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
        sub = state_pairs[state_pairs["metric"] == metric]
        if sub.empty:
            plt.close(fig)
            continue
        all_xy = []
        for model_variant, g in sub.groupby("model_variant"):
            x = g["delta_gt_off_minus_on"].to_numpy(dtype=float)
            y = g["delta_pred_off_minus_on"].to_numpy(dtype=float)
            all_xy.extend(list(x[np.isfinite(x)]))
            all_xy.extend(list(y[np.isfinite(y)]))
            ax.scatter(
                x,
                y,
                s=38,
                alpha=0.78,
                label=model_variant,
                color=colors.get(model_variant),
                edgecolor="white",
                linewidth=0.5,
            )
            if np.isfinite(x).sum() >= 3 and np.nanstd(x) > 0:
                lr = stats.linregress(x, y)
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
                ax.plot(xs, lr.intercept + lr.slope * xs, color=colors.get(model_variant), linewidth=1.6)
        if all_xy:
            lo, hi = np.nanpercentile(all_xy, [2, 98])
            pad = max((hi - lo) * 0.12, 1e-3)
            lo -= pad
            hi += pad
            ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, alpha=0.55, label="preserved (beta=1)")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        label = sub["label"].iloc[0]
        unit = sub["unit"].iloc[0]
        ax.set_title(f"Medication state preservation: {label}")
        ax.set_xlabel(f"GT off - on ({unit})")
        ax.set_ylabel(f"Prediction off - on ({unit})")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"state_preservation_{metric}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-walk", type=Path, default=DEFAULT_PER_WALK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--models", nargs="+", default=MAIN_MODELS)
    parser.add_argument("--include-qa-models", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260616)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = list(args.models)
    if args.include_qa_models:
        models += [m for m in QA_MODELS if m not in models]

    state_cols = [c for m in STATE_METRICS for c in [m.gt_col, m.pred_col]]
    distortion_cols = [m.value_col for m in DISTORTION_METRICS]
    required = [
        "model",
        "variant",
        "model_variant",
        "subject",
        "medication",
        "UPDRS_GAIT",
        *state_cols,
        *distortion_cols,
    ]

    print(f"Loading: {args.per_walk}")
    df = pd.read_csv(args.per_walk)
    require_columns(df, required)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subject_med = build_subject_med_means(df, models)
    pair_meta = paired_subjects(subject_med)
    state_pairs = make_state_pairs(subject_med, STATE_METRICS)
    state_summary = summarize_state_pairs(state_pairs, n_boot=args.bootstrap, seed=args.seed)
    distortion_pairs = make_distortion_pairs(subject_med, DISTORTION_METRICS)
    distortion_summary = summarize_distortion_pairs(distortion_pairs)

    subject_med_path = args.out_dir / "med_onoff_subject_medication_means.csv"
    pair_meta_path = args.out_dir / "med_onoff_paired_subjects.csv"
    state_pairs_path = args.out_dir / "med_onoff_state_pairs.csv"
    state_summary_path = args.out_dir / "med_onoff_state_preservation_summary.csv"
    distortion_pairs_path = args.out_dir / "med_onoff_distortion_pairs.csv"
    distortion_summary_path = args.out_dir / "med_onoff_distortion_summary.csv"

    subject_med.to_csv(subject_med_path, index=False)
    pair_meta.to_csv(pair_meta_path, index=False)
    state_pairs.to_csv(state_pairs_path, index=False)
    state_summary.to_csv(state_summary_path, index=False)
    distortion_pairs.to_csv(distortion_pairs_path, index=False)
    distortion_summary.to_csv(distortion_summary_path, index=False)
    plot_paths = write_plots(state_pairs, args.out_dir)

    meta = {
        "input_per_walk": str(args.per_walk),
        "output_dir": str(args.out_dir),
        "models": models,
        "main_models": MAIN_MODELS,
        "qa_models_available": QA_MODELS,
        "n_bootstrap": args.bootstrap,
        "state_delta_definition": "off - on",
        "primary_statistic": "linear regression slope beta in delta_pred ~ delta_gt",
        "interpretation": {
            "beta_about_1": "within-subject medication state change is preserved",
            "beta_less_than_1": "model flattens/suppresses the GT state change",
            "beta_greater_than_1": "model amplifies the GT state change",
        },
        "primary_endpoints": ["joint_velocity", "leg_amp"],
        "secondary_or_exploratory": [
            "lower_rom",
            "arm_amp",
            "trunk_lean",
            "cadence_pose",
            "phase_antiphase_error",
            "phase_locking",
        ],
        "notes": [
            "UPDRS_GAIT is coarse; primary analysis uses continuous GT off/on change.",
            "Per-subject delta_pred/delta_gt ratios are intentionally not used to avoid small-denominator blowups.",
            "Distortion analysis is auxiliary; ratio metrics use abs(log(pred/GT)), bounded/diff metrics use abs(pred-GT).",
            "BH-FDR q-values are reported for non-primary secondary/exploratory tests.",
        ],
        "outputs": {
            "subject_medication_means": str(subject_med_path),
            "paired_subjects": str(pair_meta_path),
            "state_pairs": str(state_pairs_path),
            "state_summary": str(state_summary_path),
            "distortion_pairs": str(distortion_pairs_path),
            "distortion_summary": str(distortion_summary_path),
            "plots": plot_paths,
        },
    }
    meta_path = args.out_dir / "med_onoff_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nPaired subjects:")
    print(pair_meta.groupby("model_variant").agg(n=("subject", "nunique"), responders=("updrs_responder", "sum")).to_string())
    print("\nPrimary state-preservation summary:")
    primary = state_summary[state_summary["role"] == "primary"][
        [
            "model_variant",
            "metric",
            "n_subject_pairs",
            "beta",
            "beta_ci_low",
            "beta_ci_high",
            "spearman_delta_gt_pred_rho",
            "spearman_delta_gt_pred_p",
            "signed_state_error_median",
        ]
    ]
    print(primary.to_string(index=False))
    print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
