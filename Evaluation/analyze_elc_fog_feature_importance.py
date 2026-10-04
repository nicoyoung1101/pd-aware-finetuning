"""
analyze_elc_fog_feature_importance.py

Explain E-LC freezer-trait classification by comparing RF feature importance
and feature-level GT/reconstructed fidelity.

Inputs are produced by analyze_fog_trait_reconstructed_features.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_DIR = PROJECT_ROOT / "ModelComparison/data/results/CAREPD_ELC_FOG_model_features"

ID_COLS = {
    "feature_source", "source", "dataset", "dataset_name", "subject", "label", "label_text",
    "n_walks", "walk", "other", "medication", "source_fps", "n_frames_60fps", "duration_s",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default=str(DEFAULT_IN_DIR))
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def feature_columns(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if c in ID_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(v).sum() >= max(5, int(0.5 * len(v))) and np.nanstd(v) > 1e-12:
                cols.append(c)
    return cols


def feature_group(name: str):
    base = name.replace("__walk_std", "")
    if name.endswith("__walk_std"):
        return "between-walk variability"
    if any(k in base for k in ["step_time", "inst_freq", "cadence_variability", "freezing", "freeze_band", "locomotor_band", "phase"]):
        return "within-walk rhythm/variability"
    if any(k in base for k in ["velocity", "speed"]):
        return "velocity/activity"
    if "rom_" in base:
        return "ROM"
    if "arm" in base:
        return "arm swing/asymmetry"
    if "leg" in base or "foot" in base or "cadence" in base:
        return "gait biomechanics"
    if "trunk" in base or "lean" in base:
        return "trunk/posture"
    return "other local descriptor"


def source_importance(subject_df: pd.DataFrame, source: str, repeats: int, seed0: int):
    sub = subject_df[subject_df["feature_source"] == source].copy()
    cols = feature_columns(sub)
    x = sub[cols].to_numpy(dtype=float)
    y = sub["label"].to_numpy(dtype=int)
    n_min = min(np.bincount(y))
    n_splits = max(2, min(5, int(n_min)))
    importances = []
    scores = []
    for r in range(repeats):
        seed = seed0 + r
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in splitter.split(x, y):
            clf = make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
            clf.fit(x[tr], y[tr])
            pred = clf.predict(x[te])
            scores.append(float((pred == y[te]).mean()))
            importances.append(clf.named_steps["randomforestclassifier"].feature_importances_)
    imp = np.vstack(importances)
    out = pd.DataFrame({
        "feature_source": source,
        "feature": cols,
        "group": [feature_group(c) for c in cols],
        "importance_mean": imp.mean(axis=0),
        "importance_std": imp.std(axis=0),
    })
    total = out["importance_mean"].sum()
    out["importance_norm"] = out["importance_mean"] / total if total > 0 else 0.0
    out["rank"] = out["importance_norm"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("rank"), {"source": source, "n_features": len(cols), "cv_accuracy_mean_for_importance_fits": float(np.mean(scores))}


def safe_corr(a, b, kind="spearman"):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.nanstd(a[m]) <= 1e-12 or np.nanstd(b[m]) <= 1e-12:
        return np.nan, np.nan
    if kind == "spearman":
        res = spearmanr(a[m], b[m])
        return float(res.statistic), float(res.pvalue)
    res = pearsonr(a[m], b[m])
    return float(res.statistic), float(res.pvalue)


def feature_fidelity(subject_df: pd.DataFrame, importance_df: pd.DataFrame):
    gt = subject_df[subject_df["feature_source"] == "GT_local"].copy()
    gt_imp = importance_df[importance_df["feature_source"] == "GT_local"][["feature", "rank", "importance_norm", "group"]]
    rows = []
    for source in sorted(set(subject_df["feature_source"]) - {"GT_local"}):
        pred = subject_df[subject_df["feature_source"] == source].copy()
        merged = gt.merge(pred, on="subject", suffixes=("_gt", "_pred"))
        common = []
        for c in feature_columns(gt):
            if f"{c}_gt" in merged.columns and f"{c}_pred" in merged.columns:
                common.append(c)
        for c in common:
            rho, p = safe_corr(merged[f"{c}_gt"], merged[f"{c}_pred"], "spearman")
            pr, pp = safe_corr(merged[f"{c}_gt"], merged[f"{c}_pred"], "pearson")
            diff = pd.to_numeric(merged[f"{c}_pred"], errors="coerce") - pd.to_numeric(merged[f"{c}_gt"], errors="coerce")
            rows.append({
                "feature_source": source,
                "feature": c,
                "group": feature_group(c),
                "spearman_pred_vs_gt": rho,
                "spearman_p": p,
                "pearson_pred_vs_gt": pr,
                "pearson_p": pp,
                "mean_pred_minus_gt": float(np.nanmean(diff)),
                "mae_pred_minus_gt": float(np.nanmean(np.abs(diff))),
                "n_subjects": int(merged[[f"{c}_gt", f"{c}_pred"]].dropna().shape[0]),
            })
    out = pd.DataFrame(rows).merge(gt_imp, on=["feature", "group"], how="left")
    out = out.rename(columns={"rank": "gt_importance_rank", "importance_norm": "gt_importance_norm"})
    return out.sort_values(["feature_source", "gt_importance_rank"], na_position="last")




def df_to_md(df: pd.DataFrame, index: bool = False):
    if not index:
        data = df.reset_index(drop=True).copy()
    else:
        data = df.reset_index().copy()
    data = data.fillna("")
    cols = list(data.columns)
    rows = []
    rows.append("| " + " | ".join(str(c) for c in cols) + " |")
    rows.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in data.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_md(out_dir: Path, importance: pd.DataFrame, group_imp: pd.DataFrame, fidelity: pd.DataFrame, top_k: int):
    lines = []
    lines.append("# E-LC FOG Feature Importance and Reconstructed-Feature Fidelity\n\n")
    lines.append("Purpose: check whether GT and reconstructed freezer-trait classifiers rely on the same motion descriptors, or whether reconstructed classifiers may be using model-specific artifacts.\n\n")
    lines.append("## Group Importance\n\n")
    pivot = group_imp.pivot(index="group", columns="feature_source", values="importance_norm").fillna(0)
    lines.append(df_to_md(pivot.round(3), index=True))
    lines.append("\n\n## Top GT Features and Their Reconstructed Fidelity\n\n")
    top_gt = importance[importance["feature_source"] == "GT_local"].head(top_k)[["feature", "group", "importance_norm", "rank"]]
    top_fid = fidelity[fidelity["feature"].isin(top_gt["feature"])]
    table = top_fid.pivot(index="feature", columns="feature_source", values="spearman_pred_vs_gt")
    table = top_gt.set_index("feature")[["group", "importance_norm", "rank"]].join(table).reset_index()
    lines.append(df_to_md(table.round(3), index=False))
    lines.append("\n\n## Top Features by Source\n\n")
    for source in sorted(importance["feature_source"].unique(), key=lambda x: (x != "GT_local", x)):
        sub = importance[importance["feature_source"] == source].head(10)
        lines.append(f"### {source}\n\n")
        lines.append(df_to_md(sub[["rank", "feature", "group", "importance_norm"]].round(4), index=False))
        lines.append("\n\n")
    lines.append("## Reading Notes\n\n")
    lines.append("- High importance in GT_local identifies descriptors that carry freezer-trait information in the reference motion.\n")
    lines.append("- If a reconstructed source has high classifier score but its important features differ from GT_local, classification may be artifact-driven.\n")
    lines.append("- The fidelity columns are subject-level Spearman correlations between reconstructed and GT values for the same descriptor.\n")
    lines.append("- This is task-level utility analysis; it should be interpreted together with motion-level preservation metrics.\n")
    (out_dir / "FOG_feature_importance_and_fidelity_summary.md").write_text("".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    subject_path = in_dir / "elc_fog_model_features_per_subject_all.csv"
    if not subject_path.exists():
        raise FileNotFoundError(subject_path)
    subject_df = pd.read_csv(subject_path)
    sources = sorted(subject_df["feature_source"].dropna().unique(), key=lambda x: (x != "GT_local", x))
    importance_parts = []
    meta = {"sources": [], "repeats": args.repeats}
    for source in sources:
        imp, source_meta = source_importance(subject_df, source, args.repeats, args.seed)
        importance_parts.append(imp)
        meta["sources"].append(source_meta)
    importance = pd.concat(importance_parts, ignore_index=True)
    group_imp = importance.groupby(["feature_source", "group"], as_index=False)["importance_norm"].sum()
    fidelity = feature_fidelity(subject_df, importance)

    importance.to_csv(in_dir / "elc_fog_rf_feature_importance_by_source.csv", index=False)
    group_imp.to_csv(in_dir / "elc_fog_rf_group_importance_by_source.csv", index=False)
    fidelity.to_csv(in_dir / "elc_fog_feature_fidelity_vs_gt.csv", index=False)
    (in_dir / "elc_fog_feature_importance_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_md(in_dir, importance, group_imp, fidelity, args.top_k)
    print("Saved feature-importance analysis to", in_dir)
    print(group_imp.pivot(index="group", columns="feature_source", values="importance_norm").fillna(0).round(3).to_string())


if __name__ == "__main__":
    main()
