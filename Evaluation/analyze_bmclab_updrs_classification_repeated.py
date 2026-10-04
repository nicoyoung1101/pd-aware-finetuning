#!/usr/bin/env python3
"""Repeated subject-wise BMCLab UPDRS_GAIT classification.

This is a lightweight post-processing script: it reads the existing multimodel
per-walk descriptor CSV, evaluates GT/reconstructed descriptors with repeated
subject-wise StratifiedGroupKFold, and writes summary/per-seed CSV files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "ModelComparison/data/results/CAREPD_BMCLab_extended_metrics_with_TIP"
DEFAULT_PER_WALK = DEFAULT_RESULT_ROOT / "multimodel_extended_per_walk.csv"
DEFAULT_FEATURES = DEFAULT_RESULT_ROOT / "bmclab_updrs_classification/bmclab_updrs_gait_classification_features.csv"
DEFAULT_OUT_DIR = DEFAULT_RESULT_ROOT / "bmclab_updrs_classification"

SOURCES = [
    ("GT features", "GT", "TransPose_default", "gt"),
    ("TransPose", "TransPose_default", "TransPose_default", "pred"),
    ("PIP raw", "PIP_raw", "PIP_raw", "pred"),
    ("DynaIP", "DynaIP_default", "DynaIP_default", "pred"),
    ("PNP", "PNP_default", "PNP_default", "pred"),
    ("TIP", "TIP_default", "TIP_default", "pred"),
]

METRICS = ["accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "balanced_accuracy"]


def classifier(seed: int, n_estimators: int = 150):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def run_cv(data: pd.DataFrame, feature_cols: list[str], seed: int, y_override: np.ndarray | None = None, n_estimators: int = 150) -> dict[str, float]:
    x = data[feature_cols].to_numpy(dtype=float)
    y = data["UPDRS_GAIT"].astype(int).to_numpy() if y_override is None else np.asarray(y_override, dtype=int)
    groups = data["subject"].astype(str).to_numpy()
    n_subjects = len(np.unique(groups))
    n_splits = min(5, n_subjects)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.full(len(y), -1, dtype=int)
    for train_idx, test_idx in splitter.split(x, y, groups):
        clf = classifier(seed, n_estimators=n_estimators)
        clf.fit(x[train_idx], y[train_idx])
        pred[test_idx] = clf.predict(x[test_idx])
    if np.any(pred < 0):
        raise RuntimeError("Some samples were not assigned predictions.")
    return evaluate(y, pred)


def summarize(per_seed: pd.DataFrame, source_meta: dict[tuple[str, str], dict[str, int]]) -> pd.DataFrame:
    rows = []
    for (source, source_key), group in per_seed.groupby(["source", "source_key"], sort=False):
        real = group[~group["is_permutation"]]
        perm = group[group["is_permutation"]]
        row = {"source": source, "source_key": source_key, **source_meta[(source, source_key)]}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(real[metric].mean())
            row[f"{metric}_std"] = float(real[metric].std(ddof=0))
            row[f"{metric}_p05"] = float(real[metric].quantile(0.05))
            row[f"{metric}_p95"] = float(real[metric].quantile(0.95))
        row["permutation_balanced_accuracy_mean"] = float(perm["balanced_accuracy"].mean()) if len(perm) else np.nan
        row["permutation_balanced_accuracy_std"] = float(perm["balanced_accuracy"].std(ddof=0)) if len(perm) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-walk", type=Path, default=DEFAULT_PER_WALK)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--permutations", type=int, default=10)
    ap.add_argument("--n-estimators", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    df = pd.read_csv(args.per_walk)
    features = pd.read_csv(args.features)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    source_meta: dict[tuple[str, str], dict[str, int]] = {}
    for source, source_key, model_variant, col_kind in SOURCES:
        sub = df[df["model_variant"] == model_variant].copy()
        feature_cols = features["gt_column"].tolist() if col_kind == "gt" else features["pred_column"].tolist()
        needed = ["UPDRS_GAIT", "subject", *feature_cols]
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
        sub = sub[sub["UPDRS_GAIT"].isin([0, 1, 2])].copy()
        if sub.empty:
            raise RuntimeError(f"No rows for {source_key}")
        source_meta[(source, source_key)] = {
            "n": int(len(sub)),
            "n_subjects": int(sub["subject"].nunique()),
            "n_features": int(len(feature_cols)),
        }
        y = sub["UPDRS_GAIT"].astype(int).to_numpy()
        rng = np.random.default_rng(args.seed + 10000)
        for r in range(args.repeats):
            seed = args.seed + r
            rows.append({
                "source": source,
                "source_key": source_key,
                "repeat": r,
                "is_permutation": False,
                **run_cv(sub, feature_cols, seed, n_estimators=args.n_estimators),
            })
        for p in range(args.permutations):
            seed = args.seed + 1000 + p
            y_perm = rng.permutation(y)
            rows.append({
                "source": source,
                "source_key": source_key,
                "repeat": p,
                "is_permutation": True,
                **run_cv(sub, feature_cols, seed, y_override=y_perm, n_estimators=args.n_estimators),
            })

    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed, source_meta)

    per_seed_path = args.out_dir / "bmclab_updrs_gait_classification_per_seed_repeated.csv"
    summary_path = args.out_dir / "bmclab_updrs_gait_classification_five_model_repeated.csv"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    # Also write a compact display table for manuscript drafting.
    display = summary[[
        "source", "source_key", "n", "n_subjects", "n_features",
        "accuracy_mean", "accuracy_std",
        "weighted_precision_mean", "weighted_precision_std",
        "weighted_recall_mean", "weighted_recall_std",
        "weighted_f1_mean", "weighted_f1_std",
        "balanced_accuracy_mean", "balanced_accuracy_std",
        "permutation_balanced_accuracy_mean", "permutation_balanced_accuracy_std",
    ]].copy()
    display.to_csv(args.out_dir / "bmclab_updrs_gait_classification_five_model_repeated_display.csv", index=False)

    print(f"Wrote {summary_path}")
    print(f"Wrote {per_seed_path}")
    print(summary[["source", "balanced_accuracy_mean", "balanced_accuracy_std", "permutation_balanced_accuracy_mean", "permutation_balanced_accuracy_std"]].to_string(index=False))

if __name__ == "__main__":
    main()
