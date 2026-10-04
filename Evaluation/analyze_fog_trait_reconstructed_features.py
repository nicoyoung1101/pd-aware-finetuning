"""
analyze_fog_trait_reconstructed_features.py

Second-dataset model-level FOG/freezer-trait utility analysis.

It compares subject-level classification from GT local gait descriptors against
classification from reconstructed local-pose descriptors. This deliberately uses
local/pose descriptors only (no global translation / root path), so pose-only
models such as DynaIP can be compared fairly.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

for _name, _type in [
    ("bool", bool), ("int", int), ("float", float),
    ("complex", complex), ("object", object),
    ("str", str), ("unicode", str), ("long", int),
]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            getattr(np, _name)
        except AttributeError:
            setattr(np, _name, _type)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSPOSE_ROOT = PROJECT_ROOT / "TransPose"
for p in [PROJECT_ROOT, TRANSPOSE_ROOT, Path(__file__).resolve().parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import articulate as art  # noqa: E402
from config import paths  # noqa: E402
from Evaluation.analyze_fog_trait_gt_features import (  # noqa: E402
    TARGET_FPS,
    BODY_REGIONS,
    aggregate_subject_features,
    feature_columns,
    local_rom_from_pose,
    mean_region_velocity,
    pose_only_features,
    summarize_cv,
    variability_features,
)

DEFAULT_MODEL_CONFIGS = [
    {
        "source": "TransPose",
        "predictions": PROJECT_ROOT / "TransPose/data/results/CAREPD_ELC_batch/predictions.pt",
        "role": "main",
    },
    {
        "source": "PIP_raw",
        "predictions": PROJECT_ROOT / "PIP/data/results/CAREPD_ELC_raw_batch/predictions.pt",
        "role": "main_pose_only",
    },
    {
        "source": "DynaIP",
        "predictions": PROJECT_ROOT / "DynaIP/data/results/CAREPD_ELC_batch/predictions.pt",
        "role": "main_pose_only",
    },
    {
        "source": "PNP",
        "predictions": PROJECT_ROOT / "PNP/data/results/CAREPD_ELC_batch/predictions.pt",
        "role": "main_pose_only_official_physics",
    },
    {
        "source": "TIP",
        "predictions": PROJECT_ROOT / "TIP/data/results/CAREPD_ELC_batch/predictions.pt",
        "role": "main_pose_only_transformer",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "ModelComparison/data/results/CAREPD_ELC_FOG_model_features"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cv-repeats", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--include-gt-once", action="store_true", default=True)
    return parser.parse_args()


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@torch.no_grad()
def fk_joints_from_rot(body_model, pose_rot: torch.Tensor, tran: torch.Tensor | None, device: torch.device, batch: int = 2048):
    chunks = []
    for start in range(0, pose_rot.shape[0], batch):
        rot = pose_rot[start:start + batch].to(device).float()
        tr = None if tran is None else tran[start:start + batch].to(device).float()
        _, joints = body_model.forward_kinematics(rot, shape=None, tran=tr, calc_mesh=False)
        chunks.append(joints.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def extract_local_features(body_model, pose_rot: torch.Tensor, device: torch.device):
    # Fair cross-model feature set: no global translation/root path.
    joints = fk_joints_from_rot(body_model, pose_rot, None, device)
    pose_np = pose_rot.detach().cpu().numpy()
    feat = {}
    feat.update(pose_only_features(joints))
    feat.update(local_rom_from_pose(pose_np))
    feat.update(variability_features(joints, TARGET_FPS))
    feat["joint_velocity_mean_mms"] = mean_region_velocity(joints, BODY_REGIONS["full_body_no_root"], TARGET_FPS)
    feat["lower_body_velocity_mms"] = mean_region_velocity(joints, BODY_REGIONS["lower_body"], TARGET_FPS)
    feat["upper_limbs_velocity_mms"] = mean_region_velocity(joints, BODY_REGIONS["upper_limbs"], TARGET_FPS)
    feat["n_frames_60fps"] = int(pose_rot.shape[0])
    feat["duration_s"] = float(pose_rot.shape[0] / TARGET_FPS)
    return feat


def label_from_manifest(item):
    other = str(item.get("other"))
    if other == "PD-FOG":
        return 1, "PD-FOG"
    if other == "PD-NoFOG":
        return 0, "PD-NoFOG"
    return None, other


def rows_from_predictions(body_model, pred_path: Path, source: str, device: torch.device, use_gt: bool, limit: int = 0):
    data = safe_torch_load(pred_path, map_location="cpu")
    rows = []
    n = len(data["pose_gt"])
    if limit > 0:
        n = min(n, limit)
    pose_key = "pose_gt" if use_gt else "pose_pred"
    for i in tqdm(range(n), desc=f"{source} features"):
        item = data["manifest"][i]
        label, label_text = label_from_manifest(item)
        if label is None:
            continue
        feat = extract_local_features(body_model, data[pose_key][i], device)
        feat.update({
            "dataset": "E-LC",
            "source": source,
            "subject": str(item.get("subject_id")),
            "walk": str(item.get("walk_id")),
            "label": int(label),
            "label_text": label_text,
            "other": item.get("other"),
            "medication": item.get("medication"),
            "source_fps": item.get("source_fps"),
        })
        rows.append(feat)
    return rows


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    body_model = art.ParametricModel(paths.smpl_file)

    configs = DEFAULT_MODEL_CONFIGS
    for cfg in configs:
        if not Path(cfg["predictions"]).exists():
            raise FileNotFoundError(cfg["predictions"])

    all_walks = []
    all_subjects = []
    all_summary = []
    all_seed_rows = []
    all_feature_cols = {}

    # GT upper/reference, extracted once from the same prediction manifest.
    gt_rows = rows_from_predictions(body_model, configs[0]["predictions"], "GT_local", device, use_gt=True, limit=args.limit)
    sources = [("GT_local", gt_rows, "gt_reference")]
    for cfg in configs:
        rows = rows_from_predictions(body_model, cfg["predictions"], cfg["source"], device, use_gt=False, limit=args.limit)
        sources.append((cfg["source"], rows, cfg["role"]))

    for source, rows, role in sources:
        per_walk = pd.DataFrame(rows)
        subject_df = aggregate_subject_features(per_walk)
        summary, per_seed, feat_cols = summarize_cv(subject_df, ["rf", "logistic", "svm"], args.cv_repeats, args.permutations)
        per_walk.insert(0, "feature_source", source)
        subject_df.insert(0, "feature_source", source)
        summary.insert(0, "feature_source", source)
        summary.insert(1, "role", role)
        per_seed.insert(0, "feature_source", source)
        per_seed.insert(1, "role", role)

        per_walk.to_csv(out_dir / f"{source}_fog_features_per_walk.csv", index=False)
        subject_df.to_csv(out_dir / f"{source}_fog_features_per_subject.csv", index=False)
        summary.to_csv(out_dir / f"{source}_fog_classifier_summary.csv", index=False)
        per_seed.to_csv(out_dir / f"{source}_fog_classifier_per_seed.csv", index=False)
        (out_dir / f"{source}_feature_columns.txt").write_text("\n".join(feat_cols), encoding="utf-8")

        all_walks.append(per_walk)
        all_subjects.append(subject_df)
        all_summary.append(summary)
        all_seed_rows.append(per_seed)
        all_feature_cols[source] = feat_cols

    pd.concat(all_walks, ignore_index=True).to_csv(out_dir / "elc_fog_model_features_per_walk_all.csv", index=False)
    pd.concat(all_subjects, ignore_index=True).to_csv(out_dir / "elc_fog_model_features_per_subject_all.csv", index=False)
    pd.concat(all_summary, ignore_index=True).to_csv(out_dir / "elc_fog_model_classifier_summary_all.csv", index=False)
    pd.concat(all_seed_rows, ignore_index=True).to_csv(out_dir / "elc_fog_model_classifier_per_seed_all.csv", index=False)

    meta = {
        "task": "E-LC subject-level PD-FOG vs PD-NoFOG classification from GT/reconstructed local-pose descriptors",
        "label_caveat": "Subject-level FOG-propensity/freezer-trait classification, not FOG episode detection.",
        "feature_policy": "Local pose/descriptor features only; global translation/root-path features excluded for cross-model comparability.",
        "target_fps": TARGET_FPS,
        "models": [{"source": c["source"], "predictions": str(c["predictions"]), "role": c["role"]} for c in configs],
        "feature_columns_by_source": all_feature_cols,
    }
    with (out_dir / "elc_fog_model_feature_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Saved:", out_dir)
    print(pd.concat(all_summary, ignore_index=True).query("classifier == 'rf'").to_string(index=False))


if __name__ == "__main__":
    main()
