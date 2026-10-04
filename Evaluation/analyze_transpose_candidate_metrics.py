"""
analyze_transpose_candidate_metrics.py

Generate a broad candidate-metric pool for CARE-PD / TransPose.

The goal is intentionally inclusive: compute more metrics than we expect to use
in the paper, then select the stable and interpretable ones later.

Outputs:
  - transpose_candidate_metrics_per_walk.csv
  - transpose_candidate_metrics_grouped_by_updrs.csv
  - transpose_candidate_metrics_summary.csv
  - transpose_candidate_metrics_correlations.csv
  - transpose_candidate_metric_catalog.csv

Metric families:
  1. Pose fidelity
  2. Motion dynamics
  3. Gait biomechanics
  4. Supplementary / exploratory
"""

# ========== numpy compatibility patch ==========
import warnings
import numpy as np

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

import argparse
import csv
import json
import os
import sys

import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRANSPOSE_ROOT = os.path.join(PROJECT_ROOT, "TransPose")
if TRANSPOSE_ROOT not in sys.path:
    sys.path.insert(0, TRANSPOSE_ROOT)

from analyze_gait_feature_preservation import (  # noqa: E402
    FPS,
    HORIZONTAL_AXES,
    SMPL_JOINT,
    detect_contacts,
    fk_joints,
    full_motion_features,
    pose_only_features,
    safe_torch_load,
)
from analyze_pose_fidelity import (  # noqa: E402
    IGNORED_POSITION_JOINTS,
    IGNORED_ROTATION_JOINTS,
    JOINT_NAMES,
    POSITION_BODY_PARTS,
    PREDICTED_POSITION_BODY_PARTS,
    load_body_parts,
    local_rotation_geodesic_deg,
    per_joint_mpjpe_mm,
    procrustes_aligned_mpjpe_mm,
    region_indices,
)
import articulate as art  # noqa: E402
from config import paths  # noqa: E402


PREDICTIONS_PT = os.path.join(
    TRANSPOSE_ROOT, "data", "results", "CAREPD_BMCLab_batch", "predictions.pt"
)
DRIFT_QA_DIR = os.path.join(
    TRANSPOSE_ROOT, "data", "results", "CAREPD_BMCLab_local_pose_drift_qa"
)
OUT_DIR = os.path.join(
    TRANSPOSE_ROOT, "data", "results", "CAREPD_BMCLab_candidate_metrics"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=PREDICTIONS_PT)
    parser.add_argument("--drift-qa-dir", default=DRIFT_QA_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--min-step-interval-s", type=float, default=0.25)
    parser.add_argument("--file-prefix", default="transpose_candidate_metrics")
    return parser.parse_args()


def prefixed(prefix, data):
    return {f"{prefix}_{k}": v for k, v in data.items()}


def finite_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


def finite_median(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def ratio(num, den, eps=1e-8):
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < eps:
        return np.nan
    return float(num / den)


def root_aligned(joints):
    return joints - joints[:, SMPL_JOINT["pelvis"] : SMPL_JOINT["pelvis"] + 1]


def temporal_derivative(x, fps):
    if len(x) < 2:
        return np.zeros((0,) + x.shape[1:], dtype=np.float64)
    return np.diff(x, axis=0) * fps


def mean_joint_norm(x):
    if x.size == 0:
        return np.nan
    return float(np.nanmean(np.linalg.norm(x, axis=-1)))


def p95_joint_norm(x):
    if x.size == 0:
        return np.nan
    return float(np.nanpercentile(np.linalg.norm(x, axis=-1), 95))


def motion_dynamics_features(gt_joints, pred_joints, fps=FPS):
    """
    Root-aligned joint dynamics. Units:
      velocity: mm/s
      acceleration: m/s^2
      jerk: m/s^3
    """
    gt = root_aligned(gt_joints)
    pred = root_aligned(pred_joints)

    gt_vel = temporal_derivative(gt, fps)
    pred_vel = temporal_derivative(pred, fps)
    gt_acc = temporal_derivative(gt_vel, fps)
    pred_acc = temporal_derivative(pred_vel, fps)
    gt_jerk = temporal_derivative(gt_acc, fps)
    pred_jerk = temporal_derivative(pred_acc, fps)

    vel_err = pred_vel - gt_vel
    acc_err = pred_acc - gt_acc
    jerk_err = pred_jerk - gt_jerk

    return {
        "joint_velocity_error_mean_mms": mean_joint_norm(vel_err) * 1000.0,
        "joint_velocity_error_p95_mms": p95_joint_norm(vel_err) * 1000.0,
        "joint_acceleration_error_mean_mps2": mean_joint_norm(acc_err),
        "joint_acceleration_error_p95_mps2": p95_joint_norm(acc_err),
        "joint_jerk_error_mean_mps3": mean_joint_norm(jerk_err),
        "joint_jerk_error_p95_mps3": p95_joint_norm(jerk_err),
        "gt_joint_velocity_mean_mms": mean_joint_norm(gt_vel) * 1000.0,
        "pred_joint_velocity_mean_mms": mean_joint_norm(pred_vel) * 1000.0,
        "joint_velocity_ratio": ratio(mean_joint_norm(pred_vel), mean_joint_norm(gt_vel)),
    }


def trajectory_features(gt_full_joints, pred_full_joints, fps=FPS):
    gt = gt_full_joints[:, SMPL_JOINT["pelvis"]]
    pred = pred_full_joints[:, SMPL_JOINT["pelvis"]]
    if len(gt) < 2:
        return {}
    gt_rel = gt - gt[:1]
    pred_rel = pred - pred[:1]
    diff = pred_rel - gt_rel
    gt_step = np.diff(gt[:, HORIZONTAL_AXES], axis=0)
    pred_step = np.diff(pred[:, HORIZONTAL_AXES], axis=0)
    gt_path = float(np.sum(np.linalg.norm(gt_step, axis=1)))
    pred_path = float(np.sum(np.linalg.norm(pred_step, axis=1)))
    duration = max(1e-6, (len(gt) - 1) / fps)
    final_err = float(np.linalg.norm(diff[-1, HORIZONTAL_AXES]))
    ate = float(np.sqrt(np.nanmean(np.sum(diff[:, HORIZONTAL_AXES] ** 2, axis=1))))
    return {
        "root_traj_ate_m": ate,
        "root_traj_final_error_m": final_err,
        "root_path_gt_m": gt_path,
        "root_path_pred_m": pred_path,
        "root_path_ratio": ratio(pred_path, gt_path),
        "root_straight_speed_gt_mps": float(np.linalg.norm(gt[-1, HORIZONTAL_AXES] - gt[0, HORIZONTAL_AXES]) / duration),
        "root_straight_speed_pred_mps": float(np.linalg.norm(pred[-1, HORIZONTAL_AXES] - pred[0, HORIZONTAL_AXES]) / duration),
    }


def contact_mask(foot_pos, fps=FPS):
    horiz = foot_pos[:, HORIZONTAL_AXES]
    vel = np.zeros(len(foot_pos))
    if len(foot_pos) > 1:
        vel[1:] = np.linalg.norm(np.diff(horiz, axis=0), axis=1) * fps
    height = foot_pos[:, 1] - np.nanmin(foot_pos[:, 1])
    speed_thr = np.nanpercentile(vel, 35)
    height_thr = np.nanpercentile(height, 45)
    return (vel <= speed_thr) & (height <= height_thr)


def foot_skating_features(gt_full_joints, pred_full_joints, fps=FPS):
    out = {}
    vals = []
    for side, joint_name in [("left", "lfoot"), ("right", "rfoot")]:
        j = SMPL_JOINT[joint_name]
        gt_foot = gt_full_joints[:, j]
        pred_foot = pred_full_joints[:, j]
        mask = contact_mask(gt_foot, fps)
        pred_speed = np.zeros(len(pred_foot))
        if len(pred_foot) > 1:
            pred_speed[1:] = np.linalg.norm(np.diff(pred_foot[:, HORIZONTAL_AXES], axis=0), axis=1) * fps
        contact_speed = pred_speed[mask]
        out[f"{side}_foot_skating_mean_mps"] = finite_mean(contact_speed)
        out[f"{side}_foot_skating_p95_mps"] = float(np.nanpercentile(contact_speed, 95)) if len(contact_speed) else np.nan
        out[f"{side}_gt_contact_fraction"] = float(np.mean(mask))
        if len(contact_speed):
            vals.extend(contact_speed.tolist())
    out["foot_skating_mean_mps"] = finite_mean(vals)
    out["foot_skating_p95_mps"] = float(np.nanpercentile(vals, 95)) if len(vals) else np.nan
    return out


def com_proxy_features(gt_full_joints, pred_full_joints, fps=FPS):
    """
    Exploratory COM proxies. These are not true mass-weighted COM because the
    project does not currently use segment masses. We report pelvis and
    joint-centroid proxies separately.
    """
    gt_pelvis = gt_full_joints[:, SMPL_JOINT["pelvis"]]
    pred_pelvis = pred_full_joints[:, SMPL_JOINT["pelvis"]]
    gt_centroid = np.nanmean(gt_full_joints, axis=1)
    pred_centroid = np.nanmean(pred_full_joints, axis=1)

    def proxy(prefix, gt, pred):
        gt_rel = gt - gt[:1]
        pred_rel = pred - pred[:1]
        diff = pred_rel - gt_rel
        return {
            f"{prefix}_traj_ate_m": float(np.sqrt(np.nanmean(np.sum(diff[:, HORIZONTAL_AXES] ** 2, axis=1)))),
            f"{prefix}_vertical_range_gt_m": float(np.nanpercentile(gt[:, 1], 95) - np.nanpercentile(gt[:, 1], 5)),
            f"{prefix}_vertical_range_pred_m": float(np.nanpercentile(pred[:, 1], 95) - np.nanpercentile(pred[:, 1], 5)),
            f"{prefix}_vertical_range_ratio": ratio(
                float(np.nanpercentile(pred[:, 1], 95) - np.nanpercentile(pred[:, 1], 5)),
                float(np.nanpercentile(gt[:, 1], 95) - np.nanpercentile(gt[:, 1], 5)),
            ),
            f"{prefix}_lateral_sway_gt_m": float(np.nanstd(gt_rel[:, 0])),
            f"{prefix}_lateral_sway_pred_m": float(np.nanstd(pred_rel[:, 0])),
        }

    out = {}
    out.update(proxy("pelvis_com_proxy", gt_pelvis, pred_pelvis))
    out.update(proxy("joint_centroid_com_proxy", gt_centroid, pred_centroid))
    return out


def add_preservation_columns(row, feature_names, gt_prefix, pred_prefix, out_prefix):
    for name in feature_names:
        gt = row.get(f"{gt_prefix}_{name}", np.nan)
        pred = row.get(f"{pred_prefix}_{name}", np.nan)
        row[f"{out_prefix}_diff_{name}"] = float(pred - gt) if np.isfinite(pred) and np.isfinite(gt) else np.nan
        row[f"{out_prefix}_ratio_{name}"] = ratio(pred, gt)


def summarize_dataframe(df):
    rows = []
    for col in df.columns:
        if col in {"label", "subject", "walk", "medication", "other"}:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "metric": col,
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "p05": float(vals.quantile(0.05)),
            "p95": float(vals.quantile(0.95)),
        })
    return pd.DataFrame(rows)


def correlations(df):
    rows = []
    if "UPDRS_GAIT" not in df.columns:
        return pd.DataFrame(rows)
    y = pd.to_numeric(df["UPDRS_GAIT"], errors="coerce").to_numpy()
    duration = (
        pd.to_numeric(df["duration_s"], errors="coerce").to_numpy()
        if "duration_s" in df.columns else None
    )
    for col in df.columns:
        if col in {"label", "subject", "walk", "medication", "other", "UPDRS_GAIT"}:
            continue
        x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 5 or np.nanstd(x[mask]) < 1e-12:
            continue
        rho, p = stats.spearmanr(y[mask], x[mask])
        rows.append({"metric": col, "against": "UPDRS_GAIT", "method": "spearman", "rho": rho, "p": p, "n": int(mask.sum())})
        if duration is None or col == "duration_s":
            continue
        ctrl_mask = mask & np.isfinite(duration)
        if ctrl_mask.sum() < 5 or np.nanstd(duration[ctrl_mask]) < 1e-12:
            continue
        y_rank = stats.rankdata(y[ctrl_mask])
        x_rank = stats.rankdata(x[ctrl_mask])
        d_rank = stats.rankdata(duration[ctrl_mask])
        design = np.c_[np.ones(len(d_rank)), d_rank]
        y_res = y_rank - design @ np.linalg.lstsq(design, y_rank, rcond=None)[0]
        x_res = x_rank - design @ np.linalg.lstsq(design, x_rank, rcond=None)[0]
        if np.nanstd(y_res) < 1e-12 or np.nanstd(x_res) < 1e-12:
            continue
        rho_partial, p_partial = stats.pearsonr(y_res, x_res)
        rows.append({
            "metric": col,
            "against": "UPDRS_GAIT",
            "method": "partial_spearman_control_duration_s",
            "rho": float(rho_partial),
            "p": float(p_partial),
            "n": int(ctrl_mask.sum()),
        })
    return pd.DataFrame(rows)


def metric_catalog():
    rows = [
        ("Pose fidelity", "MPJPE", "mpjpe_*", "mm", "Standard root-aligned joint position error.", "main candidate"),
        ("Pose fidelity", "PA-MPJPE", "pa_mpjpe_mm", "mm", "Standard Procrustes-aligned pose accuracy; not for amplitude preservation.", "standard auxiliary"),
        ("Pose fidelity", "Joint rotation error", "rot_*", "deg", "Local SMPL joint angle error, translation-independent.", "main candidate"),
        ("Pose fidelity", "Region-wise MPJPE", "mpjpe_trunk/upper/lower/*", "mm", "Localizes errors by body region.", "main candidate"),
        ("Motion dynamics", "Velocity error", "joint_velocity_error_*", "mm/s", "Checks whether dynamic speed of joints matches GT.", "candidate"),
        ("Motion dynamics", "Acceleration error", "joint_acceleration_error_*", "m/s^2", "Captures dynamic change mismatch; noise-sensitive.", "candidate/QA"),
        ("Motion dynamics", "Jerk error", "joint_jerk_error_*", "m/s^3", "Captures smoothness and high-frequency artifacts; very noise-sensitive.", "exploratory"),
        ("Motion dynamics", "Root trajectory error", "root_traj_*", "m", "Global path agreement; translation caveat.", "candidate with caveat"),
        ("Motion dynamics", "Foot skating", "foot_skating_*", "m/s", "Predicted foot horizontal motion during GT stance.", "candidate/QA"),
        ("Gait biomechanics", "Walking speed", "gt/pred_full_gait_speed_mps", "m/s", "Classic gait marker for bradykinesia and locomotion.", "main candidate"),
        ("Gait biomechanics", "Step/stride length", "gt/pred_full_step_length_m, stride_length_m", "m", "Classic short-step gait descriptor.", "main candidate"),
        ("Gait biomechanics", "Cadence/timing", "cadence_pose_spm, contact_cadence_spm", "steps/min", "Rhythm preservation / negative-control candidate.", "main candidate"),
        ("Gait biomechanics", "Foot clearance", "foot_lift_*", "m", "Shuffling / low foot clearance descriptor.", "candidate"),
        ("Gait biomechanics", "Arm swing", "arm_amp_*", "deg or asymmetry", "Upper-limb swing reduction/asymmetry in PD gait.", "candidate"),
        ("Gait biomechanics", "Trunk lean", "trunk_lean_*", "deg", "Axial posture / stooped posture candidate.", "candidate"),
        ("Supplementary", "Step width", "step_width_m", "m", "Stability-related but sensitive to contact detection.", "supplementary"),
        ("Supplementary", "COM proxy", "*com_proxy*", "m", "Pelvis/joint-centroid proxy only, not mass-weighted COM.", "exploratory"),
        ("Supplementary", "Leg swing", "leg_amp_*", "deg or asymmetry", "Local lower-limb amplitude; overlaps with step length but is translation-independent.", "main candidate despite overlap"),
    ]
    return pd.DataFrame(rows, columns=["level", "metric", "columns", "unit", "why_keep", "suggested_role"])


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = safe_torch_load(args.predictions, map_location="cpu")
    n = len(data["pose_pred"])
    if args.limit > 0:
        n = min(n, args.limit)

    device = torch.device(args.device)
    model = art.ParametricModel(paths.smpl_file)
    rotation_body_parts = load_body_parts(args.drift_qa_dir)
    position_regions = region_indices(POSITION_BODY_PARTS)
    ignored_position_indices = {JOINT_NAMES.index(j) for j in IGNORED_POSITION_JOINTS}
    predicted_position_regions = region_indices(
        PREDICTED_POSITION_BODY_PARTS,
        excluded_full_joints=ignored_position_indices,
    )
    ignored_rotation_indices = {JOINT_NAMES.index(j) for j in IGNORED_ROTATION_JOINTS}
    rotation_regions = region_indices(
        rotation_body_parts,
        excluded_full_joints=ignored_rotation_indices,
    )

    pose_feature_names = [
        "arm_amp_mean_p95p5_deg",
        "arm_amp_asym_p95p5",
        "leg_amp_mean_p95p5_deg",
        "leg_amp_asym_p95p5",
        "foot_lift_mean_p95p5_m",
        "foot_lift_asym_p95p5",
        "trunk_lean_mean_deg",
        "cadence_pose_spm",
        "step_time_pose_s",
    ]
    full_feature_names = [
        "gait_speed_mps",
        "contact_cadence_spm",
        "contact_step_time_s",
        "stride_time_s",
        "stride_length_m",
        "step_length_m",
        "step_width_m",
        "mos_lateral_m",
    ]

    rows = []
    for i in tqdm(range(n), desc="Candidate TransPose metrics"):
        pose_gt = data["pose_gt"][i]
        pose_pred = data["pose_pred"][i]
        tran_gt = data["tran_gt"][i]
        tran_pred = data["tran_pred"][i]

        gt_pose_j = fk_joints(model, pose_gt, device, tran=None)
        pred_pose_j = fk_joints(model, pose_pred, device, tran=None)
        gt_full_j = fk_joints(model, pose_gt, device, tran=tran_gt)
        pred_full_j = fk_joints(model, pose_pred, device, tran=tran_pred)

        per_joint_mpjpe = per_joint_mpjpe_mm(pred_pose_j, gt_pose_j)
        rot_err = local_rotation_geodesic_deg(pose_pred, pose_gt)

        item = data["manifest"][i] if data.get("manifest") and data["manifest"][i] else {}
        row = {
            "index": int(data["index"][i]) if "index" in data else i,
            "label": data["label"][i] if "label" in data else f"seq_{i:04d}",
            "subject": item.get("subject_id", ""),
            "walk": item.get("walk_id", ""),
            "UPDRS_GAIT": item.get("UPDRS_GAIT", np.nan),
            "medication": item.get("medication", ""),
            "other": item.get("other", ""),
            "frames": int(pose_gt.shape[0]),
            "duration_s": float(pose_gt.shape[0] / FPS),
            "mpjpe_full_body_no_root_mm": float(np.nanmean(per_joint_mpjpe[:, position_regions["full_body_no_root"]])),
            "mpjpe_predicted_full_body_no_root_mm": float(np.nanmean(per_joint_mpjpe[:, predicted_position_regions["full_body_no_root"]])),
            "pa_mpjpe_mm": procrustes_aligned_mpjpe_mm(pred_pose_j, gt_pose_j),
            "pa_mpjpe_predicted_mm": procrustes_aligned_mpjpe_mm(
                pred_pose_j,
                gt_pose_j,
                predicted_position_regions["full_body"],
            ),
            "rot_full_body_no_root_deg": float(np.nanmean(rot_err[:, rotation_regions["full_body_no_root"]])),
        }
        for name, idxs in position_regions.items():
            row[f"mpjpe_{name}_mm"] = float(np.nanmean(per_joint_mpjpe[:, idxs]))
        for name, idxs in predicted_position_regions.items():
            row[f"mpjpe_predicted_{name}_mm"] = float(np.nanmean(per_joint_mpjpe[:, idxs]))
        for name, idxs in rotation_regions.items():
            row[f"rot_{name}_deg"] = float(np.nanmean(rot_err[:, idxs]))
        for j, name in enumerate(JOINT_NAMES):
            row[f"joint_mpjpe_{name}_mm"] = float(np.nanmean(per_joint_mpjpe[:, j]))
            row[f"joint_rot_{name}_deg"] = (
                np.nan if name in IGNORED_ROTATION_JOINTS else float(np.nanmean(rot_err[:, j]))
            )

        gt_pose_features = pose_only_features(gt_pose_j)
        pred_pose_features = pose_only_features(pred_pose_j)
        gt_full_features = full_motion_features(gt_full_j, FPS, args.min_step_interval_s)
        pred_full_features = full_motion_features(pred_full_j, FPS, args.min_step_interval_s)
        dynamics_features = motion_dynamics_features(gt_pose_j, pred_pose_j, FPS)
        traj_features = trajectory_features(gt_full_j, pred_full_j, FPS)
        skating_features = foot_skating_features(gt_full_j, pred_full_j, FPS)
        com_features = com_proxy_features(gt_full_j, pred_full_j, FPS)

        row.update(prefixed("gt_pose", gt_pose_features))
        row.update(prefixed("pred_pose", pred_pose_features))
        row.update(prefixed("gt_full", gt_full_features))
        row.update(prefixed("pred_full", pred_full_features))
        row.update(dynamics_features)
        row.update(traj_features)
        row.update(skating_features)
        row.update(com_features)
        add_preservation_columns(row, pose_feature_names, "gt_pose", "pred_pose", "pose")
        add_preservation_columns(row, full_feature_names, "gt_full", "pred_full", "full")
        rows.append(row)

    df = pd.DataFrame(rows)
    per_walk_path = os.path.join(args.out_dir, f"{args.file_prefix}_per_walk.csv")
    df.to_csv(per_walk_path, index=False)

    numeric_cols = [c for c in df.columns if c not in {"label", "subject", "walk", "medication", "other"}]
    grouped = df.groupby("UPDRS_GAIT", dropna=False)[numeric_cols].mean(numeric_only=True)
    grouped_path = os.path.join(args.out_dir, f"{args.file_prefix}_grouped_by_updrs.csv")
    grouped.to_csv(grouped_path)

    summary = summarize_dataframe(df)
    summary_path = os.path.join(args.out_dir, f"{args.file_prefix}_summary.csv")
    summary.to_csv(summary_path, index=False)

    corr = correlations(df)
    corr_path = os.path.join(args.out_dir, f"{args.file_prefix}_correlations.csv")
    corr.to_csv(corr_path, index=False)

    catalog = metric_catalog()
    catalog_path = os.path.join(args.out_dir, f"{args.file_prefix}_catalog.csv")
    catalog.to_csv(catalog_path, index=False)

    meta = {
        "predictions": args.predictions,
        "n_sequences": int(len(df)),
        "metric_families": ["pose_fidelity", "motion_dynamics", "gait_biomechanics", "supplementary"],
        "caveats": [
            "Full-motion metrics depend on predicted global translation.",
            "Acceleration and jerk errors are noise-sensitive and should be treated as QA/exploratory.",
            "COM metrics are pelvis/joint-centroid proxies, not mass-weighted whole-body COM.",
            "Foot skating uses GT-derived contact masks to avoid relying on predicted contact labels.",
            "Correlations include raw Spearman rows and duration-controlled partial Spearman rows based on rank residuals.",
            "mpjpe_predicted_* columns exclude reduced-pose placeholder joints and should be preferred for main region-wise position error.",
        ],
        "outputs": {
            "per_walk": per_walk_path,
            "grouped_by_updrs": grouped_path,
            "summary": summary_path,
            "correlations": corr_path,
            "catalog": catalog_path,
        },
    }
    with open(os.path.join(args.out_dir, f"{args.file_prefix}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved per-walk : {per_walk_path}")
    print(f"Saved grouped  : {grouped_path}")
    print(f"Saved summary  : {summary_path}")
    print(f"Saved corr     : {corr_path}")
    print(f"Saved catalog  : {catalog_path}")


if __name__ == "__main__":
    main()
