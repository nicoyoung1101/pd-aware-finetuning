"""
analyze_pose_fidelity.py

Standard pose fidelity analysis for CARE-PD / TransPose predictions.

This complements the PD gait-feature preservation analysis. It answers:
  - How accurate is the reconstructed SMPL pose in standard joint metrics?
  - Which body regions / joints carry larger reconstruction errors?

Important implementation choices:
  - MPJPE is root-aligned by subtracting the pelvis joint before computing
    Euclidean joint errors. This keeps MPJPE in the pose-fidelity layer and
    prevents TransPose root translation from contaminating pose error.
  - PA-MPJPE uses per-frame Procrustes alignment. Because it includes scale
    alignment, it is reported only as a standard pose-shape metric, not as
    evidence for movement-amplitude preservation.
  - Region definitions are loaded from the local pose drift QA meta file so
    region-wise MPJPE and local rotation-error QA use the same body parts.
  - Region-wise local rotation error is reused from the drift QA output instead
    of recomputed, avoiding silent inconsistencies across analyses.

Example:
  python Evaluation/analyze_pose_fidelity.py
"""

# ========== numpy compatibility patch for old SMPL/chumpy-style pickles ==========
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

import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRANSPOSE_ROOT = os.path.join(PROJECT_ROOT, "TransPose")
if TRANSPOSE_ROOT not in sys.path:
    sys.path.insert(0, TRANSPOSE_ROOT)

import articulate as art
from config import paths


PREDICTIONS_PT = os.path.join(
    TRANSPOSE_ROOT, "data", "results", "CAREPD_BMCLab_batch", "predictions.pt"
)
DRIFT_QA_DIR = os.path.join(
    TRANSPOSE_ROOT, "data", "results", "CAREPD_BMCLab_local_pose_drift_qa"
)
OUT_DIR = os.path.join(
    TRANSPOSE_ROOT, "data", "results", "CAREPD_BMCLab_pose_fidelity"
)
FPS = 60.0


SMPL_JOINT = {
    "pelvis": 0,
    "lhip": 1,
    "rhip": 2,
    "spine1": 3,
    "lknee": 4,
    "rknee": 5,
    "spine2": 6,
    "lankle": 7,
    "rankle": 8,
    "spine3": 9,
    "lfoot": 10,
    "rfoot": 11,
    "neck": 12,
    "lcollar": 13,
    "rcollar": 14,
    "head": 15,
    "lshoulder": 16,
    "rshoulder": 17,
    "lelbow": 18,
    "relbow": 19,
    "lwrist": 20,
    "rwrist": 21,
    "lhand": 22,
    "rhand": 23,
}
JOINT_NAMES = [name for name, _ in sorted(SMPL_JOINT.items(), key=lambda kv: kv[1])]
IGNORED_ROTATION_JOINTS = {"pelvis", "lankle", "rankle", "lfoot", "rfoot", "lwrist", "rwrist", "lhand", "rhand"}
# These joints are identity/copy placeholders in the reduced-pose protocols used
# by TransPose/PIP/DynaIP. Keep the legacy all-position MPJPE for diagnostics,
# but use this predicted-only set for main region-wise position error.
IGNORED_POSITION_JOINTS = IGNORED_ROTATION_JOINTS
POSITION_BODY_PARTS = {
    "trunk": ["spine1", "spine2", "spine3", "neck", "head"],
    "left_arm": ["lcollar", "lshoulder", "lelbow", "lwrist", "lhand"],
    "right_arm": ["rcollar", "rshoulder", "relbow", "rwrist", "rhand"],
    "left_leg": ["lhip", "lknee", "lankle", "lfoot"],
    "right_leg": ["rhip", "rknee", "rankle", "rfoot"],
}
ROTATION_BODY_PARTS = {
    "trunk": ["spine1", "spine2", "spine3", "neck", "head"],
    "left_arm": ["lcollar", "lshoulder", "lelbow"],
    "right_arm": ["rcollar", "rshoulder", "relbow"],
    "left_leg": ["lhip", "lknee"],
    "right_leg": ["rhip", "rknee"],
}
PREDICTED_POSITION_BODY_PARTS = ROTATION_BODY_PARTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=PREDICTIONS_PT)
    parser.add_argument("--drift-qa-dir", default=DRIFT_QA_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_body_parts(drift_qa_dir):
    """Load rotation-evaluation body parts from drift QA metadata."""
    meta_path = os.path.join(drift_qa_dir, "local_pose_drift_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta["body_parts"]
    return ROTATION_BODY_PARTS


def region_indices(body_parts, *, include_full=True, excluded_full_joints=None):
    regions = {name: [SMPL_JOINT[j] for j in joints] for name, joints in body_parts.items()}
    regions["upper_limbs"] = sorted(set(regions["left_arm"] + regions["right_arm"]))
    regions["lower_body"] = sorted(set(regions["left_leg"] + regions["right_leg"]))
    if include_full:
        excluded = set(excluded_full_joints or [])
        regions["full_body"] = [i for i in range(24) if i not in excluded]
        regions["full_body_no_root"] = [i for i in range(24) if i != SMPL_JOINT["pelvis"] and i not in excluded]
    return regions


@torch.no_grad()
def fk_joints(model, pose, device, batch_size=2048):
    pose = pose.to(device).float()
    chunks = []
    for start in range(0, pose.shape[0], batch_size):
        _, joints = model.forward_kinematics(
            pose[start : start + batch_size],
            shape=None,
            tran=None,
            calc_mesh=False,
        )
        chunks.append(joints.cpu())
    return torch.cat(chunks, dim=0).numpy()


def root_align(joints):
    return joints - joints[:, SMPL_JOINT["pelvis"] : SMPL_JOINT["pelvis"] + 1]


def per_joint_mpjpe_mm(pred_joints, gt_joints):
    pred = root_align(pred_joints)
    gt = root_align(gt_joints)
    return np.linalg.norm(pred - gt, axis=-1) * 1000.0


def procrustes_aligned_mpjpe_mm(pred_joints, gt_joints, joint_indices=None):
    """
    Per-frame PA-MPJPE with similarity Procrustes alignment.

    pred_joints, gt_joints: [T, J, 3]
    returns: scalar in mm
    """
    errors = []
    for x, y in zip(pred_joints, gt_joints):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if joint_indices is not None:
            x = x[joint_indices]
            y = y[joint_indices]
        finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x = x[finite]
        y = y[finite]
        if x.shape[0] < 3:
            continue
        mu_x = x.mean(axis=0, keepdims=True)
        mu_y = y.mean(axis=0, keepdims=True)
        x0 = x - mu_x
        y0 = y - mu_y
        norm_x = np.linalg.norm(x0)
        norm_y = np.linalg.norm(y0)
        if norm_x < 1e-9 or norm_y < 1e-9:
            continue
        x0n = x0 / norm_x
        y0n = y0 / norm_y
        # Row-vector Kabsch: minimize ||X @ R - Y||, with H = X.T @ Y.
        # If H = U S Vt, the optimal rotation is R = U @ Vt.
        h = x0n.T @ y0n
        u, s, vt = np.linalg.svd(h)
        r = u @ vt
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = u @ vt
            s[-1] *= -1
        scale = (s.sum() * norm_y) / norm_x
        x_aligned = scale * (x0 @ r) + mu_y
        errors.append(np.linalg.norm(x_aligned - y, axis=1).mean() * 1000.0)
    return float(np.nanmean(errors)) if errors else np.nan


def local_rotation_geodesic_deg(pred, gt):
    pred = pred.detach().cpu().float()
    gt = gt.detach().cpu().float()
    rel = torch.matmul(pred, gt.transpose(-1, -2))
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos)).numpy()


def mean_for_indices(frame_joint_values, indices):
    return float(np.nanmean(frame_joint_values[:, indices]))


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or len(np.unique(x[mask])) < 2:
        return np.nan, np.nan, int(mask.sum())
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), int(mask.sum())


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_drift_rotation_summary(drift_qa_dir):
    path = os.path.join(drift_qa_dir, "local_pose_drift_summary.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df.set_index("index")


def make_figures(df, out_dir, regions):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Region-wise MPJPE over predicted joints only. The legacy all-joint MPJPE
    # columns are kept in CSVs for diagnostics because distal placeholder joints
    # can duplicate their upstream parent errors.
    region_order = ["full_body_no_root", "trunk", "upper_limbs", "lower_body", "left_leg", "right_leg"]
    region_labels = ["Full body", "Trunk", "Upper limbs", "Lower body", "L leg", "R leg"]
    means = [df[f"mpjpe_predicted_{r}_mm"].mean() for r in region_order]
    sems = [df[f"mpjpe_predicted_{r}_mm"].sem() for r in region_order]
    plt.figure(figsize=(9, 4.8))
    plt.bar(region_labels, means, yerr=sems, color="#4C78A8", alpha=0.85)
    plt.ylabel("Root-aligned MPJPE (mm)")
    plt.title("Region-wise pose position error (predicted joints only)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "region_wise_mpjpe.png"), dpi=180)
    plt.close()

    # Joint-wise MPJPE.
    joint_means = [df[f"joint_mpjpe_{name}_mm"].mean() for name in JOINT_NAMES]
    plt.figure(figsize=(12, 5.2))
    plt.bar(JOINT_NAMES, joint_means, color="#72B7B2", alpha=0.9)
    plt.ylabel("Root-aligned MPJPE (mm)")
    plt.title("Joint-wise pose position error (diagnostic; includes placeholders)")
    plt.xticks(rotation=55, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "joint_wise_mpjpe.png"), dpi=180)
    plt.close()

    # Region-wise local rotation error reused / matched to drift QA.
    rot_regions = ["trunk", "left_arm", "right_arm", "left_leg", "right_leg"]
    rot_labels = ["Trunk", "L arm", "R arm", "L leg", "R leg"]
    means = [df[f"rot_{r}_deg"].mean() for r in rot_regions]
    sems = [df[f"rot_{r}_deg"].sem() for r in rot_regions]
    plt.figure(figsize=(8, 4.8))
    plt.bar(rot_labels, means, yerr=sems, color="#F58518", alpha=0.85)
    plt.ylabel("Local joint rotation error (deg)")
    plt.title("Region-wise local rotation error")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "region_wise_rotation_error.png"), dpi=180)
    plt.close()

    # MPJPE by UPDRS.
    groups = sorted([g for g in df["UPDRS_GAIT"].dropna().unique()])
    plt.figure(figsize=(8.5, 4.8))
    width = 0.26
    for offset, region, label, color in [
        (-width, "trunk", "Trunk", "#4C78A8"),
        (0, "upper_limbs", "Upper limbs", "#F58518"),
        (width, "lower_body", "Lower body", "#54A24B"),
    ]:
        vals = [df.loc[df["UPDRS_GAIT"] == g, f"mpjpe_predicted_{region}_mm"].mean() for g in groups]
        plt.bar(np.asarray(groups, dtype=float) + offset, vals, width=width, label=label, color=color, alpha=0.85)
    plt.xticks(groups)
    plt.xlabel("UPDRS-Gait")
    plt.ylabel("Root-aligned MPJPE (mm)")
    plt.title("Region-wise predicted-joint MPJPE by UPDRS-Gait")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "region_mpjpe_by_updrs.png"), dpi=180)
    plt.close()

    # Left-right lower body comparison.
    plt.figure(figsize=(7.5, 4.8))
    labels = ["MPJPE L leg", "MPJPE R leg", "Rot L leg", "Rot R leg"]
    vals = [
        df["mpjpe_predicted_left_leg_mm"].mean(),
        df["mpjpe_predicted_right_leg_mm"].mean(),
        df["rot_left_leg_deg"].mean(),
        df["rot_right_leg_deg"].mean(),
    ]
    plt.bar(labels, vals, color=["#54A24B", "#B279A2", "#54A24B", "#B279A2"], alpha=0.85)
    plt.title("Left-right lower-limb reconstruction bias")
    plt.ylabel("mm for MPJPE / deg for rotation")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "left_right_lower_limb_bias.png"), dpi=180)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = safe_torch_load(args.predictions, map_location="cpu")
    rotation_body_parts = load_body_parts(args.drift_qa_dir)
    position_regions = region_indices(POSITION_BODY_PARTS)
    ignored_rotation_indices = {SMPL_JOINT[j] for j in IGNORED_ROTATION_JOINTS}
    ignored_position_indices = {SMPL_JOINT[j] for j in IGNORED_POSITION_JOINTS}
    predicted_position_regions = region_indices(
        PREDICTED_POSITION_BODY_PARTS,
        excluded_full_joints=ignored_position_indices,
    )
    rotation_regions = region_indices(
        rotation_body_parts,
        excluded_full_joints=ignored_rotation_indices,
    )
    drift_summary = load_drift_rotation_summary(args.drift_qa_dir)

    device = torch.device(args.device)
    model = art.ParametricModel(paths.smpl_file, device=device)

    n_total = len(data["pose_gt"])
    n_use = min(args.limit, n_total) if args.limit else n_total
    rows = []

    for i in tqdm(range(n_use), desc="Pose fidelity"):
        pose_gt = data["pose_gt"][i]
        pose_pred = data["pose_pred"][i]
        n = min(pose_gt.shape[0], pose_pred.shape[0])
        pose_gt = pose_gt[:n]
        pose_pred = pose_pred[:n]
        manifest = data["manifest"][i]
        fps = float(manifest.get("target_fps", FPS))

        gt_joints = fk_joints(model, pose_gt, device, args.batch_size)
        pred_joints = fk_joints(model, pose_pred, device, args.batch_size)
        joint_err = per_joint_mpjpe_mm(pred_joints, gt_joints)
        rot_err = local_rotation_geodesic_deg(pose_pred, pose_gt)

        row = {
            "index": data["index"][i],
            "label": data["label"][i],
            "subject_id": manifest.get("subject_id"),
            "walk_id": manifest.get("walk_id"),
            "UPDRS_GAIT": manifest.get("UPDRS_GAIT"),
            "medication": manifest.get("medication"),
            "other": manifest.get("other"),
            "n_frames": n,
            "fps": fps,
            "duration_s": n / fps,
            "pa_mpjpe_full_body_mm": procrustes_aligned_mpjpe_mm(pred_joints, gt_joints),
            "pa_mpjpe_predicted_full_body_mm": procrustes_aligned_mpjpe_mm(
                pred_joints,
                gt_joints,
                predicted_position_regions["full_body"],
            ),
        }

        # Legacy all-position MPJPE: useful as a diagnostic, but can include
        # reduced-pose placeholder joints such as ankle/wrist/hand/foot.
        for region, idx in position_regions.items():
            row[f"mpjpe_{region}_mm"] = mean_for_indices(joint_err, idx)

        # Main MPJPE over independently predicted joints only.
        for region, idx in predicted_position_regions.items():
            row[f"mpjpe_predicted_{region}_mm"] = mean_for_indices(joint_err, idx)

        # Region-wise local rotation error follows the official reduced-pose
        # protocol: ignored joints are excluded because they are identity
        # placeholders, not model predictions.
        for region in rotation_regions:
            if drift_summary is not None and data["index"][i] in drift_summary.index and f"{region}_mean_error_deg" in drift_summary.columns:
                row[f"rot_{region}_deg"] = float(
                    drift_summary.loc[data["index"][i], f"{region}_mean_error_deg"]
                )
            else:
                row[f"rot_{region}_deg"] = mean_for_indices(rot_err, rotation_regions[region])

        for name, idx in SMPL_JOINT.items():
            row[f"joint_mpjpe_{name}_mm"] = float(np.nanmean(joint_err[:, idx]))
            row[f"joint_rot_{name}_deg"] = (
                np.nan if name in IGNORED_ROTATION_JOINTS else float(np.nanmean(rot_err[:, idx]))
            )

        row["mpjpe_right_minus_left_leg_mm"] = row["mpjpe_right_leg_mm"] - row["mpjpe_left_leg_mm"]
        row["mpjpe_predicted_right_minus_left_leg_mm"] = (
            row["mpjpe_predicted_right_leg_mm"] - row["mpjpe_predicted_left_leg_mm"]
        )
        row["rot_right_minus_left_leg_deg"] = row["rot_right_leg_deg"] - row["rot_left_leg_deg"]
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    summary_csv = os.path.join(args.out_dir, "pose_fidelity_summary.csv")
    write_csv(summary_csv, rows, fieldnames)

    df = pd.DataFrame(rows)
    subject_cols = ["subject_id", "medication", "UPDRS_GAIT"]
    non_metric_cols = ["label", "subject_id", "walk_id", "medication", "other", *subject_cols]
    numeric_cols = [c for c in df.columns if c not in non_metric_cols]
    subject_df = df.groupby(subject_cols, dropna=False)[numeric_cols].mean().reset_index()
    subject_csv = os.path.join(args.out_dir, "pose_fidelity_subject_state_summary.csv")
    subject_df.to_csv(subject_csv, index=False)

    grouped = {
        "by_updrs": df.groupby("UPDRS_GAIT").mean(numeric_only=True).round(5).to_dict(),
        "subject_state_by_updrs": subject_df.groupby("UPDRS_GAIT").mean(numeric_only=True).round(5).to_dict(),
    }
    with open(os.path.join(args.out_dir, "pose_fidelity_grouped_means.json"), "w") as f:
        json.dump(grouped, f, indent=2)

    corr_rows = []
    metric_cols = [
        c for c in df.columns
        if c.startswith("mpjpe_") or c.startswith("pa_mpjpe_") or c.startswith("rot_") or c.startswith("joint_mpjpe_")
    ]
    for level, corr_df in [("walk", df), ("subject_state", subject_df)]:
        updrs = pd.to_numeric(corr_df["UPDRS_GAIT"], errors="coerce").to_numpy(float)
        for metric in metric_cols:
            if metric not in corr_df:
                continue
            vals = pd.to_numeric(corr_df[metric], errors="coerce").to_numpy(float)
            rho, p, n = spearman(updrs, vals)
            corr_rows.append(
                {
                    "level": level,
                    "metric": metric,
                    "rho_vs_UPDRS_GAIT": rho,
                    "p_vs_UPDRS_GAIT": p,
                    "n": n,
                }
            )
    corr_csv = os.path.join(args.out_dir, "pose_fidelity_correlations.csv")
    write_csv(corr_csv, corr_rows, ["level", "metric", "rho_vs_UPDRS_GAIT", "p_vs_UPDRS_GAIT", "n"])

    make_figures(df, args.out_dir, position_regions)

    meta = {
        "purpose": "Standard pose fidelity / joint-level comparison for CARE-PD TransPose benchmark.",
        "mpjpe": "Root-aligned by subtracting pelvis from GT and prediction before Euclidean joint error.",
        "pa_mpjpe": "Per-frame Procrustes-aligned MPJPE with scale alignment; standard pose-shape metric, not evidence of amplitude preservation.",
        "position_region_definitions": POSITION_BODY_PARTS,
        "predicted_position_region_definitions": PREDICTED_POSITION_BODY_PARTS,
        "rotation_region_definitions": rotation_body_parts,
        "ignored_rotation_joints": sorted(IGNORED_ROTATION_JOINTS),
        "ignored_position_joints_for_main_mpjpe": sorted(IGNORED_POSITION_JOINTS),
        "position_joint_policy": (
            "Legacy mpjpe_* columns include all SMPL position joints for diagnostics. "
            "Main region-wise position error should use mpjpe_predicted_* columns, "
            "which exclude reduced-pose placeholder joints that are not independently predicted."
        ),
        "rotation_joint_policy": "Ignored TransPose/PIP/DynaIP joints are excluded from local rotation error because they are identity placeholders in official reduced-pose evaluation.",
        "region_rotation_error_source": (
            os.path.join(args.drift_qa_dir, "local_pose_drift_summary.csv")
            if drift_summary is not None else "recomputed_local_geodesic"
        ),
        "uses_translation_for_mpjpe": False,
        "uses_translation_for_pa_mpjpe": False,
        "notes": [
            "MPJPE / PA-MPJPE describe geometric pose error and complement, but do not replace, PD gait feature-preservation metrics.",
            "Region-wise local rotation errors are intentionally consistent with the local pose drift QA mean-error values.",
        ],
    }
    with open(os.path.join(args.out_dir, "pose_fidelity_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved walk summary: {summary_csv}")
    print(f"Saved subject-state summary: {subject_csv}")
    print(f"Saved correlations: {corr_csv}")
    print(f"Saved figures to: {os.path.join(args.out_dir, 'figures')}")


if __name__ == "__main__":
    main()
