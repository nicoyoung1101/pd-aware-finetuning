"""
analyze_arm_swing_preservation.py

Analyze whether TransPose preserves Parkinsonian arm swing features.

Main definition:
  - Build a torso-local coordinate frame from SMPL joints for each frame.
  - Measure upper-arm forward/backward swing angle in the torso sagittal plane.
  - Use P95-P5 range as the robust arm swing amplitude.

This is a pose-only metric: GT and predicted joints are both computed with
zero translation through the same SMPL FK model, so TransPose root drift does
not affect the result.

Example:
  python Evaluation/analyze_arm_swing_preservation.py
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
from scipy.signal import savgol_filter
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRANSPOSE_ROOT = os.path.join(PROJECT_ROOT, "TransPose")
if TRANSPOSE_ROOT not in sys.path:
    sys.path.insert(0, TRANSPOSE_ROOT)

import articulate as art
from config import paths


PREDICTIONS_PT = os.path.join(TRANSPOSE_ROOT, "data", "results", "CAREPD_BMCLab_batch", "predictions.pt")
OUT_DIR = os.path.join(TRANSPOSE_ROOT, "data", "results", "CAREPD_BMCLab_arm_swing")


SMPL_JOINT = {
    "pelvis": 0,
    "neck": 12,
    "lshoulder": 16,
    "rshoulder": 17,
    "lelbow": 18,
    "relbow": 19,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=PREDICTIONS_PT, help="Batch predictions.pt from run_carepd_transpose_batch.py")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory for tables and figures")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="FK device")
    parser.add_argument("--limit", type=int, default=0, help="Optional sequence limit for smoke tests")
    parser.add_argument("--eps", type=float, default=1e-6, help="Small epsilon for degenerate-frame checks")
    parser.add_argument("--smooth-savgol", action="store_true", help="Optionally smooth angle traces before statistics")
    parser.add_argument("--savgol-window", type=int, default=11, help="Odd Savitzky-Golay window length")
    parser.add_argument("--savgol-poly", type=int, default=2, help="Savitzky-Golay polynomial order")
    parser.add_argument("--check-trans-invariance", type=int, default=5,
                        help="Number of sequences used to verify that translation does not change the angle")
    return parser.parse_args()


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_np(v, eps):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    ok = norm[..., 0] > eps
    out = np.full_like(v, np.nan, dtype=np.float64)
    out[ok] = v[ok] / norm[ok]
    return out, ok


def interpolate_nan_1d(x):
    x = np.asarray(x, dtype=np.float64).copy()
    valid = np.isfinite(x)
    if valid.sum() == 0:
        return x
    if valid.sum() == 1:
        x[~valid] = x[valid][0]
        return x
    idx = np.arange(x.shape[0])
    x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return x


def maybe_smooth_angle(angle, args):
    angle = interpolate_nan_1d(angle)
    angle = np.unwrap(angle)
    if not args.smooth_savgol:
        return angle

    n = angle.shape[0]
    window = min(args.savgol_window, n if n % 2 == 1 else n - 1)
    min_window = args.savgol_poly + 2
    if min_window % 2 == 0:
        min_window += 1
    if window < min_window:
        return angle
    return savgol_filter(angle, window_length=window, polyorder=args.savgol_poly, mode="interp")


def compute_torso_frame(joints, eps):
    pelvis = joints[:, SMPL_JOINT["pelvis"]]
    neck = joints[:, SMPL_JOINT["neck"]]
    lshoulder = joints[:, SMPL_JOINT["lshoulder"]]
    rshoulder = joints[:, SMPL_JOINT["rshoulder"]]

    x_raw = rshoulder - lshoulder
    y_raw = neck - pelvis
    x_body, ok_x = normalize_np(x_raw, eps)
    y_body, ok_y = normalize_np(y_raw, eps)

    z_raw = np.cross(x_body, y_body)
    z_body, ok_z = normalize_np(z_raw, eps)
    x_ortho_raw = np.cross(y_body, z_body)
    x_body, ok_x2 = normalize_np(x_ortho_raw, eps)

    valid = ok_x & ok_y & ok_z & ok_x2
    frame = np.stack([x_body, y_body, z_body], axis=-2)
    frame[~valid] = np.nan
    return frame, valid


def arm_angles_from_joints(joints, args):
    frame, valid_frame = compute_torso_frame(joints, args.eps)

    lvec = joints[:, SMPL_JOINT["lelbow"]] - joints[:, SMPL_JOINT["lshoulder"]]
    rvec = joints[:, SMPL_JOINT["relbow"]] - joints[:, SMPL_JOINT["rshoulder"]]

    _, ok_l = normalize_np(lvec, args.eps)
    _, ok_r = normalize_np(rvec, args.eps)

    def one_side_angle(vec, ok_vec):
        x_axis = frame[:, 0]
        y_axis = frame[:, 1]
        z_axis = frame[:, 2]
        v_up = np.einsum("ij,ij->i", vec, y_axis)
        v_forward = np.einsum("ij,ij->i", vec, z_axis)
        angle = np.arctan2(v_forward, -v_up)
        valid = valid_frame & ok_vec & np.isfinite(angle)
        angle[~valid] = np.nan
        return maybe_smooth_angle(angle, args), valid

    left_angle, left_valid = one_side_angle(lvec, ok_l)
    right_angle, right_valid = one_side_angle(rvec, ok_r)
    return left_angle, right_angle, left_valid, right_valid, valid_frame


def angle_stats(angle_rad, valid):
    a = angle_rad[valid & np.isfinite(angle_rad)]
    if a.shape[0] < 5:
        return {
            "p95p5_deg": np.nan,
            "maxmin_deg": np.nan,
            "std_deg": np.nan,
            "n_valid": int(a.shape[0]),
            "valid_frac": float(a.shape[0] / max(1, angle_rad.shape[0])),
        }
    deg = np.rad2deg(a)
    return {
        "p95p5_deg": float(np.percentile(deg, 95) - np.percentile(deg, 5)),
        "maxmin_deg": float(np.max(deg) - np.min(deg)),
        "std_deg": float(np.std(deg)),
        "n_valid": int(a.shape[0]),
        "valid_frac": float(a.shape[0] / max(1, angle_rad.shape[0])),
    }


def summarize_arm_swing(joints, args):
    left_angle, right_angle, left_valid, right_valid, frame_valid = arm_angles_from_joints(joints, args)
    left = angle_stats(left_angle, left_valid)
    right = angle_stats(right_angle, right_valid)

    out = {}
    for side, stats_dict in [("left", left), ("right", right)]:
        for key, value in stats_dict.items():
            out[f"{side}_{key}"] = value

    for stat_name in ["p95p5_deg", "maxmin_deg", "std_deg"]:
        lval = left[stat_name]
        rval = right[stat_name]
        out[f"mean_{stat_name}"] = float(np.nanmean([lval, rval]))
        out[f"asym_{stat_name}"] = float(abs(lval - rval) / (lval + rval + 1e-6)) if np.isfinite(lval + rval) else np.nan

    out["torso_frame_valid_frac"] = float(np.mean(frame_valid))
    return out, {
        "left_angle_rad": left_angle,
        "right_angle_rad": right_angle,
        "left_valid": left_valid,
        "right_valid": right_valid,
        "torso_frame_valid": frame_valid,
    }


@torch.no_grad()
def fk_joints(model, pose, device, tran=None):
    pose = pose.to(device).float()
    if tran is not None:
        tran = tran.to(device).float()
    _, joints = model.forward_kinematics(pose, shape=None, tran=tran, calc_mesh=False)
    return joints.cpu().numpy()


def add_prefixed(row, prefix, values):
    for key, value in values.items():
        row[f"{prefix}_{key}"] = value


def add_ratios(row):
    for side in ["left", "right", "mean"]:
        for stat_name in ["p95p5_deg", "maxmin_deg", "std_deg"]:
            g = row.get(f"gt_{side}_{stat_name}", np.nan)
            p = row.get(f"pred_{side}_{stat_name}", np.nan)
            row[f"ratio_{side}_{stat_name}"] = float(p / (g + 1e-6)) if np.isfinite(g) and np.isfinite(p) else np.nan

    for stat_name in ["p95p5_deg", "maxmin_deg", "std_deg"]:
        g = row.get(f"gt_asym_{stat_name}", np.nan)
        p = row.get(f"pred_asym_{stat_name}", np.nan)
        row[f"ratio_asym_{stat_name}"] = float(p / (g + 1e-6)) if np.isfinite(g) and np.isfinite(p) else np.nan
        row[f"diff_asym_{stat_name}"] = float(p - g) if np.isfinite(g) and np.isfinite(p) else np.nan


def finite_corr(x, y, method):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())
    if np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return np.nan, np.nan, int(mask.sum())
    if method == "pearson":
        r, p = stats.pearsonr(x[mask], y[mask])
    elif method == "spearman":
        r, p = stats.spearmanr(x[mask], y[mask])
    else:
        raise ValueError(method)
    return float(r), float(p), int(mask.sum())


def write_correlations(df, path):
    targets = [
        "gt_mean_p95p5_deg", "pred_mean_p95p5_deg", "ratio_mean_p95p5_deg",
        "gt_asym_p95p5_deg", "pred_asym_p95p5_deg", "ratio_asym_p95p5_deg",
    ]
    rows = []
    for metric in targets:
        if metric not in df:
            continue
        for method in ["pearson", "spearman"]:
            r, p, n = finite_corr(df[metric], df["UPDRS_GAIT"], method)
            rows.append({"x": metric, "y": "UPDRS_GAIT", "method": method, "r": r, "p": p, "n": n})

    paired = [
        ("gt_mean_p95p5_deg", "pred_mean_p95p5_deg"),
        ("gt_asym_p95p5_deg", "pred_asym_p95p5_deg"),
        ("gt_left_p95p5_deg", "pred_left_p95p5_deg"),
        ("gt_right_p95p5_deg", "pred_right_p95p5_deg"),
    ]
    for x, y in paired:
        for method in ["pearson", "spearman"]:
            r, p, n = finite_corr(df[x], df[y], method)
            rows.append({"x": x, "y": y, "method": method, "r": r, "p": p, "n": n})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y", "method", "r", "p", "n"])
        writer.writeheader()
        writer.writerows(rows)


def boxplot_by_group(df, value_cols, group_col, labels, title, ylabel, path):
    groups = sorted([g for g in df[group_col].dropna().unique()])
    if not groups:
        return
    fig, axes = plt.subplots(1, len(value_cols), figsize=(5 * len(value_cols), 4), squeeze=False)
    for ax, value_col, label in zip(axes[0], value_cols, labels):
        data = [df.loc[df[group_col] == g, value_col].dropna().values for g in groups]
        ax.boxplot(data, tick_labels=[str(g) for g in groups], showfliers=False)
        ax.scatter(
            np.concatenate([np.full(len(vals), i + 1) for i, vals in enumerate(data) if len(vals)]),
            np.concatenate([vals for vals in data if len(vals)]) if any(len(vals) for vals in data) else [],
            s=10, alpha=0.25,
        )
        ax.set_title(label)
        ax.set_xlabel(group_col)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def scatter_gt_pred(df, x_col, y_col, title, xlabel, ylabel, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    updrs_values = sorted([g for g in df["UPDRS_GAIT"].dropna().unique()])
    for g in updrs_values:
        sub = df[df["UPDRS_GAIT"] == g]
        ax.scatter(sub[x_col], sub[y_col], s=18, alpha=0.65, label=f"U={g}")
    finite = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        lo = float(finite.min().min())
        hi = float(finite.max().max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figures(df, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    boxplot_by_group(
        df,
        ["gt_mean_p95p5_deg", "pred_mean_p95p5_deg", "ratio_mean_p95p5_deg"],
        "UPDRS_GAIT",
        ["GT mean amplitude", "Pred mean amplitude", "Pred/GT amplitude"],
        "Arm Swing Amplitude By UPDRS",
        "degrees / ratio",
        os.path.join(fig_dir, "arm_swing_amplitude_by_updrs.png"),
    )
    boxplot_by_group(
        df,
        ["gt_asym_p95p5_deg", "pred_asym_p95p5_deg", "diff_asym_p95p5_deg"],
        "UPDRS_GAIT",
        ["GT asymmetry", "Pred asymmetry", "Pred - GT asymmetry"],
        "Arm Swing Asymmetry By UPDRS",
        "asymmetry",
        os.path.join(fig_dir, "arm_swing_asymmetry_by_updrs.png"),
    )
    scatter_gt_pred(
        df,
        "gt_mean_p95p5_deg",
        "pred_mean_p95p5_deg",
        "GT vs Pred Arm Swing Amplitude",
        "GT mean P95-P5 amplitude (deg)",
        "Pred mean P95-P5 amplitude (deg)",
        os.path.join(fig_dir, "scatter_gt_pred_amplitude.png"),
    )
    scatter_gt_pred(
        df,
        "gt_asym_p95p5_deg",
        "pred_asym_p95p5_deg",
        "GT vs Pred Arm Swing Asymmetry",
        "GT asymmetry",
        "Pred asymmetry",
        os.path.join(fig_dir, "scatter_gt_pred_asymmetry.png"),
    )
    if "medication" in df:
        boxplot_by_group(
            df,
            ["gt_mean_p95p5_deg", "pred_mean_p95p5_deg", "ratio_mean_p95p5_deg"],
            "medication",
            ["GT mean amplitude", "Pred mean amplitude", "Pred/GT amplitude"],
            "Arm Swing Amplitude By Medication",
            "degrees / ratio",
            os.path.join(fig_dir, "arm_swing_amplitude_by_medication.png"),
        )


def check_translation_invariance(model, pred, row_index, device, args):
    pose_gt = pred["pose_gt"][row_index]
    tran_gt = pred["tran_gt"][row_index]
    pose_pred = pred["pose_pred"][row_index]
    tran_pred = pred["tran_pred"][row_index]

    gt_zero = fk_joints(model, pose_gt, device, tran=None)
    gt_tran = fk_joints(model, pose_gt, device, tran=tran_gt)
    pred_zero = fk_joints(model, pose_pred, device, tran=None)
    pred_tran = fk_joints(model, pose_pred, device, tran=tran_pred)

    def max_angle_delta(a, b):
        la, ra, _, _, _ = arm_angles_from_joints(a, args)
        lb, rb, _, _, _ = arm_angles_from_joints(b, args)
        return float(np.nanmax(np.abs(np.concatenate([la - lb, ra - rb]))))

    return {
        "gt_max_angle_delta_rad": max_angle_delta(gt_zero, gt_tran),
        "pred_max_angle_delta_rad": max_angle_delta(pred_zero, pred_tran),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"Missing predictions file: {args.predictions}")

    device = torch.device(args.device)
    pred = safe_torch_load(args.predictions, map_location="cpu")
    model = art.ParametricModel(paths.smpl_file, device=device)

    n = len(pred["pose_pred"])
    if args.limit > 0:
        n = min(n, args.limit)

    rows = []
    angle_cache = {}
    trans_checks = []

    for i in tqdm(range(n), desc="Arm swing analysis"):
        item = pred["manifest"][i] or {}
        row = {
            "index": pred["index"][i],
            "label": pred["label"][i],
            "subject_id": item.get("subject_id"),
            "walk_id": item.get("walk_id"),
            "UPDRS_GAIT": item.get("UPDRS_GAIT"),
            "medication": item.get("medication"),
            "other": item.get("other"),
            "n_frames": int(pred["pose_gt"][i].shape[0]),
        }

        joints_gt = fk_joints(model, pred["pose_gt"][i], device, tran=None)
        joints_pred = fk_joints(model, pred["pose_pred"][i], device, tran=None)

        gt_stats, gt_angles = summarize_arm_swing(joints_gt, args)
        pred_stats, pred_angles = summarize_arm_swing(joints_pred, args)
        add_prefixed(row, "gt", gt_stats)
        add_prefixed(row, "pred", pred_stats)
        add_ratios(row)
        rows.append(row)

        angle_cache[int(pred["index"][i])] = {
            "label": pred["label"][i],
            "gt": gt_angles,
            "pred": pred_angles,
        }

        if i < args.check_trans_invariance:
            check = check_translation_invariance(model, pred, i, device, args)
            check["index"] = pred["index"][i]
            check["label"] = pred["label"][i]
            trans_checks.append(check)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "arm_swing_summary.csv")
    pt_path = os.path.join(args.out_dir, "arm_swing_summary.pt")
    corr_path = os.path.join(args.out_dir, "arm_swing_correlations.csv")
    check_path = os.path.join(args.out_dir, "translation_invariance_check.json")

    df.to_csv(csv_path, index=False)
    torch.save({"rows": rows, "angle_cache": angle_cache, "args": vars(args)}, pt_path)
    write_correlations(df, corr_path)
    with open(check_path, "w", encoding="utf-8") as f:
        json.dump(trans_checks, f, indent=2)

    make_figures(df, os.path.join(args.out_dir, "figures"))

    print(f"Saved summary      : {csv_path}")
    print(f"Saved full output  : {pt_path}")
    print(f"Saved correlations : {corr_path}")
    print(f"Saved checks       : {check_path}")
    print(f"Saved figures      : {os.path.join(args.out_dir, 'figures')}")
    print()
    print("Quick means:")
    for col in ["gt_mean_p95p5_deg", "pred_mean_p95p5_deg", "ratio_mean_p95p5_deg",
                "gt_asym_p95p5_deg", "pred_asym_p95p5_deg"]:
        print(f"  {col}: {df[col].mean():.4f}")


if __name__ == "__main__":
    main()
