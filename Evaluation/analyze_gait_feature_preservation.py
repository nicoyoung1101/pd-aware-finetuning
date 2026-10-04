"""
analyze_gait_feature_preservation.py

Comprehensive CARE-PD / TransPose clinical gait feature preservation analysis.

The output separates two feature families:
  1. Pose-only features: computed from SMPL FK with tran=None. These avoid
     TransPose root-translation drift.
  2. Full-motion features: computed with GT / predicted translations. These are
     clinically standard spatial gait features, but should be interpreted with
     the translation caveat.

Example:
  python Evaluation/analyze_gait_feature_preservation.py
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
from scipy.signal import butter, filtfilt, hilbert
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRANSPOSE_ROOT = os.path.join(PROJECT_ROOT, "TransPose")
if TRANSPOSE_ROOT not in sys.path:
    sys.path.insert(0, TRANSPOSE_ROOT)

import articulate as art
from config import paths


PREDICTIONS_PT = os.path.join(TRANSPOSE_ROOT, "data", "results", "CAREPD_BMCLab_batch", "predictions.pt")
OUT_DIR = os.path.join(TRANSPOSE_ROOT, "data", "results", "CAREPD_BMCLab_gait_features")
FPS = 60.0
UP_AXIS = 1
HORIZONTAL_AXES = [0, 2]
AMP_RATIO_MIN_DENOM = 5.0
ASYM_RATIO_MIN_DENOM = 0.05


SMPL_JOINT = {
    "pelvis": 0,
    "lhip": 1,
    "rhip": 2,
    "lknee": 4,
    "rknee": 5,
    "lankle": 7,
    "rankle": 8,
    "lfoot": 10,
    "rfoot": 11,
    "neck": 12,
    "lshoulder": 16,
    "rshoulder": 17,
    "lelbow": 18,
    "relbow": 19,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=PREDICTIONS_PT)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--min-step-interval-s", type=float, default=0.25)
    return parser.parse_args()


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@torch.no_grad()
def fk_joints(model, pose, device, tran=None):
    pose = pose.to(device).float()
    if tran is not None:
        tran = tran.to(device).float()
    _, joints = model.forward_kinematics(pose, shape=None, tran=tran, calc_mesh=False)
    return joints.cpu().numpy()


def normalize(v, eps=1e-6):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.zeros_like(v, dtype=np.float64)
    ok = n[..., 0] > eps
    out[ok] = v[ok] / n[ok]
    out[~ok] = np.nan
    return out, ok


def interp_nan(x):
    x = np.asarray(x, dtype=np.float64).copy()
    valid = np.isfinite(x)
    if valid.sum() == 0:
        return x
    if valid.sum() == 1:
        x[~valid] = x[valid][0]
        return x
    idx = np.arange(len(x))
    x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return x


def torso_frame(joints, eps=1e-6):
    pelvis = joints[:, SMPL_JOINT["pelvis"]]
    neck = joints[:, SMPL_JOINT["neck"]]
    ls = joints[:, SMPL_JOINT["lshoulder"]]
    rs = joints[:, SMPL_JOINT["rshoulder"]]

    x_body, ok_x = normalize(rs - ls, eps)
    y_body, ok_y = normalize(neck - pelvis, eps)
    z_body, ok_z = normalize(np.cross(x_body, y_body), eps)
    x_body, ok_x2 = normalize(np.cross(y_body, z_body), eps)
    valid = ok_x & ok_y & ok_z & ok_x2
    frame = np.stack([x_body, y_body, z_body], axis=-2)
    frame[~valid] = np.nan
    return frame, valid


def sagittal_angle(vec, frame, valid_frame, eps=1e-6):
    _, ok_vec = normalize(vec, eps)
    y_axis = frame[:, 1]
    z_axis = frame[:, 2]
    v_up = np.einsum("ij,ij->i", vec, y_axis)
    v_forward = np.einsum("ij,ij->i", vec, z_axis)
    angle = np.arctan2(v_forward, -v_up)
    valid = valid_frame & ok_vec & np.isfinite(angle)
    angle[~valid] = np.nan
    angle = np.unwrap(interp_nan(angle))
    return angle, valid


def range_stats_deg(angle, valid):
    a = angle[valid & np.isfinite(angle)]
    if len(a) < 5:
        return np.nan, np.nan, np.nan, len(a), len(a) / max(1, len(angle))
    d = np.rad2deg(a)
    return (
        float(np.percentile(d, 95) - np.percentile(d, 5)),
        float(np.max(d) - np.min(d)),
        float(np.std(d)),
        int(len(d)),
        float(len(d) / max(1, len(angle))),
    )


def bilateral_from_angles(left_angle, left_valid, right_angle, right_valid, prefix):
    lp95, lrom, lstd, ln, lvf = range_stats_deg(left_angle, left_valid)
    rp95, rrom, rstd, rn, rvf = range_stats_deg(right_angle, right_valid)
    out = {
        f"{prefix}_left_p95p5_deg": lp95,
        f"{prefix}_right_p95p5_deg": rp95,
        f"{prefix}_mean_p95p5_deg": float(np.nanmean([lp95, rp95])),
        f"{prefix}_asym_p95p5": float(abs(lp95 - rp95) / (lp95 + rp95 + 1e-6)) if np.isfinite(lp95 + rp95) else np.nan,
        f"{prefix}_left_maxmin_deg": lrom,
        f"{prefix}_right_maxmin_deg": rrom,
        f"{prefix}_left_std_deg": lstd,
        f"{prefix}_right_std_deg": rstd,
        f"{prefix}_left_valid_frac": lvf,
        f"{prefix}_right_valid_frac": rvf,
    }
    return out


def arm_swing_features(joints, eps=1e-6):
    frame, valid_frame = torso_frame(joints, eps)
    lvec = joints[:, SMPL_JOINT["lelbow"]] - joints[:, SMPL_JOINT["lshoulder"]]
    rvec = joints[:, SMPL_JOINT["relbow"]] - joints[:, SMPL_JOINT["rshoulder"]]
    la, lv = sagittal_angle(lvec, frame, valid_frame, eps)
    ra, rv = sagittal_angle(rvec, frame, valid_frame, eps)
    out = bilateral_from_angles(la, lv, ra, rv, "arm_amp")
    out["torso_frame_valid_frac"] = float(np.mean(valid_frame))
    return out


def leg_swing_features(joints, eps=1e-6):
    frame, valid_frame = torso_frame(joints, eps)
    lvec = joints[:, SMPL_JOINT["lknee"]] - joints[:, SMPL_JOINT["lhip"]]
    rvec = joints[:, SMPL_JOINT["rknee"]] - joints[:, SMPL_JOINT["rhip"]]
    la, lv = sagittal_angle(lvec, frame, valid_frame, eps)
    ra, rv = sagittal_angle(rvec, frame, valid_frame, eps)
    return bilateral_from_angles(la, lv, ra, rv, "leg_amp")


def foot_lift_features(joints):
    pelvis_y = joints[:, SMPL_JOINT["pelvis"], UP_AXIS]
    lf = joints[:, SMPL_JOINT["lfoot"], UP_AXIS] - pelvis_y
    rf = joints[:, SMPL_JOINT["rfoot"], UP_AXIS] - pelvis_y
    lp = float(np.percentile(lf, 95) - np.percentile(lf, 5))
    rp = float(np.percentile(rf, 95) - np.percentile(rf, 5))
    return {
        "foot_lift_left_p95p5_m": lp,
        "foot_lift_right_p95p5_m": rp,
        "foot_lift_mean_p95p5_m": float(np.nanmean([lp, rp])),
        "foot_lift_asym_p95p5": float(abs(lp - rp) / (lp + rp + 1e-6)),
    }


def trunk_posture_features(joints):
    torso = joints[:, SMPL_JOINT["neck"]] - joints[:, SMPL_JOINT["pelvis"]]
    torso_n, ok = normalize(torso)
    vertical = np.array([0.0, 1.0, 0.0])
    cosang = np.einsum("ij,j->i", torso_n, vertical)
    cosang = np.clip(cosang, -1.0, 1.0)
    lean = np.rad2deg(np.arccos(cosang))
    lean = lean[ok & np.isfinite(lean)]
    return {
        "trunk_lean_mean_deg": float(np.mean(lean)) if len(lean) else np.nan,
        "trunk_lean_std_deg": float(np.std(lean)) if len(lean) else np.nan,
    }


def cadence_from_pose(joints, fps=FPS):
    frame, valid_frame = torso_frame(joints)
    pelvis = joints[:, SMPL_JOINT["pelvis"]]
    lf = joints[:, SMPL_JOINT["lfoot"]] - pelvis
    rf = joints[:, SMPL_JOINT["rfoot"]] - pelvis
    z_axis = frame[:, 2]
    lz = np.einsum("ij,ij->i", lf, z_axis)
    rz = np.einsum("ij,ij->i", rf, z_axis)
    sig = interp_nan(lz - rz)
    sig = sig - np.nanmean(sig)
    n = len(sig)
    if n < int(fps):
        return {
            "cadence_pose_spm": np.nan,
            "step_time_pose_s": np.nan,
            "pose_step_freq_hz": np.nan,
            "cadence_pose_fft_spm": np.nan,
            "pose_step_freq_fft_hz": np.nan,
        }

    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(sig * window)) ** 2
    freq = np.fft.rfftfreq(n, d=1.0 / fps)
    mask = (freq >= 0.4) & (freq <= 3.0)
    if np.any(mask) and np.nanmax(spec[mask]) > 0:
        fft_freq = float(freq[mask][np.argmax(spec[mask])])
    else:
        fft_freq = np.nan

    cycle_freq = np.nan
    try:
        nyq = fps / 2.0
        b, a = butter(2, [0.5 / nyq, 3.0 / nyq], btype="bandpass")
        filt = filtfilt(b, a, sig, method="gust")
        analytic = hilbert(filt)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) * fps / (2.0 * np.pi)
        inst_freq = inst_freq[np.isfinite(inst_freq)]
        inst_freq = inst_freq[(inst_freq >= 0.4) & (inst_freq <= 3.0)]
        if inst_freq.shape[0] >= max(10, int(0.4 * fps)):
            lo, hi = np.percentile(inst_freq, [10, 90])
            trimmed = inst_freq[(inst_freq >= lo) & (inst_freq <= hi)]
            if trimmed.shape[0] > 0:
                cycle_freq = float(np.median(trimmed))
    except Exception:
        cycle_freq = np.nan

    if not np.isfinite(cycle_freq):
        cycle_freq = fft_freq

    cadence = 2.0 * cycle_freq * 60.0
    return {
        "cadence_pose_spm": cadence,
        "step_time_pose_s": float(60.0 / cadence) if cadence > 0 else np.nan,
        "pose_step_freq_hz": cycle_freq,
        "cadence_pose_fft_spm": 2.0 * fft_freq * 60.0 if np.isfinite(fft_freq) else np.nan,
        "pose_step_freq_fft_hz": fft_freq,
    }


def pose_only_features(joints, eps=1e-6):
    out = {}
    out.update(arm_swing_features(joints, eps))
    out.update(leg_swing_features(joints, eps))
    out.update(foot_lift_features(joints))
    out.update(trunk_posture_features(joints))
    out.update(cadence_from_pose(joints))
    return out


def path_speed(joints, fps=FPS):
    pelvis = joints[:, SMPL_JOINT["pelvis"]]
    if len(pelvis) < 2:
        return np.nan
    duration = max(1e-6, (len(pelvis) - 1) / fps)
    path_length = float(np.sum(np.linalg.norm(np.diff(pelvis[:, HORIZONTAL_AXES], axis=0), axis=1)))
    return path_length / duration


def temporal_decay_features(pose_joints, full_joints, eps=1e-6):
    n = len(pose_joints)
    if n < 40:
        return {
            "arm_amp_second_over_first": np.nan,
            "leg_amp_second_over_first": np.nan,
            "foot_lift_second_over_first": np.nan,
            "gait_speed_second_over_first": np.nan,
            "gait_speed_second_minus_first_mps": np.nan,
        }

    mid = n // 2
    pose_first, pose_second = pose_joints[:mid], pose_joints[mid:]
    full_first, full_second = full_joints[:mid], full_joints[mid:]

    def ratio(second, first):
        return float(second / (first + 1e-6)) if np.isfinite(first) and np.isfinite(second) else np.nan

    arm_first = arm_swing_features(pose_first, eps)["arm_amp_mean_p95p5_deg"]
    arm_second = arm_swing_features(pose_second, eps)["arm_amp_mean_p95p5_deg"]
    leg_first = leg_swing_features(pose_first, eps)["leg_amp_mean_p95p5_deg"]
    leg_second = leg_swing_features(pose_second, eps)["leg_amp_mean_p95p5_deg"]
    foot_first = foot_lift_features(pose_first)["foot_lift_mean_p95p5_m"]
    foot_second = foot_lift_features(pose_second)["foot_lift_mean_p95p5_m"]
    speed_first = path_speed(full_first)
    speed_second = path_speed(full_second)

    return {
        "arm_amp_second_over_first": ratio(arm_second, arm_first),
        "arm_amp_second_minus_first_deg": float(arm_second - arm_first),
        "leg_amp_second_over_first": ratio(leg_second, leg_first),
        "leg_amp_second_minus_first_deg": float(leg_second - leg_first),
        "foot_lift_second_over_first": ratio(foot_second, foot_first),
        "foot_lift_second_minus_first_m": float(foot_second - foot_first),
        "gait_speed_second_over_first": ratio(speed_second, speed_first),
        "gait_speed_second_minus_first_mps": float(speed_second - speed_first),
    }


def forward_lateral_axes(pelvis):
    horiz = pelvis[:, HORIZONTAL_AXES]
    delta = horiz[-1] - horiz[0]
    if np.linalg.norm(delta) < 1e-4 and len(horiz) > 3:
        centered = horiz - np.mean(horiz, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        delta = vt[0]
    if np.linalg.norm(delta) < 1e-4:
        delta = np.array([0.0, 1.0])
    f2 = delta / (np.linalg.norm(delta) + 1e-9)
    l2 = np.array([-f2[1], f2[0]])
    forward = np.array([f2[0], 0.0, f2[1]])
    lateral = np.array([l2[0], 0.0, l2[1]])
    return forward, lateral


def project_axis(pos, axis):
    return np.einsum("ij,j->i", pos, axis)


def group_bool_segments(mask):
    segments = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if (not value or i == len(mask) - 1) and start is not None:
            end = i if not value else i + 1
            if end > start:
                segments.append((start, end))
            start = None
    return segments


def detect_contacts(foot_pos, fps=FPS, min_interval_s=0.25):
    horiz = foot_pos[:, HORIZONTAL_AXES]
    vel = np.zeros(len(foot_pos))
    vel[1:] = np.linalg.norm(np.diff(horiz, axis=0), axis=1) * fps
    height = foot_pos[:, UP_AXIS] - np.nanmin(foot_pos[:, UP_AXIS])
    speed_thr = np.percentile(vel, 35)
    height_thr = np.percentile(height, 45)
    stance = (vel <= speed_thr) & (height <= height_thr)
    min_len = max(2, int(0.08 * fps))
    min_gap = max(1, int(min_interval_s * fps))
    contacts = []
    last = -10_000
    for start, end in group_bool_segments(stance):
        if end - start < min_len:
            continue
        local = start + int(np.argmin(vel[start:end] + 0.1 * height[start:end]))
        if local - last >= min_gap:
            contacts.append(local)
            last = local
    return np.array(contacts, dtype=np.int64)


def full_motion_features(joints, fps=FPS, min_step_interval_s=0.25):
    pelvis = joints[:, SMPL_JOINT["pelvis"]]
    lfoot = joints[:, SMPL_JOINT["lfoot"]]
    rfoot = joints[:, SMPL_JOINT["rfoot"]]
    forward, lateral = forward_lateral_axes(pelvis)

    duration = max(1e-6, (len(joints) - 1) / fps)
    path_length = float(np.sum(np.linalg.norm(np.diff(pelvis[:, HORIZONTAL_AXES], axis=0), axis=1)))
    speed = path_length / duration

    lc = detect_contacts(lfoot, fps, min_step_interval_s)
    rc = detect_contacts(rfoot, fps, min_step_interval_s)
    events = [(int(i), "L") for i in lc] + [(int(i), "R") for i in rc]
    events = sorted(events, key=lambda x: x[0])

    step_times, step_lengths, step_widths = [], [], []
    stride_times, stride_lengths = [], []
    last_by_side = {}
    for (idx, side), (prev_idx, prev_side) in zip(events[1:], events[:-1]):
        if side == prev_side:
            continue
        foot = lfoot[idx] if side == "L" else rfoot[idx]
        prev_foot = lfoot[prev_idx] if prev_side == "L" else rfoot[prev_idx]
        step_times.append((idx - prev_idx) / fps)
        step_lengths.append(abs(float(np.dot(foot - prev_foot, forward))))
        step_widths.append(abs(float(np.dot(foot - prev_foot, lateral))))
    for idx, side in events:
        foot = lfoot[idx] if side == "L" else rfoot[idx]
        if side in last_by_side:
            prev_idx, prev_foot = last_by_side[side]
            stride_times.append((idx - prev_idx) / fps)
            stride_lengths.append(abs(float(np.dot(foot - prev_foot, forward))))
        last_by_side[side] = (idx, foot)

    step_time = float(np.nanmean(step_times)) if step_times else np.nan
    cadence = float(60.0 / step_time) if step_time and np.isfinite(step_time) and step_time > 0 else np.nan

    # Approximate lateral margin of stability using pelvis as COM proxy.
    leg_len = np.nanmedian(pelvis[:, UP_AXIS] - np.minimum(lfoot[:, UP_AXIS], rfoot[:, UP_AXIS]))
    omega0 = np.sqrt(9.81 / max(0.3, float(leg_len)))
    pelvis_vel = np.zeros_like(pelvis)
    pelvis_vel[1:] = np.diff(pelvis, axis=0) * fps
    xcom = pelvis + pelvis_vel / omega0
    xcom_lat = project_axis(xcom, lateral)
    lf_lat = project_axis(lfoot, lateral)
    rf_lat = project_axis(rfoot, lateral)
    bos_min = np.minimum(lf_lat, rf_lat)
    bos_max = np.maximum(lf_lat, rf_lat)
    mos = np.minimum(xcom_lat - bos_min, bos_max - xcom_lat)

    return {
        "gait_speed_mps": speed,
        "contact_cadence_spm": cadence,
        "contact_step_time_s": step_time,
        "stride_time_s": float(np.nanmean(stride_times)) if stride_times else np.nan,
        "step_length_m": float(np.nanmean(step_lengths)) if step_lengths else np.nan,
        "stride_length_m": float(np.nanmean(stride_lengths)) if stride_lengths else np.nan,
        "step_width_m": float(np.nanmean(step_widths)) if step_widths else np.nan,
        "mos_lateral_min_m": float(np.nanmin(mos)) if len(mos) else np.nan,
        "n_contacts": int(len(events)),
        "n_steps": int(len(step_times)),
        "n_strides": int(len(stride_times)),
    }


POSE_FEATURES = [
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

FULL_FEATURES = [
    "gait_speed_mps",
    "contact_cadence_spm",
    "contact_step_time_s",
    "stride_time_s",
    "step_length_m",
    "stride_length_m",
    "step_width_m",
    "mos_lateral_min_m",
]

DECAY_FEATURES = [
    "arm_amp_second_over_first",
    "arm_amp_second_minus_first_deg",
    "leg_amp_second_over_first",
    "leg_amp_second_minus_first_deg",
    "foot_lift_second_over_first",
    "foot_lift_second_minus_first_m",
    "gait_speed_second_over_first",
    "gait_speed_second_minus_first_mps",
]


def prefixed(prefix, values):
    return {f"{prefix}_{k}": v for k, v in values.items()}


def add_preservation(row, feature_names, family):
    for name in feature_names:
        g = row.get(f"gt_{family}_{name}", np.nan)
        p = row.get(f"pred_{family}_{name}", np.nan)
        min_denom = 1e-6
        if name.endswith("_p95p5_deg"):
            min_denom = AMP_RATIO_MIN_DENOM
        elif "asym" in name:
            min_denom = ASYM_RATIO_MIN_DENOM
        elif name.endswith("_p95p5_m"):
            min_denom = 0.01

        row[f"{family}_ratio_{name}"] = float(p / (g + 1e-6)) if (
            np.isfinite(g) and np.isfinite(p) and abs(g) >= min_denom
        ) else np.nan
        row[f"{family}_diff_{name}"] = float(p - g) if np.isfinite(g) and np.isfinite(p) else np.nan
        row[f"{family}_ratio_valid_{name}"] = bool(np.isfinite(g) and abs(g) >= min_denom)


def add_decay_preservation(row):
    for name in DECAY_FEATURES:
        g = row.get(f"gt_decay_{name}", np.nan)
        p = row.get(f"pred_decay_{name}", np.nan)
        row[f"decay_diff_{name}"] = float(p - g) if np.isfinite(g) and np.isfinite(p) else np.nan
        row[f"decay_ratio_{name}"] = float(p / (g + 1e-6)) if np.isfinite(g) and np.isfinite(p) else np.nan


def finite_corr(x, y, method):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return np.nan, np.nan, int(mask.sum())
    if method == "pearson":
        r, p = stats.pearsonr(x[mask], y[mask])
    else:
        r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p), int(mask.sum())


def write_correlations(df, path):
    rows = []
    candidates = []
    for family, names in [("pose", POSE_FEATURES), ("full", FULL_FEATURES)]:
        for name in names:
            candidates.extend([f"gt_{family}_{name}", f"pred_{family}_{name}", f"{family}_ratio_{name}"])
            candidates.append(f"{family}_diff_{name}")
    for name in DECAY_FEATURES:
        candidates.extend([f"gt_decay_{name}", f"pred_decay_{name}", f"decay_diff_{name}"])
    for col in candidates:
        if col not in df:
            continue
        for method in ["pearson", "spearman"]:
            r, p, n = finite_corr(df[col], df["UPDRS_GAIT"], method)
            rows.append({"x": col, "y": "UPDRS_GAIT", "method": method, "r": r, "p": p, "n": n})
    for family, names in [("pose", POSE_FEATURES), ("full", FULL_FEATURES)]:
        for name in names:
            x, y = f"gt_{family}_{name}", f"pred_{family}_{name}"
            if x not in df or y not in df:
                continue
            for method in ["pearson", "spearman"]:
                r, p, n = finite_corr(df[x], df[y], method)
                rows.append({"x": x, "y": y, "method": method, "r": r, "p": p, "n": n})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y", "method", "r", "p", "n"])
        writer.writeheader()
        writer.writerows(rows)


def grouped_means(df, path):
    cols = [
        "gt_pose_arm_amp_mean_p95p5_deg", "pred_pose_arm_amp_mean_p95p5_deg", "pose_ratio_arm_amp_mean_p95p5_deg",
        "pose_diff_arm_amp_mean_p95p5_deg", "gt_pose_arm_amp_asym_p95p5", "pred_pose_arm_amp_asym_p95p5",
        "pose_diff_arm_amp_asym_p95p5",
        "gt_pose_leg_amp_mean_p95p5_deg", "pred_pose_leg_amp_mean_p95p5_deg", "pose_ratio_leg_amp_mean_p95p5_deg",
        "pose_diff_leg_amp_mean_p95p5_deg", "gt_pose_leg_amp_asym_p95p5", "pred_pose_leg_amp_asym_p95p5",
        "pose_diff_leg_amp_asym_p95p5",
        "gt_pose_cadence_pose_spm", "pred_pose_cadence_pose_spm", "pose_ratio_cadence_pose_spm",
        "gt_pose_cadence_pose_fft_spm", "pred_pose_cadence_pose_fft_spm",
        "gt_pose_foot_lift_asym_p95p5", "pred_pose_foot_lift_asym_p95p5", "pose_diff_foot_lift_asym_p95p5",
        "gt_pose_trunk_lean_mean_deg", "pred_pose_trunk_lean_mean_deg", "pose_diff_trunk_lean_mean_deg",
        "pose_ratio_trunk_lean_mean_deg",
        "gt_full_gait_speed_mps", "pred_full_gait_speed_mps", "full_ratio_gait_speed_mps",
        "gt_full_step_length_m", "pred_full_step_length_m", "full_ratio_step_length_m",
    ]
    out = {}
    out["by_updrs"] = df.groupby("UPDRS_GAIT")[cols].mean(numeric_only=True).round(5).to_dict()
    out["by_medication"] = df.groupby("medication")[cols].mean(numeric_only=True).round(5).to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def boxplot(df, cols, labels, group_col, title, path):
    groups = sorted([g for g in df[group_col].dropna().unique()])
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4), squeeze=False)
    for ax, col, label in zip(axes[0], cols, labels):
        data = [df.loc[df[group_col] == g, col].dropna().values for g in groups]
        ax.boxplot(data, tick_labels=[str(g) for g in groups], showfliers=False)
        xs, ys = [], []
        for i, vals in enumerate(data):
            xs.extend([i + 1] * len(vals))
            ys.extend(vals)
        ax.scatter(xs, ys, s=8, alpha=0.22)
        ax.set_title(label)
        ax.set_xlabel(group_col)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def scatter_gt_pred(df, x, y, title, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for u in sorted(df["UPDRS_GAIT"].dropna().unique()):
        sub = df[df["UPDRS_GAIT"] == u]
        ax.scatter(sub[x], sub[y], s=14, alpha=0.55, label=f"U={u}")
    finite = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        lo, hi = float(finite.min().min()), float(finite.max().max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def preservation_spearman_summary(subject_corr_path, fig_dir):
    corr = pd.read_csv(subject_corr_path)
    corr = corr[(corr["method"] == "spearman") & (corr["y"] == "UPDRS_GAIT")].copy()

    items = [
        ("Arm amplitude diff", "pose_diff_arm_amp_mean_p95p5_deg", "Pose-only"),
        ("Leg amplitude ratio", "pose_ratio_leg_amp_mean_p95p5_deg", "Pose-only"),
        ("Leg amplitude diff", "pose_diff_leg_amp_mean_p95p5_deg", "Pose-only"),
        ("Cadence ratio", "pose_ratio_cadence_pose_spm", "Pose-only"),
        ("Foot lift asym diff", "pose_diff_foot_lift_asym_p95p5", "Pose-only"),
        ("Trunk lean diff", "pose_diff_trunk_lean_mean_deg", "Pose-only"),
        ("Gait speed ratio", "full_ratio_gait_speed_mps", "Full-motion"),
        ("Step length ratio", "full_ratio_step_length_m", "Full-motion"),
        ("Arm asym diff", "pose_diff_arm_amp_asym_p95p5", "Secondary"),
        ("Leg asym diff", "pose_diff_leg_amp_asym_p95p5", "Secondary"),
    ]

    rows = []
    for label, key, family in items:
        hit = corr[corr["x"] == key]
        if hit.empty:
            continue
        rec = hit.iloc[0].to_dict()
        rec["label"] = label
        rec["family"] = family
        rows.append(rec)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("r")
    colors = {"Pose-only": "#2f7ed8", "Full-motion": "#f28e2b", "Secondary": "#8a8a8a"}
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.42 * len(df) + 1.4)))
    ax.barh(y, df["r"], color=[colors.get(f, "#999999") for f in df["family"]], alpha=0.9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Subject-state Spearman rho vs UPDRS_GAIT")
    ax.set_title("Feature Preservation Trend With PD Severity")
    ax.grid(True, axis="x", alpha=0.3)

    for i, (_, row) in enumerate(df.iterrows()):
        p = row["p"]
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = ""
        x = row["r"]
        ha = "left" if x >= 0 else "right"
        offset = 0.025 if x >= 0 else -0.025
        ax.text(x + offset, i, f"{x:.2f}{star}", va="center", ha=ha, fontsize=9)

    handles = [
        plt.Line2D([0], [0], color=colors[name], lw=6, label=name)
        for name in ["Pose-only", "Full-motion", "Secondary"]
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    path = os.path.join(fig_dir, "preservation_spearman_summary.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

    csv_path = os.path.join(os.path.dirname(fig_dir), "preservation_spearman_summary.csv")
    df[["label", "family", "x", "r", "p", "n"]].to_csv(csv_path, index=False)
    return path


def subject_state_stats(df, out_dir):
    numeric_cols = [c for c in df.columns if c.startswith(("gt_", "pred_", "pose_", "full_", "decay_"))]
    group_cols = ["subject_id", "medication", "UPDRS_GAIT"]
    subject_df = df.groupby(group_cols, dropna=False)[numeric_cols].mean(numeric_only=True).reset_index()
    summary_path = os.path.join(out_dir, "gait_feature_subject_state_summary.csv")
    subject_df.to_csv(summary_path, index=False)

    targets = [
        "gt_pose_arm_amp_mean_p95p5_deg", "pred_pose_arm_amp_mean_p95p5_deg",
        "pose_diff_arm_amp_mean_p95p5_deg", "pose_ratio_arm_amp_mean_p95p5_deg",
        "gt_pose_arm_amp_asym_p95p5", "pred_pose_arm_amp_asym_p95p5",
        "pose_diff_arm_amp_asym_p95p5",
        "gt_pose_leg_amp_mean_p95p5_deg", "pred_pose_leg_amp_mean_p95p5_deg",
        "pose_diff_leg_amp_mean_p95p5_deg", "pose_ratio_leg_amp_mean_p95p5_deg",
        "gt_pose_leg_amp_asym_p95p5", "pred_pose_leg_amp_asym_p95p5",
        "pose_diff_leg_amp_asym_p95p5",
        "gt_pose_cadence_pose_spm", "pred_pose_cadence_pose_spm", "pose_ratio_cadence_pose_spm",
        "gt_pose_foot_lift_asym_p95p5", "pred_pose_foot_lift_asym_p95p5",
        "pose_diff_foot_lift_asym_p95p5",
        "gt_pose_trunk_lean_mean_deg", "pred_pose_trunk_lean_mean_deg",
        "pose_diff_trunk_lean_mean_deg", "pose_ratio_trunk_lean_mean_deg",
        "gt_full_gait_speed_mps", "pred_full_gait_speed_mps", "full_ratio_gait_speed_mps",
        "gt_full_step_length_m", "pred_full_step_length_m", "full_ratio_step_length_m",
    ]
    rows = []
    for col in targets:
        if col not in subject_df:
            continue
        for method in ["spearman", "pearson"]:
            r, p, n = finite_corr(subject_df[col], subject_df["UPDRS_GAIT"], method)
            rows.append({"level": "subject_state", "x": col, "y": "UPDRS_GAIT", "method": method, "r": r, "p": p, "n": n})

    corr_path = os.path.join(out_dir, "gait_feature_subject_state_correlations.csv")
    with open(corr_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "x", "y", "method", "r", "p", "n"])
        writer.writeheader()
        writer.writerows(rows)
    return summary_path, corr_path


def make_figures(df, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    boxplot(
        df,
        ["pose_ratio_arm_amp_mean_p95p5_deg", "pose_ratio_leg_amp_mean_p95p5_deg", "pose_ratio_cadence_pose_spm"],
        ["Arm amp ratio", "Leg amp ratio", "Cadence ratio"],
        "UPDRS_GAIT",
        "Pose-only Preservation Ratios By UPDRS",
        os.path.join(fig_dir, "pose_preservation_by_updrs.png"),
    )
    boxplot(
        df,
        ["gt_pose_arm_amp_asym_p95p5", "pred_pose_arm_amp_asym_p95p5", "pose_diff_arm_amp_asym_p95p5"],
        ["GT arm asymmetry", "Pred arm asymmetry", "Pred - GT"],
        "UPDRS_GAIT",
        "Arm Swing Asymmetry By UPDRS",
        os.path.join(fig_dir, "arm_asymmetry_by_updrs.png"),
    )
    boxplot(
        df,
        ["gt_pose_leg_amp_asym_p95p5", "pred_pose_leg_amp_asym_p95p5", "pose_diff_leg_amp_asym_p95p5"],
        ["GT leg asymmetry", "Pred leg asymmetry", "Pred - GT"],
        "UPDRS_GAIT",
        "Leg Swing Asymmetry By UPDRS",
        os.path.join(fig_dir, "leg_asymmetry_by_updrs.png"),
    )
    boxplot(
        df,
        ["full_ratio_gait_speed_mps", "full_ratio_step_length_m", "full_ratio_step_width_m"],
        ["Speed ratio", "Step length ratio", "Step width ratio"],
        "UPDRS_GAIT",
        "Full-motion Preservation Ratios By UPDRS",
        os.path.join(fig_dir, "full_motion_preservation_by_updrs.png"),
    )
    scatter_gt_pred(
        df,
        "gt_pose_arm_amp_mean_p95p5_deg",
        "pred_pose_arm_amp_mean_p95p5_deg",
        "Arm Swing GT vs Pred",
        os.path.join(fig_dir, "scatter_arm_swing_gt_pred.png"),
    )
    scatter_gt_pred(
        df,
        "gt_pose_leg_amp_mean_p95p5_deg",
        "pred_pose_leg_amp_mean_p95p5_deg",
        "Leg Swing GT vs Pred",
        os.path.join(fig_dir, "scatter_leg_swing_gt_pred.png"),
    )
    scatter_gt_pred(
        df,
        "gt_pose_arm_amp_asym_p95p5",
        "pred_pose_arm_amp_asym_p95p5",
        "Arm Swing Asymmetry GT vs Pred",
        os.path.join(fig_dir, "scatter_arm_asymmetry_gt_pred.png"),
    )
    scatter_gt_pred(
        df,
        "gt_pose_leg_amp_asym_p95p5",
        "pred_pose_leg_amp_asym_p95p5",
        "Leg Swing Asymmetry GT vs Pred",
        os.path.join(fig_dir, "scatter_leg_asymmetry_gt_pred.png"),
    )
    scatter_gt_pred(
        df,
        "gt_full_gait_speed_mps",
        "pred_full_gait_speed_mps",
        "Gait Speed GT vs Pred",
        os.path.join(fig_dir, "scatter_gait_speed_gt_pred.png"),
    )


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pred = safe_torch_load(args.predictions, map_location="cpu")
    device = torch.device(args.device)
    model = art.ParametricModel(paths.smpl_file, device=device)

    n = len(pred["pose_pred"])
    if args.limit > 0:
        n = min(n, args.limit)

    rows = []
    for i in tqdm(range(n), desc="Gait feature preservation"):
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

        gt_pose_j = fk_joints(model, pred["pose_gt"][i], device, tran=None)
        pred_pose_j = fk_joints(model, pred["pose_pred"][i], device, tran=None)
        gt_full_j = fk_joints(model, pred["pose_gt"][i], device, tran=pred["tran_gt"][i])
        pred_full_j = fk_joints(model, pred["pose_pred"][i], device, tran=pred["tran_pred"][i])

        row.update(prefixed("gt_pose", pose_only_features(gt_pose_j, args.eps)))
        row.update(prefixed("pred_pose", pose_only_features(pred_pose_j, args.eps)))
        row.update(prefixed("gt_full", full_motion_features(gt_full_j, FPS, args.min_step_interval_s)))
        row.update(prefixed("pred_full", full_motion_features(pred_full_j, FPS, args.min_step_interval_s)))
        row.update(prefixed("gt_decay", temporal_decay_features(gt_pose_j, gt_full_j, args.eps)))
        row.update(prefixed("pred_decay", temporal_decay_features(pred_pose_j, pred_full_j, args.eps)))

        add_preservation(row, POSE_FEATURES, "pose")
        add_preservation(row, FULL_FEATURES, "full")
        add_decay_preservation(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "gait_feature_summary.csv")
    corr_path = os.path.join(args.out_dir, "gait_feature_correlations.csv")
    grouped_path = os.path.join(args.out_dir, "gait_feature_grouped_means.json")
    meta_path = os.path.join(args.out_dir, "gait_feature_meta.json")

    df.to_csv(csv_path, index=False)
    write_correlations(df, corr_path)
    grouped_means(df, grouped_path)
    subject_summary_path, subject_corr_path = subject_state_stats(df, args.out_dir)
    fig_dir = os.path.join(args.out_dir, "figures")
    make_figures(df, fig_dir)
    summary_fig_path = preservation_spearman_summary(subject_corr_path, fig_dir)

    meta = {
        "predictions": args.predictions,
        "n_sequences": len(df),
        "fps": FPS,
        "pose_only_features": POSE_FEATURES,
        "full_motion_features": FULL_FEATURES,
        "temporal_decay_features": DECAY_FEATURES,
        "notes": [
            "Pose-only features use SMPL FK with tran=None for GT and prediction.",
            "Full-motion features use GT / TransPose translations and should be interpreted with root-translation caveats.",
            "MOS is an approximate lateral margin of stability using pelvis as COM proxy.",
            "Contact-based features use heuristic low foot-speed / low foot-height stance detection.",
            "Temporal decay features compare the second half of a sequence to the first half and are exploratory for short walks.",
            "cadence_pose_spm uses 0.5-3Hz bandpass + Hilbert instantaneous phase with trimmed-median frequency; cadence_pose_fft_spm is retained as coarse FFT QA.",
            "Amplitude ratios use denominator guards to avoid small-GT explosions; diff columns are the primary amplitude/asymmetry preservation metrics.",
            "Mixed-effects modeling is not run because statsmodels is not installed; subject_state files aggregate by subject_id + medication + UPDRS_GAIT.",
            "Leg asymmetry is secondary: GT/pred preservation is weak and the amplification effect may reflect model noise, so it should not be used as a main conclusion.",
        ],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved summary      : {csv_path}")
    print(f"Saved correlations : {corr_path}")
    print(f"Saved subject stats: {subject_summary_path}")
    print(f"Saved subject corr : {subject_corr_path}")
    print(f"Saved grouped means: {grouped_path}")
    print(f"Saved figures      : {fig_dir}")
    if summary_fig_path:
        print(f"Saved summary fig  : {summary_fig_path}")
    print()
    print("Quick preservation means:")
    quick = [
        "pose_ratio_arm_amp_mean_p95p5_deg",
        "pose_ratio_leg_amp_mean_p95p5_deg",
        "pose_ratio_cadence_pose_spm",
        "pose_ratio_foot_lift_mean_p95p5_m",
        "full_ratio_gait_speed_mps",
        "full_ratio_step_length_m",
        "full_ratio_step_width_m",
        "gt_decay_gait_speed_second_over_first",
        "pred_decay_gait_speed_second_over_first",
        "gt_decay_arm_amp_second_over_first",
        "pred_decay_arm_amp_second_over_first",
    ]
    for col in quick:
        print(f"  {col}: {df[col].mean(skipna=True):.4f}")


if __name__ == "__main__":
    main()
