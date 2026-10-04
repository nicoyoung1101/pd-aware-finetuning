"""
analyze_multimodel_extended_metrics.py

Uniform extended metric pool for CARE-PD/BMCLab predictions from multiple
IMU-to-pose models.

This script is intentionally inclusive: it computes both suitable and
questionable metrics for every model, then records suitability/caveats in the
catalog. Use the catalog to decide which rows belong in main results.

Outputs:
  - multimodel_extended_per_walk.csv
  - multimodel_extended_grouped_by_updrs.csv
  - multimodel_extended_subject_state_by_updrs.csv
  - multimodel_extended_correlations.csv
  - multimodel_extended_independence.csv
  - multimodel_extended_gt_pred_consistency.csv
  - multimodel_extended_catalog.csv
  - multimodel_extended_meta.json
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
import json
import os
import sys

import pandas as pd
import torch
from scipy import stats
from scipy.signal import butter, filtfilt, hilbert
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRANSPOSE_ROOT = os.path.join(PROJECT_ROOT, "TransPose")
if TRANSPOSE_ROOT not in sys.path:
    sys.path.insert(0, TRANSPOSE_ROOT)
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from analyze_gait_feature_preservation import (  # noqa: E402
    FPS,
    HORIZONTAL_AXES,
    SMPL_JOINT,
    fk_joints,
    full_motion_features,
    interp_nan,
    pose_only_features,
    sagittal_angle,
    safe_torch_load,
    torso_frame,
)
from analyze_pose_fidelity import (  # noqa: E402
    IGNORED_ROTATION_JOINTS,
    JOINT_NAMES,
    PREDICTED_POSITION_BODY_PARTS,
    local_rotation_geodesic_deg,
    per_joint_mpjpe_mm,
    procrustes_aligned_mpjpe_mm,
    region_indices,
)
from analyze_transpose_candidate_metrics import (  # noqa: E402
    add_preservation_columns,
    com_proxy_features,
    foot_skating_features,
    motion_dynamics_features,
    prefixed,
    ratio,
    root_aligned,
    temporal_derivative,
    trajectory_features,
)
import articulate as art  # noqa: E402
from config import paths  # noqa: E402


MODEL_CONFIGS = [
    {
        "model": "TransPose",
        "variant": "default",
        "predictions": os.path.join(
            PROJECT_ROOT, "TransPose", "data", "results", "CAREPD_BMCLab_batch", "predictions.pt"
        ),
        "global_translation_role": "main_candidate",
        "global_translation_caveat": "Predicted translation is available; full-motion gait metrics are interpretable with normal caveats.",
    },
    {
        "model": "PIP",
        "variant": "raw",
        "predictions": os.path.join(
            PROJECT_ROOT, "PIP", "data", "results", "CAREPD_BMCLab_raw_batch", "predictions.pt"
        ),
        "global_translation_role": "main_candidate",
        "global_translation_caveat": "Raw neural prediction is available; more stable than the physics layer in visual QA.",
    },
    {
        "model": "PIP",
        "variant": "physics",
        "predictions": os.path.join(
            PROJECT_ROOT, "PIP", "data", "results", "CAREPD_BMCLab_batch", "predictions.pt"
        ),
        "global_translation_role": "qa_only",
        "global_translation_caveat": "Physics optimization showed visual artifacts/backward lean; full-motion metrics should be treated as QA only.",
    },
    {
        "model": "PNP",
        "variant": "default",
        "predictions": os.path.join(
            PROJECT_ROOT, "PNP", "data", "results", "CAREPD_BMCLab_batch", "predictions.pt"
        ),
        "global_translation_role": "qa_only",
        "global_translation_caveat": "PNP is an online physics model. Global translation is available, but should be visually and quantitatively checked before main-result use.",
    },
    {
        "model": "TIP",
        "variant": "default",
        "predictions": os.path.join(
            PROJECT_ROOT, "TIP", "data", "results", "CAREPD_BMCLab_batch", "predictions.pt"
        ),
        "global_translation_role": "qa_only",
        "global_translation_caveat": "TIP predicts root velocity with terrain/contact correction. Local pose descriptors are main-result candidates; full-motion/global-translation metrics are QA until trajectory is validated.",
    },
    {
        "model": "DynaIP",
        "variant": "default",
        "predictions": os.path.join(
            PROJECT_ROOT, "DynaIP", "data", "results", "CAREPD_BMCLab_batch", "predictions.pt"
        ),
        "global_translation_role": "not_suitable",
        "global_translation_caveat": "DynaIP predicts pose/global rotations but no reliable global translation in this adapter; step length/speed/root trajectory are not main-result metrics.",
    },
    {
        "model": "DIP",
        "variant": "fine_tuning",
        "predictions": os.path.join(
            PROJECT_ROOT, "DIP", "data", "results", "CAREPD_BMCLab_fine_tuning_batch", "predictions.pt"
        ),
        "global_translation_role": "not_suitable",
        "global_translation_caveat": "Original DIP predicts local pose only; tran_pred is zero, so full-motion/global-translation metrics are not suitable.",
    },
    {
        "model": "DIP",
        "variant": "original",
        "predictions": os.path.join(
            PROJECT_ROOT, "DIP", "data", "results", "CAREPD_BMCLab_original_batch", "predictions.pt"
        ),
        "global_translation_role": "not_suitable",
        "global_translation_caveat": "Original DIP checkpoint diagnostic variant; predicts local pose only and tran_pred is zero.",
    },
]


POSE_FEATURE_NAMES = [
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

FULL_FEATURE_NAMES = [
    "gait_speed_mps",
    "contact_cadence_spm",
    "contact_step_time_s",
    "stride_time_s",
    "stride_length_m",
    "step_length_m",
    "step_width_m",
    "mos_lateral_m",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=os.path.join(PROJECT_ROOT, "ModelComparison", "data", "results", "CAREPD_BMCLab_extended_metrics"),
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--models", nargs="+", choices=[f"{c['model']}_{c['variant']}" for c in MODEL_CONFIGS], default=['TransPose_default'])
    return parser.parse_args()


def finite_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


def finite_std(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) else np.nan


def rotation_angle_deg(rot):
    trace = np.trace(rot, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos))


def robust_range(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    return float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5))


def region_mean(values, idxs):
    vals = [values[i] for i in idxs if i in values and np.isfinite(values[i])]
    return float(np.nanmean(vals)) if vals else np.nan


def local_rom_features(pose_gt, pose_pred, regions):
    """Robust local rotation ROM from each joint's local angle magnitude."""
    pose_gt = pose_gt.detach().cpu().numpy()
    pose_pred = pose_pred.detach().cpu().numpy()
    out = {}
    gt_joint_rom = {}
    pred_joint_rom = {}
    for j, name in enumerate(JOINT_NAMES):
        if name in IGNORED_ROTATION_JOINTS:
            out[f"gt_rom_joint_{name}_deg"] = np.nan
            out[f"pred_rom_joint_{name}_deg"] = np.nan
            out[f"rom_ratio_joint_{name}"] = np.nan
            out[f"rom_diff_joint_{name}_deg"] = np.nan
            continue
        gt_rom = robust_range(rotation_angle_deg(pose_gt[:, j]))
        pred_rom = robust_range(rotation_angle_deg(pose_pred[:, j]))
        gt_joint_rom[j] = gt_rom
        pred_joint_rom[j] = pred_rom
        out[f"gt_rom_joint_{name}_deg"] = gt_rom
        out[f"pred_rom_joint_{name}_deg"] = pred_rom
        out[f"rom_ratio_joint_{name}"] = ratio(pred_rom, gt_rom)
        out[f"rom_diff_joint_{name}_deg"] = float(pred_rom - gt_rom) if np.isfinite(gt_rom) and np.isfinite(pred_rom) else np.nan

    for region, idxs in regions.items():
        gt = region_mean(gt_joint_rom, idxs)
        pred = region_mean(pred_joint_rom, idxs)
        out[f"gt_rom_{region}_deg"] = gt
        out[f"pred_rom_{region}_deg"] = pred
        out[f"rom_ratio_{region}"] = ratio(pred, gt)
        out[f"rom_diff_{region}_deg"] = float(pred - gt) if np.isfinite(gt) and np.isfinite(pred) else np.nan
    return out


def mean_region_velocity(joints, idxs):
    j = root_aligned(joints)
    vel = temporal_derivative(j, FPS)
    if vel.size == 0:
        return np.nan
    speed = np.linalg.norm(vel[:, idxs], axis=-1)
    return float(np.nanmean(speed) * 1000.0)


def body_part_velocity_features(gt_joints, pred_joints, regions):
    out = {}
    for region, idxs in regions.items():
        gt = mean_region_velocity(gt_joints, idxs)
        pred = mean_region_velocity(pred_joints, idxs)
        out[f"gt_velocity_{region}_mms"] = gt
        out[f"pred_velocity_{region}_mms"] = pred
        out[f"velocity_ratio_{region}"] = ratio(pred, gt)
        out[f"velocity_diff_{region}_mms"] = float(pred - gt) if np.isfinite(gt) and np.isfinite(pred) else np.nan
    return out


def bandpass_signal(x, fps=FPS, low=0.5, high=3.0):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 20:
        return None
    x = interp_nan(x)
    x = x - np.nanmean(x)
    nyq = fps / 2.0
    high = min(high, nyq * 0.95)
    if high <= low:
        return None
    b, a = butter(2, [low / nyq, high / nyq], btype="bandpass")
    padlen = min(3 * (max(len(a), len(b)) - 1), len(x) - 1)
    if padlen < 3:
        return None
    return filtfilt(b, a, x, padlen=padlen)


def circular_abs_mean_deg(angle):
    angle = np.asarray(angle, dtype=np.float64)
    angle = angle[np.isfinite(angle)]
    if len(angle) == 0:
        return np.nan
    wrapped = np.angle(np.exp(1j * angle))
    return float(np.rad2deg(np.mean(np.abs(wrapped))))


def phase_features_from_joints(joints, prefix):
    frame, valid_frame = torso_frame(joints)
    lvec = joints[:, SMPL_JOINT["lknee"]] - joints[:, SMPL_JOINT["lhip"]]
    rvec = joints[:, SMPL_JOINT["rknee"]] - joints[:, SMPL_JOINT["rhip"]]
    la, lv = sagittal_angle(lvec, frame, valid_frame)
    ra, rv = sagittal_angle(rvec, frame, valid_frame)
    lf = bandpass_signal(la)
    rf = bandpass_signal(ra)
    out = {
        f"{prefix}_leg_phase_antiphase_error_deg": np.nan,
        f"{prefix}_leg_phase_locking_value": np.nan,
        f"{prefix}_leg_inst_freq_median_hz": np.nan,
        f"{prefix}_leg_inst_freq_iqr_hz": np.nan,
    }
    if lf is None or rf is None:
        return out
    lphase = np.unwrap(np.angle(hilbert(lf)))
    rphase = np.unwrap(np.angle(hilbert(rf)))
    phase_diff = np.angle(np.exp(1j * ((lphase - rphase) - np.pi)))
    out[f"{prefix}_leg_phase_antiphase_error_deg"] = circular_abs_mean_deg(phase_diff)
    out[f"{prefix}_leg_phase_locking_value"] = float(abs(np.nanmean(np.exp(1j * phase_diff))))

    mean_phase = np.unwrap(np.angle(hilbert((lf - rf) / 2.0)))
    inst_freq = np.diff(mean_phase) * FPS / (2.0 * np.pi)
    inst_freq = inst_freq[np.isfinite(inst_freq) & (inst_freq > 0.2) & (inst_freq < 4.0)]
    if len(inst_freq):
        out[f"{prefix}_leg_inst_freq_median_hz"] = float(np.nanmedian(inst_freq))
        out[f"{prefix}_leg_inst_freq_iqr_hz"] = float(np.nanpercentile(inst_freq, 75) - np.nanpercentile(inst_freq, 25))
    return out


def add_phase_preservation(row):
    for name in [
        "leg_phase_antiphase_error_deg",
        "leg_phase_locking_value",
        "leg_inst_freq_median_hz",
        "leg_inst_freq_iqr_hz",
    ]:
        gt = row.get(f"gt_phase_{name}", np.nan)
        pred = row.get(f"pred_phase_{name}", np.nan)
        row[f"phase_diff_{name}"] = float(pred - gt) if np.isfinite(gt) and np.isfinite(pred) else np.nan
        row[f"phase_ratio_{name}"] = ratio(pred, gt)


def residualize(y, controls):
    y = np.asarray(y, dtype=np.float64)
    controls = [np.asarray(c, dtype=np.float64) for c in controls]
    mask = np.isfinite(y)
    for c in controls:
        mask &= np.isfinite(c)
    if mask.sum() < 5:
        return None, mask
    x = [np.ones(mask.sum())]
    for c in controls:
        x.append(c[mask])
    X = np.column_stack(x)
    beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    resid = y[mask] - X @ beta
    return resid, mask


def corr_pair(df, x_col, y_col, method="spearman", controls=None):
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    if controls:
        c = [pd.to_numeric(df[col], errors="coerce").to_numpy() for col in controls]
        if method == "spearman":
            x_rank = stats.rankdata(x, nan_policy="omit") if hasattr(stats, "rankdata") else x
            y_rank = stats.rankdata(y, nan_policy="omit") if hasattr(stats, "rankdata") else y
            c_rank = [stats.rankdata(v, nan_policy="omit") if hasattr(stats, "rankdata") else v for v in c]
            rx, mask_x = residualize(x_rank, c_rank)
            ry, mask_y = residualize(y_rank, c_rank)
        else:
            rx, mask_x = residualize(x, c)
            ry, mask_y = residualize(y, c)
        if rx is None or ry is None or not np.array_equal(mask_x, mask_y):
            return np.nan, np.nan, 0
        mask = mask_x
        xv, yv = rx, ry
    else:
        mask = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[mask], y[mask]
    if len(xv) < 5 or np.nanstd(xv) < 1e-12 or np.nanstd(yv) < 1e-12:
        return np.nan, np.nan, int(len(xv))
    if method == "pearson":
        r, p = stats.pearsonr(xv, yv)
    else:
        r, p = stats.spearmanr(xv, yv)
    return float(r), float(p), int(len(xv))


def updrs_correlations(df):
    rows = []
    skip = {"model", "variant", "model_variant", "label", "subject", "walk", "medication", "other"}
    numeric_cols = [c for c in df.columns if c not in skip]
    for model_variant, sub in df.groupby("model_variant"):
        for col in numeric_cols:
            if col == "UPDRS_GAIT":
                continue
            for method, controls in [
                ("spearman", None),
                ("partial_spearman_control_duration_s", ["duration_s"]),
            ]:
                r, p, n = corr_pair(sub, col, "UPDRS_GAIT", "spearman", controls)
                rows.append({
                    "model_variant": model_variant,
                    "metric": col,
                    "y": "UPDRS_GAIT",
                    "method": method,
                    "rho": r,
                    "p": p,
                    "n": n,
                })
    return pd.DataFrame(rows)


def independence_analysis(df):
    pairs = [
        ("joint_velocity_ratio", "pose_ratio_leg_amp_mean_p95p5_deg"),
        ("joint_velocity_ratio", "pose_ratio_arm_amp_mean_p95p5_deg"),
        ("joint_velocity_ratio", "pose_ratio_cadence_pose_spm"),
        ("joint_velocity_ratio", "gt_joint_velocity_mean_mms"),
        ("velocity_ratio_lower_body", "pose_ratio_leg_amp_mean_p95p5_deg"),
        ("velocity_ratio_left_leg", "pose_ratio_leg_amp_mean_p95p5_deg"),
        ("velocity_ratio_right_leg", "pose_ratio_leg_amp_mean_p95p5_deg"),
        ("rom_ratio_lower_body", "pose_ratio_leg_amp_mean_p95p5_deg"),
        ("rom_ratio_upper_limbs", "pose_ratio_arm_amp_mean_p95p5_deg"),
    ]
    rows = []
    for model_variant, sub in df.groupby("model_variant"):
        for x, y in pairs:
            if x not in sub.columns or y not in sub.columns:
                continue
            for method, controls in [
                ("spearman", None),
                ("partial_spearman_control_updrs_duration", ["UPDRS_GAIT", "duration_s"]),
            ]:
                r, p, n = corr_pair(sub, x, y, "spearman", controls)
                rows.append({
                    "model_variant": model_variant,
                    "x": x,
                    "y": y,
                    "method": method,
                    "rho": r,
                    "p": p,
                    "n": n,
                    "interpretation_hint": "high |rho| suggests the x metric may overlap with y rather than add an independent dimension",
                })
    return pd.DataFrame(rows)


def gt_pred_consistency(df):
    metric_pairs = [
        ("leg_amp_mean_p95p5_deg", "gt_pose_leg_amp_mean_p95p5_deg", "pred_pose_leg_amp_mean_p95p5_deg"),
        ("arm_amp_mean_p95p5_deg", "gt_pose_arm_amp_mean_p95p5_deg", "pred_pose_arm_amp_mean_p95p5_deg"),
        ("cadence_pose_spm", "gt_pose_cadence_pose_spm", "pred_pose_cadence_pose_spm"),
        ("trunk_lean_mean_deg", "gt_pose_trunk_lean_mean_deg", "pred_pose_trunk_lean_mean_deg"),
        ("joint_velocity_mean_mms", "gt_joint_velocity_mean_mms", "pred_joint_velocity_mean_mms"),
        ("velocity_lower_body_mms", "gt_velocity_lower_body_mms", "pred_velocity_lower_body_mms"),
        ("rom_lower_body_deg", "gt_rom_lower_body_deg", "pred_rom_lower_body_deg"),
        ("rom_upper_limbs_deg", "gt_rom_upper_limbs_deg", "pred_rom_upper_limbs_deg"),
        ("phase_inst_freq_median_hz", "gt_phase_leg_inst_freq_median_hz", "pred_phase_leg_inst_freq_median_hz"),
    ]
    rows = []
    for model_variant, sub in df.groupby("model_variant"):
        for label, gt_col, pred_col in metric_pairs:
            if gt_col not in sub.columns or pred_col not in sub.columns:
                continue
            for method, controls in [
                ("spearman", None),
                ("partial_spearman_control_updrs", ["UPDRS_GAIT"]),
                ("partial_spearman_control_updrs_duration", ["UPDRS_GAIT", "duration_s"]),
            ]:
                r, p, n = corr_pair(sub, gt_col, pred_col, "spearman", controls)
                rows.append({
                    "model_variant": model_variant,
                    "metric": label,
                    "gt_col": gt_col,
                    "pred_col": pred_col,
                    "method": method,
                    "rho": r,
                    "p": p,
                    "n": n,
                    "interpretation_hint": "partial rows ask whether per-walk individual variation is preserved beyond severity/duration",
                })
    return pd.DataFrame(rows)


def metric_catalog():
    rows = []
    base = [
        ("Pose fidelity", "MPJPE/PA-MPJPE", "mpjpe_predicted_*, pa_mpjpe_predicted_mm", "mm", "suitable", "Use predicted-joint columns for main position error."),
        ("Pose fidelity", "Local rotation error", "rot_*", "deg", "suitable", "Ignored placeholder joints are excluded."),
        ("Motion dynamics", "Joint velocity ratio", "joint_velocity_ratio", "unitless", "suitable", "Root-centered; inspect independence from amplitude metrics."),
        ("Motion dynamics", "Body-part velocity ratio", "velocity_ratio_*", "unitless", "suitable", "Root-centered; useful for part-specific compression."),
        ("Motion dynamics", "Acceleration/Jerk", "joint_acceleration_*, joint_jerk_*", "m/s^2 or m/s^3", "qa_only", "Noise-sensitive; do not use as primary clinical result."),
        ("Gait biomechanics", "Pose leg/arm swing", "pose_ratio_*_amp_*", "unitless/deg", "suitable", "Body-centered, translation-independent."),
        ("Gait biomechanics", "ROM by local rotation", "rom_ratio_*", "unitless/deg", "suitable", "Uses local rotations; ignored placeholder joints excluded."),
        ("Gait biomechanics", "Pose cadence", "pose_ratio_cadence_pose_spm", "unitless", "suitable", "Pose-based timing; not a claim that cadence itself is severity-independent."),
        ("Gait biomechanics", "Phase/timing", "phase_*", "deg/Hz/unitless", "suitable_exploratory", "Translation-independent timing signal; band-pass/Hilbert assumptions apply."),
        ("Full-motion", "Step length / gait speed / root trajectory", "full_*, root_*", "m, m/s, unitless", "model_dependent", "Requires reliable predicted global translation."),
        ("Full-motion", "Foot skating / COM proxy", "foot_skating_*, *_com_proxy_*", "m/s or m", "model_dependent_qa", "Requires reliable global trajectory/contact semantics; COM is only a proxy."),
        ("Consistency", "GT-Pred consistency", "gt_pred_consistency table", "rho", "suitable", "Use partial rows controlling UPDRS/duration to avoid common-cause inflation."),
        ("Independence", "Velocity independence", "independence table", "rho", "suitable", "Checks whether velocity ratio adds information beyond swing/cadence."),
    ]
    for level, metric, columns, unit, default_role, caveat in base:
        for cfg in MODEL_CONFIGS:
            role = default_role
            model_caveat = caveat
            if level == "Full-motion":
                role = cfg["global_translation_role"]
                model_caveat = cfg["global_translation_caveat"]
            rows.append({
                "model_variant": f"{cfg['model']}_{cfg['variant']}",
                "level": level,
                "metric": metric,
                "columns": columns,
                "unit": unit,
                "suitability": role,
                "caveat": model_caveat,
            })
    return pd.DataFrame(rows)


def selected_model_configs(limit_models=None):
    if not limit_models:
        return MODEL_CONFIGS
    wanted = set(limit_models)
    return [c for c in MODEL_CONFIGS if f"{c['model']}_{c['variant']}" in wanted]


def process_model(cfg, body_model, device, limit=0):
    path = cfg["predictions"]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = safe_torch_load(path, map_location="cpu")
    n = len(data["pose_pred"])
    if limit > 0:
        n = min(n, limit)

    ignored_rotation_indices = {JOINT_NAMES.index(j) for j in IGNORED_ROTATION_JOINTS}
    regions = region_indices(
        PREDICTED_POSITION_BODY_PARTS,
        excluded_full_joints=ignored_rotation_indices,
    )

    rows = []
    desc = f"{cfg['model']} {cfg['variant']}"
    for i in tqdm(range(n), desc=desc):
        pose_gt = data["pose_gt"][i]
        pose_pred = data["pose_pred"][i]
        tran_gt = data["tran_gt"][i]
        tran_pred = data["tran_pred"][i]

        gt_pose_j = fk_joints(body_model, pose_gt, device, tran=None)
        pred_pose_j = fk_joints(body_model, pose_pred, device, tran=None)
        gt_full_j = fk_joints(body_model, pose_gt, device, tran=tran_gt)
        pred_full_j = fk_joints(body_model, pose_pred, device, tran=tran_pred)

        item = data["manifest"][i] if data.get("manifest") and data["manifest"][i] else {}
        row = {
            "model": cfg["model"],
            "variant": cfg["variant"],
            "model_variant": f"{cfg['model']}_{cfg['variant']}",
            "index": int(data["index"][i]) if "index" in data else i,
            "label": data["label"][i] if "label" in data else f"seq_{i:04d}",
            "subject": item.get("subject_id", ""),
            "walk": item.get("walk_id", ""),
            "UPDRS_GAIT": item.get("UPDRS_GAIT", np.nan),
            "medication": item.get("medication", ""),
            "other": item.get("other", ""),
            "frames": int(pose_gt.shape[0]),
            "duration_s": float(pose_gt.shape[0] / FPS),
            "global_translation_suitability": cfg["global_translation_role"],
        }

        per_joint_mpjpe = per_joint_mpjpe_mm(pred_pose_j, gt_pose_j)
        rot_err = local_rotation_geodesic_deg(pose_pred, pose_gt)
        row["mpjpe_predicted_full_body_no_root_mm"] = float(np.nanmean(per_joint_mpjpe[:, regions["full_body_no_root"]]))
        row["pa_mpjpe_predicted_mm"] = procrustes_aligned_mpjpe_mm(pred_pose_j, gt_pose_j, regions["full_body"])
        row["rot_full_body_no_root_deg"] = float(np.nanmean(rot_err[:, regions["full_body_no_root"]]))
        for name, idxs in regions.items():
            row[f"mpjpe_predicted_{name}_mm"] = float(np.nanmean(per_joint_mpjpe[:, idxs]))
            row[f"rot_{name}_deg"] = float(np.nanmean(rot_err[:, idxs]))

        gt_pose_features = pose_only_features(gt_pose_j)
        pred_pose_features = pose_only_features(pred_pose_j)
        gt_full_features = full_motion_features(gt_full_j, FPS)
        pred_full_features = full_motion_features(pred_full_j, FPS)
        row.update(prefixed("gt_pose", gt_pose_features))
        row.update(prefixed("pred_pose", pred_pose_features))
        row.update(prefixed("gt_full", gt_full_features))
        row.update(prefixed("pred_full", pred_full_features))
        add_preservation_columns(row, POSE_FEATURE_NAMES, "gt_pose", "pred_pose", "pose")
        add_preservation_columns(row, FULL_FEATURE_NAMES, "gt_full", "pred_full", "full")

        row.update(motion_dynamics_features(gt_pose_j, pred_pose_j, FPS))
        row.update(body_part_velocity_features(gt_pose_j, pred_pose_j, regions))
        row.update(local_rom_features(pose_gt, pose_pred, regions))

        row.update(phase_features_from_joints(gt_pose_j, "gt_phase"))
        row.update(phase_features_from_joints(pred_pose_j, "pred_phase"))
        add_phase_preservation(row)

        row.update(trajectory_features(gt_full_j, pred_full_j, FPS))
        row.update(foot_skating_features(gt_full_j, pred_full_j, FPS))
        row.update(com_proxy_features(gt_full_j, pred_full_j, FPS))

        rows.append(row)
    return rows


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    body_model = art.ParametricModel(paths.smpl_file)

    configs = selected_model_configs(args.models)
    all_rows = []
    for cfg in configs:
        all_rows.extend(process_model(cfg, body_model, device, args.limit))

    df = pd.DataFrame(all_rows)
    per_walk_path = os.path.join(args.out_dir, "multimodel_extended_per_walk.csv")
    df.to_csv(per_walk_path, index=False)

    numeric_cols = [c for c in df.columns if c not in {
        "model", "variant", "model_variant", "label", "subject", "walk",
        "medication", "other", "global_translation_suitability", "UPDRS_GAIT"
    }]
    grouped = df.groupby(["model_variant", "UPDRS_GAIT"], dropna=False)[numeric_cols].mean(numeric_only=True)
    grouped_path = os.path.join(args.out_dir, "multimodel_extended_grouped_by_updrs.csv")
    grouped.to_csv(grouped_path)

    subject_state = (
        df.groupby(["model_variant", "subject", "medication", "UPDRS_GAIT"], dropna=False)[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    subject_state_path = os.path.join(args.out_dir, "multimodel_extended_subject_state.csv")
    subject_state.to_csv(subject_state_path, index=False)

    subject_state_grouped = subject_state.groupby(["model_variant", "UPDRS_GAIT"], dropna=False).mean(numeric_only=True)
    subject_state_grouped_path = os.path.join(args.out_dir, "multimodel_extended_subject_state_by_updrs.csv")
    subject_state_grouped.to_csv(subject_state_grouped_path)

    corr = updrs_correlations(df)
    corr_path = os.path.join(args.out_dir, "multimodel_extended_correlations.csv")
    corr.to_csv(corr_path, index=False)

    subject_corr = updrs_correlations(subject_state)
    subject_corr_path = os.path.join(args.out_dir, "multimodel_extended_subject_state_correlations.csv")
    subject_corr.to_csv(subject_corr_path, index=False)

    independence = independence_analysis(df)
    independence_path = os.path.join(args.out_dir, "multimodel_extended_independence.csv")
    independence.to_csv(independence_path, index=False)

    consistency = gt_pred_consistency(df)
    consistency_path = os.path.join(args.out_dir, "multimodel_extended_gt_pred_consistency.csv")
    consistency.to_csv(consistency_path, index=False)

    catalog = metric_catalog()
    catalog_path = os.path.join(args.out_dir, "multimodel_extended_catalog.csv")
    catalog.to_csv(catalog_path, index=False)

    meta = {
        "n_rows": int(len(df)),
        "models": configs,
        "important_usage_notes": [
            "All metrics are computed for every model where possible, including metrics marked unsuitable.",
            "Use multimodel_extended_catalog.csv to decide main-result vs QA vs not-suitable metrics.",
            "DynaIP full-motion/global-translation metrics are computed but marked not_suitable.",
            "PIP physics full-motion metrics are computed but marked qa_only due to visual artifacts.",
            "ROM uses local rotation angle robust range and excludes ignored placeholder joints.",
            "GT-Pred consistency should be interpreted using partial rows controlling UPDRS/duration.",
        ],
        "outputs": {
            "per_walk": per_walk_path,
            "grouped_by_updrs": grouped_path,
            "subject_state": subject_state_path,
            "subject_state_by_updrs": subject_state_grouped_path,
            "correlations": corr_path,
            "subject_state_correlations": subject_corr_path,
            "independence": independence_path,
            "gt_pred_consistency": consistency_path,
            "catalog": catalog_path,
        },
    }
    meta_path = os.path.join(args.out_dir, "multimodel_extended_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved per-walk       : {per_walk_path}")
    print(f"Saved grouped        : {grouped_path}")
    print(f"Saved subject-state  : {subject_state_path}")
    print(f"Saved correlations   : {corr_path}")
    print(f"Saved independence   : {independence_path}")
    print(f"Saved consistency    : {consistency_path}")
    print(f"Saved catalog        : {catalog_path}")
    print(f"Saved meta           : {meta_path}")


if __name__ == "__main__":
    main()
