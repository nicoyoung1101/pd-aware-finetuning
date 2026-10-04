"""
analyze_fog_trait_gt_features.py

GT-first gate for adding a second CARE-PD dataset via FOG/freezer-trait
classification. This script does not run any IMU-to-pose model. It asks whether
GT SMPL motion descriptors can classify subject-level FOG/freezer labels in
E-LC and KUL-DT-T.

Key design choices:
  - subject-level classification (labels are subject traits, not per-frame FOG)
  - native source sequences are resampled to 60 fps before feature extraction
  - no anthropometric/body-shape features are used
  - gait variability features are included because FOG is rhythm/variability-heavy
  - RF is the main classifier; logistic/SVM are optional robustness checks
  - permutation null is reported for the RF gate
"""

from __future__ import annotations

# numpy compatibility patch for old SMPL/chumpy-style pickles
import warnings
import numpy as np
for _name, _type in [
    ("bool", bool), ("int", int), ("float", float), ("complex", complex),
    ("object", object), ("str", str), ("unicode", str), ("long", int),
]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            getattr(np, _name)
        except AttributeError:
            setattr(np, _name, _type)

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from scipy.signal import butter, filtfilt, hilbert, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TRANSPOSE_ROOT = PROJECT_ROOT / "TransPose"
if str(TRANSPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSPOSE_ROOT))

import articulate as art
from config import paths

# Reuse the same clinical descriptor definitions as the BMCLab benchmark.
from Evaluation.analyze_gait_feature_preservation import (
    SMPL_JOINT,
    UP_AXIS,
    HORIZONTAL_AXES,
    pose_only_features,
    torso_frame,
    interp_nan,
)

TARGET_FPS = 60.0
OUT_DIR = PROJECT_ROOT / "ModelComparison/data/results/CAREPD_FOG_GT_feature_gate"
DATASETS = {
    "E-LC": {
        "path": PROJECT_ROOT / "Dataset/CARE-PD/Canonicalized_SMPL_pickles/E-LC_canonical.pkl",
        "positive": "PD-FOG",
        "negative": "PD-NoFOG",
        "exclude": {"NON-PD (PP-FOG)"},
        "label_name": "PD-FOG_vs_PD-NoFOG",
    },
    "KUL-DT-T": {
        "path": PROJECT_ROOT / "Dataset/CARE-PD/Canonicalized_SMPL_pickles/KUL-DT-T_canonical.pkl",
        "positive": "freezer",
        "negative": "nonfreezer",
        "exclude": set(),
        "label_name": "freezer_vs_nonfreezer",
    },
}

LOWER_ROM_JOINTS = ["lhip", "rhip", "lknee", "rknee"]
UPPER_ROM_JOINTS = ["lshoulder", "rshoulder", "lelbow", "relbow"]
BODY_REGIONS = {
    "full_body_no_root": [i for i in range(24) if i != 0],
    "lower_body": [1, 2, 4, 5, 7, 8, 10, 11],
    "upper_limbs": [16, 17, 18, 19, 20, 21, 22, 23],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()), choices=list(DATASETS.keys()))
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0, help="debug limit per dataset")
    p.add_argument("--cv-repeats", type=int, default=10)
    p.add_argument("--permutations", type=int, default=100)
    p.add_argument("--min-frames", type=int, default=60)
    return p.parse_args()


def safe_load_pickle(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty dataset: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def resample_indices(n_frames: int, src_fps: float, dst_fps: float = TARGET_FPS) -> np.ndarray:
    if n_frames <= 0:
        return np.array([], dtype=np.int64)
    duration = (n_frames - 1) / float(src_fps)
    n_out = int(np.floor(duration * dst_fps)) + 1
    times = np.arange(n_out, dtype=np.float64) / dst_fps
    idx = np.round(times * src_fps).astype(np.int64)
    return np.clip(idx, 0, n_frames - 1)


def collect_walks(data: dict, spec: dict, limit: int = 0):
    rows = []
    for sid, walks in data.items():
        if not isinstance(walks, dict):
            continue
        for wid, rec in walks.items():
            if not isinstance(rec, dict) or "pose" not in rec or "trans" not in rec:
                continue
            other = str(rec.get("other"))
            if other in spec["exclude"]:
                continue
            if other == spec["positive"]:
                label = 1
                label_text = spec["positive"]
            elif other == spec["negative"]:
                label = 0
                label_text = spec["negative"]
            else:
                continue
            rows.append((str(sid), str(wid), rec, label, label_text))
            if limit and len(rows) >= limit:
                return rows
    return rows


@torch.no_grad()
def fk_joints_batched(body_model, pose_aa: np.ndarray, trans: np.ndarray | None, device: torch.device, batch: int = 2048):
    joints_all = []
    pose_tensor = torch.from_numpy(pose_aa).float().view(-1, 24, 3)
    trans_tensor = None if trans is None else torch.from_numpy(trans).float().view(-1, 3)
    for start in range(0, pose_tensor.shape[0], batch):
        end = min(start + batch, pose_tensor.shape[0])
        p = pose_tensor[start:end].to(device)
        rot = art.math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3)
        tr = None if trans_tensor is None else trans_tensor[start:end].to(device)
        _, joints = body_model.forward_kinematics(rot, shape=None, tran=tr, calc_mesh=False)
        joints_all.append(joints.detach().cpu().numpy())
    return np.concatenate(joints_all, axis=0)


def pose_rotmat_np(pose_aa: np.ndarray, device: torch.device, batch: int = 4096):
    out = []
    pose_tensor = torch.from_numpy(pose_aa).float().view(-1, 24, 3)
    for start in range(0, pose_tensor.shape[0], batch):
        p = pose_tensor[start:start + batch].to(device)
        rot = art.math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3)
        out.append(rot.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def root_aligned(joints: np.ndarray):
    return joints - joints[:, :1]


def mean_region_velocity(joints: np.ndarray, idxs: list[int], fps: float = TARGET_FPS):
    if len(joints) < 2:
        return np.nan
    j = root_aligned(joints)
    vel = np.diff(j, axis=0) * fps
    speed = np.linalg.norm(vel[:, idxs], axis=-1)
    return float(np.nanmean(speed) * 1000.0)


def rotation_angle_deg(rot: np.ndarray):
    trace = np.trace(rot, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos))


def robust_range(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    return float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5))


def local_rom_from_pose(pose_rot: np.ndarray):
    by_joint = {}
    inv_joint = {v: k for k, v in SMPL_JOINT.items()}
    for name in set(LOWER_ROM_JOINTS + UPPER_ROM_JOINTS):
        j = SMPL_JOINT[name]
        by_joint[f"rom_{name}_deg"] = robust_range(rotation_angle_deg(pose_rot[:, j]))
    lower = [by_joint.get(f"rom_{j}_deg", np.nan) for j in LOWER_ROM_JOINTS]
    upper = [by_joint.get(f"rom_{j}_deg", np.nan) for j in UPPER_ROM_JOINTS]
    by_joint["rom_lower_body_deg"] = float(np.nanmean(lower))
    by_joint["rom_upper_limbs_deg"] = float(np.nanmean(upper))
    return by_joint


def bandpass_signal(sig, fps=TARGET_FPS, lo=0.5, hi=3.0):
    sig = np.asarray(sig, dtype=np.float64)
    sig = interp_nan(sig)
    sig = sig - np.nanmean(sig)
    if len(sig) < int(2 * fps):
        return sig
    nyq = fps / 2.0
    hi = min(hi, nyq * 0.95)
    try:
        b, a = butter(2, [lo / nyq, hi / nyq], btype="bandpass")
        return filtfilt(b, a, sig, method="gust")
    except Exception:
        return sig


def zero_crossing_intervals(sig, fps=TARGET_FPS):
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < 3:
        return np.array([])
    s = sig - np.nanmedian(sig)
    crossings = np.where(np.diff(np.signbit(s)))[0]
    times = []
    for i in crossings:
        y0, y1 = s[i], s[i + 1]
        frac = 0.0 if y1 == y0 else -y0 / (y1 - y0)
        if 0 <= frac <= 1:
            times.append((i + frac) / fps)
    times = np.asarray(times)
    intervals = np.diff(times)
    intervals = intervals[(intervals >= 0.2) & (intervals <= 1.5)]
    return intervals


def variability_features(joints: np.ndarray, fps: float = TARGET_FPS):
    frame, valid_frame = torso_frame(joints)
    pelvis = joints[:, SMPL_JOINT["pelvis"]]
    lf = joints[:, SMPL_JOINT["lfoot"]] - pelvis
    rf = joints[:, SMPL_JOINT["rfoot"]] - pelvis
    z_axis = frame[:, 2]
    lz = np.einsum("ij,ij->i", lf, z_axis)
    rz = np.einsum("ij,ij->i", rf, z_axis)
    sig = interp_nan(lz - rz)
    filt = bandpass_signal(sig, fps=fps)

    out = {}
    intervals = zero_crossing_intervals(filt, fps=fps)
    if len(intervals) >= 4:
        out["step_time_cv"] = float(np.nanstd(intervals) / (np.nanmean(intervals) + 1e-9))
        out["step_time_iqr_s"] = float(np.nanpercentile(intervals, 75) - np.nanpercentile(intervals, 25))
        out["step_time_median_s"] = float(np.nanmedian(intervals))
    else:
        out["step_time_cv"] = np.nan
        out["step_time_iqr_s"] = np.nan
        out["step_time_median_s"] = np.nan

    try:
        analytic = hilbert(filt)
        phase = np.unwrap(np.angle(analytic))
        inst = np.diff(phase) * fps / (2.0 * np.pi)
        inst = inst[np.isfinite(inst)]
        inst = inst[(inst >= 0.4) & (inst <= 3.5)]
        if len(inst) >= 10:
            out["inst_freq_median_hz"] = float(np.nanmedian(inst))
            out["inst_freq_cv"] = float(np.nanstd(inst) / (np.nanmean(inst) + 1e-9))
            out["cadence_variability_spm"] = float(np.nanstd(2.0 * inst * 60.0))
        else:
            out["inst_freq_median_hz"] = np.nan
            out["inst_freq_cv"] = np.nan
            out["cadence_variability_spm"] = np.nan
    except Exception:
        out["inst_freq_median_hz"] = np.nan
        out["inst_freq_cv"] = np.nan
        out["cadence_variability_spm"] = np.nan

    try:
        nperseg = min(len(sig), int(4 * fps))
        if nperseg >= int(1.5 * fps):
            freq, psd = welch(sig - np.nanmean(sig), fs=fps, nperseg=nperseg)
            loco = np.trapz(psd[(freq >= 0.5) & (freq <= 3.0)], freq[(freq >= 0.5) & (freq <= 3.0)])
            freeze = np.trapz(psd[(freq >= 3.0) & (freq <= 8.0)], freq[(freq >= 3.0) & (freq <= 8.0)])
            out["freezing_index_log"] = float(np.log((freeze + 1e-12) / (loco + 1e-12)))
            out["freeze_band_power"] = float(freeze)
            out["locomotor_band_power"] = float(loco)
        else:
            out["freezing_index_log"] = np.nan
            out["freeze_band_power"] = np.nan
            out["locomotor_band_power"] = np.nan
    except Exception:
        out["freezing_index_log"] = np.nan
        out["freeze_band_power"] = np.nan
        out["locomotor_band_power"] = np.nan

    # Leg phase coordination.
    try:
        lphase = np.unwrap(np.angle(hilbert(bandpass_signal(lz, fps=fps))))
        rphase = np.unwrap(np.angle(hilbert(bandpass_signal(rz, fps=fps))))
        phase_diff = np.angle(np.exp(1j * ((lphase - rphase) - np.pi)))
        out["leg_phase_antiphase_error_deg"] = float(np.rad2deg(np.nanmean(np.abs(phase_diff))))
        out["leg_phase_locking_value"] = float(abs(np.nanmean(np.exp(1j * phase_diff))))
    except Exception:
        out["leg_phase_antiphase_error_deg"] = np.nan
        out["leg_phase_locking_value"] = np.nan

    return out


def full_motion_features(joints_full: np.ndarray, fps: float = TARGET_FPS):
    out = {}
    pelvis = joints_full[:, SMPL_JOINT["pelvis"]]
    if len(pelvis) >= 2:
        duration = max(1e-6, (len(pelvis) - 1) / fps)
        horiz = pelvis[:, HORIZONTAL_AXES]
        path = float(np.sum(np.linalg.norm(np.diff(horiz, axis=0), axis=1)))
        disp = float(np.linalg.norm(horiz[-1] - horiz[0]))
        out["path_speed_mps"] = path / duration
        out["straight_speed_mps"] = disp / duration
        out["path_length_m"] = path
        out["lateral_sway_m"] = robust_range(pelvis[:, 0])
        out["vertical_range_m"] = robust_range(pelvis[:, UP_AXIS])
    else:
        out.update({
            "path_speed_mps": np.nan,
            "straight_speed_mps": np.nan,
            "path_length_m": np.nan,
            "lateral_sway_m": np.nan,
            "vertical_range_m": np.nan,
        })
    return out


def extract_features_for_walk(body_model, rec: dict, device: torch.device, min_frames: int):
    src_fps = float(rec.get("fps", TARGET_FPS))
    pose_np = np.asarray(rec["pose"], dtype=np.float32)
    trans_np = np.asarray(rec["trans"], dtype=np.float32)
    idx = resample_indices(pose_np.shape[0], src_fps, TARGET_FPS)
    if len(idx) < min_frames:
        return None
    pose = pose_np[idx].reshape(-1, 72)
    trans = trans_np[idx].reshape(-1, 3)

    pose_joints = fk_joints_batched(body_model, pose, None, device)
    full_joints = fk_joints_batched(body_model, pose, trans, device)
    pose_rot = pose_rotmat_np(pose, device)

    feat = {}
    feat.update(pose_only_features(pose_joints))
    feat.update(local_rom_from_pose(pose_rot))
    feat.update(variability_features(pose_joints, TARGET_FPS))
    feat.update(full_motion_features(full_joints, TARGET_FPS))
    feat["joint_velocity_mean_mms"] = mean_region_velocity(pose_joints, BODY_REGIONS["full_body_no_root"], TARGET_FPS)
    feat["lower_body_velocity_mms"] = mean_region_velocity(pose_joints, BODY_REGIONS["lower_body"], TARGET_FPS)
    feat["upper_limbs_velocity_mms"] = mean_region_velocity(pose_joints, BODY_REGIONS["upper_limbs"], TARGET_FPS)
    feat["n_frames_60fps"] = int(len(idx))
    feat["duration_s"] = float(len(idx) / TARGET_FPS)
    return feat


def feature_columns(df: pd.DataFrame):
    exclude = {"dataset", "subject", "walk", "label", "label_text", "other", "medication", "source_fps"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and c not in {"n_frames_60fps", "duration_s"}:
            # Exclude duration/frame count from classifier to avoid protocol leakage.
            cols.append(c)
    return cols


def aggregate_subject_features(per_walk: pd.DataFrame):
    cols = feature_columns(per_walk)
    rows = []
    for sid, g in per_walk.groupby("subject"):
        row = {
            "subject": sid,
            "dataset": g["dataset"].iloc[0],
            "label": int(g["label"].iloc[0]),
            "label_text": g["label_text"].iloc[0],
            "n_walks": int(len(g)),
        }
        for c in cols:
            row[c] = float(pd.to_numeric(g[c], errors="coerce").mean())
            # For subjects with multiple walks, between-walk variability can be a gait-stability cue.
            row[f"{c}__walk_std"] = float(pd.to_numeric(g[c], errors="coerce").std(ddof=0)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def classifier_factory(name: str, seed: int):
    if name == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=seed),
        )
    if name == "svm":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced", C=1.0, gamma="scale", random_state=seed),
        )
    raise ValueError(name)


def eval_predictions(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "weighted_precision": float(precision_score(y, pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y, pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def run_cv(subject_df: pd.DataFrame, classifier: str, seed: int, feature_cols: list[str], y_override=None):
    x = subject_df[feature_cols].to_numpy(dtype=float)
    y = subject_df["label"].to_numpy(dtype=int) if y_override is None else np.asarray(y_override, dtype=int)
    n_min = min(np.bincount(y))
    n_splits = max(2, min(5, int(n_min)))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.full(len(y), -1, dtype=int)
    for tr, te in splitter.split(x, y):
        model = classifier_factory(classifier, seed)
        model.fit(x[tr], y[tr])
        pred[te] = model.predict(x[te])
    return eval_predictions(y, pred)


def summarize_cv(subject_df: pd.DataFrame, classifiers: list[str], repeats: int, permutations: int, seed0: int = 2026):
    raw_cols = feature_columns(subject_df)
    # Drop columns with too few finite values or near-zero variance.
    feature_cols = []
    for c in raw_cols:
        v = pd.to_numeric(subject_df[c], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(v).sum() >= max(5, int(0.5 * len(v))) and np.nanstd(v) > 1e-12:
            feature_cols.append(c)

    rows = []
    per_seed = []
    for clf in classifiers:
        for r in range(repeats):
            seed = seed0 + r
            res = run_cv(subject_df, clf, seed, feature_cols)
            row = {"classifier": clf, "repeat": r, "is_permutation": False, **res}
            per_seed.append(row)
        # Permutation null only for RF by default, but allow all if requested.
        n_perm = permutations if clf == "rf" else min(20, permutations)
        rng = np.random.default_rng(seed0 + 999)
        y = subject_df["label"].to_numpy(dtype=int)
        for p in range(n_perm):
            y_perm = rng.permutation(y)
            res = run_cv(subject_df, clf, seed0 + 10000 + p, feature_cols, y_override=y_perm)
            per_seed.append({"classifier": clf, "repeat": p, "is_permutation": True, **res})

    per_seed_df = pd.DataFrame(per_seed)
    summary = []
    for (clf, is_perm), g in per_seed_df.groupby(["classifier", "is_permutation"]):
        row = {
            "classifier": clf,
            "is_permutation": bool(is_perm),
            "n_subjects": int(len(subject_df)),
            "n_positive": int(subject_df["label"].sum()),
            "n_negative": int((1 - subject_df["label"]).sum()),
            "n_features": int(len(feature_cols)),
        }
        for m in ["accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "balanced_accuracy"]:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std(ddof=0))
            row[f"{m}_p05"] = float(g[m].quantile(0.05))
            row[f"{m}_p95"] = float(g[m].quantile(0.95))
        summary.append(row)
    return pd.DataFrame(summary), per_seed_df, feature_cols


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    body_model = art.ParametricModel(paths.smpl_file)

    all_walks = []
    all_subjects = []
    all_summary = []
    all_seed_rows = []
    meta = {
        "target_fps": TARGET_FPS,
        "task": "subject-level FOG/freezer-trait classification from GT SMPL gait descriptors",
        "label_caveat": "Labels are subject-level traits, not per-frame FOG episodes. Results should not be described as FOG detection.",
        "feature_caveat": "No anthropometric/body-shape features are used; n_frames/duration are excluded from classifiers.",
        "datasets": {},
    }

    for dataset_name in args.datasets:
        spec = DATASETS[dataset_name]
        data = safe_load_pickle(spec["path"])
        walks = collect_walks(data, spec, args.limit)
        print(f"{dataset_name}: {len(walks)} labeled walks")
        rows = []
        for sid, wid, rec, label, label_text in tqdm(walks, desc=f"{dataset_name} GT features"):
            feat = extract_features_for_walk(body_model, rec, device, args.min_frames)
            if feat is None:
                continue
            feat.update({
                "dataset": dataset_name,
                "subject": sid,
                "walk": wid,
                "label": int(label),
                "label_text": label_text,
                "other": rec.get("other"),
                "medication": rec.get("medication"),
                "source_fps": rec.get("fps"),
            })
            rows.append(feat)

        per_walk = pd.DataFrame(rows)
        subject_df = aggregate_subject_features(per_walk)
        classifiers = ["rf", "logistic", "svm"]
        summary, per_seed, feat_cols = summarize_cv(subject_df, classifiers, args.cv_repeats, args.permutations)
        summary.insert(0, "dataset", dataset_name)
        per_seed.insert(0, "dataset", dataset_name)
        subject_df.insert(0, "dataset_name", dataset_name)
        per_walk.insert(0, "dataset_name", dataset_name)

        per_walk.to_csv(out_dir / f"{dataset_name}_gt_features_per_walk.csv", index=False)
        subject_df.to_csv(out_dir / f"{dataset_name}_gt_features_per_subject.csv", index=False)
        summary.to_csv(out_dir / f"{dataset_name}_gt_feature_classifier_summary.csv", index=False)
        per_seed.to_csv(out_dir / f"{dataset_name}_gt_feature_classifier_per_seed.csv", index=False)
        (out_dir / f"{dataset_name}_feature_columns.txt").write_text("\n".join(feat_cols), encoding="utf-8")

        all_walks.append(per_walk)
        all_subjects.append(subject_df)
        all_summary.append(summary)
        all_seed_rows.append(per_seed)
        counts = subject_df["label_text"].value_counts().to_dict()
        meta["datasets"][dataset_name] = {
            "source": str(spec["path"]),
            "label_name": spec["label_name"],
            "positive": spec["positive"],
            "negative": spec["negative"],
            "excluded": sorted(spec["exclude"]),
            "n_walks_used": int(len(per_walk)),
            "n_subjects_used": int(len(subject_df)),
            "subject_label_counts": counts,
            "n_features": int(len(feat_cols)),
        }

    pd.concat(all_walks, ignore_index=True).to_csv(out_dir / "fog_gt_features_per_walk_all.csv", index=False)
    pd.concat(all_subjects, ignore_index=True).to_csv(out_dir / "fog_gt_features_per_subject_all.csv", index=False)
    pd.concat(all_summary, ignore_index=True).to_csv(out_dir / "fog_gt_feature_classifier_summary_all.csv", index=False)
    pd.concat(all_seed_rows, ignore_index=True).to_csv(out_dir / "fog_gt_feature_classifier_per_seed_all.csv", index=False)
    with (out_dir / "fog_gt_feature_gate_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nSaved:", out_dir)
    print(pd.concat(all_summary, ignore_index=True).query("classifier == 'rf'").to_string(index=False))


if __name__ == "__main__":
    main()
