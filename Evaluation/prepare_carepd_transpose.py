"""
prepare_carepd_transpose.py

Convert CARE-PD canonical SMPL sequences into TransPose-style test.pt.

This adapter intentionally does not reuse the generic synthetic IMU pickle
signals. TransPose was trained with its own virtual sensor definition:
  acc: SMPL mesh vertices [1961, 5424, 1176, 4662, 411, 3021]
  ori: SMPL global joint rotations [18, 19, 4, 5, 15, 0]
  order: left forearm, right forearm, left lower leg, right lower leg, head, pelvis

The saved test.pt stores global acc/orientation. TransPose's
utils.normalize_and_concat() handles root-relative normalization before
network inference.
"""

# ========== numpy compatibility patch for chumpy / old SMPL pickles ==========
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

import json
import os
import pickle
import sys

import torch
from tqdm import tqdm

try:
    from scipy.signal import butter, sosfiltfilt
except Exception:
    butter = None
    sosfiltfilt = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRANSPOSE_ROOT = os.path.join(PROJECT_ROOT, "TransPose")
if TRANSPOSE_ROOT not in sys.path:
    sys.path.insert(0, TRANSPOSE_ROOT)

import articulate as art


INPUT_PKL = os.path.join(
    PROJECT_ROOT,
    "Dataset",
    "CARE-PD",
    "Canonicalized_SMPL_pickles",
    "BMCLab_canonical.pkl",
)
OUTPUT_DIR = os.path.join(TRANSPOSE_ROOT, "data", "dataset_work", "CAREPD_BMCLab")
SMPL_FILE = os.path.join(
    PROJECT_ROOT,
    "Models",
    "smpl_models",
    "smpl",
    "SMPL_NEUTRAL.pkl",
)

TARGET_FPS = 60.0
SOURCE_FPS_FALLBACK = 150.0

# Match TransPose AMASS preprocessing.
VI_MASK = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
JI_MASK = torch.tensor([18, 19, 4, 5, 15, 0])
ACC_SMOOTH_N = 4

# Keep this off for the first TransPose run to stay faithful to preprocess.py.
APPLY_EXTRA_LOWPASS = False
EXTRA_LOWPASS_HZ = 12.0

# Optional source-frame crop before resampling. Usually leave this at 0.
SOURCE_TRIM_FRAMES = 0

# Match TransPose DIP preprocessing's boundary cleanup after acceleration
# synthesis. This removes finite-difference edge artifacts.
OUTPUT_TRIM_FRAMES = 6


def check_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"{label} is empty: {path}")


def is_sequence_dict(obj):
    return isinstance(obj, dict) and "pose" in obj and "trans" in obj


def collect_walks(data):
    walks = []

    def visit(obj, path):
        if is_sequence_dict(obj):
            if len(path) >= 2:
                subj_id, walk_id = path[-2], path[-1]
            elif len(path) == 1:
                subj_id, walk_id = path[0], path[0]
            else:
                subj_id, walk_id = "unknown", f"walk_{len(walks):04d}"
            walks.append((str(subj_id), str(walk_id), obj))
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                visit(value, path + [key])
            return

        if isinstance(obj, (list, tuple)):
            for idx, value in enumerate(obj):
                visit(value, path + [idx])

    visit(data, [])
    return walks


def trim_arrays(pose, trans, trim_frames):
    if trim_frames <= 0:
        return pose, trans, 0
    if pose.shape[0] <= 2 * trim_frames:
        return pose, trans, 0
    return pose[trim_frames:-trim_frames], trans[trim_frames:-trim_frames], trim_frames


def trim_outputs(acc, ori, pose, trans, trim_frames):
    if trim_frames <= 0:
        return acc, ori, pose, trans, 0
    if pose.shape[0] <= 2 * trim_frames:
        return acc, ori, pose, trans, 0
    return (
        acc[trim_frames:-trim_frames],
        ori[trim_frames:-trim_frames],
        pose[trim_frames:-trim_frames],
        trans[trim_frames:-trim_frames],
        trim_frames,
    )


def resample_indices(n_frames, src_fps, dst_fps):
    if n_frames <= 0:
        return np.array([], dtype=np.int64)
    duration = (n_frames - 1) / src_fps
    n_out = int(np.floor(duration * dst_fps)) + 1
    times = np.arange(n_out, dtype=np.float64) / dst_fps
    indices = np.round(times * src_fps).astype(np.int64)
    return np.clip(indices, 0, n_frames - 1)


def lowpass_tensor(x, fps, cutoff_hz):
    if not APPLY_EXTRA_LOWPASS:
        return x
    if butter is None or sosfiltfilt is None:
        raise RuntimeError("scipy.signal butter/sosfiltfilt are required when APPLY_EXTRA_LOWPASS=True")
    nyquist = 0.5 * fps
    if cutoff_hz is None or cutoff_hz >= nyquist or x.shape[0] < 18:
        return x
    sos = butter(4, cutoff_hz / nyquist, btype="lowpass", output="sos")
    filtered = sosfiltfilt(sos, x.cpu().numpy(), axis=0).astype(np.float32)
    return torch.from_numpy(filtered).to(x.device)


def syn_acc_transpose(v, smooth_n=ACC_SMOOTH_N):
    """
    Reproduce TransPose preprocess.py::_syn_acc for 60fps sequences.
    v shape: [N, 6, 3]
    """
    if v.shape[0] < 3:
        return torch.zeros_like(v)

    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * (TARGET_FPS ** 2)
                       for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))

    if mid != 0 and v.shape[0] > smooth_n * 2:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * (TARGET_FPS ** 2) / (smooth_n ** 2)
             for i in range(0, v.shape[0] - smooth_n * 2)]
        )
    return acc.float()


def process_walk(body_model, seq):
    pose_np, trans_np, source_trim = trim_arrays(seq["pose"], seq["trans"], SOURCE_TRIM_FRAMES)
    src_fps = float(seq.get("fps", SOURCE_FPS_FALLBACK))
    idx = resample_indices(pose_np.shape[0], src_fps, TARGET_FPS)

    pose = torch.from_numpy(pose_np[idx]).float().view(-1, 24, 3)
    trans = torch.from_numpy(trans_np[idx]).float().view(-1, 3)

    pose_mat = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
    grot, _, vert = body_model.forward_kinematics(pose_mat, None, trans, calc_mesh=True)

    sensor_vertices = lowpass_tensor(vert[:, VI_MASK], TARGET_FPS, EXTRA_LOWPASS_HZ)
    acc = syn_acc_transpose(sensor_vertices)
    ori = grot[:, JI_MASK].contiguous().float()
    acc, ori, pose, trans, output_trim = trim_outputs(acc, ori, pose, trans, OUTPUT_TRIM_FRAMES)

    return {
        "acc": acc.cpu(),
        "ori": ori.cpu(),
        "pose": pose.cpu(),
        "tran": trans.cpu(),
        "source_fps": src_fps,
        "source_trim_frames": source_trim,
        "output_trim_frames": output_trim,
    }


def main():
    check_file(INPUT_PKL, "CARE-PD pickle")
    check_file(SMPL_FILE, "SMPL model")

    with open(INPUT_PKL, "rb") as f:
        data = pickle.load(f)

    walks = collect_walks(data)
    if not walks:
        raise ValueError("No CARE-PD sequences with both 'pose' and 'trans' were found.")

    print(f"Input: {INPUT_PKL}")
    print(f"SMPL: {SMPL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Walks: {len(walks)}")
    print(f"Target fps: {TARGET_FPS}")
    print(f"Extra lowpass: {APPLY_EXTRA_LOWPASS} ({EXTRA_LOWPASS_HZ} Hz)")

    body_model = art.ParametricModel(SMPL_FILE)

    accs, oris, poses, trans = [], [], [], []
    manifest = []
    for subj_id, walk_id, seq in tqdm(walks, desc="CARE-PD -> TransPose"):
        out = process_walk(body_model, seq)
        if out["pose"].shape[0] < 12:
            continue

        accs.append(out["acc"])
        oris.append(out["ori"])
        poses.append(out["pose"])
        trans.append(out["tran"])
        manifest.append({
            "subject_id": subj_id,
            "walk_id": walk_id,
            "source_fps": out["source_fps"],
            "target_fps": TARGET_FPS,
            "n_frames": int(out["pose"].shape[0]),
            "UPDRS_GAIT": seq.get("UPDRS_GAIT"),
            "medication": seq.get("medication"),
            "other": seq.get("other"),
            "source_trim_frames": out["source_trim_frames"],
            "output_trim_frames": out["output_trim_frames"],
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save({"acc": accs, "ori": oris, "pose": poses, "tran": trans},
               os.path.join(OUTPUT_DIR, "test.pt"))

    meta = {
        "source": INPUT_PKL,
        "smpl_file": SMPL_FILE,
        "target_fps": TARGET_FPS,
        "vi_mask": VI_MASK.tolist(),
        "ji_mask": JI_MASK.tolist(),
        "sensor_order": ["left_forearm", "right_forearm", "left_lower_leg", "right_lower_leg", "head", "pelvis"],
        "acc_smooth_n": ACC_SMOOTH_N,
        "apply_extra_lowpass": APPLY_EXTRA_LOWPASS,
        "extra_lowpass_hz": EXTRA_LOWPASS_HZ,
        "source_trim_frames": SOURCE_TRIM_FRAMES,
        "output_trim_frames": OUTPUT_TRIM_FRAMES,
        "n_sequences": len(accs),
    }
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(accs)} sequences to {os.path.join(OUTPUT_DIR, 'test.pt')}")
    print(f"Saved metadata to {os.path.join(OUTPUT_DIR, 'meta.json')}")


if __name__ == "__main__":
    main()
