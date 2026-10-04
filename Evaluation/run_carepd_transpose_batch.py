"""
run_carepd_transpose_batch.py

Run TransPose offline inference for all CARE-PD/BMCLab sequences prepared by
Evaluation/prepare_carepd_transpose.py.

This script does not open the viewer. It saves one consolidated predictions.pt
file that downstream clinical-feature analyses can read directly.

Examples:
  python Evaluation/run_carepd_transpose_batch.py
  python Evaluation/run_carepd_transpose_batch.py --limit 5
  python Evaluation/run_carepd_transpose_batch.py --device auto --save-per-seq
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
import json
import os
import re
import sys

import torch
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRANSPOSE_ROOT = os.path.join(PROJECT_ROOT, "TransPose")
if TRANSPOSE_ROOT not in sys.path:
    sys.path.insert(0, TRANSPOSE_ROOT)

import articulate as art
from net import TransPoseNet
from utils import normalize_and_concat


DATASET_DIR = os.path.join(TRANSPOSE_ROOT, "data", "dataset_work", "CAREPD_BMCLab")
TEST_PT = os.path.join(DATASET_DIR, "test.pt")
MANIFEST_JSON = os.path.join(DATASET_DIR, "manifest.json")
RESULT_DIR = os.path.join(TRANSPOSE_ROOT, "data", "results", "CAREPD_BMCLab_batch")
OUTPUT_PT = os.path.join(RESULT_DIR, "predictions.pt")

SENSOR_NAMES = ["LeftForeArm", "RightForeArm", "LeftLowerLeg", "RightLowerLeg", "Head", "Root"]
SENSOR_CONFIGS = {
    "Full6": ["LeftForeArm", "RightForeArm", "LeftLowerLeg", "RightLowerLeg", "Head", "Root"],
    "C5NoHead": ["LeftForeArm", "RightForeArm", "LeftLowerLeg", "RightLowerLeg", "Root"],
    "Limb4": ["LeftForeArm", "RightForeArm", "LeftLowerLeg", "RightLowerLeg"],
    "AnchorGait4": ["Root", "LeftLowerLeg", "RightLowerLeg", "Head"],
    "Asym4": ["Root", "LeftLowerLeg", "RightLowerLeg", "RightForeArm"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=TEST_PT, help="Prepared CARE-PD TransPose test.pt")
    parser.add_argument("--manifest", default=MANIFEST_JSON, help="Matching manifest.json")
    parser.add_argument("--out", default=OUTPUT_PT, help="Consolidated output .pt file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Inference device")
    parser.add_argument("--start", type=int, default=0, help="First sequence index to process")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of sequences")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap per sequence for smoke tests")
    parser.add_argument(
        "--sensor-config",
        default="Full6",
        choices=sorted(SENSOR_CONFIGS.keys()),
        help="Sensor-channel masking/imputation preset.",
    )
    parser.add_argument(
        "--mask-policy",
        default="zero",
        choices=["zero", "mean", "copy"],
        help="How to fill missing sensors before TransPose normalization.",
    )
    parser.add_argument("--save-per-seq", action="store_true", help="Also save one .pt file per sequence")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing consolidated output")
    return parser.parse_args()


def choose_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda:0")
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_manifest(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_label(manifest, index):
    if manifest is None or index >= len(manifest):
        return f"seq_{index:04d}"
    item = manifest[index]
    parts = [
        str(item.get("subject_id", "SUB")),
        str(item.get("walk_id", "walk")),
        f"U{item.get('UPDRS_GAIT', 'NA')}",
        f"med{item.get('medication', 'NA')}",
    ]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "__".join(parts))


def selected_indices(n_total, start, limit):
    if start < 0 or start >= n_total:
        raise ValueError(f"--start must be in [0, {n_total - 1}], got {start}")
    end = n_total if limit <= 0 else min(n_total, start + limit)
    return list(range(start, end))


def project_to_rotation(mat):
    u, _, vh = torch.linalg.svd(mat)
    rot = u @ vh
    det = torch.det(rot)
    if torch.any(det < 0):
        u = u.clone()
        u[det < 0, :, -1] *= -1
        rot = u @ vh
    return rot


def mean_rotation(rot):
    return project_to_rotation(rot.mean(dim=1))


def impute_sensors(acc, ori, sensor_config, mask_policy):
    keep = set(SENSOR_CONFIGS[sensor_config])
    if sensor_config == "Full6":
        return acc, ori, []

    acc = acc.clone()
    ori = ori.clone()
    keep_indices = [i for i, name in enumerate(SENSOR_NAMES) if name in keep]
    missing = []
    identity = torch.eye(3, dtype=ori.dtype, device=ori.device).expand(ori.shape[0], 3, 3)

    if keep_indices:
        mean_acc = acc[:, keep_indices].mean(dim=1)
        mean_ori = mean_rotation(ori[:, keep_indices])
    else:
        mean_acc = torch.zeros_like(acc[:, 0])
        mean_ori = identity

    for i, name in enumerate(SENSOR_NAMES):
        if name in keep:
            continue
        missing.append(name)
        if mask_policy == "zero":
            acc[:, i] = 0.0
            ori[:, i] = identity
        elif mask_policy == "mean":
            acc[:, i] = mean_acc
            ori[:, i] = mean_ori
        elif mask_policy == "copy":
            if name == "Root" and {"LeftLowerLeg", "RightLowerLeg"}.issubset(keep):
                li = SENSOR_NAMES.index("LeftLowerLeg")
                ri = SENSOR_NAMES.index("RightLowerLeg")
                acc[:, i] = 0.5 * (acc[:, li] + acc[:, ri])
                ori[:, i] = mean_rotation(ori[:, [li, ri]])
            elif name == "Head":
                root_i = SENSOR_NAMES.index("Root")
                if "Root" in keep:
                    acc[:, i] = acc[:, root_i]
                    ori[:, i] = ori[:, root_i]
                else:
                    acc[:, i] = mean_acc
                    ori[:, i] = mean_ori
            elif name == "LeftForeArm" and "RightForeArm" in keep:
                j = SENSOR_NAMES.index("RightForeArm")
                acc[:, i] = acc[:, j]
                ori[:, i] = ori[:, j]
            elif name == "RightForeArm" and "LeftForeArm" in keep:
                j = SENSOR_NAMES.index("LeftForeArm")
                acc[:, i] = acc[:, j]
                ori[:, i] = ori[:, j]
            elif name == "LeftLowerLeg" and "RightLowerLeg" in keep:
                j = SENSOR_NAMES.index("RightLowerLeg")
                acc[:, i] = acc[:, j]
                ori[:, i] = ori[:, j]
            elif name == "RightLowerLeg" and "LeftLowerLeg" in keep:
                j = SENSOR_NAMES.index("LeftLowerLeg")
                acc[:, i] = acc[:, j]
                ori[:, i] = ori[:, j]
            else:
                acc[:, i] = mean_acc
                ori[:, i] = mean_ori
    return acc, ori, missing


@torch.no_grad()
def run_one(net, data, index, device, max_frames, sensor_config, mask_policy):
    acc = data["acc"][index].float()
    ori = data["ori"][index].float()
    pose_gt_axis_angle = data["pose"][index].float()
    tran_gt = data["tran"][index].float()

    if max_frames > 0:
        acc = acc[:max_frames]
        ori = ori[:max_frames]
        pose_gt_axis_angle = pose_gt_axis_angle[:max_frames]
        tran_gt = tran_gt[:max_frames]

    acc_model, ori_model, masked_sensor_names = impute_sensors(acc, ori, sensor_config, mask_policy)

    x = normalize_and_concat(acc_model, ori_model).to(device)
    net.reset()
    pose_pred, tran_pred = net.forward_offline(x)
    pose_gt = art.math.axis_angle_to_rotation_matrix(pose_gt_axis_angle).view(-1, 24, 3, 3)

    return {
        "pose_pred": pose_pred.cpu(),
        "tran_pred": tran_pred.cpu(),
        "pose_gt": pose_gt.cpu(),
        "tran_gt": tran_gt.cpu(),
        "acc": acc.cpu(),
        "ori": ori.cpu(),
        "acc_model": acc_model.cpu(),
        "ori_model": ori_model.cpu(),
        "masked_sensor_names": masked_sensor_names,
    }


def main():
    args = parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Missing dataset: {args.dataset}. Run Evaluation/prepare_carepd_transpose.py first.")
    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.out}. Pass --overwrite to replace it.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    per_seq_dir = os.path.join(os.path.dirname(args.out), "per_sequence")
    if args.save_per_seq:
        os.makedirs(per_seq_dir, exist_ok=True)

    print(f"Dataset : {args.dataset}")
    print(f"Manifest: {args.manifest}")
    print(f"Output  : {args.out}")
    print(f"Device  : {args.device}")
    print(f"Config  : {args.sensor_config} keep={SENSOR_CONFIGS[args.sensor_config]}")
    print(f"Mask    : {args.mask_policy}")

    data = safe_torch_load(args.dataset, map_location="cpu")
    manifest = load_manifest(args.manifest)
    n_total = len(data["acc"])
    indices = selected_indices(n_total, args.start, args.limit)
    device = choose_device(args.device)
    net = TransPoseNet().to(device)

    outputs = {
        "index": [],
        "label": [],
        "manifest": [],
        "pose_pred": [],
        "tran_pred": [],
        "pose_gt": [],
        "tran_gt": [],
        "acc": [],
        "ori": [],
    }

    for index in tqdm(indices, desc="CARE-PD TransPose batch"):
        label = make_label(manifest, index)
        out = run_one(net, data, index, device, args.max_frames, args.sensor_config, args.mask_policy)
        item = manifest[index] if manifest is not None and index < len(manifest) else None

        outputs["index"].append(index)
        outputs["label"].append(label)
        outputs["manifest"].append(item)
        for key in ["pose_pred", "tran_pred", "pose_gt", "tran_gt", "acc", "ori"]:
            outputs[key].append(out[key])

        if args.save_per_seq:
            torch.save({"index": index, "label": label, "manifest": item, **out},
                       os.path.join(per_seq_dir, f"{index:04d}__{label}.pt"))

    outputs["meta"] = {
        "dataset": args.dataset,
        "manifest_path": args.manifest,
        "n_total_dataset_sequences": n_total,
        "n_processed_sequences": len(indices),
        "start": args.start,
        "limit": args.limit,
        "max_frames": args.max_frames,
        "fps": 60,
        "device": str(device),
        "sensor_order": SENSOR_NAMES,
        "sensor_config": args.sensor_config,
        "mask_policy": args.mask_policy,
        "sensor_config_keep": SENSOR_CONFIGS[args.sensor_config],
        "sensor_config_masked": [name for name in SENSOR_NAMES if name not in SENSOR_CONFIGS[args.sensor_config]],
        "sensor_mask_policy": (
            "Missing sensors are imputed before TransPose normalize_and_concat. "
            f"Policy='{args.mask_policy}'. This is an input-imputation/channel-masking ablation "
            "of a 6-IMU pretrained model, not a deployable retrained 4-IMU model."
        ),
        "format": {
            "pose_pred": "list of [T,24,3,3] local rotation matrices",
            "tran_pred": "list of [T,3] TransPose root translations",
            "pose_gt": "list of [T,24,3,3] local rotation matrices",
            "tran_gt": "list of [T,3] GT root translations",
            "acc": "list of [T,6,3] global TransPose-style accelerations",
            "ori": "list of [T,6,3,3] global IMU orientations",
        },
    }

    torch.save(outputs, args.out)
    print(f"Saved {len(indices)} sequences to {args.out}")


if __name__ == "__main__":
    main()
