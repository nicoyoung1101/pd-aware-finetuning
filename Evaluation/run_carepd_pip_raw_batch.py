"""
run_carepd_pip_raw_batch.py

Run the neural-network stage of PIP only, before physics optimization.

This produces a diagnostic pose-only PIP baseline:
  - pose_pred = raw PIP network local rotations, before PhysicsOptimizer
  - tran_pred = zeros, because raw PIP does not estimate a reliable global path
  - tran_gt is preserved for pose-only visualization and optional GT-trajectory display

Use this to separate "PIP network behavior" from "PIP physics optimizer behavior".
Do not treat tran_pred from this file as a gait-speed or step-length prediction.

Examples:
  python Evaluation/run_carepd_pip_raw_batch.py --limit 5 --overwrite
  python Evaluation/run_carepd_pip_raw_batch.py --overwrite
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import argparse
import json
import re
import sys
import warnings

import numpy as np
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


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIP_ROOT = os.path.join(PROJECT_ROOT, "PIP")
RBDL_PYTHON_DIR = os.path.join(PROJECT_ROOT, "external", "rbdl", "build", "python")
TRANSPOSE_DATASET_DIR = os.path.join(PROJECT_ROOT, "TransPose", "data", "dataset_work", "CAREPD_BMCLab")

for path in [RBDL_PYTHON_DIR, PIP_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

DATASET_DIR = os.path.join(PIP_ROOT, "data", "dataset_work", "CAREPD_BMCLab")
TEST_PT = os.path.join(DATASET_DIR, "test.pt")
MANIFEST_JSON = os.path.join(TRANSPOSE_DATASET_DIR, "manifest.json")
RESULT_DIR = os.path.join(PIP_ROOT, "data", "results", "CAREPD_BMCLab_raw_batch")
OUTPUT_PT = os.path.join(RESULT_DIR, "predictions.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=TEST_PT, help="CARE-PD PIP-compatible test.pt")
    parser.add_argument("--manifest", default=MANIFEST_JSON, help="Matching CARE-PD manifest.json")
    parser.add_argument("--out", default=OUTPUT_PT, help="Consolidated output .pt file")
    parser.add_argument("--start", type=int, default=0, help="First sequence index to process")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of sequences")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap per sequence for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing consolidated output")
    return parser.parse_args()


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


@torch.no_grad()
def run_one_raw(net, art, data, index, max_frames):
    from config import joint_set
    from utils import normalize_and_concat

    acc = data["acc"][index].float()
    ori = data["ori"][index].float()
    pose_gt_axis_angle = data["pose"][index].float()
    tran_gt = data["tran"][index].float()

    if max_frames > 0:
        acc = acc[:max_frames]
        ori = ori[:max_frames]
        pose_gt_axis_angle = pose_gt_axis_angle[:max_frames]
        tran_gt = tran_gt[:max_frames]

    init_pose = art.math.axis_angle_to_rotation_matrix(pose_gt_axis_angle[0]).view(1, 24, 3, 3)
    init_pose[0, 0] = torch.eye(3)
    lj_init = net.forward_kinematics(init_pose)[1][0, joint_set.leaf].view(-1)
    jvel_init = torch.zeros(24 * 3)

    x = (normalize_and_concat(acc, ori), lj_init, jvel_init)
    _, _, global_6d_pose, joint_velocity, contact = [_[0] for _ in net.forward([x])]
    pose_pred = net._reduced_glb_6d_to_full_local_mat(ori.view(-1, 6, 3, 3)[:, -1], global_6d_pose)
    pose_gt = art.math.axis_angle_to_rotation_matrix(pose_gt_axis_angle).view(-1, 24, 3, 3)

    return {
        "pose_pred": pose_pred.cpu(),
        "tran_pred": torch.zeros_like(tran_gt).cpu(),
        "pose_gt": pose_gt.cpu(),
        "tran_gt": tran_gt.cpu(),
        "acc": acc.cpu(),
        "ori": ori.cpu(),
        "joint_velocity_raw": joint_velocity.cpu(),
        "contact_raw": contact.cpu(),
    }


def main():
    args = parse_args()
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Missing dataset: {args.dataset}")
    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.out}. Pass --overwrite to replace it.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.chdir(PIP_ROOT)

    import articulate as art
    from net import PIP

    print(f"Dataset : {args.dataset}")
    print(f"Manifest: {args.manifest}")
    print(f"Output  : {args.out}")
    print("Model   : PIP raw network output, no physics optimizer")

    data = safe_torch_load(args.dataset, map_location="cpu")
    manifest = load_manifest(args.manifest)
    n_total = len(data["acc"])
    indices = selected_indices(n_total, args.start, args.limit)
    net = PIP()

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
        "joint_velocity_raw": [],
        "contact_raw": [],
    }

    for index in tqdm(indices, desc="CARE-PD PIP raw batch"):
        label = make_label(manifest, index)
        out = run_one_raw(net, art, data, index, args.max_frames)
        item = manifest[index] if manifest is not None and index < len(manifest) else None

        outputs["index"].append(index)
        outputs["label"].append(label)
        outputs["manifest"].append(item)
        for key in [
            "pose_pred", "tran_pred", "pose_gt", "tran_gt", "acc", "ori",
            "joint_velocity_raw", "contact_raw",
        ]:
            outputs[key].append(out[key])

    outputs["meta"] = {
        "model": "PIP-raw",
        "dataset": args.dataset,
        "manifest_path": args.manifest,
        "n_total_dataset_sequences": n_total,
        "n_processed_sequences": len(indices),
        "start": args.start,
        "limit": args.limit,
        "max_frames": args.max_frames,
        "fps": 60,
        "runtime": {
            "rbdl_python_dir": RBDL_PYTHON_DIR,
            "physics_optimizer": "disabled",
            "single_thread_env": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            },
        },
        "format": {
            "pose_pred": "list of [T,24,3,3] local rotation matrices from PIP raw network output",
            "tran_pred": "zeros; raw PIP does not provide reliable global translation",
            "pose_gt": "list of [T,24,3,3] local rotation matrices",
            "tran_gt": "list of [T,3] GT root translations",
            "acc": "list of [T,6,3] raw global accelerations",
            "ori": "list of [T,6,3,3] global IMU orientations",
            "joint_velocity_raw": "raw network joint velocity output before physics optimizer",
            "contact_raw": "raw network foot contact logits before physics optimizer",
        },
        "caveat": "Use this file for pose-only diagnostics. Do not use tran_pred for gait speed or step length.",
    }

    torch.save(outputs, args.out)
    print(f"Saved {len(indices)} sequences to {args.out}")


if __name__ == "__main__":
    main()
