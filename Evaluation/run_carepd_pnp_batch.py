"""
run_carepd_pnp_batch.py

Run PNP on CARE-PD/BMCLab sequences prepared in the TransPose adapter format.

PNP expects the same six virtual IMUs as TransPose/PIP:
  [left forearm, right forearm, left lower leg, right lower leg, head, pelvis/root]

Unlike TransPose/PIP raw, PNP also consumes angular velocity and runs an online
physics optimizer. This script derives angular velocity from the global IMU
orientation sequence and feeds raw physical units to the official PNP model.
The CARE-PD adapter acceleration follows the TransPose synthetic-AMASS
convention: second derivative of virtual sensor vertices, without a gravity
term.

Examples:
  python Evaluation/run_carepd_pnp_batch.py --limit 1 --max-frames 120 --overwrite
  python Evaluation/run_carepd_pnp_batch.py --overwrite
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
PNP_ROOT = os.path.join(PROJECT_ROOT, "PNP")
RBDL_PYTHON_DIR = os.path.join(PROJECT_ROOT, "external", "rbdl", "build", "python")
TRANSPOSE_DATASET_DIR = os.path.join(PROJECT_ROOT, "TransPose", "data", "dataset_work", "CAREPD_BMCLab")

for path in [RBDL_PYTHON_DIR, PNP_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)


DATASET_PT = os.path.join(TRANSPOSE_DATASET_DIR, "test.pt")
MANIFEST_JSON = os.path.join(TRANSPOSE_DATASET_DIR, "manifest.json")
RESULT_DIR = os.path.join(PNP_ROOT, "data", "results", "CAREPD_BMCLab_batch")
OUTPUT_PT = os.path.join(RESULT_DIR, "predictions.pt")
FPS = 60.0
G_WORLD = torch.tensor([0.0, -9.8, 0.0])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_PT)
    parser.add_argument("--manifest", default=MANIFEST_JSON)
    parser.add_argument("--out", default=OUTPUT_PT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--add-gravity",
        action="store_true",
        help="Diagnostic only: add gravity to the synthetic acceleration before feeding PNP.",
    )
    parser.add_argument("--overwrite", action="store_true")
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


def global_angular_velocity_from_ori(art, ori):
    """Compute global angular velocity from global IMU orientation matrices."""
    if ori.shape[0] < 2:
        return torch.zeros(ori.shape[0], ori.shape[1], 3, dtype=ori.dtype)
    rel_local = ori[:-1].transpose(2, 3).matmul(ori[1:])
    w_local = art.math.rotation_matrix_to_axis_angle(rel_local).view(-1, ori.shape[1], 3) * FPS
    w_global = ori[:-1].matmul(w_local.unsqueeze(-1)).squeeze(-1)
    return torch.cat((w_global, torch.zeros_like(w_global[:1])), dim=0).float()


@torch.no_grad()
def run_one_pnp(net, art, data, index, max_frames, add_gravity):
    acc = data["acc"][index].float()
    ori = data["ori"][index].float()
    pose_gt_axis_angle = data["pose"][index].float()
    tran_gt = data["tran"][index].float()

    if max_frames > 0:
        acc = acc[:max_frames]
        ori = ori[:max_frames]
        pose_gt_axis_angle = pose_gt_axis_angle[:max_frames]
        tran_gt = tran_gt[:max_frames]

    # Keep the TransPose-style synthetic vertex acceleration as-is. A 20-sequence
    # adapter ablation showed that adding gravity produces a large PNP frame /
    # physics mismatch; --add-gravity is retained only as a diagnostic switch.
    acc_pnp = acc + G_WORLD.view(1, 1, 3) if add_gravity else acc
    gyro_pnp = global_angular_velocity_from_ori(art, ori)
    pose_gt = art.math.axis_angle_to_rotation_matrix(pose_gt_axis_angle).view(-1, 24, 3, 3)

    net.rnn_initialize(pose_gt[0])
    if hasattr(net, "dynamics_optimizer"):
        net.dynamics_optimizer.quiet = True

    pose_pred = torch.zeros_like(pose_gt)
    tran_pred = torch.zeros_like(tran_gt)
    for i in range(pose_gt.shape[0]):
        pose_i, tran_i = net.forward_frame(
            acc_pnp[i].to(net.device),
            gyro_pnp[i].to(net.device),
            ori[i].to(net.device),
        )
        pose_pred[i] = pose_i.cpu()
        tran_pred[i] = tran_i.cpu()

    return {
        "pose_pred": pose_pred.cpu(),
        "tran_pred": tran_pred.cpu(),
        "pose_gt": pose_gt.cpu(),
        "tran_gt": tran_gt.cpu(),
        "acc": acc_pnp.cpu(),
        "ori": ori.cpu(),
        "gyro": gyro_pnp.cpu(),
    }


def main():
    args = parse_args()
    args.dataset = os.path.abspath(args.dataset)
    args.manifest = os.path.abspath(args.manifest)
    args.out = os.path.abspath(args.out)
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Missing dataset: {args.dataset}")
    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.out}. Pass --overwrite to replace it.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.chdir(PNP_ROOT)

    import articulate as art
    from net import PNP

    print(f"Dataset : {args.dataset}")
    print(f"Manifest: {args.manifest}")
    print(f"Output  : {args.out}")
    print("Model   : PNP official online physics model")
    print(f"Gravity : {'added to synthetic acc (diagnostic)' if args.add_gravity else 'not added'}")

    data = safe_torch_load(args.dataset, map_location="cpu")
    manifest = load_manifest(args.manifest)
    n_total = len(data["acc"])
    indices = selected_indices(n_total, args.start, args.limit)
    net = PNP()
    if hasattr(net, "dynamics_optimizer"):
        net.dynamics_optimizer.quiet = True

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
        "gyro": [],
    }

    for index in tqdm(indices, desc="CARE-PD PNP batch"):
        label = make_label(manifest, index)
        out = run_one_pnp(net, art, data, index, args.max_frames, add_gravity=args.add_gravity)
        item = manifest[index] if manifest is not None and index < len(manifest) else None

        outputs["index"].append(index)
        outputs["label"].append(label)
        outputs["manifest"].append(item)
        for key in ["pose_pred", "tran_pred", "pose_gt", "tran_gt", "acc", "ori", "gyro"]:
            outputs[key].append(out[key])

    outputs["meta"] = {
        "model": "PNP",
        "dataset": args.dataset,
        "manifest_path": args.manifest,
        "n_total_dataset_sequences": n_total,
        "n_processed_sequences": len(indices),
        "start": args.start,
        "limit": args.limit,
        "max_frames": args.max_frames,
        "fps": FPS,
        "sensor_order": [
            "left_forearm",
            "right_forearm",
            "left_lower_leg",
            "right_lower_leg",
            "head",
            "pelvis_root",
        ],
        "ignored_joints": [0, 7, 8, 10, 11, 20, 21, 22, 23],
        "input_conventions": {
            "acc": "[T,6,3] global synthetic vertex acceleration in physical units; no gravity term in the main adapter",
            "ori": "[T,6,3,3] global IMU orientation",
            "gyro": "[T,6,3] global angular velocity derived from ori at 60 fps",
        },
        "runtime": {
            "pnp_root": PNP_ROOT,
            "rbdl_python_dir": RBDL_PYTHON_DIR,
            "smpl_file": os.path.join(PNP_ROOT, "models", "SMPL_male.pkl"),
            "weights": os.path.join(PNP_ROOT, "data", "weights", "PNP", "weights.pt"),
            "physics_model": os.path.join(PNP_ROOT, "models", "physics.urdf"),
            "device": str(net.device),
        },
        "caveat": (
            "PNP is an online physics model. Inspect visualizations and metrics before treating "
            "global translation as comparable to TransPose/PIP raw/DynaIP."
        ),
        "notes": [
            "Main adapter does not add gravity to CARE-PD synthetic acceleration.",
            "The --add-gravity mode is diagnostic only; it produced large pose-position errors in a first-20 ablation.",
        ],
    }

    torch.save(outputs, args.out)
    print(f"Saved {len(indices)} sequences to {args.out}")


if __name__ == "__main__":
    main()
