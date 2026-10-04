"""
run_carepd_dynaip_batch.py

Run DynaIP inference on CARE-PD/BMCLab sequences prepared in the
TransPose adapter format.

Important conventions:
  - DynaIP expects sensor order:
      [Root, LeftLowerLeg, RightLowerLeg, Head, LeftForeArm, RightForeArm]
  - Our TransPose adapter stores:
      [LeftForeArm, RightForeArm, LeftLowerLeg, RightLowerLeg, Head, Root]
    so this script reorders sensors as [5, 2, 3, 4, 0, 1].
  - DynaIP uses its own root-normalized IMU input via utils.data.normalize_imu.
  - DynaIP predicts global SMPL rotations but no reliable global translation.
    tran_pred is therefore stored as zeros; do not use it for gait speed or
    step length.

Examples:
  python Evaluation/run_carepd_dynaip_batch.py --limit 3 --overwrite
  python Evaluation/run_carepd_dynaip_batch.py --sensor-config Limb4 --limit 3 --overwrite
  python Evaluation/run_carepd_dynaip_batch.py --overwrite
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
DYNAIP_ROOT = os.path.join(PROJECT_ROOT, "DynaIP")
TRANSPOSE_DATASET_DIR = os.path.join(PROJECT_ROOT, "TransPose", "data", "dataset_work", "CAREPD_BMCLab")

if DYNAIP_ROOT not in sys.path:
    sys.path.insert(0, DYNAIP_ROOT)

DATASET_PT = os.path.join(TRANSPOSE_DATASET_DIR, "test.pt")
MANIFEST_JSON = os.path.join(TRANSPOSE_DATASET_DIR, "manifest.json")
SMPL_NEUTRAL = os.path.join(
    PROJECT_ROOT, "Models", "smpl_models", "smpl", "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
)
RESULT_DIR = os.path.join(DYNAIP_ROOT, "data", "results", "CAREPD_BMCLab_batch")
OUTPUT_PT = os.path.join(RESULT_DIR, "predictions.pt")

# TransPose sensor order -> DynaIP sensor order.
SENSOR_REORDER = torch.tensor([5, 2, 3, 4, 0, 1], dtype=torch.long)
DYNAIP_SENSOR_NAMES = ["Root", "LeftLowerLeg", "RightLowerLeg", "Head", "LeftForeArm", "RightForeArm"]
SENSOR_CONFIGS = {
    "Full6": ["Root", "LeftLowerLeg", "RightLowerLeg", "Head", "LeftForeArm", "RightForeArm"],
    "Limb4": ["LeftLowerLeg", "RightLowerLeg", "LeftForeArm", "RightForeArm"],
    "AnchorGait4": ["Root", "LeftLowerLeg", "RightLowerLeg", "Head"],
    "Asym4": ["Root", "LeftLowerLeg", "RightLowerLeg", "RightForeArm"],
}

# DynaIP/DIP SMPL joint conventions.
P_INIT_JOINT_MASK = torch.tensor([1, 2, 3, 4, 5, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19])
P_INIT_SELECT = torch.tensor([0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13])
V_INIT_JOINT_MASK = torch.tensor([0, 15, 20, 21, 7, 8])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_PT)
    parser.add_argument("--manifest", default=MANIFEST_JSON)
    parser.add_argument("--out", default=OUTPUT_PT)
    parser.add_argument("--smpl", default=SMPL_NEUTRAL)
    parser.add_argument("--weights", default=os.path.join(DYNAIP_ROOT, "weights", "DynaIP_s.pth"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--sensor-config",
        default="Full6",
        choices=sorted(SENSOR_CONFIGS.keys()),
        help="DynaIP sensor-channel masking preset. Masking is applied after DynaIP root normalization.",
    )
    parser.add_argument(
        "--mask-policy",
        default="zero",
        choices=["zero", "mean", "copy"],
        help=(
            "How to fill masked DynaIP sensor channels: zero = current baseline; "
            "mean = per-frame mean over kept channels; copy = heuristic copy/average from related sensors."
        ),
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


@torch.no_grad()
def fk_global_and_joints(body_model, pose_local, tran=None, batch_size=2048):
    chunks_g = []
    chunks_j = []
    for start in range(0, pose_local.shape[0], batch_size):
        glb, joints = body_model.forward_kinematics(
            pose_local[start : start + batch_size],
            tran=None if tran is None else tran[start : start + batch_size],
            calc_mesh=False,
        )
        chunks_g.append(glb.cpu())
        chunks_j.append(joints.cpu())
    return torch.cat(chunks_g, dim=0), torch.cat(chunks_j, dim=0)


def rotation_6d_from_matrix(rot):
    return rot[:, :, :, :2].transpose(2, 3).clone().flatten(1)


def build_initial_states(body_model, pose_gt, tran_gt, root_ori):
    glb_pose, joints = fk_global_and_joints(body_model, pose_gt, tran=tran_gt)

    root_inv = glb_pose[:, :1].transpose(2, 3)
    glb_pose_norm = root_inv.matmul(glb_pose)
    p_all = rotation_6d_from_matrix(glb_pose_norm[:, P_INIT_JOINT_MASK])
    p_init = p_all.view(p_all.shape[0], -1, 6)[:, P_INIT_SELECT][:1].float()

    joints_rel = joints.clone()
    joints_rel[:, :, 0] = joints_rel[:, :, 0] - joints_rel[:, :1, 0]
    joints_rel[:, :, 2] = joints_rel[:, :, 2] - joints_rel[:, :1, 2]
    velocity = (joints_rel[1:] - joints_rel[:-1]) * 60.0
    velocity = torch.cat((velocity[:1], velocity), dim=0)
    velocity = torch.cat((velocity[:, :1], velocity[:, 1:] - velocity[:, :1]), dim=1)
    velocity = velocity.bmm(root_ori)
    v_init = velocity[:1, V_INIT_JOINT_MASK].float()
    return v_init, p_init


def _channel_mean(dynaip_imu, keep_indices):
    if not keep_indices:
        return torch.zeros_like(dynaip_imu[:, 0])
    return dynaip_imu[:, keep_indices].mean(dim=1)


def apply_sensor_mask(dynaip_imu, sensor_config, mask_policy):
    keep = set(SENSOR_CONFIGS[sensor_config])
    masked = dynaip_imu.clone()
    keep_indices = [i for i, name in enumerate(DYNAIP_SENSOR_NAMES) if name in keep]
    mean_channel = _channel_mean(dynaip_imu, keep_indices)
    masked_names = []
    for i, name in enumerate(DYNAIP_SENSOR_NAMES):
        if name not in keep:
            if mask_policy == "zero":
                masked[:, i] = 0.0
            elif mask_policy == "mean":
                masked[:, i] = mean_channel
            elif mask_policy == "copy":
                # Heuristic imputation in DynaIP-normalized feature space.
                # This is only an adapter ablation, not a deployable 4-IMU model.
                if name == "Root" and {"LeftLowerLeg", "RightLowerLeg"}.issubset(keep):
                    li = DYNAIP_SENSOR_NAMES.index("LeftLowerLeg")
                    ri = DYNAIP_SENSOR_NAMES.index("RightLowerLeg")
                    masked[:, i] = 0.5 * (dynaip_imu[:, li] + dynaip_imu[:, ri])
                elif name == "Head":
                    root_i = DYNAIP_SENSOR_NAMES.index("Root")
                    masked[:, i] = masked[:, root_i] if "Root" in keep or "Root" in masked_names else mean_channel
                elif name == "LeftForeArm" and "RightForeArm" in keep:
                    masked[:, i] = dynaip_imu[:, DYNAIP_SENSOR_NAMES.index("RightForeArm")]
                elif name == "RightForeArm" and "LeftForeArm" in keep:
                    masked[:, i] = dynaip_imu[:, DYNAIP_SENSOR_NAMES.index("LeftForeArm")]
                elif name == "LeftLowerLeg" and "RightLowerLeg" in keep:
                    masked[:, i] = dynaip_imu[:, DYNAIP_SENSOR_NAMES.index("RightLowerLeg")]
                elif name == "RightLowerLeg" and "LeftLowerLeg" in keep:
                    masked[:, i] = dynaip_imu[:, DYNAIP_SENSOR_NAMES.index("LeftLowerLeg")]
                else:
                    masked[:, i] = mean_channel
            masked_names.append(name)
    return masked, masked_names


@torch.no_grad()
def run_one(net, body_model, data, index, device, max_frames, sensor_config, mask_policy):
    from utils.data import normalize_imu

    acc = data["acc"][index].float()
    ori = data["ori"][index].float()
    pose_aa = data["pose"][index].float()
    tran_gt = data["tran"][index].float()

    if max_frames > 0:
        acc = acc[:max_frames]
        ori = ori[:max_frames]
        pose_aa = pose_aa[:max_frames]
        tran_gt = tran_gt[:max_frames]

    acc = acc[:, SENSOR_REORDER]
    ori = ori[:, SENSOR_REORDER]
    dynaip_imu = normalize_imu(acc, ori).float()
    dynaip_imu_model, masked_sensor_names = apply_sensor_mask(dynaip_imu, sensor_config, mask_policy)

    import articulate as art

    pose_gt = art.math.axis_angle_to_rotation_matrix(pose_aa).view(-1, 24, 3, 3).float()
    v_init, p_init = build_initial_states(body_model, pose_gt, tran_gt, ori[:, 0])

    glb_xsens, glb_smpl = net.predict(
        dynaip_imu_model.to(device),
        v_init.to(device),
        p_init.to(device),
    )
    pose_pred = body_model.inverse_kinematics_R(glb_smpl.cpu()).view(-1, 24, 3, 3)

    return {
        "pose_pred": pose_pred.cpu(),
        "tran_pred": torch.zeros_like(tran_gt).cpu(),
        "pose_gt": pose_gt.cpu(),
        "tran_gt": tran_gt.cpu(),
        "acc": acc.cpu(),
        "ori": ori.cpu(),
        "dynaip_imu": dynaip_imu.cpu(),
        "dynaip_imu_model": dynaip_imu_model.cpu(),
        "masked_sensor_names": masked_sensor_names,
        "glb_pose_smpl_pred": glb_smpl.cpu(),
        "glb_pose_xsens_pred": glb_xsens.cpu(),
    }


def main():
    args = parse_args()
    args.dataset = os.path.abspath(args.dataset)
    args.manifest = os.path.abspath(args.manifest)
    args.out = os.path.abspath(args.out)
    args.smpl = os.path.abspath(args.smpl)
    args.weights = os.path.abspath(args.weights)
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Missing dataset: {args.dataset}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Missing weights: {args.weights}")
    if not os.path.exists(args.smpl):
        raise FileNotFoundError(f"Missing SMPL model: {args.smpl}")
    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.out}. Pass --overwrite to replace it.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.chdir(DYNAIP_ROOT)

    import articulate as art
    from model.model import Poser

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    data = safe_torch_load(args.dataset, map_location="cpu")
    manifest = load_manifest(args.manifest)
    n_total = len(data["acc"])
    indices = selected_indices(n_total, args.start, args.limit)

    print(f"Dataset : {args.dataset}")
    print(f"Manifest: {args.manifest}")
    print(f"Weights : {args.weights}")
    print(f"SMPL    : {args.smpl}")
    print(f"Output  : {args.out}")
    print(f"Device  : {device}")
    print("Model   : DynaIP pose-only output, no global translation")
    print(f"Config  : {args.sensor_config} keep={SENSOR_CONFIGS[args.sensor_config]}")
    print(f"Mask    : {args.mask_policy}")

    net = Poser().to(device)
    net.load_state_dict(safe_torch_load(args.weights, map_location=device))
    net.eval()
    body_model = art.ParametricModel(args.smpl, device=device)

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
        "dynaip_imu": [],
        "dynaip_imu_model": [],
        "glb_pose_smpl_pred": [],
        "glb_pose_xsens_pred": [],
    }

    for index in tqdm(indices, desc="CARE-PD DynaIP batch"):
        label = make_label(manifest, index)
        out = run_one(net, body_model, data, index, device, args.max_frames, args.sensor_config, args.mask_policy)
        item = manifest[index] if manifest is not None and index < len(manifest) else None
        outputs["index"].append(index)
        outputs["label"].append(label)
        outputs["manifest"].append(item)
        for key in [
            "pose_pred", "tran_pred", "pose_gt", "tran_gt", "acc", "ori",
            "dynaip_imu", "dynaip_imu_model", "glb_pose_smpl_pred", "glb_pose_xsens_pred",
        ]:
            outputs[key].append(out[key])

    outputs["meta"] = {
        "model": "DynaIP",
        "dataset": args.dataset,
        "manifest_path": args.manifest,
        "weights": args.weights,
        "smpl": args.smpl,
        "n_total_dataset_sequences": n_total,
        "n_processed_sequences": len(indices),
        "start": args.start,
        "limit": args.limit,
        "max_frames": args.max_frames,
        "fps": 60,
        "sensor_order": ["Root", "LeftLowerLeg", "RightLowerLeg", "Head", "LeftForeArm", "RightForeArm"],
        "sensor_config": args.sensor_config,
        "mask_policy": args.mask_policy,
        "sensor_config_keep": SENSOR_CONFIGS[args.sensor_config],
        "sensor_config_masked": [name for name in DYNAIP_SENSOR_NAMES if name not in SENSOR_CONFIGS[args.sensor_config]],
        "sensor_mask_policy": (
            "DynaIP IMU is first normalized with the true root sensor using utils.data.normalize_imu; "
            f"masked sensor channels are filled using policy='{args.mask_policy}'. "
            "This is an input-imputation/channel-masking ablation of a 6-IMU pretrained model, "
            "not a deployable retrained 4-IMU model."
        ),
        "source_sensor_order": ["LeftForeArm", "RightForeArm", "LeftLowerLeg", "RightLowerLeg", "Head", "Root"],
        "source_to_dynaip_reorder": SENSOR_REORDER.tolist(),
        "translation_policy": "DynaIP predicts pose only here; tran_pred is zero and should not be used for full-motion gait metrics.",
        "ignored_rotation_joints": ["pelvis", "lankle", "rankle", "lfoot", "rfoot", "lwrist", "rwrist", "lhand", "rhand"],
        "format": {
            "pose_pred": "list of [T,24,3,3] local SMPL rotation matrices from DynaIP global prediction",
            "tran_pred": "list of [T,3] zeros; no reliable global translation",
            "pose_gt": "list of [T,24,3,3] GT local SMPL rotation matrices",
            "tran_gt": "list of [T,3] GT root translations",
        },
    }
    torch.save(outputs, args.out)
    print(f"Saved {len(indices)} sequences to {args.out}")


if __name__ == "__main__":
    main()
