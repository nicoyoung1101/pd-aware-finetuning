"""
Run Transformer Inertial Poser (TIP) on CARE-PD C6 synthetic IMU sequences.

This adapter keeps the output format aligned with the other model runners:
{
    "pose_gt":   list[T, 24, 3, 3],
    "pose_pred": list[T, 24, 3, 3],
    "tran_gt":   list[T, 3],
    "tran_pred": list[T, 3],
    "manifest": list[dict],
}

TIP is a causal/real-time model. It predicts active SMPL-local joints in a
PyBullet/Nimble state representation; this script maps those predictions back
to 24-joint SMPL rotation matrices. Joints not represented by TIP (toe, wrist,
hand) are left as identity and should be excluded from local rotation metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIP_ROOT = PROJECT_ROOT / "TIP"
TRANSPOSE_ROOT = PROJECT_ROOT / "TransPose"

for p in [PROJECT_ROOT, TIP_ROOT, TRANSPOSE_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _aa_to_rotmat(axis_angle: torch.Tensor) -> torch.Tensor:
    import articulate as art

    flat = axis_angle.reshape(-1, 3)
    return art.math.axis_angle_to_rotation_matrix(flat).reshape(*axis_angle.shape[:-1], 3, 3)


def _load_manifest(dataset_path: Path, n: int) -> list[dict]:
    manifest_path = dataset_path.with_name("manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if len(manifest) >= n:
            return manifest[:n]
    return [{"index": i, "label": f"seq_{i:04d}"} for i in range(n)]


def _make_tip_model(checkpoint: Path, device: torch.device):
    from simple_transformer_with_state import TF_RNN_Past_State

    model = TF_RNN_Past_State(
        input_size_imu=6 * (9 + 3),
        size_s=18 * 6 + 3 + 20,
        rnn_hid_size=512,
        tf_hid_size=1024,
        tf_in_dim=256,
        n_heads=16,
        tf_layers=4,
        dropout=0.0,
        in_dropout=0.0,
        past_state_dropout=0.8,
        with_acc_sum=True,
    )
    state = _safe_torch_load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _init_tip_character():
    from render_funcs import COLOR_OURS, init_viz
    import constants as cst

    spec = importlib.util.spec_from_file_location("char_info", TIP_ROOT / "amass_char_info.py")
    char_info = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(char_info)

    map_bound = cst.MAP_BOUND * 2.0
    grid_num = int(map_bound / cst.GRID_SIZE) * 2
    init_grid_list = list(np.zeros((grid_num, grid_num)).flatten())

    cwd = os.getcwd()
    os.chdir(TIP_ROOT)
    try:
        _, char, _, _, _, _ = init_viz(
            char_info,
            init_grid_list,
            hmap_scale=cst.GRID_SIZE,
            gui=False,
            compare_gt=False,
            color=COLOR_OURS,
            viz_h_map=False,
        )
    finally:
        os.chdir(cwd)
    return char, char_info


def _dataset_pose_to_rotmat(pose: torch.Tensor) -> torch.Tensor:
    # dataset pose is [T, 24, 3] axis-angle for CARE-PD.
    if pose.ndim == 4:
        return pose.float()
    return _aa_to_rotmat(pose.float())


def _pose_rot_to_tip_state(char, pose_rot: np.ndarray, tran: np.ndarray | None):
    from data_utils import dip_pose_2_bullet_format
    import constants as cst

    bullet_q = dip_pose_2_bullet_format(char, pose_rot, tran)
    s = np.zeros(cst.n_dofs * 2, dtype=np.float64)
    s[:6] = bullet_q[:6]
    cursor = 6
    for idx in char.non_root_active_idx:
        start = (char.get_char_info().nimble_state_map[idx] - 1) * 3 + 6
        s[start : start + 3] = bullet_q[cursor : cursor + 3]
        cursor += 3
    assert cursor == cst.n_dofs
    return s


def _tip_state_to_smpl_pose(char, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from fairmotion.ops import conversions
    import constants as cst

    pose = np.tile(np.eye(3, dtype=np.float64), (24, 1, 1))
    q = state[: cst.n_dofs]

    # Inverse of data_utils.dip_pose_2_bullet_format root conversion.
    pose[0] = cst.rot_up_R.T @ conversions.A2R(q[3:6])

    for idx in char.non_root_active_idx:
        name = char.get_char_info().bvh_map[idx]
        smpl_idx = cst.SMPL_JOINT_IDX_MAPPING[name]
        start = (char.get_char_info().nimble_state_map[idx] - 1) * 3 + 6
        pose[smpl_idx] = conversions.A2R(q[start : start + 3])

    tran = cst.rot_up_R.T @ np.array([q[0], q[1], q[2] - cst.root_z_offset], dtype=np.float64)
    return pose.astype(np.float32), tran.astype(np.float32)


def _carepd_imu_to_tip(ori: torch.Tensor, acc: torch.Tensor, rotate_to_tip_frame: bool = True) -> np.ndarray:
    # CARE-PD C6 order: left forearm, right forearm, left lower leg,
    # right lower leg, head, pelvis/root.
    # TIP order: pelvis/root, left wrist/forearm, right wrist/forearm,
    # left knee/lower leg, right knee/lower leg, head.
    order = [5, 0, 1, 2, 3, 4]
    ori_np = ori[:, order].detach().cpu().numpy()
    acc_np = acc[:, order].detach().cpu().numpy()
    if rotate_to_tip_frame:
        import constants as cst

        # TIP's official preprocessing left-multiplies real IMU orientation and
        # acceleration by rot_up_R before feeding the network. CARE-PD synthetic
        # IMUs are in the TransPose/SMPL global frame, so we apply the same frame
        # conversion after reordering sensors into TIP order.
        R = np.asarray(cst.rot_up_R, dtype=ori_np.dtype)
        ori_np = np.einsum("ij,tbjk->tbik", R, ori_np)
        acc_np = np.einsum("ij,tbj->tbi", R, acc_np)
    return np.concatenate([ori_np.reshape(ori_np.shape[0], 6 * 9), acc_np.reshape(acc_np.shape[0], 6 * 3)], axis=1)


@torch.no_grad()
def run_one_sequence(
    model,
    char,
    pose_gt_rot: torch.Tensor,
    tran_gt: torch.Tensor,
    ori: torch.Tensor,
    acc: torch.Tensor,
    rotate_imu_to_tip_frame: bool = True,
):
    from real_time_runner_minimal import RTRunnerMin
    import constants as cst

    imu = _carepd_imu_to_tip(ori, acc, rotate_to_tip_frame=rotate_imu_to_tip_frame)
    s_init = _pose_rot_to_tip_state(
        char,
        pose_gt_rot[0].detach().cpu().numpy(),
        tran_gt[0].detach().cpu().numpy() if tran_gt is not None else None,
    )

    runner = RTRunnerMin(char, model, max_input_l=40, s_init=s_init, with_acc_sum=True)
    pred_states = np.zeros((imu.shape[0], cst.n_dofs * 2), dtype=np.float64)
    pred_states[0] = s_init

    for t in range(imu.shape[0] - 1):
        res = runner.step(imu[t], pred_states[t, :3])
        pred_states[t + 1] = res["qdq"]

    # Match TIP's built-in delay compensation.
    trim = runner.IMU_n_smooth + 2
    if pred_states.shape[0] > trim + 1:
        pred_states[:-trim] = pred_states[trim:]
        pred_states[-trim:] = pred_states[-trim - 1]

    pose_pred = []
    tran_pred = []
    for s in pred_states:
        p, tr = _tip_state_to_smpl_pose(char, s)
        pose_pred.append(p)
        tran_pred.append(tr)
    return torch.from_numpy(np.stack(pose_pred)), torch.from_numpy(np.stack(tran_pred))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(PROJECT_ROOT / "TransPose/data/dataset_work/CAREPD_BMCLab/test.pt"))
    p.add_argument("--checkpoint", default=str(TIP_ROOT / "output/model-without-dip9and10.pt"))
    p.add_argument("--out", default=str(PROJECT_ROOT / "TIP/data/results/CAREPD_BMCLab_batch/predictions.pt"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument(
        "--no-imu-frame-rotation",
        action="store_true",
        help="Disable TIP official rot_up_R IMU frame conversion for ablation/debugging.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cpu":
        # TIP's original runner was written for CUDA and calls tensor.cuda()
        # inside the real-time step. On CPU/M-series Macs we keep tensors on CPU.
        torch.Tensor.cuda = lambda self, *a, **kw: self

    data = _safe_torch_load(Path(args.dataset), map_location="cpu")
    manifest = _load_manifest(Path(args.dataset), len(data["pose"]))
    n = len(data["pose"]) if args.limit <= 0 else min(args.limit, len(data["pose"]))

    model = _make_tip_model(Path(args.checkpoint), device)
    char, _ = _init_tip_character()

    pose_gt_out, pose_pred_out, tran_gt_out, tran_pred_out, manifest_out = [], [], [], [], []
    for i in tqdm(range(n), desc="TIP CARE-PD"):
        pose_gt_rot = _dataset_pose_to_rotmat(data["pose"][i])
        tran_gt = data.get("tran", [None] * len(data["pose"]))[i]
        pose_pred, tran_pred = run_one_sequence(
            model,
            char,
            pose_gt_rot,
            tran_gt,
            data["ori"][i].float(),
            data["acc"][i].float(),
            rotate_imu_to_tip_frame=not args.no_imu_frame_rotation,
        )
        pose_gt_out.append(pose_gt_rot.cpu())
        pose_pred_out.append(pose_pred.cpu())
        tran_gt_out.append(tran_gt.cpu() if isinstance(tran_gt, torch.Tensor) else torch.zeros(pose_pred.shape[0], 3))
        tran_pred_out.append(tran_pred.cpu())
        manifest_out.append(manifest[i])

    out = {
        "pose_gt": pose_gt_out,
        "pose_pred": pose_pred_out,
        "tran_gt": tran_gt_out,
        "tran_pred": tran_pred_out,
        "manifest": manifest_out,
        "model": "TIP",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "notes": {
            "carepd_order": ["l_forearm", "r_forearm", "l_lower_leg", "r_lower_leg", "head", "pelvis"],
            "tip_order": ["pelvis", "l_forearm", "r_forearm", "l_lower_leg", "r_lower_leg", "head"],
            "imu_frame_rotation": "rot_up_R" if not args.no_imu_frame_rotation else "disabled",
            "unpredicted_joints_identity": ["ltoe", "rtoe", "lwrist", "rwrist", "lhand", "rhand"],
            "translation": "TIP root velocity/SBP correction; treat as QA until validated",
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
