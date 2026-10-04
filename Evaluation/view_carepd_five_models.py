#!/usr/bin/env python3
"""Interactive CARE-PD viewer for GT + five model reconstructions.

Default layout is root-aligned/in-place so models with unreliable global
translation can still be compared fairly for pose and motion dynamics.

Usage:
  KMP_DUPLICATE_LIB_OK=TRUE python \
      Evaluation/view_carepd_five_models.py --index 0

Keys:
  N / Right : next sequence
  P / Left  : previous sequence
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np

# Compatibility for older SMPL/chumpy-style dependencies.
if not hasattr(np, "bool"):
    np.bool = np.bool_
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "Evaluation") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "Evaluation"))

from aitviewer.configuration import CONFIG as C
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.spheres import Spheres

C.update_conf({"smplx_models": str(PROJECT_ROOT / "Models" / "smpl_models")})

from view_carepd_transpose import (  # noqa: E402
    TEST_PT,
    PLAYBACK_FPS,
    TransPoseNet,
    choose_device,
    find_index,
    get_manifest_item,
    load_manifest,
    make_label,
    align_to_floor,
    apply_path_mode,
    make_smpl_sequence,
    run_transpose,
    safe_torch_load,
)
import articulate as art  # noqa: E402

PREDICTION_FILES = {
    "PIP raw": PROJECT_ROOT / "PIP" / "data" / "results" / "CAREPD_BMCLab_raw_batch" / "predictions.pt",
    "DynaIP": PROJECT_ROOT / "DynaIP" / "data" / "results" / "CAREPD_BMCLab_batch" / "predictions.pt",
    "PNP": PROJECT_ROOT / "PNP" / "data" / "results" / "CAREPD_BMCLab_batch" / "predictions.pt",
    "TIP": PROJECT_ROOT / "TIP" / "data" / "results" / "CAREPD_BMCLab_batch" / "predictions.pt",
}

COLORS = {
    # High contrast: GT stays cool blue; all reconstructions use a warm gradient.
    "GT": (0.18, 0.58, 1.0, 1.0),
    "TransPose": (1.00, 0.78, 0.30, 1.0),
    "PIP raw": (1.00, 0.58, 0.18, 1.0),
    "DynaIP": (0.95, 0.36, 0.12, 1.0),
    "PNP": (0.82, 0.17, 0.10, 1.0),
    "TIP": (0.58, 0.06, 0.06, 1.0),
}

IMU_VERTEX_MASK = torch.tensor([1961, 5424, 1176, 4662, 411, 3021], dtype=torch.long)
IMU_SENSOR_NAMES = ["L forearm", "R forearm", "L shank", "R shank", "head", "pelvis"]
IMU_MARKER_COLOR = (0.02, 0.02, 0.02, 1.0)

ORDER = ["GT", "TransPose", "PIP raw", "DynaIP", "PNP", "TIP"]


def _as_tensor_list_item(obj, idx):
    item = obj[idx]
    if isinstance(item, np.ndarray):
        item = torch.from_numpy(item)
    return item.detach().cpu().float()


def _slice_frames(tensor, n):
    if tensor is None:
        return None
    return tensor[:n].clone()


def load_prediction_file(path, idx):
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    pred = safe_torch_load(str(path), map_location="cpu")
    n = len(pred["pose_pred"])
    if idx < 0 or idx >= n:
        raise IndexError(f"index {idx} outside prediction file range 0..{n-1}: {path}")
    return {
        "pose_pred": _as_tensor_list_item(pred["pose_pred"], idx),
        "tran_pred": _as_tensor_list_item(pred.get("tran_pred", pred.get("trans_pred")), idx)
        if ("tran_pred" in pred or "trans_pred" in pred)
        else None,
        "pose_gt": _as_tensor_list_item(pred["pose_gt"], idx) if "pose_gt" in pred else None,
        "tran_gt": _as_tensor_list_item(pred["tran_gt"], idx) if "tran_gt" in pred else None,
    }


def zero_translation_like(pose):
    return torch.zeros((pose.shape[0], 3), dtype=torch.float32)


def mesh_color(name, alpha):
    c = COLORS[name]
    return (c[0], c[1], c[2], alpha)


def make_imu_marker_sequence(
    name,
    pose_mat,
    tran,
    smpl_layer,
    x_offset=0.0,
    floor_mode="first",
    path_mode="center",
    radius=0.035,
    color=IMU_MARKER_COLOR,
):
    """Create animated IMU marker spheres at the exact synthetic-IMU SMPL vertices.

    The vertex mask matches Evaluation/prepare_carepd_transpose.py:
    left forearm, right forearm, left lower leg, right lower leg, head, pelvis.
    The same path/floor/x-offset transformation as make_smpl_sequence() is applied.
    """
    trans = tran.clone().float()
    pose_mat, trans = apply_path_mode(pose_mat, trans, path_mode)
    pose_axis_angle = art.math.rotation_matrix_to_axis_angle(pose_mat).view(-1, 24, 3)
    trans[:, 0] += x_offset
    trans = align_to_floor(pose_axis_angle, trans, smpl_layer, mode=floor_mode)

    with torch.no_grad():
        out = smpl_layer.bm(
            body_pose=pose_axis_angle[:, 1:].reshape(pose_axis_angle.shape[0], -1),
            global_orient=pose_axis_angle[:, 0],
            transl=trans,
        )
    verts = out.vertices.detach().cpu()[:, IMU_VERTEX_MASK].numpy().astype(np.float32)
    return Spheres(
        verts,
        radius=radius,
        color=color,
        name=name,
        cast_shadow=False,
    )


class FiveModelViewer(Viewer):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.playback_fps = PLAYBACK_FPS
        self.scene.fps = PLAYBACK_FPS
        self.scene.floor.enabled = True

        self.manifest = load_manifest()
        self.data = safe_torch_load(TEST_PT, map_location="cpu")
        self.index = find_index(self.manifest, args.subject, args.walk, args.index)
        self.max_index = len(self.manifest) - 1

        self.device = choose_device(args.device)
        self.transpose_net = TransPoseNet().to(self.device).eval()
        self.smpl_layer = SMPLLayer(model_type="smpl", gender="neutral", device="cpu")

        self.current_nodes = []
        self.load_sequence(self.index, first=True)

    def _clear_nodes(self):
        for node in self.current_nodes:
            try:
                self.scene.remove(node)
            except Exception:
                pass
        self.current_nodes = []

    def _x_offsets(self):
        center = (len(ORDER) - 1) / 2.0
        return {name: (i - center) * self.args.spacing for i, name in enumerate(ORDER)}

    def _common_length(self, outputs):
        lengths = []
        for out in outputs.values():
            lengths.append(int(out["pose_pred"].shape[0]))
        lengths.append(int(outputs["TransPose"]["pose_gt"].shape[0]))
        n = min(lengths)
        if self.args.max_frames > 0:
            n = min(n, self.args.max_frames)
        return n

    def _prepare_translation(self, pose, tran, gt_tran=None):
        if not self.args.use_trans:
            return zero_translation_like(pose)
        if tran is None:
            return zero_translation_like(pose)
        tran = tran.clone()
        if self.args.gt_height and gt_tran is not None and len(gt_tran) == len(tran):
            tran[:, 1] = gt_tran[:, 1]
        return tran

    def load_sequence(self, index, first=False):
        self.index = max(0, min(index, self.max_index))
        item = get_manifest_item(self.manifest, self.index)
        label = make_label(self.manifest, self.index)

        if not first:
            self._clear_nodes()

        with torch.no_grad():
            tp = run_transpose(
                self.data,
                self.index,
                self.device,
                max_frames=-1,
                net=self.transpose_net,
            )
        outputs = {"TransPose": tp}
        for name, path in PREDICTION_FILES.items():
            outputs[name] = load_prediction_file(path, self.index)

        n = self._common_length(outputs)
        pose_gt = _slice_frames(tp["pose_gt"], n)
        tran_gt = _slice_frames(tp["tran_gt"], n)

        offsets = self._x_offsets()
        nodes = []

        gt_tran_view = self._prepare_translation(pose_gt, tran_gt, tran_gt)
        nodes.append(
            make_smpl_sequence(
                "GT",
                pose_gt,
                gt_tran_view,
                self.smpl_layer,
                mesh_color("GT", self.args.mesh_alpha),
                x_offset=offsets["GT"],
                use_trans=True,
                floor_mode=self.args.floor_mode,
                path_mode=self.args.path_mode,
            )
        )
        if self.args.show_imu:
            nodes.append(
                make_imu_marker_sequence(
                    "GT IMUs",
                    pose_gt,
                    gt_tran_view,
                    self.smpl_layer,
                    x_offset=offsets["GT"],
                    floor_mode=self.args.floor_mode,
                    path_mode=self.args.path_mode,
                    radius=self.args.imu_radius,
                )
            )

        for name in ["TransPose", "PIP raw", "DynaIP", "PNP", "TIP"]:
            pose = _slice_frames(outputs[name]["pose_pred"], n)
            tran = _slice_frames(outputs[name]["tran_pred"], n) if outputs[name]["tran_pred"] is not None else None
            tran_view = self._prepare_translation(pose, tran, tran_gt)
            nodes.append(
                make_smpl_sequence(
                    name,
                    pose,
                    tran_view,
                    self.smpl_layer,
                    mesh_color(name, self.args.mesh_alpha),
                    x_offset=offsets[name],
                    use_trans=True,
                    floor_mode=self.args.floor_mode,
                    path_mode=self.args.path_mode,
                )
            )
            if self.args.show_imu:
                nodes.append(
                    make_imu_marker_sequence(
                        f"{name} IMUs",
                        pose,
                        tran_view,
                        self.smpl_layer,
                        x_offset=offsets[name],
                        floor_mode=self.args.floor_mode,
                        path_mode=self.args.path_mode,
                        radius=self.args.imu_radius,
                    )
                )

        for node in nodes:
            self.scene.add(node)
        self.current_nodes = nodes
        self.scene.camera.target = np.array([0.0, 0.8, 0.0])

        print("=" * 72)
        print(f"Loaded index {self.index}: {label}")
        print(f"Frames: {n} | fps: {PLAYBACK_FPS} | translation: {'raw/gt-height' if self.args.use_trans else 'root-aligned in-place'}")
        print("Left to right: " + " | ".join(ORDER))
        print(f"Mesh alpha: {self.args.mesh_alpha:.2f}")
        print("IMU markers: " + ("on (black dots: L/R forearm, L/R shank, head, pelvis)" if self.args.show_imu else "off"))
        print("Keys: N/Right next, P/Left previous")

    def key_event(self, key, action, modifiers):
        # Keep viewer defaults, then add simple browsing keys.
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS:
            if key == keys.N:
                self.load_sequence((self.index + 1) % (self.max_index + 1))
                return
            if key == keys.P:
                self.load_sequence((self.index - 1) % (self.max_index + 1))
                return
        super().key_event(key, action, modifiers)


def smoke_test(args):
    manifest = load_manifest()
    data = safe_torch_load(TEST_PT, map_location="cpu")
    idx = find_index(manifest, args.subject, args.walk, args.index)
    device = choose_device(args.device)
    net = TransPoseNet().to(device).eval()
    with torch.no_grad():
        tp = run_transpose(data, idx, device, max_frames=args.max_frames if args.max_frames > 0 else 20, net=net)
    print(f"TransPose OK: {tp['pose_pred'].shape}")
    for name, path in PREDICTION_FILES.items():
        out = load_prediction_file(path, idx)
        print(f"{name} OK: pose={tuple(out['pose_pred'].shape)}, tran={None if out['tran_pred'] is None else tuple(out['tran_pred'].shape)}")
    print("Smoke test complete.")


def parse_args():
    ap = argparse.ArgumentParser(description="View GT + five CARE-PD model reconstructions side by side.")
    ap.add_argument("--index", type=int, default=0, help="CARE-PD manifest index.")
    ap.add_argument("--subject", type=str, default=None, help="Optional subject id filter, e.g. SUB01.")
    ap.add_argument("--walk", type=str, default=None, help="Optional walk id filter, e.g. SUB01_off_walk_1.")
    ap.add_argument("--max-frames", type=int, default=-1, help="Limit frames for faster loading/debugging.")
    ap.add_argument("--device", type=str, default="cpu", help="Device for TransPose inference: cpu/cuda/mps if supported.")
    ap.add_argument("--spacing", type=float, default=1.45, help="Horizontal spacing between bodies.")
    ap.add_argument("--mesh-alpha", type=float, default=0.55, help="Body mesh opacity. Lower values make SMPL meshes more transparent.")
    ap.add_argument("--use-trans", action="store_true", help="Use each model's translation when available. Default is in-place/root-aligned.")
    ap.add_argument("--gt-height", action="store_true", default=True, help="When --use-trans, use GT vertical height for all predictions.")
    ap.add_argument("--no-gt-height", action="store_false", dest="gt_height")
    ap.add_argument("--floor-mode", choices=["first", "min", "none"], default="first")
    ap.add_argument("--path-mode", choices=["align-forward", "center", "raw"], default="center")
    ap.add_argument("--hide-imu", action="store_false", dest="show_imu", help="Hide the six synthetic-IMU sensor markers.")
    ap.add_argument("--imu-radius", type=float, default=0.035, help="Radius of IMU marker spheres in meters.")
    ap.set_defaults(show_imu=True)
    ap.add_argument("--smoke", action="store_true", help="Load all files and exit without opening the GUI.")
    return ap.parse_args()


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    if args.smoke:
        smoke_test(args)
        return
    viewer = FiveModelViewer(args)
    viewer.run()


if __name__ == "__main__":
    main()
