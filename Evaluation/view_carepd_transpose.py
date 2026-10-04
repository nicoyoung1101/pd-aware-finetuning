"""
view_carepd_transpose.py

Run TransPose on one CARE-PD sequence and visualize the reconstructed body.

Default view:
  - left/body 1: CARE-PD SMPL ground truth
  - right/body 2: TransPose prediction from synthetic IMU

Examples:
  python Evaluation/view_carepd_transpose.py --index 0
  python Evaluation/view_carepd_transpose.py --index 0 --no-view
  python Evaluation/view_carepd_transpose.py --subject SUB01 --walk SUB01_off_walk_1
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
import imgui


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRANSPOSE_ROOT = os.path.join(PROJECT_ROOT, "TransPose")
if TRANSPOSE_ROOT not in sys.path:
    sys.path.insert(0, TRANSPOSE_ROOT)

import articulate as art
from config import paths
from net import TransPoseNet
from utils import normalize_and_concat

from aitviewer.configuration import CONFIG as C
C.update_conf({"smplx_models": os.path.join(PROJECT_ROOT, "Models", "smpl_models")})
from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer


DATASET_DIR = os.path.join(TRANSPOSE_ROOT, "data", "dataset_work", "CAREPD_BMCLab")
TEST_PT = os.path.join(DATASET_DIR, "test.pt")
MANIFEST_JSON = os.path.join(DATASET_DIR, "manifest.json")
RESULT_DIR = os.path.join(TRANSPOSE_ROOT, "data", "results", "CAREPD_BMCLab_visual")


PLAYBACK_FPS = 60
ORTHOGRAPHIC = False
FLOOR_SIDE_LENGTH = 400.0
FLOOR_N_TILES = 800
PATIENT_INFO_RIGHT_MARGIN = 24
PATIENT_INFO_WIDTH = 430
PATIENT_INFO_Y = 120
UPDRS_LABEL = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}
UPDRS_GUI_COLOR = {
    0: (0.20, 0.90, 0.30, 1.0),
    1: (1.00, 0.85, 0.20, 1.0),
    2: (1.00, 0.55, 0.10, 1.0),
    3: (1.00, 0.20, 0.20, 1.0),
    None: (0.65, 0.65, 0.65, 1.0),
}


class TC:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    ORANGE = "\033[33m"
    RED = "\033[91m"
    CYAN = "\033[96m"


UPDRS_TERM_COLOR = {0: TC.GREEN, 1: TC.YELLOW, 2: TC.ORANGE, 3: TC.RED, None: TC.DIM}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Sequence index in test.pt")
    parser.add_argument("--subject", default=None, help="Optional subject id, e.g. SUB01")
    parser.add_argument("--walk", default=None, help="Optional walk id, e.g. SUB01_off_walk_1")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Inference device")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap for faster preview")
    parser.add_argument("--pred-only", action="store_true", help="Visualize only prediction")
    parser.add_argument("--no-view", action="store_true", help="Save tensors only; do not open viewer")
    parser.add_argument("--distance", type=float, default=1.0, help="Distance between GT and prediction in viewer")
    parser.add_argument("--no-trans", action="store_true", help="Zero out root translation in visualization")
    parser.add_argument("--no-fix-floor", action="store_true", help="Do not lift visualized bodies to the floor")
    parser.add_argument(
        "--path-mode",
        choices=["align-forward", "center", "raw"],
        default="align-forward",
        help="Visual path mode: rotate walking direction to +Z, show in-place, or keep raw trajectory",
    )
    parser.add_argument(
        "--translation-mode",
        choices=["gt-height", "flat-height", "raw"],
        default="gt-height",
        help="Prediction height display: use GT vertical motion, keep flat height, or raw TransPose translation",
    )
    parser.add_argument(
        "--floor-mode",
        choices=["first", "sequence", "none"],
        default="first",
        help="Floor alignment for visualization: first frame, whole sequence, or raw/no alignment",
    )
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


def load_manifest():
    if not os.path.exists(MANIFEST_JSON):
        return None
    with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def find_index(manifest, subject, walk, fallback_index):
    if subject is None and walk is None:
        return fallback_index
    if manifest is None:
        raise ValueError("Cannot search by subject/walk because manifest.json was not found.")

    for i, item in enumerate(manifest):
        subject_ok = subject is None or str(item.get("subject_id")) == subject
        walk_ok = walk is None or str(item.get("walk_id")) == walk
        if subject_ok and walk_ok:
            return i

    raise ValueError(f"No sequence matched subject={subject!r}, walk={walk!r}")


def get_manifest_item(manifest, index):
    if manifest is None or index < 0 or index >= len(manifest):
        return {}
    return manifest[index]


def make_label(manifest, index):
    if manifest is None or index >= len(manifest):
        return f"seq_{index:04d}"
    item = get_manifest_item(manifest, index)
    parts = [
        str(item.get("subject_id", "SUB")),
        str(item.get("walk_id", "walk")),
        f"U{item.get('UPDRS_GAIT', 'NA')}",
        f"med{item.get('medication', 'NA')}",
    ]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "__".join(parts))


def make_seq_name(manifest, index):
    item = get_manifest_item(manifest, index)
    subj = item.get("subject_id", "SUB")
    walk = item.get("walk_id", "walk")
    updrs = item.get("UPDRS_GAIT")
    med = item.get("medication")
    other = item.get("other")
    parts = [f"[{index + 1}]", f"{subj}/{walk}"]
    parts.append(f"U={updrs}" if updrs is not None else "U=?")
    if med:
        parts.append(f"med={med}")
    if other:
        parts.append(f"({other})")
    return "  ".join(parts)


def make_output_path(label, max_frames):
    suffix = f"__first{max_frames}" if max_frames > 0 else ""
    return os.path.join(RESULT_DIR, f"{label}{suffix}.pt")


def print_info_box(info):
    updrs = info["updrs"]
    color = UPDRS_TERM_COLOR.get(updrs, TC.DIM)
    updrs_label = UPDRS_LABEL.get(updrs, "unknown")
    bar = "-" * 72
    print()
    print(f"  +{bar}+")
    print(f"  | {TC.BOLD}{TC.CYAN}[{info['idx'] + 1:3d} / {info['total']:3d}]{TC.RESET}  dataset = {TC.BOLD}CAREPD_BMCLab / TransPose{TC.RESET}")
    print("  |")
    print(f"  |   subject          : {info['subject']}")
    print(f"  |   walk             : {info['walk']}")
    print(f"  |   fps              : {info['fps']}   ({info['frames']} frames, {info['duration']:.2f}s)")
    print(f"  |   {TC.BOLD}UPDRS_GAIT       : {color}{updrs}  ({updrs_label}){TC.RESET}")
    print(f"  |   medication       : {info['medication']}")
    print(f"  |   other            : {info['other']}")
    print(f"  |   path mode        : {info['path_mode']}")
    print(f"  |   pred height mode : {info['translation_mode']}")
    print(f"  +{bar}+")
    print(f"  {TC.DIM}N=next  P=prev  M=path mode  T=pred height  Space=play/pause  F=fit view{TC.RESET}\n")


@torch.no_grad()
def run_transpose(data, index, device, max_frames, net=None):
    if index < 0 or index >= len(data["acc"]):
        raise IndexError(f"index={index} is out of range for {len(data['acc'])} sequences")

    acc = data["acc"][index].float()
    ori = data["ori"][index].float()
    pose_gt_axis_angle = data["pose"][index].float()
    tran_gt = data["tran"][index].float()

    if max_frames > 0:
        acc = acc[:max_frames]
        ori = ori[:max_frames]
        pose_gt_axis_angle = pose_gt_axis_angle[:max_frames]
        tran_gt = tran_gt[:max_frames]

    x = normalize_and_concat(acc, ori).to(device)
    if net is None:
        net = TransPoseNet().to(device)
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
        "network_input": x.cpu(),
    }


def align_to_floor(pose_axis_angle, trans, smpl_layer, up_axis=1, mode="first"):
    if mode == "none":
        return trans.clone()

    with torch.no_grad():
        out = smpl_layer.bm(
            body_pose=pose_axis_angle[:, 1:].reshape(pose_axis_angle.shape[0], -1),
            global_orient=pose_axis_angle[:, 0],
            transl=trans,
        )

        if mode == "first":
            min_up = float(out.vertices[0, :, up_axis].min().item())
        elif mode == "sequence":
            min_up = float(out.vertices[..., up_axis].min().item())
        else:
            raise ValueError(f"Unknown floor mode: {mode}")

    trans = trans.clone()
    trans[:, up_axis] -= min_up
    return trans


def make_pred_visual_translation(out, mode):
    trans = out["tran_pred"].clone().float()
    if mode == "gt-height":
        trans[:, 1] = out["tran_gt"][:, 1].float()
    elif mode == "flat-height":
        trans[:, 1] = trans[0, 1]
    elif mode == "raw":
        pass
    else:
        raise ValueError(f"Unknown translation mode: {mode}")
    return trans


def apply_path_mode(pose_mat, trans, mode):
    pose_vis = pose_mat.clone()
    trans_vis = trans.clone().float()
    trans_vis = trans_vis - trans_vis[:1]

    if mode == "center":
        trans_vis[:, 0] = 0
        trans_vis[:, 2] = 0
        return pose_vis, trans_vis

    if mode == "raw":
        return pose_vis, trans_vis

    if mode != "align-forward":
        raise ValueError(f"Unknown path mode: {mode}")

    delta = trans_vis[-1] - trans_vis[0]
    dx = float(delta[0].item())
    dz = float(delta[2].item())
    if (dx * dx + dz * dz) < 1e-8:
        return pose_vis, trans_vis

    angle = np.arctan2(dx, dz)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rot = torch.tensor(
        [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]],
        dtype=trans_vis.dtype,
        device=trans_vis.device,
    )
    trans_vis = trans_vis @ rot.T
    pose_vis[:, 0] = rot.unsqueeze(0).matmul(pose_vis[:, 0])
    return pose_vis, trans_vis


def make_smpl_sequence(
    name,
    pose_mat,
    tran,
    smpl_layer,
    color,
    x_offset=0.0,
    use_trans=True,
    floor_mode="first",
    path_mode="align-forward",
):
    trans = tran.clone().float()
    if not use_trans:
        trans.zero_()
    pose_mat, trans = apply_path_mode(pose_mat, trans, path_mode)
    pose_axis_angle = art.math.rotation_matrix_to_axis_angle(pose_mat).view(-1, 24, 3)
    trans[:, 0] += x_offset
    trans = align_to_floor(pose_axis_angle, trans, smpl_layer, mode=floor_mode)

    return SMPLSequence(
        poses_body=pose_axis_angle[:, 1:].reshape(pose_axis_angle.shape[0], -1),
        poses_root=pose_axis_angle[:, 0],
        trans=trans,
        betas=None,
        smpl_layer=smpl_layer,
        name=name,
        color=color,
        z_up=False,
    )


class TransPoseBrowser(Viewer):
    title = "CARE-PD TransPose Viewer"

    def __init__(self, data, manifest, net, device, args, start_index, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.manifest = manifest
        self.net = net
        self.device = device
        self.max_frames = args.max_frames
        self.pred_only = args.pred_only
        self.distance = args.distance
        self.use_trans = not args.no_trans
        self.path_mode = args.path_mode
        self.floor_mode = "none" if args.no_fix_floor else args.floor_mode
        self.translation_mode = args.translation_mode
        self.current_idx = start_index
        self.current_nodes = []
        self.current_info = {}
        self.current_out = None
        self.show_patient_info = True
        self.smpl_layer = SMPLLayer(model_type="smpl", gender="neutral", device="cpu")
        self.playback_fps = PLAYBACK_FPS
        self.gui_controls["patient_info"] = self.gui_patient_info
        self._kill_shadows()
        self._expand_floor()
        self.load_sequence(self.current_idx, is_first=True)
        if ORTHOGRAPHIC:
            self._try_set_orthographic(True)

    def _kill_shadows(self):
        try:
            for light in self.scene.lights:
                for attr in ["shadow_enabled", "casts_shadow", "shadows"]:
                    if hasattr(light, attr):
                        try:
                            setattr(light, attr, False)
                        except Exception:
                            pass
        except Exception:
            pass

    def _expand_floor(self):
        floor = getattr(self.scene, "floor", None)
        if floor is None:
            return
        try:
            floor.side_length = FLOOR_SIDE_LENGTH
            floor.n_tiles = FLOOR_N_TILES
            if floor.plane == "xz":
                v1 = np.array([1, 0, 0], dtype=np.float32)
                v2 = np.array([0, 0, 1], dtype=np.float32)
            elif floor.plane == "xy":
                v1 = np.array([0, 1, 0], dtype=np.float32)
                v2 = np.array([1, 0, 0], dtype=np.float32)
            else:
                v1 = np.array([0, 1, 0], dtype=np.float32)
                v2 = np.array([0, 0, 1], dtype=np.float32)
            floor.vertices, floor.normals, floor.uvs = floor._get_renderable_data(v1, v2, FLOOR_SIDE_LENGTH)
        except Exception as e:
            print(f"  (floor expand warn: {e})")

    def _try_set_orthographic(self, ortho):
        cam = self.scene.camera
        for attr in ["is_ortho", "is_orthographic", "orthographic"]:
            if hasattr(cam, attr):
                try:
                    setattr(cam, attr, ortho)
                    return
                except Exception:
                    pass

    def _fit_camera(self, node):
        try:
            self.center_view_on_node(node)
        except Exception:
            pass

    def _update_title(self, label):
        try:
            self.window.title = f"CARE-PD TransPose: {label}"
        except Exception:
            pass

    def _make_info(self, idx, out):
        item = get_manifest_item(self.manifest, idx)
        frames = int(out["pose_pred"].shape[0])
        fps = float(item.get("target_fps", PLAYBACK_FPS))
        updrs = item.get("UPDRS_GAIT")
        return {
            "idx": idx,
            "total": len(self.data["acc"]),
            "dataset": "CAREPD_BMCLab / TransPose",
            "subject": item.get("subject_id", "-"),
            "walk": item.get("walk_id", "-"),
            "fps": fps,
            "frames": frames,
            "duration": frames / fps if fps else 0.0,
            "updrs": updrs,
            "updrs_label": UPDRS_LABEL.get(updrs, "unknown"),
            "medication": item.get("medication") or "-",
            "other": item.get("other") or "-",
            "path_mode": self.path_mode,
            "translation_mode": self.translation_mode,
            "floor_mode": self.floor_mode,
        }

    def _remove_current_nodes(self):
        for node in self.current_nodes:
            try:
                self.scene.remove(node)
            except Exception:
                pass
        self.current_nodes = []

    def _build_nodes(self, out):
        nodes = []
        if not self.pred_only:
            nodes.append(make_smpl_sequence(
                "GT CARE-PD",
                out["pose_gt"],
                out["tran_gt"],
                self.smpl_layer,
                color=(0.35, 0.62, 1.0, 1.0),
                x_offset=-self.distance / 2,
                use_trans=self.use_trans,
                floor_mode=self.floor_mode,
                path_mode=self.path_mode,
            ))
        pred_tran = make_pred_visual_translation(out, self.translation_mode)
        nodes.append(make_smpl_sequence(
            f"TransPose Prediction ({self.translation_mode})",
            out["pose_pred"],
            pred_tran,
            self.smpl_layer,
            color=(1.0, 0.48, 0.20, 1.0),
            x_offset=self.distance / 2 if not self.pred_only else 0.0,
            use_trans=self.use_trans,
            floor_mode=self.floor_mode,
            path_mode=self.path_mode,
        ))
        return nodes

    def _save_output(self, idx, label, out):
        os.makedirs(RESULT_DIR, exist_ok=True)
        out_path = make_output_path(label, self.max_frames)
        torch.save({
            "index": idx,
            "label": label,
            "manifest": get_manifest_item(self.manifest, idx),
            "path_mode": self.path_mode,
            "translation_mode": self.translation_mode,
            "floor_mode": self.floor_mode,
            **out,
        }, out_path)
        return out_path

    def load_sequence(self, idx, is_first=False):
        idx = idx % len(self.data["acc"])
        label = make_label(self.manifest, idx)
        scene_name = make_seq_name(self.manifest, idx)
        print(f"Running TransPose: index={idx}, label={label}")

        out = run_transpose(self.data, idx, self.device, self.max_frames, net=self.net)
        out_path = self._save_output(idx, label, out)
        nodes = self._build_nodes(out)

        self._remove_current_nodes()
        for node in nodes:
            self.scene.add(node)

        self.current_idx = idx
        self.current_nodes = nodes
        self.current_out = out
        self.current_info = self._make_info(idx, out)
        try:
            self.scene.current_frame_id = 0
        except Exception:
            pass
        if is_first and nodes:
            self._fit_camera(nodes[0])
        self._update_title(scene_name)
        print(f"Saved   : {out_path}")
        print_info_box(self.current_info)

    def toggle_translation_mode(self):
        modes = ["gt-height", "flat-height", "raw"]
        self.translation_mode = modes[(modes.index(self.translation_mode) + 1) % len(modes)]
        self.load_sequence(self.current_idx)

    def toggle_path_mode(self):
        modes = ["align-forward", "center", "raw"]
        self.path_mode = modes[(modes.index(self.path_mode) + 1) % len(modes)]
        self.load_sequence(self.current_idx)

    def gui_patient_info(self):
        if not self.show_patient_info or not self.current_info:
            return

        info = self.current_info
        flags = (
            imgui.WINDOW_ALWAYS_AUTO_RESIZE
            | imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_SAVED_SETTINGS
        )
        try:
            display_w = imgui.get_io().display_size[0]
            info_x = max(520, display_w - PATIENT_INFO_WIDTH - PATIENT_INFO_RIGHT_MARGIN)
        except Exception:
            info_x = 520
        imgui.set_next_window_position(info_x, PATIENT_INFO_Y, imgui.FIRST_USE_EVER)

        expanded, self.show_patient_info = imgui.begin("Patient Info", self.show_patient_info, flags=flags)
        if expanded:
            imgui.text(f"[{info['idx'] + 1:3d} / {info['total']:3d}]  dataset = {info['dataset']}")
            imgui.separator()
            imgui.text(f"subject     : {info['subject']}")
            imgui.text(f"walk        : {info['walk']}")
            imgui.text(f"fps         : {info['fps']:.1f}   ({info['frames']} frames, {info['duration']:.2f}s)")

            updrs_color = UPDRS_GUI_COLOR.get(info["updrs"], UPDRS_GUI_COLOR[None])
            imgui.text("UPDRS_GAIT  :")
            imgui.same_line()
            imgui.text_colored(f"{info['updrs']}  ({info['updrs_label']})", *updrs_color)

            imgui.text(f"medication  : {info['medication']}")
            imgui.text(f"other       : {info['other']}")
            imgui.separator()
            imgui.text("blue        : GT CARE-PD")
            imgui.text("orange      : TransPose")
            imgui.text(f"path mode   : {info['path_mode']}")
            imgui.text(f"pred height : {info['translation_mode']}")
            imgui.text(f"floor       : {info['floor_mode']}")
        imgui.end()

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS:
            if key == keys.N:
                self.load_sequence(self.current_idx + 1)
                return
            if key == keys.P:
                self.load_sequence(self.current_idx - 1)
                return
            if key == keys.T:
                self.toggle_translation_mode()
                return
            if key == keys.M:
                self.toggle_path_mode()
                return
        super().key_event(key, action, modifiers)


def main():
    args = parse_args()
    if not os.path.exists(TEST_PT):
        raise FileNotFoundError(f"Missing {TEST_PT}. Run Evaluation/prepare_carepd_transpose.py first.")

    manifest = load_manifest()
    index = find_index(manifest, args.subject, args.walk, args.index)
    label = make_label(manifest, index)
    device = choose_device(args.device)

    print(f"Dataset : {TEST_PT}")
    print(f"Sequence: index={index}, label={label}")
    print(f"Device  : {device}")

    data = safe_torch_load(TEST_PT, map_location="cpu")
    net = TransPoseNet().to(device)

    if not args.no_view:
        print("Opening aitviewer browser")
        TransPoseBrowser(data, manifest, net, device, args, index).run()
        return

    out = run_transpose(data, index, device, args.max_frames, net=net)

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = make_output_path(label, args.max_frames)
    torch.save({
        "index": index,
        "label": label,
        "manifest": manifest[index] if manifest and index < len(manifest) else None,
        "path_mode": args.path_mode,
        "translation_mode": args.translation_mode,
        "floor_mode": "none" if args.no_fix_floor else args.floor_mode,
        **out,
    }, out_path)

    print(f"Frames  : {out['pose_pred'].shape[0]}")
    print(f"Saved   : {out_path}")


if __name__ == "__main__":
    main()
