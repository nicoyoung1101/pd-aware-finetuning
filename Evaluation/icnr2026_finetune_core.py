"""Shared CARE-PD/TransPose fine-tuning utilities for the ICNR 2026 study.

The module keeps the training/evaluation contract explicit:

* one sample is one complete, variable-length walk;
* subject-balanced sampling is deterministic per epoch;
* C5-noHead uses raw-space head <- pelvis copying before TransPose
  normalization, producing a constant normalized head channel (0, I);
* only the pose_s3 capacity ladder can be made trainable;
* clinical losses operate on differentiable SMPL forward kinematics;
* acceleration noise is added in raw m/s^2 before normalization.

It intentionally contains no command-line entry point. Training and sanity
checks import these functions so both paths use exactly the same code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

for _name, _type in [
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
    ("unicode", str),
    ("long", int),
]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            getattr(np, _name)
        except AttributeError:
            setattr(np, _name, _type)

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSPOSE_ROOT = PROJECT_ROOT / "TransPose"
if str(TRANSPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSPOSE_ROOT))

import articulate as art  # noqa: E402
from config import joint_set, paths  # noqa: E402
from net import TransPoseNet  # noqa: E402
from utils import normalize_and_concat  # noqa: E402


DATASET_PATH = TRANSPOSE_ROOT / "data/dataset_work/CAREPD_BMCLab/test.pt"
MANIFEST_PATH = TRANSPOSE_ROOT / "data/dataset_work/CAREPD_BMCLab/manifest.json"
SPLIT_PATH = (
    PROJECT_ROOT
    / "ModelComparison/data/splits/CAREPD_BMCLab_ICNR2026_nested_cv/nested_splits.json"
)
REFERENCE_METRICS_PATH = (
    PROJECT_ROOT
    / "ModelComparison/data/results/CAREPD_BMCLab_extended_metrics_with_TIP/"
    "multimodel_extended_per_walk.csv"
)
PRETRAINED_WEIGHTS_PATH = Path(paths.weights_file)

SENSOR_NAMES = [
    "LeftForeArm",
    "RightForeArm",
    "LeftLowerLeg",
    "RightLowerLeg",
    "Head",
    "Root",
]
HEAD_INDEX = SENSOR_NAMES.index("Head")
ROOT_INDEX = SENSOR_NAMES.index("Root")
PREDICTED_JOINTS = tuple(joint_set.reduced)

CONDITIONS = {
    "P6": {"sensor_config": "Full6", "clinical": False},
    "C6": {"sensor_config": "Full6", "clinical": True},
    "P5": {"sensor_config": "C5NoHead", "clinical": False},
    "C5": {"sensor_config": "C5NoHead", "clinical": True},
}

CAPACITY_PARAMETER_COUNTS = {
    "S1": 23_130,
    "S1.5": 41_306,
    "S2": 700_762,
}

# Frozen before formal outer-fold training. The two clinical terms jointly
# contribute approximately one pose-gradient unit at initialization.
FROZEN_VELOCITY_WEIGHT = 0.0583
FROZEN_SWING_WEIGHT = 0.0352
FROZEN_POSE_WEIGHT = 1.0
FROZEN_JOINT_POSITION_WEIGHT = 1.0
# Calibrated once on fold-0 inner-train, C6/S1, with lambda_SP=0. At the
# clinical-selected pilot checkpoint this makes the summed-L2-SP restoring
# gradient 10% of the median complete-optimizer-group task-gradient RMS.
FROZEN_L2SP_WEIGHT = 0.0014783522259721978
JOINT_POSITION_HUBER_BETA_M = 0.02
SWING_TOP_BOTTOM_FRACTION = 0.05
SWING_TOP_BOTTOM_MIN_FRAMES = 3


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda:0")
    if name != "cpu":
        raise ValueError(f"Unknown device: {name}")
    return torch.device("cpu")


def load_dataset_and_manifest(
    dataset_path: str | Path = DATASET_PATH,
    manifest_path: str | Path = MANIFEST_PATH,
):
    data = safe_torch_load(dataset_path, map_location="cpu")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    keys = ("acc", "ori", "pose", "tran")
    lengths = {key: len(data[key]) for key in keys}
    if len(set(lengths.values())) != 1 or lengths["acc"] != len(manifest):
        raise ValueError(f"Dataset/manifest length mismatch: {lengths}, manifest={len(manifest)}")
    return data, manifest


def load_nested_splits(path: str | Path = SPLIT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fold_subjects(split_payload: dict, fold: int, role: str) -> list[str]:
    if role not in {"inner_train", "inner_val", "outer_test"}:
        raise ValueError(f"Unknown split role: {role}")
    try:
        item = next(x for x in split_payload["folds"] if int(x["outer_fold"]) == int(fold))
    except StopIteration as exc:
        raise ValueError(f"Outer fold {fold} is absent") from exc
    return list(item[f"{role}_subjects"])


def indices_for_subjects(manifest: list[dict], subjects: Iterable[str]) -> list[int]:
    wanted = set(subjects)
    return [i for i, item in enumerate(manifest) if str(item["subject_id"]) in wanted]


def condition_spec(name: str) -> dict:
    if name not in CONDITIONS:
        raise ValueError(f"Unknown condition {name}; choose from {sorted(CONDITIONS)}")
    return dict(CONDITIONS[name])


@dataclass(frozen=True)
class NoiseSpec:
    """Raw-IMU augmentation; pelvis/other multipliers enable separate curves."""

    acc_white_std_mps2: float = 0.0
    acc_bias_std_mps2: float = 0.0
    ori_jitter_std_deg: float = 0.0
    ori_mount_std_deg: float = 0.0
    yaw_drift_std_deg_per_min: float = 0.0
    pelvis_multiplier: float = 1.0
    other_multiplier: float = 1.0
    fps: float = 60.0

    def active(self) -> bool:
        return any(
            value > 0
            for value in (
                self.acc_white_std_mps2,
                self.acc_bias_std_mps2,
                self.ori_jitter_std_deg,
                self.ori_mount_std_deg,
                self.yaw_drift_std_deg_per_min,
            )
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _sensor_scales(spec: NoiseSpec, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    scales = torch.full((6,), float(spec.other_multiplier), dtype=dtype, device=device)
    scales[ROOT_INDEX] = float(spec.pelvis_multiplier)
    return scales


def _random_rotations(
    shape: tuple[int, ...],
    std_deg: float,
    scales: torch.Tensor,
    generator: torch.Generator,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if std_deg <= 0:
        eye = torch.eye(3, dtype=dtype, device=device)
        return eye.expand(*shape, 3, 3).clone()
    rotvec = torch.randn(*shape, 3, generator=generator, dtype=dtype, device=device)
    scale_shape = [1] * len(shape) + [1]
    scale_shape[-2] = 6
    rotvec = rotvec * math.radians(std_deg) * scales.view(*scale_shape)
    return art.math.axis_angle_to_rotation_matrix(rotvec).view(*shape, 3, 3)


def augment_raw_imu(
    acc: torch.Tensor,
    ori: torch.Tensor,
    spec: NoiseSpec,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply noise before root-relative normalization; units remain m/s^2 and SO(3)."""
    if not spec.active():
        return acc, ori
    acc = acc.clone()
    ori = ori.clone()
    scales = _sensor_scales(spec, dtype=acc.dtype, device=acc.device)

    if spec.acc_bias_std_mps2 > 0:
        bias = torch.randn(1, 6, 3, generator=generator, dtype=acc.dtype, device=acc.device)
        acc = acc + bias * spec.acc_bias_std_mps2 * scales.view(1, 6, 1)
    if spec.acc_white_std_mps2 > 0:
        white = torch.randn(acc.shape, generator=generator, dtype=acc.dtype, device=acc.device)
        acc = acc + white * spec.acc_white_std_mps2 * scales.view(1, 6, 1)

    if spec.ori_mount_std_deg > 0:
        mount = _random_rotations(
            (1, 6), spec.ori_mount_std_deg, scales, generator,
            dtype=ori.dtype, device=ori.device,
        )
        ori = ori @ mount
    if spec.ori_jitter_std_deg > 0:
        jitter = _random_rotations(
            (ori.shape[0], 6), spec.ori_jitter_std_deg, scales, generator,
            dtype=ori.dtype, device=ori.device,
        )
        ori = jitter @ ori
    if spec.yaw_drift_std_deg_per_min > 0:
        slopes = torch.randn(6, generator=generator, dtype=ori.dtype, device=ori.device)
        slopes = slopes * math.radians(spec.yaw_drift_std_deg_per_min) * scales
        time_min = torch.arange(ori.shape[0], dtype=ori.dtype, device=ori.device) / spec.fps / 60.0
        rotvec = torch.zeros(ori.shape[0], 6, 3, dtype=ori.dtype, device=ori.device)
        rotvec[:, :, 1] = time_min[:, None] * slopes[None, :]
        drift = art.math.axis_angle_to_rotation_matrix(rotvec).view(ori.shape[0], 6, 3, 3)
        ori = drift @ ori
    return acc, ori


def impute_sensor_config(
    acc: torch.Tensor,
    ori: torch.Tensor,
    sensor_config: str,
    c5_policy: str = "root_copy",
) -> tuple[torch.Tensor, torch.Tensor]:
    if sensor_config == "Full6":
        return acc, ori
    if sensor_config != "C5NoHead":
        raise ValueError(f"Unsupported training sensor config: {sensor_config}")
    if c5_policy not in {"root_copy", "raw_zero"}:
        raise ValueError("C5 policy must be root_copy (M1) or raw_zero (M0 diagnostic)")

    acc = acc.clone()
    ori = ori.clone()
    if c5_policy == "root_copy":
        acc[:, HEAD_INDEX] = acc[:, ROOT_INDEX]
        ori[:, HEAD_INDEX] = ori[:, ROOT_INDEX]
    else:
        acc[:, HEAD_INDEX] = 0.0
        ori[:, HEAD_INDEX] = torch.eye(3, dtype=ori.dtype, device=ori.device)
    return acc, ori


def prepare_imu_input(
    acc: torch.Tensor,
    ori: torch.Tensor,
    sensor_config: str,
    *,
    c5_policy: str = "root_copy",
    noise: NoiseSpec | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    acc = acc.float()
    ori = ori.float()
    if noise is not None and noise.active():
        if generator is None:
            raise ValueError("An explicit generator is required for stochastic augmentation")
        acc, ori = augment_raw_imu(acc, ori, noise, generator)
    acc, ori = impute_sensor_config(acc, ori, sensor_config, c5_policy)
    return normalize_and_concat(acc, ori)


def load_cycle_counts(path: str | Path = REFERENCE_METRICS_PATH) -> dict[int, float]:
    counts: dict[int, float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model_variant") != "TransPose_default":
                continue
            index = int(row["index"])
            cadence = float(row["gt_pose_cadence_pose_spm"])
            duration = float(row["duration_s"])
            counts[index] = cadence * duration / 120.0 if np.isfinite(cadence) else math.nan
    return counts


class WholeWalkDataset:
    """A lightweight index view over the in-memory CARE-PD tensors."""

    def __init__(
        self,
        data: dict,
        manifest: list[dict],
        indices: Iterable[int],
        cycle_counts: dict[int, float] | None = None,
    ):
        self.data = data
        self.manifest = manifest
        self.indices = list(indices)
        self.cycle_counts = cycle_counts or {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        index = self.indices[position]
        item = self.manifest[index]
        lengths = [len(self.data[key][index]) for key in ("acc", "ori", "pose", "tran")]
        if len(set(lengths)) != 1:
            raise ValueError(f"Sequence {index} has inconsistent lengths: {lengths}")
        cycles = self.cycle_counts.get(index, math.nan)
        return {
            "index": index,
            "subject": str(item["subject_id"]),
            "updrs": int(item["UPDRS_GAIT"]),
            "medication": str(item["medication"]),
            "cycles": float(cycles),
            "acc": self.data["acc"][index],
            "ori": self.data["ori"][index],
            "pose": self.data["pose"][index],
            "tran": self.data["tran"][index],
        }


class SubjectBalancedEpochSampler:
    """Exactly equal sample counts per subject while retaining whole walks."""

    def __init__(
        self,
        dataset: WholeWalkDataset,
        *,
        seed: int,
        samples_per_subject: int | None = None,
    ):
        positions: dict[str, list[int]] = defaultdict(list)
        for position, index in enumerate(dataset.indices):
            subject = str(dataset.manifest[index]["subject_id"])
            positions[subject].append(position)
        if not positions:
            raise ValueError("Cannot sample an empty dataset")
        self.positions = dict(sorted(positions.items()))
        self.seed = int(seed)
        if samples_per_subject is None:
            samples_per_subject = math.ceil(len(dataset) / len(self.positions))
        if samples_per_subject <= 0:
            raise ValueError("samples_per_subject must be positive")
        self.samples_per_subject = int(samples_per_subject)

    def _draw_subject(self, values: list[int], rng: np.random.Generator) -> list[int]:
        result: list[int] = []
        while len(result) < self.samples_per_subject:
            result.extend(rng.permutation(values).tolist())
        return result[: self.samples_per_subject]

    def epoch_positions(self, epoch: int) -> list[int]:
        rng = np.random.default_rng(self.seed + int(epoch))
        result = []
        for values in self.positions.values():
            result.extend(self._draw_subject(values, rng))
        return rng.permutation(result).tolist()

    def subject_counts(self, epoch: int) -> dict[str, int]:
        lookup = {
            position: subject for subject, values in self.positions.items() for position in values
        }
        counts: dict[str, int] = defaultdict(int)
        for position in self.epoch_positions(epoch):
            counts[lookup[position]] += 1
        return dict(sorted(counts.items()))

    def __iter__(self) -> Iterator[int]:
        return iter(self.epoch_positions(0))

    def __len__(self) -> int:
        return len(self.positions) * self.samples_per_subject


def load_pretrained_net(device: torch.device) -> TransPoseNet:
    return TransPoseNet().to(device)


def configure_capacity(net: TransPoseNet, capacity: str) -> list[torch.nn.Parameter]:
    if capacity not in CAPACITY_PARAMETER_COUNTS:
        raise ValueError(f"Unknown capacity {capacity}")
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    if capacity == "S1":
        modules = [net.pose_s3.linear2]
    elif capacity == "S1.5":
        modules = [net.pose_s3.linear1, net.pose_s3.linear2]
    else:
        modules = [net.pose_s3]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    trainable = [p for p in net.parameters() if p.requires_grad]
    count = sum(p.numel() for p in trainable)
    if count != CAPACITY_PARAMETER_COUNTS[capacity]:
        raise AssertionError(
            f"Capacity {capacity} expected {CAPACITY_PARAMETER_COUNTS[capacity]:,} parameters, got {count:,}"
        )
    return trainable


def set_finetune_mode(net: TransPoseNet, *, enable_pose_s3_dropout: bool) -> None:
    """Keep every frozen branch deterministic; optionally enable pose_s3 dropout."""
    net.eval()
    if enable_pose_s3_dropout:
        net.pose_s3.train()


def forward_pose_6d(net: TransPoseNet, imu: torch.Tensor) -> torch.Tensor:
    leaf = net.pose_s1(imu)[0]
    full = net.pose_s2(torch.cat((leaf, imu), dim=1))[0]
    return net.pose_s3(torch.cat((full, imu), dim=1))[0]


def reduced_global_6d_to_local_pose(
    net: TransPoseNet,
    root_rotation: torch.Tensor,
    reduced_6d: torch.Tensor,
) -> torch.Tensor:
    reduced = art.math.r6d_to_rotation_matrix(reduced_6d).view(-1, joint_set.n_reduced, 3, 3)
    eye = torch.eye(3, dtype=reduced.dtype, device=reduced.device)
    global_full = eye.expand(reduced.shape[0], 24, 3, 3).clone()
    global_full[:, joint_set.reduced] = reduced
    local = net.global_to_local_pose(global_full).view(-1, 24, 3, 3)
    for ignored_joint in joint_set.ignored:
        local[:, ignored_joint] = eye
    local[:, 0] = root_rotation.view(-1, 3, 3)
    return local


def pose_axis_angle_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    return art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)


def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)


def _unwrap_angle(angle: torch.Tensor) -> torch.Tensor:
    if angle.shape[0] < 2:
        return angle
    delta = angle[1:] - angle[:-1]
    wrapped = torch.remainder(delta + math.pi, 2 * math.pi) - math.pi
    correction = torch.where(delta.abs() < math.pi, torch.zeros_like(delta), wrapped - delta)
    return torch.cat((angle[:1], angle[1:] + torch.cumsum(correction, dim=0)), dim=0)


def _leg_swing_angles(joints: torch.Tensor) -> list[torch.Tensor]:
    pelvis = joints[:, 0]
    neck = joints[:, 12]
    left_shoulder = joints[:, 16]
    right_shoulder = joints[:, 17]
    x_body = _normalize(right_shoulder - left_shoulder)
    y_body = _normalize(neck - pelvis)
    z_body = _normalize(torch.cross(x_body, y_body, dim=-1))
    x_body = _normalize(torch.cross(y_body, z_body, dim=-1))
    del x_body  # completes the same orthogonalization as the benchmark torso frame

    angles = []
    for hip, knee in ((1, 4), (2, 5)):
        vector = joints[:, knee] - joints[:, hip]
        up = torch.sum(vector * y_body, dim=-1)
        forward = torch.sum(vector * z_body, dim=-1)
        angles.append(_unwrap_angle(torch.atan2(forward, -up)))
    return angles


def top_bottom_mean_range(
    values: torch.Tensor,
    fraction: float = SWING_TOP_BOTTOM_FRACTION,
    min_frames: int = SWING_TOP_BOTTOM_MIN_FRAMES,
) -> torch.Tensor:
    """Differentiable p95-p5 surrogate with gradient on about 10% of frames."""
    n = int(values.shape[0])
    if n < 2:
        raise ValueError("A range requires at least two frames")
    k = max(int(min_frames), int(math.ceil(n * fraction)))
    k = min(k, max(1, n // 2))
    top = torch.topk(values, k=k, largest=True, sorted=False).values.mean()
    bottom = torch.topk(values, k=k, largest=False, sorted=False).values.mean()
    return top - bottom


def differentiable_leg_swing_amplitude(joints: torch.Tensor) -> torch.Tensor:
    """Training surrogate: bilateral top/bottom-5% mean sagittal range."""
    amplitudes = [top_bottom_mean_range(angle) for angle in _leg_swing_angles(joints)]
    return torch.stack(amplitudes).mean()


def evaluation_leg_swing_amplitude(joints: torch.Tensor) -> torch.Tensor:
    """Held-out metric: exact bilateral p95-p5 sagittal range."""
    amplitudes = [
        torch.quantile(angle, 0.95) - torch.quantile(angle, 0.05)
        for angle in _leg_swing_angles(joints)
    ]
    return torch.stack(amplitudes).mean()


def differentiable_joint_velocity(joints: torch.Tensor, fps: float = 60.0) -> torch.Tensor:
    root_aligned = joints - joints[:, :1]
    velocity = torch.diff(root_aligned, dim=0) * fps
    return torch.linalg.vector_norm(velocity, dim=-1).mean()


def absolute_log_ratio(predicted: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.abs(torch.log((predicted + eps) / (target + eps)))


def swing_reliability(cycles: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not math.isfinite(cycles):
        return torch.tensor(1.0, device=device, dtype=dtype)
    return torch.tensor(min(1.0, max(0.0, cycles / 2.0)), device=device, dtype=dtype)


def snapshot_trainable_parameters(net: TransPoseNet) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in net.named_parameters()
        if parameter.requires_grad
    }


def l2sp_loss(net: TransPoseNet, reference: dict[str, torch.Tensor]) -> torch.Tensor:
    squared_sum = None
    for name, parameter in net.named_parameters():
        if parameter.requires_grad:
            value = torch.sum((parameter - reference[name]) ** 2)
            squared_sum = value if squared_sum is None else squared_sum + value
    if squared_sum is None:
        raise RuntimeError("L2-SP requested without trainable parameters")
    # The unnormalized sum keeps the restoring gradient on each parameter equal
    # to 2 * lambda_sp * (theta_i - theta_pre_i), independent of how many other
    # parameters are unfrozen in S1/S1.5/S2.
    return squared_sum


def root_relative_joint_position_huber(
    predicted_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    beta_m: float = JOINT_POSITION_HUBER_BETA_M,
) -> torch.Tensor:
    predicted = predicted_joints - predicted_joints[:, :1]
    target = gt_joints - gt_joints[:, :1]
    distance = torch.linalg.vector_norm(
        predicted[:, list(PREDICTED_JOINTS)] - target[:, list(PREDICTED_JOINTS)], dim=-1
    )
    return F.smooth_l1_loss(distance, torch.zeros_like(distance), beta=beta_m)


class FineTuneLoss:
    def __init__(self, net: TransPoseNet, body_model: art.ParametricModel):
        self.net = net
        self.body_model = body_model

    def __call__(
        self,
        predicted_6d: torch.Tensor,
        imu: torch.Tensor,
        gt_axis_angle: torch.Tensor,
        cycles: float,
        *,
        pose_weight: float,
        velocity_weight: float,
        swing_weight: float,
        l2sp_weight: float,
        pretrained: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        gt_local = pose_axis_angle_to_matrix(gt_axis_angle)
        gt_global = self.body_model.forward_kinematics_R(gt_local)
        gt_reduced = gt_global[:, joint_set.reduced]
        predicted_reduced = art.math.r6d_to_rotation_matrix(predicted_6d).view_as(gt_reduced)
        rotation_chordal = torch.mean((predicted_reduced - gt_reduced) ** 2)

        predicted_local = reduced_global_6d_to_local_pose(
            self.net, imu[:, -9:].view(-1, 3, 3), predicted_6d
        )
        _, predicted_joints = self.body_model.forward_kinematics(
            predicted_local, shape=None, tran=None, calc_mesh=False
        )
        with torch.no_grad():
            _, gt_joints = self.body_model.forward_kinematics(
                gt_local, shape=None, tran=None, calc_mesh=False
            )
        joint_position = root_relative_joint_position_huber(predicted_joints, gt_joints)
        pose_total = rotation_chordal + FROZEN_JOINT_POSITION_WEIGHT * joint_position

        predicted_velocity = differentiable_joint_velocity(predicted_joints)
        gt_velocity = differentiable_joint_velocity(gt_joints)
        velocity = absolute_log_ratio(predicted_velocity, gt_velocity)

        predicted_swing_surrogate = differentiable_leg_swing_amplitude(predicted_joints)
        gt_swing_surrogate = differentiable_leg_swing_amplitude(gt_joints)
        swing_surrogate = absolute_log_ratio(predicted_swing_surrogate, gt_swing_surrogate)
        predicted_swing_eval = evaluation_leg_swing_amplitude(predicted_joints)
        gt_swing_eval = evaluation_leg_swing_amplitude(gt_joints)
        swing_eval = absolute_log_ratio(predicted_swing_eval, gt_swing_eval)
        reliability = swing_reliability(
            cycles, device=predicted_6d.device, dtype=predicted_6d.dtype
        )
        swing = swing_surrogate * reliability
        l2sp = l2sp_loss(self.net, pretrained)

        total = (
            pose_weight * pose_total
            + velocity_weight * velocity
            + swing_weight * swing
            + l2sp_weight * l2sp
        )
        return {
            "total": total,
            "pose_total": pose_total,
            "rotation_chordal": rotation_chordal,
            "joint_position_huber": joint_position,
            "velocity_log_ratio": velocity,
            "swing_log_ratio": swing_eval,
            "swing_surrogate_log_ratio": swing_surrogate,
            "swing_surrogate_weighted": swing,
            "swing_reliability": reliability,
            "clinical_distortion": 0.5 * (velocity + swing_eval),
            "l2sp": l2sp,
            "predicted_velocity": predicted_velocity,
            "gt_velocity": gt_velocity,
            "predicted_leg_swing_eval": predicted_swing_eval,
            "gt_leg_swing_eval": gt_swing_eval,
            "predicted_leg_swing_surrogate": predicted_swing_surrogate,
            "gt_leg_swing_surrogate": gt_swing_surrogate,
        }


@torch.no_grad()
def no_root_mpjpe_mm(
    body_model: art.ParametricModel,
    predicted_local_pose: torch.Tensor,
    gt_axis_angle: torch.Tensor,
) -> float:
    gt_local = pose_axis_angle_to_matrix(gt_axis_angle)
    _, predicted_joints = body_model.forward_kinematics(
        predicted_local_pose, shape=None, tran=None, calc_mesh=False
    )
    _, gt_joints = body_model.forward_kinematics(gt_local, shape=None, tran=None, calc_mesh=False)
    predicted = (predicted_joints - predicted_joints[:, :1]).cpu().numpy()
    target = (gt_joints - gt_joints[:, :1]).cpu().numpy()
    per_joint = np.linalg.norm(predicted - target, axis=-1) * 1000.0
    return float(np.nanmean(per_joint[:, PREDICTED_JOINTS]))


def make_body_model(device: torch.device) -> art.ParametricModel:
    return art.ParametricModel(paths.smpl_file, device=device)


def chunked(values: list[int], size: int) -> Iterator[list[int]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]
