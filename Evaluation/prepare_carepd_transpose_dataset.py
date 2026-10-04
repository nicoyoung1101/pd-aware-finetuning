"""
prepare_carepd_transpose_dataset.py

Generic CARE-PD canonical-SMPL -> TransPose-style C6 IMU adapter.

This is a thin wrapper around prepare_carepd_transpose.py so that second
CARE-PD cohorts (e.g. E-LC / KUL-DT-T) use exactly the same virtual IMU
pipeline as BMCLab:
  acc vertices: [1961, 5424, 1176, 4662, 411, 3021]
  ori joints  : [18, 19, 4, 5, 15, 0]
  order       : left forearm, right forearm, left lower leg, right lower leg, head, pelvis
  target fps  : 60
  SMPL shape  : neutral (shape=None), matching the existing C6 benchmark path
"""

import argparse
import json
import os
import pickle
import sys

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Evaluation import prepare_carepd_transpose as base

CANONICAL_DIR = os.path.join(PROJECT_ROOT, "Dataset", "CARE-PD", "Canonicalized_SMPL_pickles")
DEFAULT_SMPL = base.SMPL_FILE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True, help="CARE-PD canonical cohort name, e.g. E-LC or KUL-DT-T")
    parser.add_argument("--input", default=None, help="Optional explicit canonical .pkl path")
    parser.add_argument("--output", default=None, help="Output dataset_work directory")
    parser.add_argument("--smpl", default=DEFAULT_SMPL)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test sequence limit")
    parser.add_argument(
        "--include-labels",
        default="",
        help="Comma-separated values matched against sequence['other']; empty keeps all sequences.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sanitize_dataset_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def main():
    args = parse_args()
    input_pkl = args.input or os.path.join(CANONICAL_DIR, f"{args.dataset_name}_canonical.pkl")
    output_dir = args.output or os.path.join(base.TRANSPOSE_ROOT, "data", "dataset_work", f"CAREPD_{sanitize_dataset_name(args.dataset_name)}")
    include_labels = {s.strip() for s in args.include_labels.split(",") if s.strip()}

    base.check_file(input_pkl, "CARE-PD canonical pickle")
    base.check_file(args.smpl, "SMPL model")
    if os.path.exists(os.path.join(output_dir, "test.pt")) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to replace.")

    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    walks = base.collect_walks(data)
    if include_labels:
        walks = [(sid, wid, seq) for sid, wid, seq in walks if str(seq.get("other")) in include_labels]
    if args.limit > 0:
        walks = walks[: args.limit]
    if not walks:
        raise ValueError("No sequences matched the requested dataset/filter.")

    print(f"Input       : {input_pkl}")
    print(f"SMPL        : {args.smpl}")
    print(f"Output      : {output_dir}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Walks       : {len(walks)}")
    print(f"Target fps  : {base.TARGET_FPS}")
    print(f"Labels kept : {sorted(include_labels) if include_labels else 'all'}")

    body_model = base.art.ParametricModel(args.smpl)
    accs, oris, poses, trans = [], [], [], []
    manifest = []
    for subj_id, walk_id, seq in tqdm(walks, desc=f"CARE-PD {args.dataset_name} -> TransPose"):
        out = base.process_walk(body_model, seq)
        if out["pose"].shape[0] < 12:
            continue
        accs.append(out["acc"])
        oris.append(out["ori"])
        poses.append(out["pose"])
        trans.append(out["tran"])
        manifest.append({
            "dataset_name": args.dataset_name,
            "subject_id": subj_id,
            "walk_id": walk_id,
            "source_fps": out["source_fps"],
            "target_fps": base.TARGET_FPS,
            "n_frames": int(out["pose"].shape[0]),
            "UPDRS_GAIT": seq.get("UPDRS_GAIT"),
            "medication": seq.get("medication"),
            "other": seq.get("other"),
            "source_trim_frames": out["source_trim_frames"],
            "output_trim_frames": out["output_trim_frames"],
        })

    os.makedirs(output_dir, exist_ok=True)
    torch.save({"acc": accs, "ori": oris, "pose": poses, "tran": trans}, os.path.join(output_dir, "test.pt"))
    meta = {
        "source": input_pkl,
        "dataset_name": args.dataset_name,
        "smpl_file": args.smpl,
        "target_fps": base.TARGET_FPS,
        "vi_mask": base.VI_MASK.tolist(),
        "ji_mask": base.JI_MASK.tolist(),
        "sensor_order": ["left_forearm", "right_forearm", "left_lower_leg", "right_lower_leg", "head", "pelvis"],
        "acc_smooth_n": base.ACC_SMOOTH_N,
        "source_trim_frames": base.SOURCE_TRIM_FRAMES,
        "output_trim_frames": base.OUTPUT_TRIM_FRAMES,
        "shape_policy": "neutral_shape_none_to_match_existing_C6_benchmark",
        "include_labels": sorted(include_labels),
        "n_sequences": len(accs),
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(accs)} sequences to {os.path.join(output_dir, 'test.pt')}")
    print(f"Saved manifest to {os.path.join(output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
