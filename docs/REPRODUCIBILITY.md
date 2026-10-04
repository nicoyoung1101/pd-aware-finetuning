# Reproduction guide

## Aggregate-result demo
Run from the repository root:

```bash
python -m pip install -r requirements-demo.txt
python scripts/make_figures.py
```

This recreates the effect plot from the bundled aggregate table without patient data or a GPU.

## Research pipeline
Acquire external assets as described in `DATA.md`, then install the research runtime. PyTorch 2.6+ changes the default checkpoint loading behavior; the release targets PyTorch 2.0–2.5 for the original TransPose loader. Legacy SMPL/chumpy pickles may require the upstream Python environment.

```bash
python -m pip install -r requirements.txt
python Evaluation/prepare_carepd_transpose.py
python Evaluation/run_carepd_transpose_batch.py
python Evaluation/analyze_multimodel_extended_metrics.py --out-dir ModelComparison/data/results/CAREPD_BMCLab_extended_metrics_with_TIP
python Evaluation/generate_icnr2026_nested_splits.py
python Evaluation/run_icnr2026_sanity_gates.py --device cpu
python Evaluation/train_icnr2026_transpose.py --help
```

The cycle reliability input is `ModelComparison/data/results/CAREPD_BMCLab_extended_metrics_with_TIP/multimodel_extended_per_walk.csv`; generate the shared baseline metric file at that path using the metric script's output option. It supplies the per-walk cycle counts used in training. Reuse one generated split artifact for all arms. The sanity gates verify the data contract, healthy-pretrained benchmark regression and loss/gradient isolation. Formal runs refuse to proceed unless gates pass.

For each of the six folds, run P6/C6 and P5/C5 with the same seed and S1 capacity:

```bash
python Evaluation/train_icnr2026_transpose.py --fold 0 --condition C6 --capacity S1 --device cuda --epochs 160 --patience 8 --out-root runs/Full6
python Evaluation/train_icnr2026_transpose.py --fold 0 --condition C5 --capacity S1 --device cuda --epochs 80 --patience 8 --out-root runs/C5
```

Repeat with P6/P5 and the remaining folds. The epoch values are ceilings; verify genuine dual-monitor patience stops and rerun sensor-matched arms if capped before outer-test inference. Refer to `PROTOCOL.md` for the complete contract.

## Held-out inference
`scripts/icnr2026_oof_inference.py` keeps the original outer-test unlock guard. Path adaptation changes the script hash. A new run therefore needs its own reviewed unlock manifest containing `approved: true` and the current script hash in `inference_script_sha256`, according to the guard in that script; do not reuse the historical unlock artifact. Inference accepts `--full6-root` and `--c5-root` to locate completed runs. Inspect the guard's exact JSON schema before creating the manifest.

The included analysis scripts accept local per-walk result tables to reproduce participant aggregation and clustered bootstrap inference. Participant-level inputs and trained fold checkpoints are excluded from this public software release. Exact numerical reruns require the original asset versions, canonical ordering, manifest, split hashes and frozen checkpoints. `results/` provides the unchanged aggregate result snapshot, not replacement synthetic patient data.

## Release validation
Synthetic tests verify root-copy normalization, exact S1 freezing, whole-walk subject-balanced sampling, log-ratio direction and L2-SP anchoring. They need the Python runtime but no licensed SMPL model, patient data or GPU:

```bash
python -m unittest discover -s tests -v
```

A fresh GPU training rerun was not performed during release packaging. Historical full-data audit evidence is documented in `OUTER_TEST_REPORT.md`.
