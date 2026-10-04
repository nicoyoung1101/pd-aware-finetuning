# Portfolio release — 2026-10-04

This release packages the completed local research and frozen aggregate results. The original source data and experiment outputs remain separately managed.

- Curated English README, generated banners, result figures and protocol documentation.
- Public qualitative MP4/GIF assets; no patient-level tensors or clinical tables are bundled.
- Relative paths replace machine-specific inference paths. Research losses and frozen coefficients retain their study values.
- Shared metric runner now exposes the existing model selector as `--models`; the benchmark defaults to its five main models and the fine-tuning package defaults to TransPose only. This prevents unrelated diagnostic outputs from becoming mandatory inputs.
- Final reported signed-trunk results and legacy unsigned implementation are explicitly distinguished.
- Optional CI configuration is provided as `docs/ci-example.yml`; place it under `.github/workflows/` to enable it with appropriate GitHub workflow credentials.
- Source syntax, result-demo regeneration, local Markdown links and video integrity were checked. Fine-tuning additionally has data-free contract tests. No new cohort evaluation or GPU training is claimed by this packaging release.

## SMPL portfolio display update

Replaced the README skeleton previews with SMPL surface-mesh animations. The fine-tuning preview is generated from the original labeled ICNR MP4; the benchmark uses the original prediction tensors and SMPL mesh topology to render the common-walk five-model comparison.
