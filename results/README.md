# Frozen aggregate results

The paired-effect table is copied without numeric changes from the 2026-08-15 benchmark-aligned descriptor analysis of frozen outer-test checkpoints. Each row uses 23 equally weighted participants and 781 paired walks. `clinical_minus_pose` is the difference in dimensionless absolute log-ratio reconstruction distortion; negative means lower error.

`ci_low`/`ci_high` are nominal subject-clustered 95% bootstrap intervals. The separate Bonferroni columns contain 99% intervals for the five-descriptor family within a sensor configuration. The CSV preserves both interpretations.

The five-descriptor analysis was added **post hoc** following methodological review. Its descriptors were selected from the preceding benchmark's GT–severity associations; the trained checkpoints were unchanged. This timing is distinct from the original frozen Clinical Distortion primary endpoint, preserved in `docs/OUTER_TEST_REPORT.md`.

GT–UPDRS correlations and prediction–UPDRS correlations are descriptive. Stronger correlation with severity does not by itself establish faithful reconstruction.
