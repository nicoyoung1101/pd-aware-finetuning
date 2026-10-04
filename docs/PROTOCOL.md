# PD-aware fine-tuning protocol

## Study design
The study compares pose-only (`P6`, `P5`) and clinical-aware (`C6`, `C5`) fine-tuning from the same healthy-pretrained TransPose initialization. Six outer subject-wise folds contain 4/4/4/4/4/3 participants; each has four inner-validation participants. A participant appears in exactly one outer-test fold. Walks and medication states stay grouped by participant.

One training sample is one complete variable-length walk; sequences are not cropped, padded or concatenated. Subject-balanced sampling draws equal numbers of whole walks per participant. Validation first averages within each subject and then weights subjects equally.

## Model and sensor controls
Formal main runs use S1, the final `pose_s3.linear2` layer: 23,130 trainable parameters, approximately 0.48% of the 4.80-million-parameter base model. S1.5 and S2 are implemented capacity options; they are not additional completed main-result arms. Translation branches and the other pose stages stay frozen.

The head-free five-IMU condition retains the architecture and input width. Head acceleration and orientation are copied from the pelvis in raw space before normalization, producing normalized head channels `(0, I)`.

## Loss and checkpoint selection
The pose term combines rotation-matrix MSE and root-relative joint-position Smooth L1. Clinical-aware conditions add joint velocity and a differentiable sagittal leg-swing surrogate. L2-SP anchors trainable parameters to the pretrained initialization.

Frozen weights: pose=1, position=1, velocity=0.0583, swing=0.0352, L2-SP=0.0014783522259721978. Velocity/swing are whole-walk absolute log-ratio errors; the swing surrogate uses the top/bottom 5% mean while evaluation uses the original p95–p5 descriptor. Short-sequence cycle reliability weights the training swing term only.

All four arms select their main checkpoint with the same subject-equal inner-validation Clinical Distortion rule. An own-objective checkpoint is retained as a sensitivity analysis. Dual-monitor patience determines termination; capped trajectories are excluded and affected matched arms are rerun before outer-test access.

## Statistical analysis and result timing
The frozen 2026-08-09 analysis uses 20,000 subject-clustered bootstrap replicates. The original composite Clinical Distortion endpoint averages the two optimized descriptor distortions. The 2026-08-15 methodological review adds a **post-hoc benchmark-aligned analysis** reporting five descriptors separately, without a new composite or checkpoint selection. Both records are preserved; the README presents the descriptor-level evidence and its limits.

## Findings and limits
Joint-velocity and leg-swing distortion decrease in both sensor configurations. Lower-body velocity improves clearly with six IMUs; five-IMU effects are uncertain. Lower-body RoM remains inconclusive. Foot-lift distortion increases in both configurations. This supports targeted feature preservation, not a claim of universal clinical improvement. The between-configuration treatment-effect interval crosses zero and is not an equivalence/non-inferiority result. Real-IMU validation remains outstanding.
