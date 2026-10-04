# ICNR 2026 one-time outer-test report

Date: 2026-08-09  
Status: **final frozen outer-test analysis completed once**

## Main conclusion

Clinical-aware fine-tuning reduced the preregistered Clinical Distortion endpoint relative to pose-only fine-tuning in both sensor configurations. No-root MPJPE also decreased slightly in both comparisons. Effects outside the clinical loss were mixed: arm-swing distortion improved, lower-body RoM was inconclusive, and foot-lift distortion showed nominal/potential worsening in both configurations.

The correct result branch is therefore:

- Full6 primary: `improvement_supported`;
- head-free C5 primary: `improvement_supported`;
- Full6 held-out descriptors: `nominal_potential_decline: foot_lift`;
- C5 held-out descriptors: `nominal_potential_decline: foot_lift`;
- C5 versus Full6 treatment-effect difference: `no evidence that treatment effects differ`—not an equivalence claim.

## Primary and safety endpoints

All differences are `clinical-aware − pose-only`; negative values indicate lower distortion/error.

| Configuration | Clinical Distortion difference (95% CI) | No-root MPJPE difference, mm (95% CI) |
|---|---:|---:|
| Full6: C6−P6 | −0.02597 (−0.03333, −0.01889) | −0.552 (−0.936, −0.220) |
| C5/M1: C5−P5 | −0.02931 (−0.03619, −0.02276) | −0.501 (−0.861, −0.158) |

The corresponding subject-equal condition means were:

| Condition | Clinical Distortion | No-root MPJPE, mm |
|---|---:|---:|
| P6 | 0.05518 | 8.570 |
| C6 | 0.02920 | 8.018 |
| P5 | 0.06847 | 10.338 |
| C5 | 0.03916 | 9.837 |

The C5-minus-Full6 treatment-effect difference in Clinical Distortion was −0.00334 (95% CI −0.00808, +0.00152). The interval crosses zero, so the permitted statement is only that the between-configuration treatment-effect difference was inconclusive. It does not license `comparable`, `equivalent`, `same effect`, `non-inferior`, or retained-percentage wording.

## Held-out descriptors

| Descriptor | Full6 C6−P6 (95% CI) | C5 C5−P5 (95% CI) | Frozen interpretation |
|---|---:|---:|---|
| Arm-swing distortion | −0.00839 (−0.01554, −0.00173) | −0.01156 (−0.01805, −0.00586) | Improved in both configurations |
| Foot-lift distortion | +0.01820 (+0.00836, +0.02769) | +0.03890 (+0.02317, +0.05512) | Nominal/potential worsening in both |
| Lower-body RoM distortion | −0.00370 (−0.00997, +0.00254) | −0.00497 (−0.01417, +0.00439) | Inconclusive in both |
| Arm-asymmetry absolute error | −0.00148 (−0.00462, +0.00138) | +0.00254 (−0.00516, +0.00959) | Inconclusive in both |

The foot-lift result is mechanistically plausible but not a causal test. The optimized leg-swing descriptor constrains the sagittal hip-to-knee vector; foot lift depends on the coupled hip–knee chain. Increasing thigh swing without a matching knee constraint can alter vertical foot trajectory. Ankle rotation is an ignored identity placeholder in the reduced TransPose protocol. A future loss should constrain knee angle or the hip-to-ankle trajectory.

The three held-out descriptor intervals are nominal unadjusted 95% CIs. Foot-lift worsening must be disclosed, but it is not labelled confirmatory harm.

## Imputation sensitivity

M1 remains the main five-IMU method. M2 applies a fold-specific subject-equal normalized head-channel mean to checkpoints trained with M1 and is therefore an inference-mismatch stress test.

| M2 endpoint, C5-M2−P5-M2 | Difference (95% CI) |
|---|---:|
| Clinical Distortion | −0.03255 (−0.04016, −0.02510) |
| No-root MPJPE | +1.674 mm (+0.611, +2.856) |
| Arm-swing distortion | −0.00787 (−0.01371, −0.00275) |
| Foot-lift distortion | +0.02962 (+0.01354, +0.04602) |
| Lower-body RoM distortion | −0.01332 (−0.02422, −0.00265) |

Required wording: clinical-distortion improvement persisted under fold-mean imputation, whereas pose accuracy degraded. M2 is not evidence that results are imputation-invariant.

## Checkpoint sensitivity

Own-objective checkpoint selection preserved the main Clinical Distortion conclusion:

- Full6: −0.02755 (−0.03448, −0.02089);
- C5/M1: −0.03108 (−0.03842, −0.02415).

Foot-lift distortion remained nominally worse under the sensitivity checkpoints. Thus the main result is not an artifact of choosing only the clinical-distortion-selected checkpoint.

## Severity trends

Ratios were aggregated to 42 subject–medication states. CIs resampled the 23 subjects, keeping each subject’s ON/OFF states together. These are descriptive cross-validated trends; no analytic p-values or formal differences between correlations were tested.

| Metric | P6 ρ (95% CI) | C6 ρ (95% CI) | P5 ρ (95% CI) | C5 ρ (95% CI) |
|---|---:|---:|---:|---:|
| Velocity ratio | −0.679 (−0.847, −0.419) | −0.244 (−0.606, +0.150) | −0.650 (−0.817, −0.399) | −0.272 (−0.601, +0.100) |
| Leg-swing ratio | −0.730 (−0.861, −0.539) | −0.513 (−0.721, −0.256) | −0.697 (−0.850, −0.475) | −0.479 (−0.729, −0.174) |

The reduced magnitude is consistent with less severity-dependent attenuation, but no inferential Δρ claim is made.

## B0 descriptive baseline

The all-781-walk B0-C6 path reproduced Gate 2 exactly: U0/U1/U2 no-root MPJPE was 16.740611/18.482270/26.961447 mm. Subject-equal B0-C6 MPJPE was 19.828 mm, compared with 8.570 mm for P6 and 8.018 mm for C6. These large B0-to-fine-tuned changes may be displayed descriptively in Panel B, but they were not the preregistered pose-safety contrast and cannot become a headline claim.

Naive head removal increased pretrained Clinical Distortion: B0-C5-M1 minus B0-C6 was +0.02780 (+0.02126, +0.03521). Fine-tuned C5 minus B0-C5-M1 was −0.05884 (−0.06443, −0.05336), also post-hoc descriptive.

## Integrity audit

- Outer folds contained exactly 108/122/151/139/145/116 walks; total 781.
- Every participant appeared in exactly one outer fold: 23 unique subjects and 23 fold-subject appearances.
- Nine main conditions plus four own-objective sensitivities produced exactly 10,153 rows (`781 × 13`).
- Every paired comparison used 781/781 walks.
- All GT and prediction validity fractions were 1.0; no endpoint was non-finite.
- Maximum benchmark-versus-Torch leg-swing difference was `1.35e-5°`; maximum Clinical Distortion difference was `3.80e-7`.
- B0 preview and Gate helper agreed per walk with maximum MPJPE difference `0`.
- Bootstrap used 20,000 subject-clustered replicates with frozen seed `20260808`.

No alternative checkpoint, endpoint, missingness rule, bootstrap hierarchy, figure type or headline branch was selected after viewing outer-test results.
