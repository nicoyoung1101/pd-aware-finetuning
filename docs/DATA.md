# Data and external assets

The experiments use **virtual IMUs synthesized from MoCap-derived canonical SMPL motion**, not recorded wearable IMU streams. BMCLab contains 23 participants and 781 walks; E-LC supports subject-level freezer-trait classification (53 participants: 36 PD-FOG and 17 PD-NoFOG).

Acquire CARE-PD canonical SMPL sequences through the dataset authors under their data terms. The expected local layout is:

```text
Dataset/CARE-PD/Canonicalized_SMPL_pickles/BMCLab_canonical.pkl
Dataset/CARE-PD/Canonicalized_SMPL_pickles/E-LC_canonical.pkl
Models/smpl_models/smpl/SMPL_NEUTRAL.pkl
TransPose/data/weights.pt
```

Obtain SMPL from [the SMPL project](https://smpl.is.tue.mpg.de/) and healthy-pretrained TransPose weights via [the upstream instructions](https://github.com/Xinyu-Yi/TransPose). These assets are acquired separately. Generated motion tensors, participant-level tables, clinical labels and checkpoints belong in the ignored local data directories.

Public `results/` files contain aggregate research results. Videos show qualitative reconstructions, rather than identifiable camera recordings. They are examples, not evidence of performance across the entire cohort.

Preparation is fixed at 60 Hz. The six sensor locations are left/right forearms, left/right shanks, head and pelvis. Preserve the canonical sequence ordering and preprocessing when reproducing the frozen study.
