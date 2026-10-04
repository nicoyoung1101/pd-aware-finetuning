# Attribution and software provenance

The base **TransPose architecture, network and articulate toolkit** are by Xinyu Yi and collaborators, from [Xinyu-Yi/TransPose](https://github.com/Xinyu-Yi/TransPose), local source revision `2b31fd1bf1534c6f9c973fa091c0c8cbcccc7310`. Its GPL-3.0 license is preserved in `TransPose/LICENSE` and at the repository root. The adapted `config.py` resolves CARE-PD paths relative to this checkout; its upstream origin remains credited. Toolkit comments retain their own source attribution.

Research contributions in `Evaluation/` and the analysis scripts comprise CARE-PD data adaptation, virtual sensor synthesis, evaluation, and the PD fine-tuning protocol. TransPose itself is an existing baseline, not a new architecture introduced by this project.

Additional benchmark models are acquired independently from their original repositories and retain their own license and dependencies:

| Baseline | Upstream |
|---|---|
| PIP (pre-physics neural output) | https://github.com/Xinyu-Yi/PIP |
| DynaIP | https://github.com/dx118/dynaip |
| PNP | https://github.com/Xinyu-Yi/PNP |
| TIP | https://github.com/jyf588/transformer-inertial-poser |

SMPL, CARE-PD, pretrained weights and any external recordings are governed by their respective providers' terms; the software license does not grant rights to redistribute those assets.

The fine-tuning release uses the archived GPU study version of `net.py` and the fine-tuning core, including per-joint ignored-rotation assignment required by the differentiable training path.
