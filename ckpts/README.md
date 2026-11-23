# Model checkpoints

**Base** is the rigid SeFMol-FT checkpoint. PocketPref-DPO was initialized
from Base and used a frozen copy of Base as the DPO reference. The
packaged PocketPref-DPO weights were obtained via joint optimization on
the spatial and chemical preference data (2,207 spatial and 947 chemical
preference pairs).

Each diffusion checkpoint contains the model state and its
configuration. Select one diffusion checkpoint per inference run.
PocketPref-DPO can be run directly with the [Quick Start](../README.md#quick-start).

| File | SHA-256 |
| --- | --- |
| [`ft.pt`](ft.pt) | `21ccb6ae27e88aca3b905bfb0f0bc6079be6a208967f516e739d800c4da38235` |
| [`dpo/pocketpref-dpo.pt`](dpo/pocketpref-dpo.pt) | `e40e2f1ddad8b7e5340e4948cd7ff85105d9474b943a50d34661989c371ecf99` |
| [`atom_num_model.pt`](atom_num_model.pt) | `e16256fc90b491bddb4f8eccf34cc1174623cf84e7702037802b80d759ba9099` |

Base originates from [SeFMol](https://github.com/sii-pilab/SeFMol). Both
diffusion checkpoints use this repository's inference export format.

## PAFlow atom-count predictor

Base and PocketPref-DPO use the same pretrained
[PAFlow atom-count predictor](https://github.com/CMACH508/PAFlow) in the
official evaluation. The reproduction archive includes the original
checkpoint used in those runs.
