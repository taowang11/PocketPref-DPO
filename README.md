# PocketPref-DPO

**Learning pocket-sensitive molecular generation through joint spatial and chemical response preference optimization**

PocketPref-DPO generates 3D small-molecule ligands from a protein binding
pocket. It fine-tunes a diffusion model with preferences derived from
spatial and chemical perturbations of training pockets. At inference,
the model takes a single pocket structure and generates ligands without
calling a docking scorer.

## Setup

### Environment

Create the Conda environment and install the PyTorch and PyG wheels for
Linux with CUDA 12.1.

```bash
conda env create -f environment.yml
conda activate pocketpref-dpo

python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter==2.1.2 torch-cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

[CPU setup](scripts/README.md#cpu-installation) is also available.

### Model checkpoints

| Model | File | Role |
| --- | --- | --- |
| Base (SeFMol-FT) | [`ckpts/ft.pt`](ckpts/ft.pt) | Starting weights and frozen reference for DPO |
| PocketPref-DPO | [`ckpts/dpo/pocketpref-dpo.pt`](ckpts/dpo/pocketpref-dpo.pt) | Joint spatial and chemical preference fine-tuned model |
| PAFlow atom-count predictor | [`ckpts/atom_num_model.pt`](ckpts/atom_num_model.pt) | Shared ligand-size prior used for evaluation |

Both diffusion checkpoints use the same architecture and can be run
independently with the same generation script. The script defaults to
PocketPref-DPO. Use `--checkpoint ckpts/ft.pt` to run Base. See the
[checkpoint details](ckpts/README.md) for provenance and file hashes.

## Quick Start

### 1. Prepare a pocket

Try the [prepared JAK2 pocket](data/examples/jak2_8bm2_pocket.pdb)
shipped with the repository. For your own target, use a cropped PDB with
standard amino acid records and one conformation for each atom.

### 2. Generate ligands

Run five generation attempts from the repository root. The sampler uses
the bundled PAFlow predictor to initialize each ligand's atom count, then
runs 50 reverse diffusion steps.

```bash
bash scripts/inference.sh \
  --pocket data/examples/jak2_8bm2_pocket.pdb \
  --output-dir outputs/example
```

The output directory contains `molecules.sdf` with successfully
reconstructed 3D ligands and `run.json` with the settings and outcome
of every attempt. The defaults for sampling are in
[configs/sampling.json](configs/sampling.json).

See the [inference guide](scripts/README.md) for a Base comparison,
predicted ligand sizes, and output details.

## Evaluation

Use a manifest with the same test pockets for all models. Each entry
points to a prepared receptor and the already generated molecules for
that pocket. The default budget is 100 pockets × 100 attempts per model.

```bash
python evaluate.py --manifest evaluation.json --output-dir outputs/evaluation --workers 32
```

Add `--dock` for full Vina docking. Use `--analysis structure` with the
same manifest to measure PoseBusters, ProLIF, steric clashes, ring-size
occurrence, and shared Morgan PCA on the saved poses.
See the [evaluation guide](data/evaluation.md) for receptor and reference
inputs, dependencies, and metric definitions.

## Training

Use the preference archive described in
[data/README.md](data/README.md). Extract it from the repository root and
start from the Base checkpoint.

```bash
unzip data/pocketpref-dpo-train.zip -d data
python train.py --data-dir data/pocketpref-dpo-train \
  --output-dir outputs/joint_dpo --save-every 5000
```

Each update samples one spatial pair and one chemical pair. For each
kind, sample a pocket uniformly and then a pair uniformly within that
pocket. The policy and frozen Base reference use the original pocket
and share the diffusion timestep and noise within each pair. DPO
compares the winner and loser on their reduction in denoising error
relative to Base. Training uses 1,000 diffusion timesteps, DPO β = 10, AdamW with learning rate and
weight decay both `1e-5`, gradient clipping at 2, and seed `20260929`.
The checkpoints saved by `train.py` can be passed to `generate.py`
with `--checkpoint`.

## Preference Pair Construction

Prepare a manifest of candidate SDFs, original pocket PDBs, and
prespecified edit sites. The builder edits each pocket, scores the
same saved ligand pose in both environments, and exports the tensor
inputs and preference pairs consumed by `train.py`.

```bash
python build_preferences.py --manifest sites.json --output-dir outputs/my_pairs
```

See [input format, scoring dependencies, and selection rules](data/preference_construction.md).
