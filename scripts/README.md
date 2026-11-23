# Inference Guide

Start with the [Quick Start](../README.md#quick-start) for predicted-size
generation from a prepared pocket. Run the commands below from the
repository root.

## CPU installation

Create and activate the Conda environment, then install the CPU wheels.

```bash
conda env create -f environment.yml
conda activate pocketpref-dpo

python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-scatter==2.1.2 torch-cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

Set `--device cpu` when running the generation command.

## Predict ligand size

Both diffusion checkpoints use the bundled `ckpts/atom_num_model.pt`
by default.

```bash
bash scripts/inference.sh \
  --pocket /path/to/pocket.pdb \
  --num-samples 5 \
  --seed 20260831 \
  --device cuda:0 \
  --output-dir outputs/predicted_size
```

Pass `--pocket-volume` and `--pocket-area` to use measured pocket
features. The benchmark sampler looks up these features in the PAFlow
metadata for each test pocket. The standalone example uses the fallback
values of 232.63 Å³ and 229.08 Å². For a fixed initial atom count, use
`--atom-count 24`.
`--atom-count-checkpoint` selects custom predictor weights.

## Run Base

Use the same pocket, atom count, and sampling defaults as the Quick
Start. Change the checkpoint and output directory.

```bash
bash scripts/inference.sh \
  --pocket data/examples/jak2_8bm2_pocket.pdb \
  --checkpoint ckpts/ft.pt \
  --output-dir outputs/example_base
```

Use `--checkpoint` to select any compatible inference export. Run
`python generate.py --help` for the complete argument list.

## Sampling defaults

[configs/sampling.json](../configs/sampling.json) sets the default number
of attempts, batch size, seed, device, and CPU thread count. Command-line
options override these defaults.

## Outputs

Use a new directory for each run.

| File | Contents |
| --- | --- |
| `molecules.sdf` | Connected, successfully reconstructed molecules in the input pocket coordinate frame |
| `run.json` | Input and checkpoint hashes, including the predictor hash when used, sampling settings, and each attempt's initial atom count and outcome |

`--num-samples` sets the number of attempts. A failed reconstruction is
recorded in `run.json` and reduces the number of structures in the SDF.
