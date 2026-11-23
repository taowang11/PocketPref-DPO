"""Generate three-dimensional ligands for a prepared protein pocket."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
DEFAULT_ATOM_COUNT_CHECKPOINT = ROOT / 'ckpts' / 'atom_num_model.pt'
DEFAULT_STEPS = 50
PROPERTY_TARGETS = (1.0, 1.0, 1.0, 50.0, 3.0, 2.0, 0.5, 2.0)
LIGAND_TYPES = (
    (1, False), (6, False), (6, True), (7, False), (7, True), (8, False),
    (8, True), (9, False), (15, False), (15, True), (16, False), (16, True), (17, False),
)


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('Expected a positive integer.')
    return result


def parse_args():
    defaults = json.loads((ROOT / 'configs/sampling.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pocket', required=True, type=Path, help='Prepared pocket PDB in the input coordinate frame.')
    parser.add_argument('--checkpoint', required=True, type=Path, help='Base FT or PocketPref-DPO inference checkpoint.')
    parser.add_argument('--output-dir', required=True, type=Path, help='New output directory for molecules.sdf and run.json.')
    parser.add_argument('--num-samples', type=positive_int, default=str(defaults['num_samples']), help='Number of generation attempts.')
    parser.add_argument('--batch-size', type=positive_int, default=str(defaults['batch_size']))
    parser.add_argument('--seed', type=int, default=str(defaults['seed']))
    parser.add_argument('--device', default=str(defaults['device']), help='PyTorch device, e.g. cuda:0 or cpu.')
    parser.add_argument('--num-threads', type=positive_int, default=str(defaults['num_threads']))
    size = parser.add_mutually_exclusive_group()
    size.add_argument('--atom-count', type=positive_int, help='Fixed initial ligand atom count.')
    size.add_argument('--atom-count-checkpoint', type=Path,
                      help='Trusted PAFlow predictor weights (default: ckpts/atom_num_model.pt).')
    parser.add_argument('--pocket-volume', type=float, default=232.63)
    parser.add_argument('--pocket-area', type=float, default=229.08)
    return parser.parse_args()


def load_model(checkpoint, device):
    import torch
    from easydict import EasyDict
    from models.sefmol import SeFMol, initialize_diffusion_params

    # The published format stores tensors and a plain model configuration.
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('format') != 'pocketpref-dpo-inference-v1':
        raise ValueError('Expected an inference export. Load one of the published or newly trained PocketPref-DPO weights.')
    config = EasyDict(payload['model_config'])
    model = SeFMol(config, protein_atom_feature_dim=27,
                   ligand_atom_feature_dim=len(LIGAND_TYPES), n_timesteps=DEFAULT_STEPS)
    state = payload['model']
    initialize_diffusion_params(state, config, DEFAULT_STEPS)
    model.load_state_dict(state, strict=True)
    model.refine_net.safe_knn = True
    return model.to(device).eval()


def draw_batch(model, positions, features, counts):
    import torch
    from models.sefmol import log_sample_categorical
    from torch_scatter import scatter_mean

    device = positions.device
    graphs = torch.arange(len(counts), device=device)
    protein_batch = graphs.repeat_interleave(positions.shape[0])
    ligand_batch = graphs.repeat_interleave(torch.tensor(counts, device=device))
    protein_pos = positions.repeat(len(counts), 1)
    center = scatter_mean(protein_pos, protein_batch, dim=0)
    initial = torch.randn((len(ligand_batch), 3), device=device)
    types = log_sample_categorical(torch.zeros((len(ligand_batch), len(LIGAND_TYPES)), device=device))
    result = model.sample_diffusion(
        protein_pos=protein_pos, protein_v=features.repeat(len(counts), 1),
        batch_protein=protein_batch, init_ligand_pos=center[ligand_batch] + initial,
        init_ligand_v=types, batch_ligand=ligand_batch,
        properties=torch.tensor(PROPERTY_TARGETS, device=device).repeat(len(counts)),
        num_steps=DEFAULT_STEPS,
    )
    coords = result['pos'].detach().cpu().double().split(counts)
    categories = result['v'].detach().cpu().split(counts)
    return [(c.numpy(), t.tolist()) for c, t in zip(coords, categories)]


def generate(args):
    import numpy as np
    import torch

    from datasets.pocket import featurize
    from rdkit import Chem
    from utils.atom_count_predictor import PAFlowAtomCountSampler
    from utils.reconstruct import MolReconsError, reconstruct_from_generated

    if not 0 <= args.seed < 2**32:
        raise ValueError('Seed must be in [0, 2**32).')
    atom_count_checkpoint = args.atom_count_checkpoint
    if args.atom_count is None and atom_count_checkpoint is None:
        atom_count_checkpoint = DEFAULT_ATOM_COUNT_CHECKPOINT
        if not atom_count_checkpoint.is_file():
            raise FileNotFoundError(f'Missing bundled PAFlow predictor at {atom_count_checkpoint}. '
                                    'Restore it from the reproduction archive or pass --atom-count.')
    for path in (args.pocket, args.checkpoint, atom_count_checkpoint):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {args.output_dir}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu or install a CUDA-enabled runtime.')
    torch.set_num_threads(args.num_threads)
    model = load_model(args.checkpoint, device)
    positions, features = featurize(args.pocket)
    pocket = SimpleNamespace(protein_pos=positions, volume=args.pocket_volume, area=args.pocket_area)
    positions, features = positions.to(device), features.to(device)
    predictor = None
    if atom_count_checkpoint is not None:
        predictor = PAFlowAtomCountSampler(str(atom_count_checkpoint), device=device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True)
    attempts = []
    with Chem.SDWriter(str(args.output_dir / 'molecules.sdf')) as writer:
        for begin in range(0, args.num_samples, args.batch_size):
            count = min(args.batch_size, args.num_samples - begin)
            counts = predictor.sample(pocket, count) if predictor else [args.atom_count] * count
            for index, (xyz, categories) in enumerate(draw_batch(model, positions, features, counts), begin):
                record = dict(attempt=index, initial_atom_count=len(categories), valid=False)
                atoms, aromatic = zip(*(LIGAND_TYPES[i] for i in categories))
                try:
                    mol = reconstruct_from_generated(xyz, list(atoms), list(aromatic))
                    Chem.SanitizeMol(mol)
                    if len(Chem.GetMolFrags(mol)) != 1:
                        raise MolReconsError('Reconstructed molecule has multiple fragments.')
                    mol.SetProp('_Name', f'{args.checkpoint.stem} attempt {index}')
                    mol.SetIntProp('attempt', index)
                    writer.write(mol)
                    record.update(valid=True, smiles=Chem.MolToSmiles(mol, isomericSmiles=True))
                except (MolReconsError, ValueError, RuntimeError) as error:
                    record['error'] = f'{type(error).__name__}: {error}'
                attempts.append(record)
    valid = sum(row['valid'] for row in attempts)
    report = dict(checkpoint_file=args.checkpoint.name,
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  pocket_sha256=hashlib.sha256(args.pocket.read_bytes()).hexdigest(),
                  reverse_steps=DEFAULT_STEPS, seed=args.seed, device=str(device),
                  batch_size=args.batch_size, atom_count_mode='predict' if predictor else 'fixed',
                  pocket_volume=args.pocket_volume, pocket_area=args.pocket_area,
                  property_targets=PROPERTY_TARGETS, attempted=len(attempts), valid=valid,
                  attempts=attempts)
    if atom_count_checkpoint is not None:
        report['atom_count_checkpoint_sha256'] = hashlib.sha256(atom_count_checkpoint.read_bytes()).hexdigest()
    (args.output_dir / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    return valid


if __name__ == '__main__':
    try:
        opts = parse_args()
        valid = generate(opts)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise SystemExit(f'Generation failed: {exc}')
    print(f'Saved {valid}/{opts.num_samples} reconstructed molecules to {opts.output_dir}')
