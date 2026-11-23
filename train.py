"""Train PocketPref-DPO from Base on spatial and chemical preference pairs."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from generate import positive_int

ROOT = Path(__file__).resolve().parent
SEED = 20260929
BETA = 10.0
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 2.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=ROOT / "ckpts/ft.pt")
    parser.add_argument("--steps", type=positive_int, default=40000)
    parser.add_argument("--save-every", type=positive_int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def matched_noise(model, batch):
    import torch

    ligand_pos = batch["ligand_pos"]
    ligand_batch = batch["batch_ligand"]
    num_pairs = len(batch["properties"]) // 2
    steps = model.sample_time(num_pairs, ligand_pos.device)
    position_noise = torch.empty_like(ligand_pos)
    type_noise = ligand_pos.new_empty((len(ligand_pos), model.num_classes))
    for pair_index in range(num_pairs):
        a = torch.where(ligand_batch == 2 * pair_index)[0]
        b = torch.where(ligand_batch == 2 * pair_index + 1)[0]
        count = max(len(a), len(b))
        positions = torch.randn((count, 3), device=ligand_pos.device)
        uniforms = torch.rand((count, model.num_classes), device=ligand_pos.device)
        for atoms in (a, b):
            position_noise[atoms] = positions[:len(atoms)]
            type_noise[atoms] = uniforms[:len(atoms)]
    return {
        "time_step": steps.repeat_interleave(2),
        "position_noise": position_noise,
        "categorical_uniform": type_noise,
    }


def save_checkpoint(model, config, step, base_sha256, output):
    import torch

    target = output / f"step_{step:05d}.pt"
    temporary = target.with_suffix(".pt.tmp")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({
        "format": "pocketpref-dpo-inference-v1",
        "model_config": config,
        "model": state,
        "training_step": step,
        "base_checkpoint_sha256": base_sha256,
    }, temporary)
    temporary.replace(target)


def train(args):
    import numpy as np
    import torch
    from easydict import EasyDict
    from torch.utils.data import DataLoader

    from datasets.preferences import JointPairSampler, PreferenceDataset, collate_pairs
    from models.sefmol import SeFMol

    if not 0 <= args.seed < 2**32:
        raise ValueError("Seed must be in [0, 2**32).")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = PreferenceDataset(args.data_dir)
    sampler = JointPairSampler(data.pairs, args.seed)
    loader = DataLoader(data, batch_sampler=sampler, collate_fn=collate_pairs, num_workers=0)
    batches = iter(loader)
    payload = torch.load(args.base, map_location="cpu", weights_only=True)
    if payload.get("format") != "pocketpref-dpo-inference-v1":
        raise ValueError("Use the exported Base checkpoint, ckpts/ft.pt.")
    config = payload["model_config"]
    state = payload["model"]

    def model_from_base():
        model = SeFMol(EasyDict(config), protein_atom_feature_dim=27,
                       ligand_atom_feature_dim=13, n_timesteps=1000)
        model.load_state_dict(state, strict=True)
        return model.to(args.device)

    policy = model_from_base().train()
    reference = model_from_base().eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    base_sha256 = hashlib.sha256(args.base.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "training.json").write_text(json.dumps({
        "base_checkpoint_sha256": base_sha256,
        "pair_manifest_sha256": hashlib.sha256((args.data_dir / "pairs.json").read_bytes()).hexdigest(),
        "pair_count": len(data),
        "seed": args.seed,
        "steps": args.steps,
        "beta": BETA,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
    }, indent=2) + "\n")
    with (args.output_dir / "train.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            try:
                batch = next(batches)
            except StopIteration:
                sampler.epoch += 1
                batches = iter(loader)
                batch = next(batches)
            batch = {key: value.to(args.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            kwargs = {**batch, **matched_noise(policy, batch)}
            kwargs["properties"] = batch["properties"].flatten()
            with torch.no_grad():
                baseline = reference.get_diffusion_loss(**kwargs)["loss_diffusion_graph"]
            current = policy.get_diffusion_loss(**kwargs)["loss_diffusion_graph"]
            if step == 1 and not torch.allclose(current, baseline, atol=1e-5, rtol=1e-5):
                raise RuntimeError("Policy and reference must agree at initialization.")
            gain = baseline - current
            gap = gain[::2] - gain[1::2]
            loss = torch.nn.functional.softplus(-BETA * gap).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite DPO loss at step {step}.")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step == args.steps or step % 50 == 0:
                line = json.dumps(dict(step=step, loss=float(loss.detach()), gap=float(gap.detach().mean()),
                                       grad_norm=float(grad_norm)))
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            if step == args.steps or step % args.save_every == 0:
                save_checkpoint(policy, config, step, base_sha256, args.output_dir)


if __name__ == "__main__":
    train(parse_args())
