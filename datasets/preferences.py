"""Portable tensors and within-pocket pairs for joint DPO."""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

KINDS = ("spatial", "chemical")
PAIR_FORMAT = "pocketpref-dpo-pairs-v1"
INPUT_FORMAT = "pocketpref-dpo-inputs-v1"


class PreferenceDataset:
    def __init__(self, directory):
        directory = Path(directory)
        document = json.loads((directory / "pairs.json").read_text())
        inputs = torch.load(directory / "inputs.pt", map_location="cpu", weights_only=True)
        if document["format"] != PAIR_FORMAT or inputs["format"] != INPUT_FORMAT:
            raise ValueError("Unsupported preference data format.")
        self.pairs = document["pairs"]
        self.pockets = inputs["pockets"]
        self.ligands = inputs["ligands"]
        for pair in self.pairs:
            pocket = pair["data_id"]
            a, b = pair["A"], pair["B"]
            if pair["preference_kind"] not in KINDS or a == b or pocket not in self.pockets:
                raise ValueError(f"Invalid preference pair: {pair!r}")
            if self.ligands[a]["pocket"] != pocket or self.ligands[b]["pocket"] != pocket:
                raise ValueError(f"Ligand/pocket identity mismatch for {a!r}, {b!r}.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        pocket = self.pockets[pair["data_id"]]
        return pocket, self.ligands[pair["A"]], self.ligands[pair["B"]]


class JointPairSampler:
    """Each batch draws one pair per preference kind, balancing pockets."""

    def __init__(self, pairs, seed):
        self.seed = seed
        self.epoch = 0
        self.batches = math.ceil(len(pairs) / len(KINDS))
        self.groups = {kind: defaultdict(list) for kind in KINDS}
        for index, pair in enumerate(pairs):
            self.groups[pair["preference_kind"]][pair["data_id"]].append(index)
        self.pockets = {kind: sorted(self.groups[kind]) for kind in KINDS}
        if not all(self.pockets.values()):
            raise ValueError("Joint training requires at least one pair of each preference kind.")

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = random.Random(self.seed + 1000003 * self.epoch)
        for _ in range(self.batches):
            indices = []
            for kind in KINDS:
                pocket = rng.choice(self.pockets[kind])
                indices.append(rng.choice(self.groups[kind][pocket]))
            rng.shuffle(indices)
            yield indices


def collate_pairs(pairs):
    pockets = [pocket for pocket, _, _ in pairs for _ in range(2)]
    ligands = [ligand for _, winner, loser in pairs for ligand in (winner, loser)]
    graph = torch.arange(len(ligands))
    return {
        "protein_pos": torch.cat([p["position"] for p in pockets]),
        "protein_v": torch.cat([p["features"] for p in pockets]),
        "batch_protein": graph.repeat_interleave(torch.tensor([len(p["position"]) for p in pockets])),
        "properties": torch.stack([p["properties"] for p in pockets]),
        "ligand_pos": torch.cat([ligand["position"] for ligand in ligands]),
        "ligand_v": torch.cat([ligand["types"] for ligand in ligands]),
        "batch_ligand": graph.repeat_interleave(torch.tensor([len(ligand["types"]) for ligand in ligands])),
    }
