"""Encode pocket PDB atoms for the SeFMol backbone."""

import torch
import torch.nn.functional as F

from utils.data import PDBProtein

PROTEIN_ELEMENTS = (1, 6, 7, 8, 16, 34)


def featurize(pocket):
    data = PDBProtein(str(pocket), mode='path').to_dict_atom()
    positions = torch.from_numpy(data['pos'])
    if positions.ndim != 2 or positions.shape[0] < 2 or not torch.isfinite(positions).all():
        raise ValueError('The pocket PDB must contain at least two protein atoms with finite coordinates.')
    elements = torch.from_numpy(data['element'])[:, None] == torch.tensor(PROTEIN_ELEMENTS)
    residue = F.one_hot(torch.from_numpy(data['atom_to_aa_type']), num_classes=20)
    backbone = torch.from_numpy(data['is_backbone'])[:, None]
    return positions, torch.cat((elements, residue, backbone), dim=-1).float()
