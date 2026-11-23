"""Chemical metrics shared by preference construction and evaluation."""

import math
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


@lru_cache(maxsize=1)
def rdkit_sa_scorer():
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return sascorer
    except ModuleNotFoundError:
        from importlib.util import module_from_spec, spec_from_file_location
        from rdkit import RDConfig

        path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
        if not path.is_file():
            raise RuntimeError(f"RDKit SA_Score is missing at {path}.")
        spec = spec_from_file_location("sascorer", path)
        sascorer = module_from_spec(spec)
        spec.loader.exec_module(sascorer)
        return sascorer


def normalized_sa(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return round((10 - rdkit_sa_scorer().calculateScore(mol)) / 9, 2)


def descriptors(mol, smiles):
    weight = Descriptors.ExactMolWt(mol)
    logp = Crippen.MolLogP(mol)
    donors = Lipinski.NumHDonors(mol)
    acceptors = Lipinski.NumHAcceptors(mol)
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return {
        "qed": QED.qed(mol),
        "sa": normalized_sa(smiles),
        "lipinski": sum((weight < 500, donors <= 5, acceptors <= 10, -2 <= logp <= 5, rotatable <= 10)),
        "logp": logp,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mw": Descriptors.MolWt(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rings": mol.GetRingInfo().NumRings(),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "rotatable_bonds": rotatable,
    }


def internal_diversity(mols):
    # SeFMol's N² denominator includes ordered pairs and zero self-distances.
    fingerprints = [Chem.RDKFingerprint(mol) for mol in mols]
    n = len(fingerprints)
    if n == 0:
        return None
    total = math.fsum(
        n_i - math.fsum(DataStructs.BulkTanimotoSimilarity(fp, fingerprints[:n_i]))
        for n_i, fp in enumerate(fingerprints)
    )
    return 2 * total / n**2
