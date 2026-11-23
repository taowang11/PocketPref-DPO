"""Fixed-pose Vina 1.2.7 with preserved heavy-atom coordinates."""

import contextlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist

from utils.evaluation.receptor_prep import (
    assert_heavy_atom_coordinates_preserved,
    filter_pqr_to_source_heavy_atoms,
    write_pdb_with_unique_chains,
)

PDBQT_ELEMENTS = {"A": "C", "C": "C", "N": "N", "NA": "N", "OA": "O",
                  "O": "O", "S": "S", "SA": "S", "P": "P", "F": "F",
                  "Cl": "Cl", "Br": "Br", "I": "I"}


def run_preparation(command, output):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode or not output.is_file() or not output.stat().st_size:
        raise RuntimeError(f"{command[0]} did not prepare {output.name}: {result.stderr[-1500:]}")


def prepare_receptor(source, directory):
    import AutoDockTools

    directory.mkdir()
    fixed = directory / "unique_chains.pdb"
    raw = directory / "raw.pqr"
    filtered = directory / "source_heavy.pqr"
    output = directory / "receptor.pdbqt"
    write_pdb_with_unique_chains(source, fixed)
    pdb2pqr = os.environ.get("PDB2PQR30_EXECUTABLE", "pdb2pqr30")
    run_preparation([pdb2pqr, "--ff=AMBER", "--nodebump", "--noopt", "--keep-chain",
                     str(fixed), str(raw)], raw)
    filter_pqr_to_source_heavy_atoms(source, raw, filtered)
    adt = Path(AutoDockTools.__path__[0]) / "Utilities24/prepare_receptor4.py"
    run_preparation([sys.executable, str(adt), "-r", str(filtered), "-o", str(output)], output)
    assert_heavy_atom_coordinates_preserved(source, output, allow_extra=False)
    return output


def prepare_ligand(mol, output):
    from meeko import MoleculePreparation
    from openbabel import pybel

    prepared = Chem.AddHs(mol, addCoords=True)
    # Use the Open Babel-backed Meeko path from the historical score protocol.
    babel = pybel.readstring("sdf", Chem.MolToMolBlock(prepared) + "\n$$$$\n")
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        preparator = MoleculePreparation()
        preparator.prepare(babel.OBMol)
        preparator.write_pdbqt_file(str(output))
    xyz, elements = [], []
    for line in output.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            atom_type = line.split()[-1]
            if atom_type not in {"H", "HD", "HS"}:
                elements.append(PDBQT_ELEMENTS[atom_type])
                xyz.append([float(line[offset:offset + 8]) for offset in (30, 38, 46)])
    xyz = np.asarray(xyz)
    original = mol.GetConformer().GetPositions()
    expected = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if sorted(elements) != sorted(expected):
        raise ValueError("PDBQT preparation changed the set of heavy atoms.")
    for element in set(elements):
        first = xyz[np.array(elements) == element]
        second = original[np.array(expected) == element]
        distances = cdist(first, second)
        if max(distances.min(0).max(), distances.min(1).max()) > 0.002:
            raise ValueError("PDBQT preparation changed the input heavy-atom coordinates.")


def engine(receptor, box, seed, ligand=None):
    from vina import Vina

    vina = Vina(sf_name="vina", seed=seed, cpu=1, verbosity=0)
    vina.set_receptor(str(receptor))
    if ligand is not None:
        vina.set_ligand_from_file(str(ligand))
    vina.compute_vina_maps(center=box["center"], box_size=box["size"])
    return vina


def receptor_heavy(path):
    table = Chem.GetPeriodicTable()
    positions, radii = [], []
    for line in Path(path).read_text().splitlines():
        if not line.startswith("ATOM  ") or line[16] not in " A":
            continue
        element = line[76:78].strip().capitalize()
        if element not in {"", "H", "D"}:
            positions.append([float(line[offset:offset + 8]) for offset in (30, 38, 46)])
            radii.append(table.GetRvdw(table.GetAtomicNumber(element)))
    if not positions:
        raise ValueError(f"No heavy receptor atoms in {path}.")
    return np.asarray(positions), np.asarray(radii)


def maximum_overlap(mol, receptor):
    positions, radii = receptor
    table = Chem.GetPeriodicTable()
    ligand_radii = np.asarray([table.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()])
    return float((ligand_radii[:, None] + radii - cdist(mol.GetConformer().GetPositions(), positions)).max())
