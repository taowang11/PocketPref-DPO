"""Structure checks, interaction recovery, ring occurrence, and shared Morgan PCA."""

from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist

CLASH_OVERLAP = 0.8
PCA_SEED = 20261026
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
RING_BINS = ("3", "4", "5", "6", "7", "8+")
INTERACTIONS = [
    "HBDonor", "HBAcceptor", "Hydrophobic", "PiStacking", "Cationic",
    "Anionic", "CationPi", "PiCation", "XBDonor", "XBAcceptor",
]
STRUCTURE_METRICS = (
    "posebusters_pass", "prolif_interaction_count", "prolif_recovery",
    "prolif_complete_recovery", "prolif_jaccard",
    "clash_pairs", "clash_atom_fraction", "max_overlap_A",
)
ADT_ELEMENTS = {
    "A": "C", "NA": "N", "NS": "N", "OA": "O", "OS": "O", "SA": "S",
    "CL": "Cl", "BR": "Br",
}
STANDARD_RESIDUES = set((
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
    "HID HIE HIP CYX CYM ASH GLH LYN HSD HSE HSP"
).split())


def heavy_receptor(path):
    table = Chem.GetPeriodicTable()
    positions, radii = [], []
    is_pdbqt = path.suffix.lower() == ".pdbqt"
    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in " A":
            continue
        element = line.split()[-1] if is_pdbqt else line[76:78].strip()
        if element.upper() in {"H", "D", "HD", "HS"}:
            continue
        if not element:
            raise ValueError(f"Missing receptor element at {path}, atom {line[6:11].strip()}.")
        element = ADT_ELEMENTS.get(element.upper(), element.capitalize()) if is_pdbqt else element.capitalize()
        positions.append([float(line[offset:offset + 8]) for offset in (30, 38, 46)])
        radii.append(table.GetRvdw(table.GetAtomicNumber(element)))
    if not positions:
        raise ValueError(f"No heavy receptor atoms in {path}.")
    xyz = np.asarray(positions)
    if not np.isfinite(xyz).all():
        raise ValueError(f"Nonfinite receptor coordinates in {path}.")
    return xyz, np.asarray(radii)


def clash_metrics(mol, receptor):
    protein_xyz, protein_radii = receptor
    table = Chem.GetPeriodicTable()
    heavy = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if not heavy:
        raise ValueError("No ligand heavy atoms for steric evaluation.")
    ligand_xyz = mol.GetConformer().GetPositions()[[atom.GetIdx() for atom in heavy]]
    ligand_radii = [table.GetRvdw(atom.GetAtomicNum()) for atom in heavy]
    overlap = np.asarray(ligand_radii)[:, None] + protein_radii - cdist(ligand_xyz, protein_xyz)
    clashing = overlap > CLASH_OVERLAP
    return {
        "clash_pairs": int(clashing.sum()),
        "clash_atom_fraction": float(clashing.any(axis=1).mean()),
        "max_overlap_A": float(max(0.0, overlap.max())),
    }


def write_protein_input(source, target):
    """Prepare a protein-only heavy-atom PDB without changing coordinates."""
    lines, mapping, translated = [], {}, {}
    for line in source.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in " A":
            continue
        name, chain = line[17:20].strip(), line[21].strip()
        element = line[76:78].strip().upper()
        if name not in STANDARD_RESIDUES or element in {"H", "D"}:
            continue
        number = int(line[22:26])
        converted = 9000 - number if number < 0 else number
        if converted > 9999:
            raise ValueError(f"Residue number {number} cannot be represented in the prepared PDB.")
        key = (chain, converted, line[26])
        if translated.setdefault(key, number) != number:
            raise ValueError(f"Residue number collision after ProLIF preparation in {source}.")
        if number < 0:
            suffix = f".{chain}" if chain else ""
            mapping[f"{converted}{suffix}"] = f"{number}{suffix}"
            line = line[:22] + f"{converted:4d}" + line[26:]
        lines.append(line)
    if not lines:
        raise ValueError(f"No standard protein residues for ProLIF in {source}.")
    target.write_text("\n".join(lines) + "\nEND\n")
    return mapping


class StructureEvaluation:
    """Load shared PoseBusters and ProLIF configuration once per evaluation."""

    def __init__(self):
        from posebusters import PoseBusters
        from posebusters.tools.loading import safe_load_mol
        from prolif import Fingerprint, Molecule
        from prolif.io import MoleculeStandardizer

        # Parallelism is at the evaluator's outer process level.
        self.posebusters = PoseBusters(config="dock", max_workers=0)
        self.fingerprint = Fingerprint(interactions=INTERACTIONS, implicit_hydrogens=True)
        self.standardize = MoleculeStandardizer()
        self.prolif_molecule = Molecule.from_rdkit
        self.load_receptor = safe_load_mol

    def features(self, mol, context):
        pattern = self.fingerprint.generate(self.prolif_molecule(mol), context["protein"], metadata=True)
        labels = context["residue_labels"]
        features = set()
        for pair, interactions in pattern.items():
            residue = pair[1]
            suffix = f".{residue.chain}" if residue.chain else ""
            identity = f"{residue.number}{suffix}"
            label = f"{residue.name}{labels.get(identity, identity)}"
            features.update(f"{label}|{kind.removeprefix('Implicit')}" for kind in interactions)
        return features

    def prepare(self, receptor, clash_receptor, prolif_receptor, reference, work):
        context = {"errors": {}, "reference": None}
        try:
            # Loading with PoseBusters' own receptor options preserves its atom treatment.
            parameters = self.posebusters.config["loading"]["mol_cond"]
            context["cond"] = self.load_receptor(receptor, **parameters)
            if context["cond"] is None:
                raise ValueError(f"Could not load {receptor} for PoseBusters.")
        except (OSError, ValueError, RuntimeError) as error:
            context["errors"]["posebusters"] = str(error)
        try:
            context["steric"] = heavy_receptor(clash_receptor)
        except (OSError, ValueError, RuntimeError) as error:
            context["errors"]["steric"] = str(error)
        try:
            prepared = work / "protein.pdb"
            context["residue_labels"] = write_protein_input(prolif_receptor, prepared)
            context["protein"] = self.standardize(prepared)
        except (OSError, ValueError, RuntimeError, TypeError, AssertionError) as error:
            context["errors"]["prolif"] = str(error)
        if reference is not None and "protein" in context:
            try:
                context["reference"] = self.features(reference, context)
            except (ValueError, RuntimeError, TypeError, AssertionError, KeyError) as error:
                context["errors"]["prolif_reference"] = str(error)
        return context

    def measure(self, mol, context, row):
        errors = row["errors"]
        errors.update(context["errors"])
        if "cond" in context and context["cond"] is not None:
            try:
                checks = self.posebusters.bust(
                    mol_pred=Chem.Mol(mol), mol_cond=context["cond"], full_report=False
                ).iloc[0].to_dict()
                checks = {key: bool(value) if isinstance(value, (bool, np.bool_)) else None
                          for key, value in checks.items()}
                row["posebusters_checks"] = checks
                if len(checks) < 15 or any(value is None for value in checks.values()):
                    raise ValueError("Missing applicable PoseBusters dock checks.")
                row["posebusters_pass"] = int(all(checks.values()))
            except (ValueError, RuntimeError, IndexError, TypeError, AssertionError) as error:
                errors["posebusters"] = str(error)
        if "protein" in context:
            try:
                features = self.features(Chem.Mol(mol), context)
                row["prolif_features"] = sorted(features)
                row["prolif_interaction_count"] = len(features)
                reference = context["reference"]
                if reference is not None:
                    row["prolif_reference_count"] = len(reference)
                    if reference:
                        hits = len(reference & features)
                        row["prolif_recovered_count"] = hits
                        row["prolif_recovery"] = hits / len(reference)
                        row["prolif_complete_recovery"] = int(reference <= features)
                        row["prolif_jaccard"] = hits / len(reference | features)
                    else:
                        errors["prolif_recovery"] = "Reference interaction set is empty."
            except (ValueError, RuntimeError, TypeError, AssertionError, KeyError) as error:
                errors["prolif"] = str(error)
        if "steric" in context:
            try:
                row.update(clash_metrics(mol, context["steric"]))
            except (ValueError, RuntimeError) as error:
                errors["steric"] = str(error)


def ring_sizes(mol):
    return sorted(len(ring) for ring in mol.GetRingInfo().AtomRings())


class ChemicalSpace:
    def __init__(self, capacity):
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS, fpSize=MORGAN_BITS, includeChirality=False,
        )
        self.fingerprints = np.empty((capacity, MORGAN_BITS), dtype=np.uint8)
        self.records = []
        self.rings = defaultdict(list)

    def add(self, row, mol, role="generated"):
        model, pocket = row["model"], row["pocket"]
        self.records.append(dict(model=model, pocket=pocket, attempt=row["attempt"], role=role))
        features = self.generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(features, self.fingerprints[len(self.records) - 1])
        sizes = ring_sizes(mol)
        self.rings[model, pocket, role].append(sizes)
        return sizes

    def ring_tables(self):
        per_pocket, by_model = [], defaultdict(list)
        for (model, pocket, role), sizes in self.rings.items():
            for key in RING_BINS:
                bin_size = 8 if key == "8+" else int(key)
                matches = sum(any(min(n, 8) == bin_size for n in rings) for rings in sizes)
                rate = matches / len(sizes)
                per_pocket.append(dict(model=model, pocket=pocket, role=role, ring_size=key,
                                       molecules=len(sizes), containing_ring=matches, occurrence=rate))
                by_model[model, role, key].append(rate)
        summary = [
            dict(model=model, role=role, ring_size=key, pockets=len(values), pocket_equal_occurrence=fmean(values))
            for (model, role, key), values in by_model.items()
        ]
        return per_pocket, summary

    def pca(self):
        n = len(self.records)
        if not n:
            return [], dict(records=0, explained_variance_ratio=[None, None]), None
        matrix = self.fingerprints[:n].astype(np.float32)
        center = matrix.mean(0)
        matrix -= center
        total = float(np.einsum("ij,ij->", matrix, matrix, dtype=np.float64))
        coords = np.zeros((n, 2))
        components = np.zeros((2, MORGAN_BITS))
        variance = np.zeros(2)
        if total:
            if n > 2:
                u, s, vt = svds(matrix, k=2, which="LM", tol=1e-5, random_state=PCA_SEED)
                order = np.argsort(s)[::-1]
                u, s, vt = u[:, order], s[order], vt[order]
            else:
                u, s, vt = np.linalg.svd(matrix, full_matrices=False)
            for index in range(len(s)):
                pivot = vt[index, np.argmax(np.abs(vt[index]))]
                sign = 1 if pivot >= 0 else -1
                coords[:, index] = u[:, index] * s[index] * sign
                components[index] = vt[index] * sign
                variance[index] = float(s[index])**2 / total
        rows = [dict(row, pc1=float(point[0]), pc2=float(point[1]))
                for row, point in zip(self.records, coords)]
        metadata = dict(records=n, radius=MORGAN_RADIUS, bits=MORGAN_BITS, chirality=False,
                        seed=PCA_SEED, explained_variance_ratio=variance.tolist())
        return rows, metadata, {"center": center, "components": components, "explained_variance_ratio": variance}
