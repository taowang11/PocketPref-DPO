"""Build DPO inputs and fixed-pose response preferences from candidate SDFs."""

import argparse
import json
import math
import re
import tempfile
from pathlib import Path

MARGINS = {"spatial": 1.0, "chemical": 0.7}
QUALITY_MIN = 0.5
MAX_OVERLAP = 0.8


def rank_pairs(rows, pocket, kind, site):
    """Deduplicate complete scores, rank by E(P′) − E(P), then pair extremes."""
    by_smiles = {}
    for row in sorted(rows, key=lambda r: r["key"]):
        response = row["edited"] - row["original"]
        by_smiles.setdefault(row["smiles"], (row["key"], response))
    ranking = sorted(by_smiles.values(), key=lambda item: (-item[1], item[0]))
    pairs = []
    for index in range(len(ranking) // 2):
        a, b = ranking[index], ranking[-1 - index]
        if a[1] - b[1] >= MARGINS[kind] - 1e-9:
            pairs.append(dict(data_id=pocket, preference_kind=kind, site=site,
                              A=a[0], B=b[0], A_loss=a[1], B_loss=b[1], gap=a[1] - b[1]))
    return pairs


def remove_conflicts(pairs):
    directions = {(p["data_id"], p["A"], p["B"]) for p in pairs}
    seen = set()
    kept = []
    for pair in pairs:
        pocket, a, b = pair["data_id"], pair["A"], pair["B"]
        identity = (pair["preference_kind"], pocket, a, b)
        if (pocket, b, a) not in directions and identity not in seen:
            kept.append(pair)
            seen.add(identity)
    return kept


def score_site(item, resolve, site_number, output):
    import numpy as np
    import torch
    from rdkit import Chem
    from rdkit.Chem import QED
    from datasets.pocket import featurize
    from generate import LIGAND_TYPES, PROPERTY_TARGETS
    from utils.evaluation import fixed_vina
    from utils.evaluation.chemistry import normalized_sa
    from utils.evaluation.receptor_prep import assert_heavy_atom_coordinates_preserved
    from utils.pocket_edits import edit_pocket

    kind = item["kind"]
    if kind not in MARGINS:
        raise ValueError(f"Unknown intervention kind {kind!r}.")
    pocket = str(item["pocket"])
    if not re.fullmatch(r"[\w.-]+", pocket):
        raise ValueError(f"Invalid pocket ID {pocket!r}.")
    original = resolve(item["receptor"])
    source_sdf = resolve(item["candidates"])
    site_id = f"{pocket}:{site_number}"
    edited = output / "edited_pockets" / f"{pocket}_{site_number}.pdb"
    label = edit_pocket(original, edited, item["site"], kind)
    if item.get("intervention") not in (None, label):
        raise ValueError(f"Unexpected intervention at {site_id}: {label}.")
    position, features = featurize(original)
    properties = torch.tensor(item.get("properties", PROPERTY_TARGETS), dtype=torch.float32)
    if properties.shape != (8,) or not torch.isfinite(properties).all():
        raise ValueError(f"Expected eight finite property targets for {pocket}.")
    receptor = {"position": position, "features": features, "properties": properties}
    steric = [fixed_vina.receptor_heavy(path) for path in (original, edited)]
    ligand_types = {value: index for index, value in enumerate(LIGAND_TYPES)}

    accepted, errors, seen = [], [], set()
    for index, mol in enumerate(Chem.SDMolSupplier(str(source_sdf), removeHs=True)):
        if mol is None or mol.GetNumConformers() != 1 or len(Chem.GetMolFrags(mol)) != 1:
            errors.append(dict(record=index, reason="invalid_or_disconnected"))
            continue
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        local_id = mol.GetProp("candidate_key") if mol.HasProp("candidate_key") else str(index)
        key = f"{pocket}:{local_id}"
        if key in seen:
            raise ValueError(f"Repeated candidate ID {key!r} in {source_sdf}.")
        seen.add(key)
        xyz = mol.GetConformer().GetPositions()
        try:
            types = [ligand_types[(atom.GetAtomicNum(), atom.GetIsAromatic())] for atom in mol.GetAtoms()]
        except KeyError:
            errors.append(dict(key=key, reason="unsupported_atom_type"))
            continue
        if not np.isfinite(xyz).all():
            errors.append(dict(key=key, reason="nonfinite_coordinates"))
            continue
        if min(QED.qed(mol), normalized_sa(smiles)) < QUALITY_MIN:
            errors.append(dict(key=key, reason="quality"))
            continue
        if max(fixed_vina.maximum_overlap(mol, state) for state in steric) > MAX_OVERLAP:
            errors.append(dict(key=key, reason="initial_overlap"))
            continue
        data = {"pocket": pocket, "position": torch.tensor(xyz, dtype=torch.float32),
                "types": torch.tensor(types, dtype=torch.long)}
        accepted.append((key, smiles, mol, xyz, data))

    scores, ligands = [], {}
    box = item.get("box")
    if accepted:
        if box is None:
            xyz = np.concatenate([row[3] for row in accepted])
            lower, upper = xyz.min(0), xyz.max(0)
            box = {"center": ((lower + upper) / 2).tolist(),
                   "size": np.maximum(upper - lower + 12, 20).tolist()}
        center = np.asarray(box["center"], dtype=float)
        size = np.asarray(box["size"], dtype=float)
        if center.shape != (3,) or size.shape != (3,) or not np.isfinite([center, size]).all() or (size <= 0).any():
            raise ValueError(f"Invalid Vina box at {site_id}.")
        for row in accepted:
            if np.any(np.abs(row[3] - center) >= size / 2):
                raise ValueError(f"The box at {site_id} does not contain {row[0]!r}.")
        box = {"center": center.tolist(), "size": size.tolist()}
        with tempfile.TemporaryDirectory(dir=output) as temp:
            work = Path(temp)
            prepared = item.get("pdbqt", {})
            engines = []
            for name, pdb in (("original", original), ("edited", edited)):
                receptor_pdbqt = resolve(prepared[name]) if name in prepared else (
                    fixed_vina.prepare_receptor(pdb, work / name))
                assert_heavy_atom_coordinates_preserved(pdb, receptor_pdbqt, allow_extra=False)
                engines.append(fixed_vina.engine(receptor_pdbqt, box, 20260831))
            for index, (key, smiles, mol, _, data) in enumerate(accepted):
                try:
                    ligand = work / f"ligand_{index}.pdbqt"
                    fixed_vina.prepare_ligand(mol, ligand)
                    values = []
                    for vina in engines:
                        vina.set_ligand_from_file(str(ligand))
                        values.append(float(vina.score()[0]))
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError("Nonfinite Vina score.")
                except (ValueError, RuntimeError, OSError, KeyError) as error:
                    errors.append(dict(key=key, reason=str(error)))
                    continue
                scores.append(dict(key=key, smiles=smiles, original=values[0], edited=values[1]))
                ligands[key] = data
    pairs = rank_pairs(scores, pocket, kind, site_id)
    report = dict(pocket=pocket, kind=kind, site=site_id, intervention=label,
                  box=box, pairs_before_conflicts=len(pairs), scores=scores, exclusions=errors)
    return pocket, receptor, ligands, pairs, report


def build(manifest, output):
    import torch
    from datasets.preferences import INPUT_FORMAT, PAIR_FORMAT

    manifest = Path(manifest)
    sites = json.loads(manifest.read_text())["sites"]
    if not sites:
        raise ValueError("The manifest must supply at least one edit site.")

    def resolve(value):
        return manifest.parent / value

    output = Path(output)
    output.mkdir(parents=True)
    (output / "edited_pockets").mkdir()
    pairs, pockets, ligands, reports = [], {}, {}, []
    for number, item in enumerate(sites):
        pocket, receptor, candidates, selected, report = score_site(item, resolve, number, output)
        if pocket in pockets and any(not torch.equal(receptor[key], pockets[pocket][key]) for key in receptor):
            raise ValueError(f"Pocket input changed across sites for {pocket!r}.")
        pockets[pocket] = receptor
        for key, candidate in candidates.items():
            if key in ligands and any(not torch.equal(candidate[k], ligands[key][k]) for k in ("position", "types")):
                raise ValueError(f"Candidate coordinates or types changed across sites for {key!r}.")
            ligands[key] = candidate
        pairs.extend(selected)
        reports.append(report)
        print(f"{report['site']}: {len(report['scores'])} scored, {len(selected)} pairs", flush=True)
    pairs = remove_conflicts(pairs)
    (output / "build_report.json").write_text(json.dumps({"sites": reports, "kept_pairs": len(pairs)}, indent=2) + "\n")
    if not pairs:
        raise ValueError(f"No pairs meet the selection rules. See {output / 'build_report.json'}.")
    candidate_ids = sorted({pair[field] for pair in pairs for field in ("A", "B")})
    pocket_ids = sorted({pair["data_id"] for pair in pairs})
    torch.save(dict(format=INPUT_FORMAT,
                    pockets={key: pockets[key] for key in pocket_ids},
                    ligands={key: ligands[key] for key in candidate_ids}), output / "inputs.pt")
    (output / "pairs.json").write_text(json.dumps(dict(format=PAIR_FORMAT, pairs=pairs), indent=2) + "\n")
    print(f"Saved {len(pairs)} preferences in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    build(args.manifest, args.output_dir)
