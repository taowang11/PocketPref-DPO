"""Validation helpers for rigid-receptor preparation."""

from collections import defaultdict

import numpy as np


_PDB_CHAIN_IDS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def _is_heavy_atom(atom_name):
    element = atom_name.lstrip("0123456789")[:1].upper()
    return element not in {"H", "D"}


def read_heavy_atom_coordinates(path):
    """Read heavy-atom identities and coordinates from a PDB-like file."""
    atoms = []
    with open(path) as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if not _is_heavy_atom(atom_name):
                continue
            key = (
                line[21:22].strip(),
                line[22:26].strip(),
                line[26:27].strip(),
                atom_name,
            )
            try:
                coordinates = np.asarray(
                    [line[30:38], line[38:46], line[46:54]], dtype=float)
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid receptor coordinates for {key!r} in {path}") from exc
            atoms.append((key, coordinates))
    if not atoms:
        raise RuntimeError(f"No receptor heavy atoms found in {path}")
    return atoms


def assert_heavy_atom_coordinates_preserved(source_path, prepared_path,
                                            tolerance=0.01,
                                            allow_extra=True):
    """Reject receptor preparation that removes or moves source heavy atoms."""
    source_atoms = read_heavy_atom_coordinates(source_path)
    prepared_atoms = read_heavy_atom_coordinates(prepared_path)

    prepared_by_name = defaultdict(list)
    for key, coordinates in prepared_atoms:
        prepared_by_name[key[-1]].append((key, coordinates))

    missing = []
    displaced = []
    matched_distances = []
    for key, source_coordinates in source_atoms:
        candidates = prepared_by_name[key[-1]]
        if not candidates:
            missing.append(key)
            continue
        distances = [
            float(np.linalg.norm(source_coordinates - coordinates))
            for _, coordinates in candidates
        ]
        closest_index = int(np.argmin(distances))
        distance = distances[closest_index]
        candidates.pop(closest_index)
        matched_distances.append(distance)
        if distance > tolerance:
            displaced.append((distance, key))

    if missing:
        preview = ", ".join(repr(key) for key in missing[:5])
        raise RuntimeError(
            f"Prepared receptor {prepared_path} is missing {len(missing)} "
            f"source heavy atoms: {preview}")
    if displaced:
        displaced.sort(reverse=True)
        preview = ", ".join(
            f"{key!r}={distance:.3f}A"
            for distance, key in displaced[:5]
        )
        raise RuntimeError(
            f"Prepared receptor {prepared_path} moved "
            f"{len(displaced)} source heavy atoms beyond {tolerance:.3f}A: "
            f"{preview}")

    extra_count = len(prepared_atoms) - len(source_atoms)
    if not allow_extra and extra_count:
        raise RuntimeError(
            f"Prepared receptor {prepared_path} contains {extra_count} "
            "heavy atoms absent from the source receptor")

    return {
        "source_heavy_atoms": len(source_atoms),
        "prepared_heavy_atoms": len(prepared_atoms),
        "max_displacement": max(matched_distances),
    }


def filter_pqr_to_source_heavy_atoms(source_path, prepared_pqr_path,
                                     output_path, tolerance=0.01,
                                     max_hydrogen_bond=1.35):
    """Remove PDB2PQR-repaired heavy atoms and their attached hydrogens."""
    source_atoms = read_heavy_atom_coordinates(source_path)
    with open(prepared_pqr_path) as handle:
        lines = handle.readlines()

    prepared_records = []
    prepared_heavy = []
    for line_index, line in enumerate(lines):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        coordinates = np.asarray(
            [line[30:38], line[38:46], line[46:54]], dtype=float)
        hydrogen = not _is_heavy_atom(atom_name)
        record = (line_index, atom_name, coordinates, hydrogen)
        prepared_records.append(record)
        if not hydrogen:
            prepared_heavy.append(record)

    available_by_name = defaultdict(list)
    for heavy_index, (_, atom_name, coordinates, _) in enumerate(
            prepared_heavy):
        available_by_name[atom_name].append((heavy_index, coordinates))

    keep_heavy = set()
    for key, source_coordinates in source_atoms:
        atom_name = key[-1]
        candidates = available_by_name[atom_name]
        if not candidates:
            raise RuntimeError(
                f"Prepared receptor {prepared_pqr_path} is missing source "
                f"heavy atom {key!r}")
        distances = [
            float(np.linalg.norm(source_coordinates - coordinates))
            for _, coordinates in candidates
        ]
        closest_index = int(np.argmin(distances))
        distance = distances[closest_index]
        heavy_index, _ = candidates.pop(closest_index)
        if distance > tolerance:
            raise RuntimeError(
                f"Prepared receptor {prepared_pqr_path} moved source heavy "
                f"atom {key!r} by {distance:.3f}A")
        keep_heavy.add(heavy_index)

    keep_lines = set()
    for heavy_index in keep_heavy:
        keep_lines.add(prepared_heavy[heavy_index][0])

    heavy_coordinates = [record[2] for record in prepared_heavy]
    for line_index, _, coordinates, hydrogen in prepared_records:
        if not hydrogen:
            continue
        distances = [
            float(np.linalg.norm(coordinates - heavy_coordinates_i))
            for heavy_coordinates_i in heavy_coordinates
        ]
        parent_index = int(np.argmin(distances))
        if (parent_index in keep_heavy and
                distances[parent_index] <= max_hydrogen_bond):
            keep_lines.add(line_index)

    with open(output_path, "w") as handle:
        for line_index, line in enumerate(lines):
            if (not line.startswith(("ATOM  ", "HETATM")) or
                    line_index in keep_lines):
                handle.write(line)

    return assert_heavy_atom_coordinates_preserved(
        source_path, output_path, tolerance=tolerance, allow_extra=False)


def write_pdb_with_unique_chains(source_path, output_path):
    """Map each source chain/segment pair to a unique PDB chain ID.

    CrossDocked assembly files can store symmetry-copy identity in the PDB
    segment field while reusing the same chain and residue identifiers.
    PDB2PQR otherwise merges those distinct residues.
    """
    with open(source_path) as handle:
        lines = handle.readlines()

    chain_keys = []
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        original_chain = line[21:22].strip()
        segment = line[72:76].strip() if len(line) >= 76 else ""
        key = (original_chain, segment)
        if key not in chain_keys:
            chain_keys.append(key)
    if len(chain_keys) > len(_PDB_CHAIN_IDS):
        raise RuntimeError(
            f"Cannot represent {len(chain_keys)} receptor chain/segment "
            "pairs with one-character PDB chain IDs")

    used = set()
    chain_map = {}
    for key in chain_keys:
        original_chain, segment = key
        preferred = [segment[-1:] or "", original_chain]
        chain_id = next(
            (candidate for candidate in preferred
             if candidate and candidate in _PDB_CHAIN_IDS
             and candidate not in used),
            None,
        )
        if chain_id is None:
            chain_id = next(
                candidate for candidate in _PDB_CHAIN_IDS
                if candidate not in used)
        chain_map[key] = chain_id
        used.add(chain_id)

    normalized = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            original_chain = line[21:22].strip()
            segment = line[72:76].strip() if len(line) >= 76 else ""
            line = line[:21] + chain_map[(original_chain, segment)] + line[22:]
        normalized.append(line)
    with open(output_path, "w") as handle:
        handle.writelines(normalized)
    return chain_map
