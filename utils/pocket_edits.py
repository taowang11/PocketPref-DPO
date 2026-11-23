"""Local side-chain edits on sites selected before scoring."""

from pathlib import Path

import numpy as np

BACKBONE = {"N", "CA", "C", "O", "OXT"}
# Replacement residue, converted atom, parent, replacement atom and element,
# idealized bond length, and maximum radial displacement in Å.
MUTATIONS = {
    "ASP": ("ASN", "OD2", "CG", "ND2", "N", 1.33, 0.25),
    "GLU": ("GLN", "OE2", "CD", "NE2", "N", 1.33, 0.25),
    "SER": ("CYS", "OG", "CB", "SG", "S", 1.81, 0.50),
    "THR": ("VAL", "OG1", "CB", "CG1", "C", 1.53, 0.25),
    "ASN": ("ASP", "ND2", "CG", "OD2", "O", 1.25, 0.25),
    "GLN": ("GLU", "NE2", "CD", "OE2", "O", 1.25, 0.25),
    "TYR": ("PHE", "OH", "CZ", None, None, 1.36, 0.25),
}

SIDE_CHAINS = {
    "ASP": {"CB", "CG", "OD1", "OD2"},
    "ASN": {"CB", "CG", "OD1", "ND2"},
    "GLU": {"CB", "CG", "CD", "OE1", "OE2"},
    "GLN": {"CB", "CG", "CD", "OE1", "NE2"},
    "SER": {"CB", "OG"},
    "THR": {"CB", "OG1", "CG2"},
    "TYR": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
}


def atom_position(line):
    return np.array([float(line[offset:offset + 8]) for offset in (30, 38, 46)])


def with_position(line, xyz):
    return line[:30] + "".join(f"{value:8.3f}" for value in xyz) + line[54:]


def edit_pocket(original, output, site, kind):
    lines = Path(original).read_text().splitlines(keepends=True)
    selected = (site["chain"], str(site["residue"]), site.get("insertion", " "))
    residue_lines = [i for i, line in enumerate(lines) if line.startswith("ATOM  ")
                     and (line[21], line[22:26].strip(), line[26]) == selected]
    if not residue_lines:
        raise ValueError(f"Residue {selected!r} is absent from {original}.")
    names = {lines[i][17:20].strip() for i in residue_lines}
    if len(names) != 1:
        raise ValueError(f"Residue {selected!r} has ambiguous identities.")
    residue_name = names.pop()
    indices = {}
    for i in residue_lines:
        if lines[i][76:78].strip().upper() in {"H", "D"}:
            lines[i] = ""
            continue
        name = lines[i][12:16].strip()
        if name in indices or lines[i][16] not in " A":
            raise ValueError(f"Resolve alternate locations for residue {selected!r}.")
        indices[name] = i
    coordinates = {name: atom_position(lines[i]) for name, i in indices.items()}

    if kind == "spatial":
        if residue_name == "PRO":
            raise ValueError("A single χ1 rotation cannot preserve the proline ring.")
        moving = indices.keys() - BACKBONE - {"CB"}
        if not {"CA", "CB"} <= indices.keys() or not moving:
            raise ValueError(f"Residue {selected!r} has no editable distal side chain.")
        center = coordinates["CB"]
        axis = center - coordinates["CA"]
        length = np.linalg.norm(axis)
        angle = float(site["angle_deg"])
        if length == 0 or not np.isfinite(angle) or angle % 360 == 0:
            raise ValueError("Use a finite, nonzero rotation around a valid CA–CB axis.")
        axis /= length
        theta = np.deg2rad(angle)
        for name in moving:
            delta = coordinates[name] - center
            xyz = center + (np.cos(theta) * delta + np.sin(theta) * np.cross(axis, delta)
                            + (1 - np.cos(theta)) * axis * np.dot(axis, delta))
            i = indices[name]
            lines[i] = with_position(lines[i], xyz)
        label = f"{residue_name} chi1 {angle:g}"
    elif kind == "chemical":
        replacement, atom, parent, renamed, element, bond, max_move = MUTATIONS[residue_name]
        required = {"N", "CA", "C", "O"} | SIDE_CHAINS[residue_name]
        if not required <= indices.keys() <= required | {"OXT"}:
            raise ValueError(f"Residue {selected!r} is incomplete or has unexpected atoms.")
        delta = coordinates[atom] - coordinates[parent]
        length = np.linalg.norm(delta)
        if length == 0 or abs(length - bond) > max_move:
            raise ValueError(f"Terminal bond at {selected!r} is outside the edit tolerance.")
        for i in indices.values():
            lines[i] = lines[i][:17] + replacement + lines[i][20:]
        i = indices[atom]
        if renamed is None:
            lines[i] = ""
        else:
            xyz = coordinates[parent] + bond * delta / length
            line = with_position(lines[i], xyz)
            lines[i] = line[:12] + f"{renamed:>4}" + line[16:76] + f"{element:>2}" + line[78:]
        label = f"{residue_name}->{replacement}"
    else:
        raise ValueError(f"Unknown intervention kind {kind!r}.")
    Path(output).write_text("".join(lines))
    return label
