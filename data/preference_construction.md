# Preference Pair Construction

## Inputs

Supply a JSON manifest with one entry for each pocket intervention. Use
paths relative to the manifest. Supply a cropped, single-conformer
protein pocket PDB and an SDF containing the original 3D candidate
poses. If the same SDF is used for two edits of a pocket, keep the same
pocket ID and molecule order. The optional SDF property
`candidate_key` identifies records across different SDFs for the same
pocket.

```json
{
  "sites": [
    {
      "pocket": "training_pocket_01",
      "receptor": "pocket.pdb",
      "candidates": "candidates.sdf",
      "kind": "spatial",
      "site": {"chain": "A", "residue": "123", "angle_deg": 120}
    },
    {
      "pocket": "training_pocket_01",
      "receptor": "pocket.pdb",
      "candidates": "candidates.sdf",
      "kind": "chemical",
      "site": {"chain": "A", "residue": "145"}
    }
  ]
}
```

Fill in actual residue numbers for your structure. Select sites before
scoring the candidates. For spatial edits, select a clash-checked
side-chain rotation in the ligand-contact region. The historical
chemical sites were selected using reference-ligand interactions with
the changed atom in PLIP 3.0.1. The hydrogen-bond criterion was
donor–acceptor distance ≤3.5 Å and angle ≥120°. The Asp/Glu salt-bridge
criterion used a ≤4.0 Å charge-center distance and required the changed
atom in the charged group. The manifest records the site selection.
The builder applies the intervention at the supplied site.

Each site can supply `insertion` in its `site` object and an optional
`box` with `center` and `size` vectors in Å. If omitted, one common
box is computed from the eligible ligand coordinates with at least
12 Å total padding and a 20 Å minimum side length. An optional
`properties` array overrides the eight default generation targets in
this order: QED, normalized SA, logP, TPSA, HBA, HBD, Fsp3, rotatable
bonds. Use the same pocket and property vector for every edit of a
pocket. The packaged training data already contain the original
reference-derived property vectors.

## Scoring dependencies

The builder uses Vina 1.2.7, the Open Babel backend of Meeko
0.1.dev3, PDB2PQR 3.6.2 and AutoDockTools_py3. The training and generation
commands only need the main Conda environment.

```bash
python -m pip install vina==1.2.7 meeko==0.1.dev3 pdb2pqr==3.6.2
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

Set `PDB2PQR30_EXECUTABLE` if `pdb2pqr30` is installed in a separate
environment. Sites may also supply a `pdbqt` object with `original`
and `edited` receptor paths prepared with the same protocol. The
builder checks that their heavy-atom coordinates agree with the
respective PDBs.

## Selection and outputs

- **Spatial.** Rotate the side chain beyond Cβ around the Cα–Cβ axis.
- **Chemical.** Apply the supported substitution at the supplied
  residue. The seven directions are Asp→Asn, Glu→Gln, Ser→Cys,
  Thr→Val, Asn→Asp, Gln→Glu, and Tyr→Phe. The terminal-atom edit
  preserves the remaining heavy coordinates. Tyr→Phe removes OH.
- **Quality and scores.** Require QED and normalized SA ≥0.5 and
  initial ligand–pocket heavy-atom van der Waals overlap ≤0.8 Å in
  both pockets. Score each passing pose with Vina `score()`, keeping
  the same grid and input pose across both states.
- **Pairs.** Deduplicate finite scored candidates by canonical
  isomeric SMILES per site. Sort by signed response
  `E(edited) - E(original)`, pair from the two ends of the ranking,
  and keep response gaps ≥1.0 kcal/mol for spatial preferences and
  ≥0.7 kcal/mol for chemical preferences. For custom multi-site
  manifests, remove conflicting orientations within each pocket and
  repeated pairs of the same kind.

The output includes the edited PDBs and `build_report.json` with finite
scores and exclusions. The training files are `pairs.json` and
`inputs.pt`. Supply both preference kinds to train the joint model.
The packaged archived pairs preserve the historical candidate inventory,
site ranking and class-expansion decisions used in the released dataset.
The custom manifest workflow takes those decisions as inputs.
