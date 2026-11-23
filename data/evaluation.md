# Evaluation

Run `evaluate.py` on saved samples from any number of models. The default
protocol expects 100 test pockets and 100 generation attempts for each
model at each pocket. Failed generation attempts stay in the validity
denominator. For a smaller run, set `--expected-pockets` and
`--expected-attempts` explicitly.

## Manifest

Provide every model on the same list of pockets. The labels in `runs`
are the model names in the output table. Paths are relative to the
manifest.

```json
{
  "pockets": [
    {
      "id": "pocket_001",
      "pocket": "test/pocket_001.pdb",
      "receptor_pdbqt": "test/full_receptor_001.pdbqt",
      "structure_receptor": "test/full_receptor_001.pdb",
      "reference_sdf": "test/reference_001.sdf",
      "runs": {
        "Base": "samples/base/pocket_001",
        "PocketPref-DPO": "samples/dpo/pocket_001",
        "Other model": "samples/other/pocket_001.sdf"
      }
    }
  ]
}
```

Repeat the pocket entry for each test pocket. Generation directories
contain `molecules.sdf` and `run.json`, as written by `generate.py`.
The evaluator checks the reported attempt count and compares the
input-pocket hash when `pocket` is supplied. A plain multi-record SDF
also works. For plain SDFs, the supplied generation budget is used to
count missing records as invalid attempts. An optional integer
`attempt` property identifies the original attempt, from 0 to 99 for
the default budget. Otherwise, records are numbered in SDF order.
For published SDFs filtered by the original authors before release,
report the counts of available structures. Recovering the original
generation validity requires the original attempt records.

A supplied `receptor_pdbqt` is used directly for all models at that pocket.
Use the same prepared receptor as the benchmark you are comparing with.
Alternatively, provide a full receptor PDB under `receptor`. The program
prepares it with PDB2PQR and AutoDockTools. With only `pocket`, the
program prepares and evaluates that cropped PDB.

For structure evaluation, supply a full receptor PDB under
`structure_receptor` or `receptor`. The fallback is the cropped
`pocket` PDB. Supply an optional protein-only PDB under
`prolif_receptor` to use a separately prepared protein for ProLIF.
`reference_sdf` optionally supplies one reference ligand pose per
pocket. Reference recovery requires this ligand. Receptors, reference
ligands, and generated poses must share the same coordinate frame.

The default Vina box is centered on the bounding box of each generated
pose after adding hydrogens, with 5 Å added to each box dimension.
To use a common box at a pocket, supply `"box": {"center": [x, y, z],
"size": [sx, sy, sz]}` in Å. Evaluation preserves the saved heavy-atom
pose for fixed scoring and as the starting pose of the other Vina modes.

## Run

```bash
python evaluate.py --manifest evaluation.json --output-dir outputs/evaluation --workers 32
```

The default computes fixed-pose Vina score and ligand-only Vina min.
Add `--dock` to run a separate full search from the saved pose at
exhaustiveness 16. For High Affinity, also supply an optional
`reference_vina_dock` score for each pocket. This metric is the
percentage of successful generated docks scoring at least as favorably
as the reference. The default includes pockets with at least 50
successful docks and averages those pockets equally.

The Vina preparation dependencies are described in the
[preference construction guide](preference_construction.md#scoring-dependencies).
For comparison with the archived 100 × 100 evaluation, use Vina 1.2.2,
RDKit 2023.9.6, Open Babel 3.1.1, Meeko 0.1.dev3, and the same complete
prepared receptor PDBQTs. The available scoring environment and input
hashes are recorded in `protocol.json`. The default seed 0 follows
Vina's random-seed convention. Use `--vina-seed` with a nonzero integer
for a fixed seed. Full docking records the actual seed per molecule.

## Structure evaluation

Use Python 3.10 or later in a separate environment with the structure
dependencies. The versions below reproduce the structure evaluator's
tested environment.

```bash
python -m pip install rdkit==2025.9.5 posebusters==0.4.6 prolif==2.2.0 scipy==1.15.2 numpy==2.2.6 pandas==2.3.3 MDAnalysis==2.9.0 gemmi==0.7.5
python evaluate.py --analysis structure --manifest evaluation.json --output-dir outputs/structure --workers 32
```

The structure mode evaluates the original saved poses. It requires
neither model weights nor Vina dependencies. Use `--analysis all` to run
both evaluations in an environment that supports both dependency sets.
For the archived Vina protocol and the newer structure tools, run the
two commands separately with the corresponding environments.

| Metric | Definition |
| --- | --- |
| PoseBusters | `dock` checks against the receptor. A molecule passes when all required Boolean checks pass. The reference pose is omitted from the checks. |
| ProLIF | Unique residue–interaction-type pairs on a template-standardized protein using implicit hydrogens. Reports the interaction count, reference recovery (intersection divided by reference count), complete recovery, and Jaccard similarity. Recovery and Jaccard are left blank when the reference is absent or has no fingerprint features. |
| Steric clashes | Counts protein–ligand heavy-atom pairs with RDKit van der Waals overlap greater than 0.8 Å. Also reports the fraction of ligand heavy atoms involved and the maximum overlap in Å. Uses `receptor_pdbqt` if supplied, or the structure PDB. |
| Ring size | RDKit ring sizes grouped into 3, 4, 5, 6, 7, and 8+ atoms. Each bin reports the fraction of valid molecules containing at least one ring in that bin. A molecule can contribute to multiple bins. |
| Morgan PCA | One shared fit on the pooled, valid generated molecules and any supplied references, using centered radius-2, 2,048-bit Morgan fingerprints without chirality. Use the same fitted axes for model panels. |

## Outputs

| File | Contents |
| --- | --- |
| `molecules.csv` | One row per model, pocket, and generation attempt, including failures and the enabled metrics |
| `pockets.csv` | Metric counts, means, medians, diversity, and optional High Affinity for each model and pocket |
| `summary.csv`, `summary.json` | One row per model, with counts and summary statistics |
| `protocol.json` | Inputs, model checkpoint hashes when available, versions, and scoring settings |
| `ring_sizes_by_pocket.csv`, `ring_size_summary.csv` | Per-pocket ring occurrence and equal-pocket means |
| `morgan_pca.csv`, `morgan_pca.json`, `morgan_pca_model.npz` | PC1/PC2 coordinates, variance explained, and the fitted center and loadings |

Molecule-level means and medians pool molecules with a finite value for
the corresponding metric. Counts report the denominator for each
metric. Validity uses all generation attempts. SA is normalized to the
0–1 scale. Lipinski counts five rules using exact molecular weight,
H-bond donor and acceptor counts, logP, and rotatable bonds. Diversity is
the within-pocket RDK-fingerprint mean of 1 minus Tanimoto over all
ordered molecule pairs, including zero self-distances. Model-level
diversity and High Affinity weight each contributing pocket equally.
In structure mode, the summary also reports each structure metric
averaged over contributing pockets with equal weight and gives the
corresponding pocket count. PoseBusters pass averages are proportions
over evaluable structures. Missing metric values and generation failures
remain identifiable in the per-attempt output.
