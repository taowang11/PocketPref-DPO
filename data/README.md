# Example data

`examples/jak2_8bm2_pocket.pdb` is a prepared binding pocket for the
[Quick Start](../README.md#quick-start). It contains 307 heavy atoms from
39 residues in chain A of JAK2, selected within 10 Å of the reference
ligand. The pocket was prepared from the TarPass benchmark structure.
[PDB entry 8BM2](https://www.rcsb.org/structure/8BM2) ·
[TarPass source files](https://github.com/sorui-qin/TarPass/tree/da5058666fb36c0c275ba1d4bd9b8906b0a702fb/Targets/JAK2)

## Training data

`pocketpref-dpo-train.zip` contains the exact ordered set of **3,154**
preferences used for training. The archive contains **429**
training pockets and **6,056** distinct generated ligand records, with
**2,207 spatial** and **947 chemical** pairs. The original pockets are
from the CrossDocked v1.1-derived training data. Ligand coordinates and
atom types were exported from the saved Base candidate pool.

| File in the archive | Contents |
| --- | --- |
| `pairs.json` | Ordered preferences with pocket ID, winner, loser, intervention kind, and signed score-loss gap |
| `inputs.pt` | CPU tensors for the original pocket positions/features, reference property vectors, and saved ligand positions/types |
| `metadata.json` | Record counts, property order, source hashes, and archive-member checksums |

The repository includes this 4.6 MiB archive. Its SHA-256 is
`aa74175dc6fe5d8482a9112afef0b3760b95f6cc0a451a0aa24350365f541f74`.
Extract it under `data/` and
see [Training](../README.md#training). For constructing preferences from
your own candidate poses, see [Preference Pair Construction](preference_construction.md).
