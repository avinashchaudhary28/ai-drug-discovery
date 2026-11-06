# Schema — `mutation_ddg_v1.0.csv`

14,391 rows × 19 columns. One row = one (drug, gene, mutation, assay type)
measurement pair: a wild-type affinity and a mutant affinity for the same
compound against the same target, plus their difference.

## Columns

| Column | Type | Meaning |
|---|---|---|
| `source` | str | `chembl` or `davis` — where this pair came from. |
| `gene` | str | HGNC gene symbol of the target. |
| `mutation` | str | Normalised mutation. Single substitution `T315I`; multiple substitutions joined with `+`, sorted (`L858R+T790M`). |
| `mut_note` | str | The parenthesised suffix from the source label, if any. 340 rows; the only value present is `p` (Davis's notation for phosphorylated). Empty otherwise. |
| `mutation_raw` | str | The source's original mutation string, before normalisation. Kept so every row is traceable. |
| `target_id` | str | ChEMBL target identifier. Empty for Davis rows (2,496 of them). |
| `drug_key` | str | First 14 characters of the compound's InChIKey — the skeleton block. This is the join key for "same compound", and it ignores stereochemistry and salt form. |
| `drug_id` | str | ChEMBL molecule identifier. |
| `smiles` | str | Compound SMILES. |
| `assay_type` | str | `IC50` (8,901), `Kd` (4,198), or `Ki` (1,292). **Never mix these** — see the datasheet. |
| `pK_wt` | float | Wild-type affinity as `9 − log10(value in nM)`. Higher = binds more tightly. |
| `pK_mut` | float | Mutant affinity, same transform. |
| `ddG` | float | `pK_mut − pK_wt`. **Negative means the mutation weakened binding** (resistance). Positive means the compound held up or improved. |
| `censored_wt` | bool | The wild-type measurement was reported as `>` or `<` a limit, not an exact value. |
| `censored_mut` | bool | Same, for the mutant measurement. |
| `same_doc` | bool | ChEMBL only: the wild-type reference came from the **same publication** as the mutant measurement. `False` means the wild-type value is a ChEMBL-wide median across papers instead, which mixes assay conditions. 96% of the benchmark subset is `True`. Always `True` for Davis (one panel, one protocol). |
| `is_kinase` | bool | The target is classified as a protein kinase (via ChEMBL protein classification; all Davis targets are kinases). |
| `mut_parsed` | bool | The mutation string parsed into substitutions. **Always `True` in this file** — unparseable rows were dropped during construction, and the column is retained only so the filter is explicit and reproducible. |
| `in_both_sources` | bool | This pair exists in both ChEMBL and Davis. On a duplicate, the ChEMBL row was kept (and among those, a `same_doc=True` one preferred). 1,073 rows in the benchmark subset. |

## Sign convention, stated once because it is easy to invert

```
ddG = pK_mut − pK_wt

ddG strongly negative  →  the mutant binds the compound far worse  →  resistance
ddG near zero          →  the mutation did not change binding much
ddG positive           →  the mutant binds at least as well
```

Sanity check anyone can run: filter to `gene == "ABL1"` and rank mutations by
mean `ddG` ascending. The most negative should be the canonical
imatinib-resistance set — G250E, F317I, T315I, E255K, Y253F, F317L. If they come
out at the other end, the sign has been flipped somewhere.

## Recommended filters

```python
import pandas as pd
d = pd.read_csv("mutation_ddg_v1.0.csv")

# Kinase mutation pairs, parseable mutations           -> 10,201 rows, 56 genes
kinase = d[d.is_kinase & d.mut_parsed & d.smiles.notna() & d.ddG.notna()]

# The modelling set: same, minus censored measurements ->  8,110 rows, 56 genes
clean = kinase[~(kinase.censored_wt | kinase.censored_mut)]

# Leave-one-gene-out folds: genes with >= 20 pairs     ->     22 genes, 7,954 rows
folds = clean.gene.value_counts()
folds = folds[folds >= 20].index
```

**Both row counts are correct and they are not interchangeable.** 10,201 is the
dataset size including censored measurements; 8,110 is the uncensored subset, and
every model result reported in this repository was trained and evaluated on the
8,110. If you read 10,201 anywhere and your filter gives 8,110, this is why.

Never pool `assay_type` values. Mixing Kd with Ki was measured to destroy the
gain from mutation-aware inputs (ΔPearson +0.0028, t = 0.12, versus +0.0800,
t = 4.13 on clean Kd alone).
