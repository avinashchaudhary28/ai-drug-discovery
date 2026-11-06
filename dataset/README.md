# Kinase mutation–affinity (ΔΔG) dataset — v1.0

Paired wild-type / mutant binding affinities for small molecules against human
protein targets, assembled and audited from ChEMBL and the Davis kinase panel.

**10,201 kinase ΔΔG pairs across 56 genes.** The dataset behind the published
state of the art on mutation effects, Platinum, contains **16** protein-kinase
pairs across 3 proteins.

```
mutation_ddg_v1.0.csv     14,391 rows x 19 columns
MANIFEST.sha256           checksum
SCHEMA.md                 column-by-column, sign convention, recommended filters
DATASHEET.md              provenance, construction, validation, limitations
benchmark/
  logo_benchmark.py       leave-one-gene-out harness + baselines (standalone)
  REFERENCE_RESULTS.md    numbers to compare against
```

## Start here

```bash
python benchmark/logo_benchmark.py --data mutation_ddg_v1.0.csv
```

That prints the baselines, runs a sign-convention check against known
imatinib-resistance biology, and tells you how to plug in your own model.

```python
import pandas as pd
d = pd.read_csv("mutation_ddg_v1.0.csv")

# The modelling set: kinase, parseable mutation, uncensored -> 8,110 rows
clean = d[d.is_kinase & d.mut_parsed & d.smiles.notna() & d.ddG.notna()
          & ~(d.censored_wt | d.censored_mut)]
```

## Three things to know before you use it

**1. `ddG = pK_mut − pK_wt`, so negative means the mutation weakened binding.**
Inverting this raises no error — it just reports the least-affected compounds as
the worst. The harness checks it for you.

**2. Two row counts are in circulation and they are not interchangeable.**
10,201 is the kinase subset *including* censored (`>` / `<`) measurements; 8,110
is the uncensored subset, and every model result in this repository used the
8,110. `SCHEMA.md` has the exact filters.

**3. The gene is the unit, not the row.** EGFR contributes 2,376 of 8,110 pairs.
Statistics computed per row manufacture confidence. Evaluate leave-one-gene-out
and aggregate across genes.

## What models on this data actually achieve

Reported here because a dataset that implies an unearned capability is worse than
no dataset.

| Setting | Result |
|---|---|
| Random split (gene on both sides) | Spearman **0.752** — and meaningless |
| Leave-one-gene-out, zero measurements on the target | Spearman **0.067**; four representations all failed a pre-registered gate |
| 20 measurements on the target, ranking | **+0.255** Spearman over the best no-model estimate (t = 4.07, 15/16 genes) — a real statistical result |
| The same, as a decision (top-3 shortlist) | **34%** hit rate vs **26%** with no model — margin not established (t = 1.97) |
| Magnitude ("by how much") | Does not work |

**The 0.752-versus-0.067 contrast is the most useful single number here.** It
shows how easily this task can be made to look solved by splitting the data the
wrong way.

## Licence

**[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) — not a
choice.** 11,895 of the 14,391 rows derive from ChEMBL, which is CC BY-SA 3.0,
and that licence is share-alike: adaptations must carry the same terms. So this
file is free to use, redistribute and adapt, including commercially, provided
you attribute and keep the licence. **No one can be granted exclusive rights to
it.**

Attribute ChEMBL (Zdrazil et al., *Nucleic Acids Research* 2024, 52(D1),
D1180–D1192) and, for the 2,496 Davis-derived rows, Davis et al., *Nature
Biotechnology* 2011. The code in this repository is a separate work and carries
no licence.

No dataset DOI yet — cite the repository and commit. `DATASHEET.md` has the full
citation note and the related preprint, which reports the Davis sequence defect
and is *not* a paper about this dataset.
