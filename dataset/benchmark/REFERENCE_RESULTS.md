# Reference results — numbers to compare against

All produced by `logo_benchmark.py` on `mutation_ddg_v1.0.csv` with the default
filter: **8,110 pairs, 56 genes, 22 leave-one-gene-out folds** (genes with ≥ 20
pairs). Gene is the unit; Spearman is computed within each held-out gene and then
averaged.

Reproduce with:

```
python logo_benchmark.py --data ../mutation_ddg_v1.0.csv
```

## Baselines

| Model | MAE ↓ | Spearman (within gene) |
|---|---|---|
| per-drug mean | **0.720** | +0.055 |
| **predict 0 — the floor** | **0.750** | — (constant: no ranking) |
| training mean | 0.756 | — (constant) |
| per-substitution mean | 0.846 | +0.079 |

Against the floor, gene-paired:

| Model | ΔMAE | t | folds won | jackknife min t |
|---|---|---|---|---|
| per-drug mean | +0.0295 | +0.57 | 9/22 | −1.26 (PDGFRA) |
| training mean | −0.0059 | −0.57 | 13/22 | −0.91 (PDGFRA) |
| per-substitution mean | −0.0958 | −1.82 | 5/22 | −2.14 (PDGFRA) |

**Nothing beats predicting zero.** The per-drug mean's +0.0295 at t = +0.57 on
9 of 22 folds is noise, and the jackknife shows it going negative when one gene
is removed.

## Trained models

A gradient-boosted model on drug + mutation features, same 22 folds:

| | MAE ↓ | Spearman |
|---|---|---|
| XGBoost (drug + mutation) | 0.771 | +0.067 |

**Worse than the floor.** Four successive representations were tried — amino-acid
property changes; KLIFS-aligned pocket position; ligand contact frequency from
crystal structures; ESM-2 embeddings of the mutated pocket — and none cleared a
pre-registered gate. The closest (ESM-2) reached ΔMAE +0.0748 against the floor
at t = +1.12, 8 of 14 folds, with Spearman unmoved (−0.004).

## The number that matters most

| Split | MAE | Spearman |
|---|---|---|
| **Random** (gene appears on both sides) | 0.445 vs floor 0.729 | **+0.752** |
| **Leave-one-gene-out** (gene removed) | 0.771 vs floor 0.750 | **+0.067** |

Same model, same features, same data. The model fits the task well and collapses
when the gene is removed: **it memorises genes rather than learning mutation
effects.**

If your method reports a number near 0.75 on this dataset, check your split
before believing it. 0.752 is what a random split gives and it means nothing.

## Few-shot: when the target gene has some measurements

The leave-one-gene-out setting above assumes **zero** measurements on the target
gene. A different, more realistic protocol gives the model *k* measurements on
the held-out gene and tests on the rest of it, with the test set held fixed
across all *k* so the curve is readable.

Spearman within the held-out gene (17-gene set, 5 seeds):

| k measurements given | model | best no-model estimate |
|---|---|---|
| 0 | 0.141 | 0.052 |
| 5 | 0.206 | 0.062 |
| 10 | 0.266 | 0.070 |
| **20** | **0.340** | 0.089 |

At k = 20, against the best no-model estimate (a drug's own measurement where it
exists, otherwise its per-drug mean from other genes): **+0.255 Spearman,
t = +4.07, 15 of 16 genes, jackknife +3.71.** The k = 0 → k = 20 lift wins on
**16 of 17 genes** (t = +4.32).

**That is a statistical result, not a product.** Converted into the decision a
chemist makes — shortlist three compounds for one resistance mutation — the
genuinely least-affected compound is in the model's top three **34%** of the
time, against **26%** for the no-model estimate and 19% by chance. The +0.080
margin is not statistically established (t = +1.97) and does not survive the
jackknife.

Magnitude prediction ("by how much") fails or is fragile in every variant tried.

## Honest accounting

The pre-registered few-shot gate was evaluated **nine times** across gene sets,
model variants and two baseline definitions. One configuration passed, and
fragilely. The first version of the "hardest" baseline was wrong in a way that
flattered the result — it used only the *k* supplied measurements and therefore
*degraded* as *k* grew, making it weakest exactly at the gate point. Correcting
it roughly halved the headline margin (+0.150, t = 3.38 → +0.080, t = 1.97 on the
top-3 metric).

Full detail: `claude/phase3_results.md` in this repository.

## Reporting your own result

Report all of: mean MAE and mean within-gene Spearman over folds; the paired t
against **both** `predict 0` and `per-drug mean`; how many folds you win; and the
jackknife, so a reader can see whether the result rests on a single gene. State
how your split was made.

And remember the noise floor: roughly **0.5 log units** of irreducible label
noise sits in this data — within one publication, ChEMBL holds four wild-type
measurements for one compound and target spanning 0.8 log units. An MAE far
below 0.5 is evidence of leakage, not skill.
