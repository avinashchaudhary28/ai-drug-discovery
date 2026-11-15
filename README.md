# AI Drug Discovery — binding affinity, ADMET, and kinase drug resistance

An independent, fully public attempt at predicting how kinase inhibitors bind —
and how that binding breaks when the target mutates. The code, the data, the
benchmarks and **every failed attempt** are here.

**TDC DTI-DG (BindingDB_Patent): PCC 0.591 ± 0.004** — statistically tied with
the top leaderboard entry (0.588 ± 0.002, 0.67 SE apart), clearly ahead of
second place (0.538 ± 0.008, 5.9 SE). TDC's own split and evaluator, 5 seeds,
reproduced on an independent run. See [`REPRODUCE.md`](REPRODUCE.md).

Preprint: *Frozen sequence encoders with gradient boosting match the state of
the art on drug–target affinity domain generalization, and a sequence defect in
the Davis kinase benchmark*, A. Kumar, ChemRxiv 2026,
[doi:10.26434/chemrxiv.15011207/v1](https://doi.org/10.26434/chemrxiv.15011207/v1)
([local PDF](paper_preprint.pdf)).

## The dataset — [`dataset/`](dataset/)

**10,201 kinase mutation–affinity (ΔΔG) pairs across 56 genes**, assembled from
ChEMBL and Davis and audited end to end. Cross-source agreement **Pearson
0.950** on the 1,108 pairs present in both.

For scale: **Platinum**, the dataset behind the published state of the art on
mutation effects, contains **16 protein-kinase pairs** across 3 proteins.

Ships with a [datasheet](dataset/DATASHEET.md), a [schema](dataset/SCHEMA.md),
checksums, and a standalone [leave-one-gene-out benchmark
harness](dataset/benchmark/logo_benchmark.py) with
[reference results](dataset/benchmark/REFERENCE_RESULTS.md).

```bash
python dataset/benchmark/logo_benchmark.py --data dataset/mutation_ddg_v1.0.csv
```

That reproduces the baselines and runs a sign-convention check against known
imatinib-resistance biology before it reports anything.

## Two findings worth the click

**A sequence defect in the Davis kinase benchmark.** All 45 mutation-labelled
kinases in Davis carry the wild-type sequence, so ABL1 and ABL1(E255K) produce
**bit-identical** ESM-2 embeddings (L2 = 0.0000, cosine = 1.000000). Any
sequence-based model trained on it has been learning drug resistance from inputs
containing no mutation. Reconstructing the sequences cut contradictory labels
from 9.6% to 4.0% of rows and improved concordance index on mutant proteins by
+0.0173, consistent across all 5 seeds (t = 23.16).

**Random split 0.752, leave-one-gene-out 0.067.** Same model, same data, same
features. Split the data randomly and the task looks solved; remove the gene
entirely — the real case, a target the model has never seen — and it collapses.
It memorises genes rather than learning mutation effects. That contrast is the
most useful single number here for anyone else attempting this task.

## What does not work, stated plainly

Six attempts at predicting ΔΔG for an unseen kinase, each a different
mechanistic hypothesis: amino-acid property changes; KLIFS-aligned pocket
position; ligand contact frequency from crystal structures; ESM-2 embeddings of
the mutated pocket; reframing it as resistance classification; and explicit
drug × mutation interaction features. **Each had its pass/fail gate written into
the code before the run. None cleared it.**

Given ~20 measurements on the target gene, *ranking* works (+0.255 Spearman over
the best no-model estimate, t = 4.07, 15/16 genes). Converted into the decision
a chemist makes — shortlist three compounds — the margin over no model at all is
+0.080 and not statistically established. Magnitude prediction does not work at
all.

Eleven gate evaluations, the baselines that were wrong and got corrected, and
the claims withdrawn are all in [`claude/phase3_results.md`](claude/).

## Other modules

| Module | Model | Data | Result |
|---|---|---|---|
| Binding v2 | Morgan FP + RDKit descriptors + ChemBERTa + ESM-2 → XGBoost | Davis + KIBA | Davis CI 0.888 (random), 0.675 (new drug) · KIBA CI 0.856 |
| ADMET v2 | Morgan count FP + MACCS + descriptors + ChemBERTa → XGBoost | TDC ADMET Benchmark Group (22 datasets) | Re-scored with TDC's own `evaluate_many`, 5 seeds. 20 of 22 reproduce the earlier self-computed numbers (median 0.66% apart); `vdss_lombardo` and `half_life_obach` do **not** and the earlier values for those two should not be used |
| Toxicity | Morgan FP + descriptors → XGBoost | Tox21 (12 assays) | mean ROC-AUC 0.852 (random), 0.772 (scaffold) |
| Solubility | same | ESOL | RMSE 0.666 (random), 0.880 (scaffold) |

Demo app: `streamlit run streamlit_app.py`

The repurposing tab is labelled a **diagnostic, not a discovery feature**, and
says so in the app: 75 of 134 checkable targets still have no known drug in the
model's top 50.

## Licensing — the code and the data differ

**Code: all rights reserved.** Published for reproducibility and verification.
You are welcome to read, run and check it. For commercial use, redistribution or
inclusion in a product, please get in touch first.

**Data ([`dataset/`](dataset/)): [CC BY-SA
3.0](https://creativecommons.org/licenses/by-sa/3.0/)** — and that is not a
choice. 11,895 of the 14,391 rows derive from ChEMBL, which is CC BY-SA 3.0 and
share-alike, so adaptations must carry the same terms. You may use, redistribute
and adapt the data, including commercially, provided you attribute and keep the
licence. Attribute ChEMBL (Zdrazil et al., *Nucleic Acids Research* 2024,
52(D1), D1180–D1192) and, for the 2,496 Davis-derived rows, Davis et al.,
*Nature Biotechnology* 2011. Full terms in the
[datasheet](dataset/DATASHEET.md).

---

Research work, not medical advice. Nothing here has wet-lab validation.
