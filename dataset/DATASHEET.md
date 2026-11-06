# Datasheet — kinase mutation–affinity (ΔΔG) dataset v1.0

A collection of paired wild-type / mutant binding affinities for small molecules
against human protein targets, assembled and audited from ChEMBL and the Davis
kinase panel. The quantity of interest is **ΔΔG = pK_mut − pK_wt**: how much a
target mutation changes a compound's measured affinity.

- **File:** `mutation_ddg_v1.0.csv` — 14,391 rows × 19 columns
- **Schema:** `SCHEMA.md` (read this before filtering — two row counts are in
  circulation and they mean different things)
- **Checksum:** `MANIFEST.sha256`
- **Built by:** Avinash Kumar. Code: `experiments/phase2/` in this repository.

## Why this exists

Predicting how a kinase inhibitor's affinity changes when its target mutates is
the drug-resistance question, and it is commercially central — imatinib → T315I,
EGFR inhibitors → T790M, ALK inhibitors → G1202R. The published literature names
**data scarcity, not architecture**, as the bottleneck.

The dataset behind the published state of the art on mutation effects is
**Platinum**. We harvested Platinum ourselves (984 pairs, reading verified
against its own `fold_change` column at 99.7% sign agreement) and counted its
composition:

| | Platinum | This dataset |
|---|---|---|
| Total ΔΔG pairs | 984 | 14,391 |
| **Protein-kinase pairs** | **16** (PRKACA 12, ABL1 3, AURKA 1) | **10,201** |
| Kinase genes | 3 | **56** |

Platinum is dominated by oxidoreductases, hydrolases and transferases — aldose
reductase, thymidylate synthase, HIV protease, DHFR. It is a fine general-protein
benchmark and it is **not** a kinase benchmark. That gap is what this dataset
fills.

## Composition

| | Pairs | Genes | Mutations | Compounds |
|---|---|---|---|---|
| Full file | 14,391 | 198 | 903 | 6,617 |
| Kinase subset (`is_kinase`) | **10,201** | **56** | 220 | 4,450 |
| Kinase, uncensored — the modelling set | **8,110** | 56 | 220 | 4,450 |
| Kinase, uncensored, ≥ 20 pairs per gene (LOGO folds) | 7,954 | **22** | — | — |

By source: ChEMBL 11,895, Davis 2,496 (after de-duplication).
By assay type: IC50 8,901 · Kd 4,198 · Ki 1,292.

On the modelling set: mean |ΔΔG| 0.729, median 0.492, sd 1.028, range −4.44 to
+4.22. **26% of pairs have |ΔΔG| > 1.0** — so this is not a dataset of
near-zero effects. 96% take their wild-type reference from the same publication
as the mutant measurement.

The largest effects are the textbook resistance mutations — PDGFRA D842V, ABL1
T315I and E255K, KIT D816V, ALK L1196M. None of that was prompted; it is the
strongest available check that the extraction is correct.

## How it was built

ChEMBL activity records carry an `assay_variant_mutation` field. Every human
mutation-annotated Kd/Ki/IC50 measurement in nM was pulled, together with
wild-type measurements for each target that appeared, and paired.
`pAct = 9 − log10(nM)`. Davis's kinase panel was processed the same way and
merged.

Four construction rules, **each of them the scar of a specific earlier mistake**:

1. **Never pair across assay types.** A Kd mutant is never paired with an IC50
   wild type. Mixing was measured to destroy the signal: ΔPearson +0.0028
   (t = 0.12) mixed, versus +0.0800 (t = 4.13) on clean Kd.
2. **Only `standard_relation = '='`.** A `>10000` entry is a detection limit, not
   a measurement. In Davis, pKd 5.0 is a censoring floor and 52% of apparent
   pairs were floor-to-floor.
3. **Same-document pairing.** Each mutant measurement is paired with the
   wild-type value **from its own publication**; failing that, a cross-document
   median flagged `same_doc=False`. Mutant measurements are grouped per document
   too — one mutation measured in two papers is two observations, not one to
   average.
4. **Fail-closed mutation parsing.** A chunk is accepted only if it is *entirely*
   a chain of mutation tokens.

Rows deliberately excluded: 950 ChEMBL pairs whose mutation field is literally
`UNDEFINED MUTATION` (keeping them would collapse different mutations into one
key), and 10,869 mutant measurements with no wild-type counterpart under these
rules — discarded rather than paired loosely.

## Validation

**Cross-source agreement on the 1,108 pairs present in both ChEMBL and Davis:
Pearson 0.950, Spearman 0.913, median difference 0.00, 10 pairs (1%) differing
by more than 1.0.** Two independent sources, two independent extraction paths.

Overlapping pairs are **never averaged** — ChEMBL's value is kept and the
agreement is reported rather than smoothed away.

## Errors found during construction, and what they cost

Every one of these returned a plausible wrong answer with no error raised. They
are listed because the correction is part of the artifact.

| Error | Effect | Resolution |
|---|---|---|
| ChEMBL refuses offsets past ~20,000; two slices exceeded it | ~3,800 mutant measurements silently lost; the run exited green, reporting 11,585 pairs | Split on `assay_type` then `pchembl_value` bands, and **reconcile every split** against its parent count. Current runs report `shortfall_slices: 0` |
| A half-finished download treated as complete | Partial data read as whole | Downloads write a `done` marker; the wild-type pass resumes per target |
| Wild type taken as a **median across papers and assays** | Manufactured offsets between sources, which were then **reported as an audit finding about a possible unit error** | Claim **withdrawn**. Same-document pairing adopted; every ΔΔG value changed |
| Mutation parser reading residues out of surrounding text | Wrote a *wrong residue* with no error: `H1D` from inside `JH1DOMAIN`; `N771I` from `D770_N771insNPG` (the `I` of `INS` is an amino-acid letter) | Full-chain matching only; both cases are now test cases |

The wild-type fix is the one worth dwelling on, because the prediction it made
came true. If the cross-source disagreements were an artifact of the median,
correcting the pairing had to shrink them:

| | before | after |
|---|---|---|
| Pearson (overlap) | 0.911 | **0.950** |
| Spearman | 0.856 | **0.913** |
| pairs differing > 1.0 | 32 (3%) | **10 (1%)** |
| constant-offset disagreements | 9 | **3** |

Six of the nine "disagreements" were ours.

## The noise floor — read this before reporting an error metric

Within a **single publication**, ChEMBL holds four wild-type measurements for one
compound and target spanning **5.29 to 6.10 — 0.8 log units inside one paper.**
Same-document pairing does not remove this, because one paper contains many
assay records.

So roughly **0.5 log units of irreducible noise** sits in these labels. **A
reported error below that should be treated as evidence of leakage, not skill.**
This argues for evaluating on the |ΔΔG| ≥ 1.0 subset and for reporting ranking
metrics alongside regression error.

## Limitations

1. **Label noise ~0.5 log units**, as above. This is a property of the
   underlying measurements, not of the assembly.
2. **IC50 dominates** (8,901 of 14,391). IC50 depends on assay conditions more
   than Kd does. Filter by `assay_type` for anything sensitive to this.
3. **Censored measurements are flagged, not dropped.** `censored_wt` /
   `censored_mut` mark values reported as `>` or `<`. The 10,201 headline
   includes them; the 8,110 modelling set does not.
4. **Severe gene imbalance.** EGFR contributes 2,376 pairs; several genes have
   fewer than 20. Any statistic computed per row rather than per gene will be
   dominated by a handful of genes.
5. **Some of the most important mutations sit outside the ATP pocket.** BRAF
   V600E and LRRK2 G2019S are not inside the 85-residue KLIFS pocket, so
   pocket-based representations cannot describe them — and LRRK2 (677 pairs) and
   BRAF (431 pairs) have *no* in-pocket mutation at all.
6. **`is_kinase` comes from ChEMBL's protein classification.** Two targets
   returned no classification and are counted separately rather than folded into
   "not a kinase".
7. **`drug_key` is the InChIKey skeleton block**, so it ignores stereochemistry
   and salt form. Two stereoisomers collapse to one compound.
8. **No wet-lab validation.** Everything here is retrospective, assembled from
   published measurements.
9. **BindingDB is not included.** Its 640-column header contains no mutation
   column; mutation information, where present, is free text inside
   `Target Name`, and in the ~130,000 rows actually read those were HIV Gag-Pol,
   not kinases. This is a cost judgement, **not** a finding that BindingDB holds
   nothing — the host throttled hard enough that the file was never read in full.
10. **Platinum is not merged in.** Where it is used it is held out only
    (`hold_out_only=True`); training on it and then claiming to beat the
    published number would be dishonest.

## What models on this dataset actually achieve

Stated here because a dataset that implies an unearned capability is worse than
no dataset. Full detail in `claude/phase3_results.md`; headline:

- **Zero measurements on the target gene: nothing works.** Four representations
  (amino-acid properties; KLIFS pocket position; ligand contact frequency; ESM-2
  on the mutated pocket) all failed a pre-registered gate under
  leave-one-gene-out. The same model reaches Spearman **0.75** when the gene is
  in training and **0.067** when it is removed — it memorises genes.
- **Twenty measurements on the target gene: ranking works, statistically.** The
  model beats the best no-model estimate by **+0.255** Spearman (t = 4.07,
  **15 of 16 genes**, jackknife +3.71), and the lift from supplying measurements
  wins on **16 of 17 genes**.
- **The same result, judged as a decision, is not enough.** Shortlist three
  compounds and the genuinely least-affected one is present **34%** of the time,
  against **26%** with no model and 19% by chance. That +0.080 margin is not
  statistically established (t = 1.97) and is fragile.
- **Magnitude prediction — "by how much" — does not work** in any variant.

The pre-registered gate was evaluated **nine times** across gene sets, model
variants and two baselines. One row passed, fragilely. The first version of the
"hardest" baseline was wrong in a way that flattered the result — it degraded as
more measurements were supplied — and correcting it roughly halved the headline
margin.

**The 0.752-versus-0.067 contrast is the most useful single number here for
anyone else attempting this task:** it shows how easily this problem can be made
to look solved by splitting the data the wrong way.

## Uses

**Appropriate:** benchmarking mutation-effect models under leave-one-gene-out;
few-shot / transfer evaluation; studying label noise in public affinity data;
cross-source agreement analysis.

**Inappropriate:** training on it and evaluating with a random split (that
reports ~0.75 and means nothing); quoting an error metric below ~0.5 log units
as skill; clinical or dosing decisions of any kind; pooling assay types.

## Licence and citation

**This dataset is released under [Creative Commons
Attribution–ShareAlike 3.0 Unported (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/),
because it has to be.**

11,895 of the 14,391 rows derive from **ChEMBL**, which is itself licensed
CC BY-SA 3.0. That licence requires attribution and is **share-alike**:
adaptations must be redistributed under the same terms. This file is an
adaptation of ChEMBL, so no other licence is available for it — including "all
rights reserved", which an earlier version of this document incorrectly claimed.

What that means in practice:

- You may use, redistribute and adapt this file, including commercially.
- You must attribute ChEMBL (and this compilation) and keep the same licence on
  anything derived from it.
- **The compiler cannot grant anyone exclusive rights to this data**, and cannot
  stop a recipient from redistributing it.

Attribution for ChEMBL, as its documentation asks:

> Zdrazil B. et al., *The ChEMBL Database in 2023: a drug discovery platform
> spanning multiple bioactivity data types and time periods*, Nucleic Acids
> Research, 2024, 52(D1), D1180–D1192.

The 2,496 Davis-derived rows come from Davis M.I. et al., *Comprehensive
analysis of kinase inhibitor selectivity*, Nature Biotechnology, 2011 — cite it
alongside. Both sources' measurements originate in the publications they index,
and ChEMBL notes separately that bundled **commercial** property calculations
should not be extracted to train models replicating the commercial process; this
compilation contains no such calculated properties, only measured affinities,
identifiers and SMILES.

**The code in this repository is a separate work** and is not covered by the
above; no licence is attached to it.

**No dataset DOI has been issued yet.** Until one is, cite the repository and the
commit:

```
A. Kumar, kinase mutation–affinity (ΔΔG) dataset v1.0, CC BY-SA 3.0,
https://github.com/avinashchaudhary28/ai-drug-discovery, commit <sha>.
```

**No dataset DOI has been issued yet.** Until one is, cite the repository and the
commit:

```
A. Kumar, kinase mutation–affinity (ΔΔG) dataset v1.0,
https://github.com/avinashchaudhary28/ai-drug-discovery, commit <sha>.
```

The related preprint — *Frozen sequence encoders with gradient boosting match the
state of the art on drug–target affinity domain generalization, and a sequence
defect in the Davis kinase benchmark*, ChemRxiv,
[doi:10.26434/chemrxiv.15011207/v1](https://doi.org/10.26434/chemrxiv.15011207/v1)
— reports the Davis sequence defect and the affinity benchmark result. **It is
not a paper about this dataset**, so cite it for those findings rather than as
the dataset's reference.
