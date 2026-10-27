---
title: "Frozen sequence encoders with gradient boosting match the state of the art on drug–target affinity domain generalization, and a sequence defect in the Davis kinase benchmark"
author: "Avinash Kumar"
date: "October 2026"
geometry: margin=1in
fontsize: 11pt
linkcolor: blue
urlcolor: blue
---

## Abstract

We report two results on public kinase drug–target interaction (DTI) benchmarks.

First, a deliberately simple model — frozen pretrained encoders for molecules
and proteins, with gradient-boosted trees on top and no end-to-end training —
reaches **PCC 0.591 ± 0.004** on the Therapeutics Data Commons DTI-DG benchmark
(BindingDB_Patent), evaluated with TDC's own split and evaluator over five
seeds. This is statistically indistinguishable from the best published entry
(0.588 ± 0.002; difference 0.003 against a combined standard error of 0.0045,
0.67 SE) and separated from the second entry (0.538 ± 0.008) by 5.9 SE. The
result reproduced exactly on an independent run.

Second, we document a defect in the widely used Davis kinase dataset that
affects any work using its mutant kinases. Davis lists 442 protein names but
contains only **379 distinct sequences**: all 45 mutation-labelled entries are
assigned the *wild-type* sequence of their gene. ABL1 and ABL1(E255K) have
bit-identical ESM-2 embeddings (L2 = 0.0000, cosine = 1.000000). The affinity
labels differ while the model input does not, so these pairs are irreducible
contradictions in training — 9.63% of all Davis rows. Reconstructing the mutant
sequences from the names raises the distinct-sequence count to 423, reduces
contradictory rows to 3.97%, and improves concordance index on mutant proteins
by **+0.0173 (5/5 seeds, t = 23.16)** under an otherwise identical protocol.

We also report negative results that constrain what these models can do:
enlarging the training set with BindingDB does not improve within-target
ranking, and the resulting predictor is largely target-blind — the same twelve
molecules occupy the top 50 of 250–290 of 442 targets.

---

## 1. Introduction

Predicting the binding affinity between a small molecule and a protein is a
standard task in computational drug discovery, and public leaderboards now
exist to compare methods on fixed splits. Two things motivated this work. The
first was to establish whether a straightforward combination of existing
pretrained encoders is competitive on a benchmark designed to be hard —
Therapeutics Data Commons' DTI-DG task, which trains on older patents and tests
on newer ones, so the test set contains chemistry the model has not seen.

The second emerged while diagnosing why the resulting model performed poorly at
a different task — ranking drugs within a single target. That investigation led
to a property of the Davis dataset that, as far as we can establish, has not
been reported: its mutant kinases carry wild-type sequences.

Throughout, we report the seed-to-seed spread rather than a single number, and
we retain the results that did not work. Several of the findings below are
negative, and two of them overturned conclusions we had drawn from single runs.

## 2. Methods

### 2.1 Representations

All encoders are frozen and pretrained; nothing is fine-tuned.

**Molecules (1,625 dimensions).** Morgan fingerprint, radius 2, 1024 bits;
217 RDKit descriptors, clipped to ±10⁶ with non-finite values set to zero; and
the mean-pooled final hidden state of ChemBERTa-77M-MTR (384 dimensions).

**Proteins (1,280 dimensions).** The mean-pooled final hidden state of ESM-2
650M (`facebook/esm2_t33_650M_UR50D`), computed in half precision. Sequences
longer than 1,000 residues are split into 1,000-residue windows, each window
embedded separately, and the window embeddings averaged.

### 2.2 Predictor

XGBoost regression on the concatenated drug and protein vectors: 1,200 trees,
maximum depth 8, learning rate 0.05, subsample 0.8, colsample_bytree 0.6,
histogram tree method. One model per seed.

### 2.3 Protocol

For the DTI-DG benchmark we use `dti_dg_group`, take the benchmark's `test`
split unchanged, and for each of seeds 1–5 obtain the train/valid split through
`group.get_train_valid_split`. We train on the **train** portion only, not
train+valid, which is the conservative choice. Predictions for all five seeds
are passed to `group.evaluate_many`, so the reported metric is computed by TDC
rather than by us.

Features are computed over the union of train and test. All three featurizers
are deterministic functions of structure or sequence and use no labels, so this
is not leakage; it is the same transductive featurization used by pretrained
encoders generally. Every molecule (140,746) and protein (477) appearing in
either split is featurized, so no test row requires imputation.

## 3. Result 1: DTI-DG benchmark

`group.evaluate_many` returns `{'bindingdb_patent': [0.591, 0.004]}`.

| Entry | PCC | Difference from ours |
|---|---|---|
| **This work** | **0.591 ± 0.004** | — |
| Otter-Knowledge-Ensemble | 0.588 ± 0.002 | 0.003 |
| OrangeBalls-ProtBertMorgan | 0.538 ± 0.008 | 0.053 |
| MMD | 0.433 ± 0.010 | 0.158 |
| CORAL | 0.432 ± 0.010 | 0.159 |
| ERM | 0.427 ± 0.012 | 0.164 |
| MTL | 0.425 ± 0.010 | 0.166 |
| GroupDRO | 0.384 ± 0.006 | 0.207 |
| AndMASK | 0.288 ± 0.019 | 0.303 |
| IRM | 0.284 ± 0.021 | 0.307 |

(Leaderboard values as published on tdcommons.ai, October 2026.)

**We claim a tie, not a win.** The margin over the leading entry is 0.003
against a combined standard error of 0.0045 — 0.67 SE, with fully overlapping
intervals. The margin over the second entry is 5.9 SE and is unambiguous. The
run was repeated on a fresh runtime and returned 0.591 ± 0.004 again.

The result is consistent with the structure of the leaderboard rather than
surprising given it. The entries built on domain-generalization algorithms —
MMD, CORAL, IRM, GroupDRO — sit at or below plain empirical risk minimization
(0.427). The two entries that separate do so through representation: a
knowledge graph, and ProtBert with Morgan fingerprints. Our recipe belongs to
the second family with a stronger protein encoder, ESM-2 650M rather than
ProtBert. The benchmark's shape favours this: 140,746 molecules against only
477 proteins, split by patent year, so the test years bring mostly new
chemistry against a largely shared protein set.

## 4. Result 2: mutant kinases in Davis carry wild-type sequences

### 4.1 The observation

The Davis dataset provides 442 named kinases. Computing ESM-2 embeddings for
all of them and comparing within gene families gives:

| | |
|---|---|
| Protein names | 442 |
| Distinct sequences | **379** |
| Names duplicating another name's sequence | 63 |
| Mutation-labelled names | 45 |
| Mutation-labelled names with their own sequence | **0** |

The largest collision groups are ABL1 (15 names), EGFR (12, including L858R,
T790M and the exon-19 deletions), PIK3CA (10), KIT (8), FLT3 (7) and RET (4).
Within each group every name maps to one sequence. For ABL1 and ABL1(E255K) the
embeddings are bit-identical: L2 distance 0.0000, cosine similarity 1.000000,
against a mean cosine of 0.924 between randomly chosen distinct kinases.

The consequence is not a representation limitation but a label contradiction: a
model receives identical input for ABL1 and ABL1(E255K) and is asked to predict
different affinities. **9.63% of Davis rows (2,894 of 30,056) participate in such
contradictions**, with a mean within-group range of 1.12 pKd units. No model
architecture can resolve this, pocket-aware representations included, because
the pocket is derived from the same sequence.

We note that the affinity *labels* appear to be correct; only the sequences
supplied as model input are wrong.

### 4.2 Reconstruction

Mutations are recoverable from the names. We parse point mutations (`E255K`),
deletions (`E746A750del`) and multi-mutants (`V559D-T670I`, `L858RT790M`) and
apply them to the wild-type sequence.

Residue numbering in the names does not always match Davis's sequence indexing.
An initial implementation searched ±10 positions for the stated residue; this
silently mutated wrong positions. PIK3CA's eight edits all resolved at offset
+1, a consistent isoform shift, but ABL1's ten resolved at +3, +3, +3, +3, +2,
+2, +4, −1, −1 and −1, while E255K and Q252H matched nowhere — ABL1's true
offset lies outside ±10, its two isoforms differing by 19 residues. The scattered
offsets were coincidental matches on same-letter residues nearby.

We replaced this with a single offset per gene, chosen over −60…+60 as the
offset satisfying the most of that gene's mutations, and skip any mutation that
does not match at that offset. Writing no residue is preferable to writing the
wrong one. Final status: 50 sequences edited (all exact at the gene's offset),
25 skipped as not point mutations (`FLT3(ITD)`,
`JAK2(JH1domain-catalytic)`, `RSK1(KinDom.1-N-terminal)`,
`PFCDPK1(Pfalciparum)`), distinct sequences 379 → **423**, contradictory rows
9.63% → **3.97%**.

### 4.3 Effect

Identical model, features and splits; only the protein sequences differ. ESM-2
embeddings were recomputed for both arms under the same precision so the
comparison is not confounded by numerics. Five seeds; *t* is mean over standard
error.

| Split | Metric | Wild-type sequences | Reconstructed | Δ | Wins | *t* |
|---|---|---|---|---|---|---|
| random | CI, all pairs | 0.8862 | 0.8878 | +0.0016 | 4/5 | 2.31 |
| random | **CI, mutant proteins** | 0.9229 | **0.9402** | **+0.0173** | **5/5** | **23.16** |
| cold-target | CI, all pairs | 0.8483 | 0.8497 | +0.0014 | 3/5 | 0.93 |
| cold-target | **CI, mutant proteins** | 0.9187 | **0.9258** | **+0.0071** | **5/5** | **2.86** |

The effect is large and consistent where the defect lies, and absent elsewhere —
the all-pairs cold-target change is noise (3/5, *t* = 0.93) because mutant pairs
are a minority of the data. This is a data correction with a measurable
consequence, not a better model.

### 4.4 Remaining issue

`JAK1/2/3(JH1domain-catalytic)` versus `(JH2domain-pseudokinase)`, and
`RSK1–4(KinDom.1-N-terminal)` versus `(KinDom.2-C-terminal)`, are *different
domains* of one protein and bind differently. Davis assigns both the same
full-length sequence. The correct treatment is the domain subsequence rather
than a point edit; 18 names are affected and we have not addressed them.

## 5. Negative results

These bound what the approach can do and are reported because they were
measured, not because they are flattering.

**More data does not improve within-target ranking.** Adding BindingDB (Kd and
Ki) to Davis training data, holding model, features and splits fixed and
evaluating on Davis cold-drug splits over 12 seeds, changes concordance index by
amounts indistinguishable from noise for every variant tried (|*t*| < 1.5; the
raw mixture is if anything worse at −0.0177, winning 2/12 seeds). Absolute
accuracy does improve, but only when assay types are clean: ΔPearson is +0.0028
(*t* = 0.12) for mixed Kd+Ki with no assay indicator, +0.0462 (*t* = 2.41) with
an indicator, and +0.0800 (*t* = 4.13) for Kd alone. Pooling heterogeneous assay
types cancels the gain entirely.

A single-seed run of this experiment gave +0.0195 CI and would have been
reported as a positive result; five seeds reversed it (−0.0122 ± 0.0515). We
mention this because the resolution required to detect an effect of this size
exceeds what a single split provides.

**The predictor is largely target-blind.** Scoring 10,465 ChEMBL phase 1+
molecules against all 442 Davis targets and keeping the top 50 each: of the 134
targets for which a known-drug list could be resolved from ChEMBL mechanism
data, **105 have no known drug anywhere in the top 50**. The same twelve
molecules occupy the top 50 of 250–290 of the 442 targets. Median molecular
weight of the top 50 is 973 Da against 368 Da for the library, and 47% exceed
1,000 Da — peptides and glycopeptides that cannot occupy a kinase ATP site. The
model promotes large unusual molecules largely independently of which protein is
asked about.

Two post-hoc corrections recover part of this without retraining: restricting to
250–700 Da and subtracting each drug's mean predicted score across all targets.
Rank-1 accuracy rises from 13/442 to 37/442 and top-10 recall from 24/134 to
52/134. Seventy-five of 134 targets still fail. As a correctness check,
subtracting a per-target constant produced results identical to the
uncorrected ranking, as it must, since a constant per column cannot reorder
within that column.

## 6. Limitations

The DTI-DG result is one benchmark with a particular shape — many molecules,
few proteins, split by time — and should not be read as general affinity
prediction ability. Good performance there coexists with the target-blindness
described above; the two are measured on different axes and both are true.

The Davis correction covers point mutations and simple deletions. Insertions
(`FLT3(ITD)`) and domain-specific entries are not handled. The mutant set spans
roughly ten genes, so generalization of mutant-aware models to unseen genes
remains untested.

No wet-lab validation was performed. All evaluation is retrospective on held-out
public data.

## 7. Data and code

All code, including the notebooks that reproduce both results, the mutation
reconstruction, and the experiments reporting the negative results, is at
<https://github.com/avinashchaudhary28/ai-drug-discovery>. Runs that failed are
retained alongside those that succeeded.

Benchmarks are from Therapeutics Data Commons (<https://tdcommons.ai>). Davis is
obtained through the DeepDTA repository. Pretrained encoders are ESM-2 650M and
ChemBERTa-77M-MTR, used as published.

## 8. Conclusion

A frozen-encoder, gradient-boosted baseline matches the best published entry on
the TDC DTI-DG benchmark, which suggests the margin on that task currently comes
from representation quality rather than from domain-generalization machinery.

The Davis defect is the more consequential finding. A widely used kinase
benchmark supplies wild-type sequences for all 45 of its mutant entries, making
9.63% of its rows self-contradictory, and correcting this produces a measurable
improvement on exactly the subset affected. Work that reports mutant or
selectivity performance on Davis should check whether its inputs distinguish the
variants it is scoring.

## References

Therapeutics Data Commons leaderboards:
<https://tdcommons.ai/benchmark/dti_dg_group/bindingdb_patent/>,
<https://tdcommons.ai/benchmark/admet_group/>

Davis M.I. et al. *Comprehensive analysis of kinase inhibitor selectivity.*
Nature Biotechnology 29, 1046–1051 (2011).

Lin Z. et al. *Evolutionary-scale prediction of atomic-level protein structure
with a language model.* Science 379, 1123–1130 (2023). (ESM-2)

Öztürk H., Özgür A., Ozkirimli E. *DeepDTA: deep drug–target binding affinity
prediction.* Bioinformatics 34, i821–i829 (2018).
