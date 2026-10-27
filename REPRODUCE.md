# Reproducing the TDC benchmark results

This repository contains everything needed to reproduce our entries on the
Therapeutics Data Commons leaderboards. Both use TDC's own splits and TDC's own
evaluator; we never compute the leaderboard metric ourselves.

## DTI-DG / BindingDB_Patent — PCC 0.591 ± 0.004

**Notebook:** `notebooks/binding_tdc_dtidg.ipynb`
**Runtime:** Google Colab with a T4 GPU. Open the notebook, set
*Runtime → Change runtime type → T4 GPU*, then *Run all*.
**Time:** 1–2 h on the first run (protein and molecule embeddings), ~25 min
afterwards — all features are cached to Google Drive, and the cache is
checkpointed so a disconnect resumes rather than restarts.

What the notebook does:

1. `group = dti_dg_group(path=...)`, `benchmark = group.get('BindingDB_Patent')`.
   The split is TDC's; we do not touch it.
2. Featurizes every molecule and protein appearing in train ∪ test. All
   featurizers are frozen, pretrained and label-free:
   - molecule: Morgan fingerprint (radius 2, 1024 bits) + 217 RDKit descriptors
     + ChemBERTa-77M-MTR mean-pooled embedding → 1,625 dims
   - protein: ESM-2 650M (`facebook/esm2_t33_650M_UR50D`) mean-pooled, with
     sequences over 1,000 residues split into windows and averaged → 1,280 dims
3. For seeds 1–5: `group.get_train_valid_split(..., seed=seed)`, trains XGBoost
   (1200 trees, depth 8, lr 0.05, subsample 0.8, colsample_bytree 0.6) on the
   **train** split only, predicts the test split.
4. `group.evaluate_many(predictions_list)` returns the official number.

Expected output:

```
{'bindingdb_patent': [0.591, 0.004]}
```

We ran this twice on independent runtimes and obtained the same value to three
decimals. The dataset contains 140,746 unique molecules and 477 unique
proteins; features are built for all of them, so no test row falls back to
imputation.

## ADMET Benchmark Group

**Notebook:** `notebooks/admet_v2_tdc.ipynb`
Runs all 22 benchmarks in the group with 5 seeds each and reports both our own
metric computation and TDC's `group.evaluate_many` output side by side. The
official numbers are written to `admet_official.json`; those are the ones that
belong in a submission.

Features: Morgan **count** fingerprint (2048) + MACCS keys (167) + 217 RDKit
descriptors + ChemBERTa-77M-MTR, with one XGBoost model per dataset.

## Other material in this repository

| Path | What |
|---|---|
| `notebooks/pocket_aware.ipynb` | KLIFS 85-residue pocket representation vs whole-sequence ESM-2 |
| `notebooks/ddg_feasibility.ipynb` | Counts the ΔΔG pairs extractable from Davis |
| `experiments/mutants/` | The Davis mutant-sequence correction and its measured effect |
| `experiments/repurposing/` | Per-target failure accounting; ranking corrections |
| `experiments/v3_bindingdb/` | Whether adding BindingDB data helps (it does not, for ranking) |
| `streamlit_app.py`, `core.py` | The demo application |
| `models/` | Trained models for the demo |

## Environment

Colab defaults plus: `rdkit`, `transformers`, `xgboost`, `scikit-learn`, and
`PyTDC`. Install PyTDC with `--no-deps` — installing it normally upgrades
`fsspec` and Colab then requires a session restart. The notebooks do this
already.

## A note on how results are reported here

Numbers in this repository are reported with the seed-to-seed spread, and
negative results are kept rather than dropped. `experiments/*/results/` contains
the runs that failed as well as the ones that worked, including the case where a
promising single-seed gain (+0.0195 CI) reversed once five seeds were run. The
reasoning is in `docs/` and in the commit messages.
