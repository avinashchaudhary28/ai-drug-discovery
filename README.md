# 💊 AI Drug Discovery

Molecule (SMILES) do → ~670 kinase proteins pe binding prediction, 22 ADMET properties, 12 Tox21 toxicity tests aur solubility.

**TDC DTI-DG (BindingDB_Patent): PCC 0.591 ± 0.004** — statistically tied with
the top leaderboard entry (0.588 ± 0.002; 0.67 SE apart), clearly ahead of
second place (0.538 ± 0.008; 5.9 SE). Evaluated with TDC's own split and
evaluator over 5 seeds, reproduced on an independent run.
See [`REPRODUCE.md`](REPRODUCE.md) and the preprint
([`paper_preprint.pdf`](paper_preprint.pdf)).

> **Licensing.** This code is published for **reproducibility and verification**
> of the results reported in the preprint. No licence is granted: all rights are
> reserved. You are welcome to read, run and check the work. For any commercial
> use, redistribution, or inclusion in a product, please get in touch first.

| Module | Model | Data | Result |
|---|---|---|---|
| Binding v2 | Morgan FP + RDKit descriptors + ChemBERTa + ESM-2 → XGBoost | Davis + KIBA | Davis CI 0.888 (random), 0.675 (new drug) · KIBA CI 0.856 |
| ADMET v2 | Morgan count FP + MACCS + descriptors + ChemBERTa → XGBoost | TDC ADMET Benchmark Group (22 datasets) | 17/22 within 5% of a top leaderboard model (5 seeds, official test split) |
| Toxicity | Morgan FP + descriptors → XGBoost | Tox21 (12 assays) | mean ROC-AUC 0.852 (random), 0.772 (scaffold) |
| Solubility | same | ESOL | RMSE 0.666 (random), 0.880 (scaffold) |

Run: `streamlit run streamlit_app.py`

Research demo only — not medical advice.
