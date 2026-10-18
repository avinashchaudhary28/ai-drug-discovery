# 💊 AI Drug Discovery — Phase 1

Molecule (SMILES) do → ~670 kinase proteins pe binding prediction, 22 ADMET properties, 12 Tox21 toxicity tests aur solubility.

| Module | Model | Data | Result |
|---|---|---|---|
| Binding v2 | Morgan FP + RDKit descriptors + ChemBERTa + ESM-2 → XGBoost | Davis + KIBA | Davis CI 0.888 (random), 0.675 (new drug) · KIBA CI 0.856 |
| ADMET v2 | Morgan count FP + MACCS + descriptors + ChemBERTa → XGBoost | TDC ADMET Benchmark Group (22 datasets) | 17/22 within 5% of a top leaderboard model (5 seeds, official test split) |
| Toxicity | Morgan FP + descriptors → XGBoost | Tox21 (12 assays) | mean ROC-AUC 0.852 (random), 0.772 (scaffold) |
| Solubility | same | ESOL | RMSE 0.666 (random), 0.880 (scaffold) |

Run: `streamlit run streamlit_app.py`

Research demo only — not medical advice.
