"""Phase 3, kadam 1: farsh banao, phir uspe pehla model rakho.

Sawal ek hi hai, aur nirmam hai:

    ANDEKHE gene pe, kya koi bhi cheez "hamesha 0 bolo" se behtar hai?

Ye farsh isliye zaroori hai kyunki ddG ka median lagbhag 0 hai. Aadha
data bahut chhote asar ka hai, to "0 bolo" ka MAE pehle hi achha dikhta
hai. Kaafi published mutation models is farsh se bas thoda hi upar hain.

Jaanch leave-one-gene-out hai: ek gene poora hata kar train, phir usi pe
test. Yahi wo haalat hai jo asli istemaal me aati hai - koi company apne
gene pe poochhegi, jis pe humne train nahi kiya.

Ikai GENE hai, row nahi. EGFR ki 2,376 rows ko 2,376 swatantra naap maan
lena jhootha bharosa deta hai - sab ek hi protein se aayi hain.

Nateeja: results/baseline_logo.csv, results/baseline_report.json
"""
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build                                # noqa: E402
from logo import (BASELINES, eligible_genes, logo_splits,  # noqa: E402
                  metrics, paired_t)

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
SRC = HERE.parent / "phase2" / "results" / "mutation_ddg_merged.csv"

MIN_PAIRS = int(os.environ.get("MIN_PAIRS", "20"))
N_TREES = int(os.environ.get("N_TREES", "400"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]

# Tulna ke liye published aankde. Ye SEEDHE comparable nahi hain (alag
# dataset, alag protocol) - isliye reference ki tarah chhapte hain,
# daawe ki tarah nahi.
PUBLISHED = {
    "TKI-ABL1, 8 inhibitors (best reported)": 0.39,
    "FEP+ (industry gold standard)": 0.37,
    "Platinum 10-fold CV (Pearson)": 0.20,
}


def load():
    if not SRC.exists():
        raise RuntimeError(f"{SRC} nahi mila - pehle Phase 2 ka merge chalao")
    df = pd.read_csv(SRC)
    n0 = len(df)
    df = df[df.is_kinase == True]                          # noqa: E712
    n1 = len(df)
    df = df[~(df.censored_wt | df.censored_mut)]
    n2 = len(df)
    df = df[df.mut_parsed == True]                         # noqa: E712
    df = df[df.smiles.notna() & df.ddG.notna()].copy()
    print(f"  {n0:,} -> kinase {n1:,} -> censored hataaye {n2:,} -> "
          f"istemaal layak {len(df):,}")
    print(f"  genes {df.gene.nunique()}, "
          f">= {MIN_PAIRS} jode wale {len(eligible_genes(df, MIN_PAIRS))}")
    return df


def fit_model(train, test, Xtr, Xte, seed):
    import xgboost as xgb
    m = xgb.XGBRegressor(
        n_estimators=N_TREES, max_depth=6, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
        tree_method="hist", max_bin=128, random_state=seed, n_jobs=0)
    m.fit(Xtr, train.ddG.to_numpy(dtype=float))
    return m.predict(Xte)


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    print("[1/3] data")
    df = load()

    print("\n[2/3] features (bina protein ke - sirf dawa aur mutation)")
    X, ok, names = build(df)
    if (~ok).sum():
        print(f"  !! {int((~ok).sum())} rows ki SMILES nahi padhi gayi - "
              f"wo gir gayin (shunya se NAHI bhari)")
    df = df[ok].copy().reset_index(drop=True)
    print(f"  X: {X.shape[0]:,} x {X.shape[1]:,}")

    print(f"\n[3/3] leave-one-gene-out ({len(SEEDS)} seeds)")
    rows = []
    for gene, train, test in logo_splits(df, MIN_PAIRS):
        itr = train.index.to_numpy()
        ite = test.index.to_numpy()
        y = test.ddG.to_numpy(dtype=float)

        for name, fn in BASELINES.items():
            rows.append({"gene": gene, "model": name, "seed": -1,
                         **metrics(y, fn(train, test))})

        for seed in SEEDS:
            p = fit_model(train, test, X[itr], X[ite], seed)
            rows.append({"gene": gene, "model": "XGB (dawa + mutation)",
                         "seed": seed, **metrics(y, p)})
        print(f"  {gene:<8} n={len(test):>5,}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    r = pd.DataFrame(rows)
    # Seeds ka ausat, taaki har gene pe har model ki ek line bane
    per_gene = (r.groupby(["gene", "model"])
                 .agg(n=("n", "first"), mae=("mae", "mean"),
                      rmse=("rmse", "mean"), pearson=("pearson", "mean"),
                      spearman=("spearman", "mean")).reset_index())
    per_gene.to_csv(RES / "baseline_logo.csv", index=False)

    print(f"\n{'=' * 70}\nNATEEJA (har gene ek ikai)\n{'=' * 70}\n")
    summ = (per_gene.groupby("model")
            .agg(genes=("gene", "nunique"), mae=("mae", "mean"),
                 rmse=("rmse", "mean"), spearman=("spearman", "mean"))
            .sort_values("mae"))
    print(summ.to_string())

    # Asli sawal: farsh se behtar? Har gene pe jodi banakar.
    floor = "0 bolo (farsh)"
    base = per_gene[per_gene.model == floor].set_index("gene")
    print(f"\n  '{floor}' se tulna (har gene pe MAE kitna GHATA):\n")
    cmp_rows = []
    for model in summ.index:
        if model == floor:
            continue
        m = per_gene[per_gene.model == model].set_index("gene")
        common = base.index.intersection(m.index)
        d = (base.loc[common, "mae"] - m.loc[common, "mae"]).to_numpy()
        t = paired_t(d)
        sp = float(np.nanmean(m.loc[common, "spearman"]))
        cmp_rows.append({"model": model, "d_mae": t["mean"], "t": t["t"],
                         "wins": t["wins"], "genes": t["n"], "spearman": sp})
        print(f"    {model:<26} dMAE {t['mean']:+.4f}  t={t['t']:+6.2f}  "
              f"jeeta {t['wins']}/{t['n']} genes  spearman {sp:+.3f}")

    best = max(cmp_rows, key=lambda c: c["t"]) if cmp_rows else None
    print(f"\n{'=' * 70}\nFAISLA\n{'=' * 70}")
    if best is None:
        print("  koi tulna nahi ho saki")
    elif best["t"] >= 2.0 and best["wins"] >= 0.6 * best["genes"]:
        print(f"  Farsh TOOTA: '{best['model']}' ne {best['wins']}/"
              f"{best['genes']} genes pe jeeta, t={best['t']:+.2f}")
        print(f"  Matlab andekhe gene pe kuch seekha ja raha hai. Ab "
              f"pocket-aware ko ISSE jeetna hoga.")
    else:
        print(f"  Farsh NAHI toota. Sabse achha: '{best['model']}', "
              f"t={best['t']:+.2f}, jeeta {best['wins']}/{best['genes']}")
        print(f"  Matlab bina protein ke roop ke kuch khaas nahi milta - "
              f"ye apne aap me ek nateeja hai, aur pocket-aware ka asli "
              f"kaaran bhi.")

    print(f"\n  Spearman (gene ke andar ranking) - yahi company poochhegi:")
    for c in sorted(cmp_rows, key=lambda c: -(c["spearman"] or -9)):
        print(f"    {c['model']:<26} {c['spearman']:+.3f}")
    print(f"\n  Reference (SEEDHE comparable NAHI - alag data, alag "
          f"protocol):")
    for k, v in PUBLISHED.items():
        print(f"    {k:<42} {v:.2f}")

    rep = {
        "pairs": int(len(df)), "genes_tested": int(per_gene.gene.nunique()),
        "min_pairs": MIN_PAIRS, "seeds": SEEDS, "n_trees": N_TREES,
        "summary": json.loads(summ.reset_index().to_json(orient="records")),
        "vs_floor": cmp_rows,
        "floor_broken": bool(best and best["t"] >= 2.0
                             and best["wins"] >= 0.6 * best["genes"]),
        "published_reference": PUBLISHED,
        "note": ("Ikai gene hai, row nahi. Published aankde seedhe "
                 "comparable nahi hain."),
    }
    (RES / "baseline_report.json").write_text(json.dumps(rep, indent=2))
    md = ["**Phase 3 kadam 1 - leave-one-gene-out farsh**", "",
          f"- {len(df):,} kinase jode, {per_gene.gene.nunique()} genes test",
          "", "```", summ.to_string(), "```"]
    (RES / "RESULT.md").write_text("\n".join(md))
    print(f"\n  files: {RES}/baseline_logo.csv, baseline_report.json "
          f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
