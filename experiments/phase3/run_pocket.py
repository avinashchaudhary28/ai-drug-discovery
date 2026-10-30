"""Phase 3 kadam 2: kya KLIFS pocket position farsh todti hai?

Kadam 1 ka nateeja: model gene yaad kar leta hai (random split pe
Spearman 0.752) par andekhe gene pe kuch nahi karta (0.067). Wajah:
raw position cross-gene tulnay nahi hai.

KLIFS pocket position wahi tulna deti hai - har kinase me "position 45"
ek hi structural jagah. Sawal: isse farak padta hai ya nahi?

YAHAN EK CHEEZ NATEEJE KO JHOOTHA BANA SAKTI THI, aur usse bacha gaya
hai. KLIFS mapping 22 me se sirf 14 folds pe bani. Agar pocket wala
model in 14 pe chale aur usse purane 22-fold ke farsh se tulna ki jaye,
to sudhaar pocket ka nahi - kathin genes (LRRK2, BRAF, FLT3, MET) hata
dene ka hoga. Isliye SAB kuch - farsh, bina-pocket model, pocket model -
ek hi 14 folds pe chalta hai. Asli tulna ye hai:

    XGB bina pocket   vs   XGB pocket ke saath     (ek hi folds pe)

Nateeja: results/pocket_logo.csv, results/pocket_report.json
"""
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_with_pocket                    # noqa: E402
from logo import (BASELINES, eligible_genes, logo_splits,  # noqa: E402
                  metrics, paired_t)

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
SRC = HERE.parent / "phase2" / "results" / "mutation_ddg_merged.csv"
MAPS = RES / "klifs_map.json"

MIN_PAIRS = int(os.environ.get("MIN_PAIRS", "20"))
N_TREES = int(os.environ.get("N_TREES", "400"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]


def fit(Xtr, ytr, Xte, seed):
    import xgboost as xgb
    m = xgb.XGBRegressor(
        n_estimators=N_TREES, max_depth=6, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
        tree_method="hist", max_bin=128, random_state=seed, n_jobs=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    if not MAPS.exists():
        raise RuntimeError(f"{MAPS} nahi mila - pehle run_klifs_map.py")
    maps = json.loads(MAPS.read_text())
    print(f"[1/3] data aur mapping ({len(maps)} genes ki mapping hai)")

    df = pd.read_csv(SRC)
    df = df[(df.is_kinase == True)                         # noqa: E712
            & ~(df.censored_wt | df.censored_mut)
            & (df.mut_parsed == True)                      # noqa: E712
            & df.smiles.notna() & df.ddG.notna()]
    n_all = len(df)
    # SIRF wahi genes jinki mapping jaanch paas kar chuki hai - aur
    # farsh bhi inhi pe, warna tulna jhoothi ho jayegi
    df = df[df.gene.isin(maps)].copy().reset_index(drop=True)
    folds = eligible_genes(df, MIN_PAIRS)
    print(f"  {n_all:,} -> mapping wale genes pe {len(df):,} jode")
    print(f"  LOGO folds: {len(folds)}  ({', '.join(folds)})")
    if len(folds) < 8:
        raise RuntimeError(f"sirf {len(folds)} folds - tulna bharosemand "
                           f"nahi hogi")

    print("\n[2/3] features")
    X0, X1, ok, (n0, n1) = build_with_pocket(df, maps)
    df = df[ok].copy().reset_index(drop=True)
    pf = X1[:, X0.shape[1]:]
    in_pocket = float(pf[:, 0].mean())
    print(f"  bina pocket : {X0.shape[1]:,} features")
    print(f"  pocket ke saath: {X1.shape[1]:,} features "
          f"(+{X1.shape[1] - X0.shape[1]})")
    print(f"  jin jodon ka mutation pocket me hai: {in_pocket:.0%}")
    if in_pocket < 0.2:
        print(f"  !! sirf {in_pocket:.0%} - pocket feature ke paas kehne "
              f"ko kuch hai hi nahi, nateeja uske hisaab se padhna")

    print(f"\n[3/3] leave-one-gene-out, {len(SEEDS)} seeds")
    rows = []
    for gene, train, test in logo_splits(df, MIN_PAIRS):
        itr, ite = train.index.to_numpy(), test.index.to_numpy()
        y = test.ddG.to_numpy(dtype=float)
        ytr = train.ddG.to_numpy(dtype=float)

        for name, fn in BASELINES.items():
            rows.append({"gene": gene, "model": name, "seed": -1,
                         **metrics(y, fn(train, test))})
        for seed in SEEDS:
            rows.append({"gene": gene, "model": "XGB bina pocket",
                         "seed": seed,
                         **metrics(y, fit(X0[itr], ytr, X0[ite], seed))})
            rows.append({"gene": gene, "model": "XGB + pocket", "seed": seed,
                         **metrics(y, fit(X1[itr], ytr, X1[ite], seed))})
        print(f"  {gene:<10} n={len(test):>5,}  ({time.time()-t0:.0f}s)",
              flush=True)

    r = pd.DataFrame(rows)
    per_gene = (r.groupby(["gene", "model"])
                 .agg(n=("n", "first"), mae=("mae", "mean"),
                      rmse=("rmse", "mean"), spearman=("spearman", "mean"))
                 .reset_index())
    per_gene.to_csv(RES / "pocket_logo.csv", index=False)

    summ = (per_gene.groupby("model")
            .agg(genes=("gene", "nunique"), mae=("mae", "mean"),
                 rmse=("rmse", "mean"), spearman=("spearman", "mean"))
            .sort_values("mae"))
    print(f"\n{'=' * 72}\nNATEEJA (ek hi {len(folds)} folds pe sab)\n"
          f"{'=' * 72}\n")
    print(summ.to_string())

    def cmp(a, b):
        A = per_gene[per_gene.model == a].set_index("gene")
        B = per_gene[per_gene.model == b].set_index("gene")
        c = A.index.intersection(B.index)
        t_mae = paired_t((B.loc[c, "mae"] - A.loc[c, "mae"]).to_numpy())
        sa = A.loc[c, "spearman"].to_numpy()
        sb = B.loc[c, "spearman"].to_numpy()
        t_sp = paired_t(sa - sb)
        return t_mae, t_sp, float(np.nanmean(sa)), float(np.nanmean(sb))

    print(f"\n  ASLI TULNA - pocket se farak pada ya nahi")
    print(f"  (dono ek hi folds, ek hi seeds, sirf features ka farak)\n")
    t_mae, t_sp, sp_p, sp_n = cmp("XGB + pocket", "XGB bina pocket")
    print(f"    MAE      : {t_mae['mean']:+.4f} kam  t={t_mae['t']:+6.2f}  "
          f"jeeta {t_mae['wins']}/{t_mae['n']} genes")
    print(f"    Spearman : {t_sp['mean']:+.4f} zyada t={t_sp['t']:+6.2f}  "
          f"jeeta {t_sp['wins']}/{t_sp['n']} genes")
    print(f"    Spearman: pocket {sp_p:+.3f}  vs  bina pocket {sp_n:+.3f}")

    fm, fs, sp_pk, _ = cmp("XGB + pocket", "0 bolo (farsh)")
    print(f"\n  farsh se tulna (pocket model vs '0 bolo'):")
    print(f"    MAE      : {fm['mean']:+.4f}  t={fm['t']:+6.2f}  "
          f"jeeta {fm['wins']}/{fm['n']} genes")

    broke = (fm["t"] >= 2.0 and fm["wins"] >= 0.6 * fm["n"])
    helped = (t_sp["t"] >= 2.0 or t_mae["t"] >= 2.0)
    print(f"\n{'=' * 72}\nFAISLA\n{'=' * 72}")
    if broke and helped:
        print(f"  Pocket se FARAK PADA aur farsh TOOTA.")
    elif helped:
        print(f"  Pocket se farak pada, par farsh phir bhi nahi toota - "
              f"sudhaar asli hai par kaafi nahi.")
    elif broke:
        print(f"  Farsh toota, par pocket ki wajah se NAHI (t={t_sp['t']:+.2f}).")
    else:
        print(f"  Pocket se farak NAHI pada (MAE t={t_mae['t']:+.2f}, "
              f"Spearman t={t_sp['t']:+.2f}), aur farsh bhi nahi toota.")
        print(f"  Yaani aligned position akela kaafi nahi hai. Agla sawal: "
              f"poora mutated pocket sequence (ESM-2, GPU) kuch deta hai "
              f"ya nahi - par ab wo bhi maan kar nahi, naap kar.")

    print(f"\n  Yaad rahe: ye {len(folds)} genes wo hain jinki mapping "
          f"jaanch paas kar payi. LRRK2, BRAF, FLT3, MET bahar hain - "
          f"unke mutations ya to pocket ke bahar hain ya numbering nahi "
          f"milti. Ye nateeja un genes ke baare me kuch nahi kehta.")

    rep = {"folds": folds, "pairs": int(len(df)),
           "in_pocket_frac": in_pocket,
           "n_features_base": int(X0.shape[1]),
           "n_features_pocket": int(X1.shape[1]),
           "summary": json.loads(summ.reset_index().to_json(orient="records")),
           "pocket_vs_nopocket": {"mae": t_mae, "spearman": t_sp,
                                  "spearman_pocket": sp_p,
                                  "spearman_nopocket": sp_n},
           "pocket_vs_floor_mae": fm,
           "floor_broken": bool(broke), "pocket_helped": bool(helped)}
    (RES / "pocket_report.json").write_text(json.dumps(rep, indent=2))
    print(f"\n  files: {RES}/pocket_logo.csv, pocket_report.json "
          f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
