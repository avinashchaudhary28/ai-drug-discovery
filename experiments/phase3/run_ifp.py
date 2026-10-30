"""Phase 3 kadam 3: "mutation wahan hai jahan dawa lagti hai?"

Kadam 2 me pocket position akeli fail hui (Spearman -0.004, t=-0.16).
Soch ye hai ki sawal hi galat tha: position structural SLOT batati hai,
chemistry nahi.

Ab teesra arm: us position pe ligand kitni baar contact karta hai
(KLIFS ke interaction fingerprints se, 7 types ke saath). Ye mechanistic
hai aur cross-gene tulnay bhi - aur crystal structures se aata hai,
humare affinity labels se nahi, to leak mumkin nahi.

Teeno arm EK HI folds, EK HI seeds pe. Farak sirf features ka:
    XGB bina pocket  |  XGB + pocket  |  XGB + pocket + IFP

Nateeja: results/ifp_logo.csv, results/ifp_report.json
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
from klifs_ifp import IFP_NAMES, _get, fetch_gene_ifp, ifp_features  # noqa: E402
from logo import (BASELINES, eligible_genes, logo_splits,  # noqa: E402
                  metrics, paired_t)

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
WORK = HERE / "work_klifs"
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
    maps = json.loads(MAPS.read_text())
    df = pd.read_csv(SRC)
    df = df[(df.is_kinase == True)                         # noqa: E712
            & ~(df.censored_wt | df.censored_mut)
            & (df.mut_parsed == True)                      # noqa: E712
            & df.smiles.notna() & df.ddG.notna()]
    df = df[df.gene.isin(maps)].copy().reset_index(drop=True)
    folds = eligible_genes(df, MIN_PAIRS)
    print(f"[1/4] {len(df):,} jode, {len(folds)} folds")

    print("\n[2/4] features (bina pocket / pocket ke saath)")
    X0, X1, ok, _ = build_with_pocket(df, maps)
    df = df[ok].copy().reset_index(drop=True)
    # pocket block ka doosra column klifs_index hai (features.py:
    # POCKET_NAMES = in_pocket, klifs_index, klifs_index_frac)
    kidx = X1[:, X0.shape[1] + 1].astype(int)
    print(f"  bina pocket {X0.shape[1]:,} | pocket {X1.shape[1]:,} features")
    print(f"  pocket me mutations: {(kidx > 0).mean():.0%}")

    print("\n[3/4] IFP utaro (har gene ki ligand wali structures se)")
    aggs = {}
    for g in sorted(df.gene.unique()):
        ks = _get("/kinase_ID", {"kinase_name": g, "species": "HUMAN"})
        if not ks:
            print(f"  {g:<10} kinase_ID nahi mili - IFP nahi")
            continue
        a = fetch_gene_ifp(g, ks[0]["kinase_ID"], WORK)
        aggs[g] = a
        print(f"  {g:<10} {a['n']:>3} structures ka IFP"
              + (f"  ({a['n_bad_ifp']} kharab chhooti)"
                 if a.get("n_bad_ifp") else "")
              + f"  ({time.time()-t0:.0f}s)", flush=True)

    rows = [ifp_features(aggs.get(g), i) for g, i in zip(df.gene, kidx)]
    IFP = np.vstack(rows).astype(np.float32)
    X2 = np.hstack([X1, IFP]).astype(np.float32)
    nz = float((np.abs(IFP[:, :8]).sum(axis=1) > 0).mean())
    print(f"\n  IFP feature kitne jodon pe kuch kehta hai: {nz:.0%}")
    if nz < 0.15:
        print(f"  !! sirf {nz:.0%} - IFP ke paas kehne ko kuch hai hi "
              f"nahi, nateeja uske hisaab se padhna")

    print(f"\n[4/4] leave-one-gene-out, {len(SEEDS)} seeds")
    recs = []
    for gene, train, test in logo_splits(df, MIN_PAIRS):
        itr, ite = train.index.to_numpy(), test.index.to_numpy()
        y = test.ddG.to_numpy(dtype=float)
        ytr = train.ddG.to_numpy(dtype=float)
        for name, fn in BASELINES.items():
            recs.append({"gene": gene, "model": name, "seed": -1,
                         **metrics(y, fn(train, test))})
        for seed in SEEDS:
            for nm, X in (("XGB bina pocket", X0), ("XGB + pocket", X1),
                          ("XGB + pocket + IFP", X2)):
                recs.append({"gene": gene, "model": nm, "seed": seed,
                             **metrics(y, fit(X[itr], ytr, X[ite], seed))})
        print(f"  {gene:<10} n={len(test):>5,}  ({time.time()-t0:.0f}s)",
              flush=True)

    r = pd.DataFrame(recs)
    per_gene = (r.groupby(["gene", "model"])
                 .agg(n=("n", "first"), mae=("mae", "mean"),
                      rmse=("rmse", "mean"), spearman=("spearman", "mean"))
                 .reset_index())
    per_gene.to_csv(RES / "ifp_logo.csv", index=False)
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
        return (paired_t((B.loc[c, "mae"] - A.loc[c, "mae"]).to_numpy()),
                paired_t((A.loc[c, "spearman"]
                          - B.loc[c, "spearman"]).to_numpy()))

    out = {}
    print(f"\n  TULNA (dono taraf ek hi folds aur seeds)\n")
    for a, b, label in (
            ("XGB + pocket + IFP", "XGB + pocket", "IFP ne pocket me kuch joda?"),
            ("XGB + pocket + IFP", "XGB bina pocket", "IFP+pocket vs kuch nahi"),
            ("XGB + pocket + IFP", "0 bolo (farsh)", "farsh se")):
        tm, ts = cmp(a, b)
        out[label] = {"mae": tm, "spearman": ts}
        print(f"    {label}")
        print(f"      MAE      {tm['mean']:+.4f} kam  t={tm['t']:+6.2f}  "
              f"jeeta {tm['wins']}/{tm['n']}")
        print(f"      Spearman {ts['mean']:+.4f} zyada t={ts['t']:+6.2f}  "
              f"jeeta {ts['wins']}/{ts['n']}")

    ifp_t = out["IFP ne pocket me kuch joda?"]
    floor_t = out["farsh se"]
    helped = (ifp_t["mae"]["t"] >= 2.0 or ifp_t["spearman"]["t"] >= 2.0)
    broke = (floor_t["mae"]["t"] >= 2.0
             and floor_t["mae"]["wins"] >= 0.6 * floor_t["mae"]["n"])

    print(f"\n{'=' * 72}\nFAISLA\n{'=' * 72}")
    if helped and broke:
        print("  IFP se FARAK PADA aur farsh TOOTA.")
    elif helped:
        print("  IFP se farak pada, par farsh phir bhi nahi toota.")
    elif broke:
        print("  Farsh toota, par IFP ki wajah se NAHI.")
    else:
        print(f"  IFP se farak NAHI pada (MAE t={ifp_t['mae']['t']:+.2f}, "
              f"Spearman t={ifp_t['spearman']['t']:+.2f}), aur farsh bhi "
              f"nahi toota.")
        print(f"  Ye teesri koshish hai. Teen alag tareeke - raw position,"
              f"\n  aligned pocket position, aur ab 'dawa yahan lagti hai "
              f"ya nahi' -\n  sabhi andekhe gene pe '0 bolo' se nahi jeet "
              f"paye.")

    (RES / "ifp_report.json").write_text(json.dumps({
        "folds": folds, "pairs": int(len(df)),
        "ifp_coverage": nz,
        "summary": json.loads(summ.reset_index().to_json(orient="records")),
        "comparisons": out, "ifp_helped": bool(helped),
        "floor_broken": bool(broke)}, indent=2))
    print(f"\n  files: {RES}/ifp_logo.csv, ifp_report.json "
          f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
