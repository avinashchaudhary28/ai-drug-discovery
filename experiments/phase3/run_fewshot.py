"""Phase 3 kadam 5: naye gene ke k maap dekar baaki ki bhavishyvani.

Zero-shot chaar koshishon me fail hua. Par koi company kabhi aise nahi
aati - uske paas apne target ke kuch maap hote hi hain. Asli sawal:

    "Humare paas apne kinase pe 20 maap hain. Baaki ke liye bata do."

Ye model ka badlaav nahi, SAWAL ka badlaav hai - aur ye wahi sawal hai
jo khareedaar poochta hai.

GATE PEHLE SE TAY, CODE ME LIKHA:
    paas = k=20 pe Spearman >= 0.30, AUR teeno baseline se saaf upar
           (0 bolo | shots ka ausat | shots se har dawa ka ausat),
           jisme sabse kathin wale se t >= 2.0

Teesra baseline hi asli imtihan hai. Agar model usse nahi jeeta, to
faida model ka nahi un k numbers ka hai - aur tab bechne ko kuch nahi.

Nateeja: results/fewshot_curve.csv, results/fewshot_report.json
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
from fewshot import (eligible, shots_drug_mean, shots_mean,  # noqa: E402
                     split_gene, take_shots)
from logo import metrics, paired_t                        # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
SRC = HERE.parent / "phase2" / "results" / "mutation_ddg_merged.csv"

KS = [int(x) for x in os.environ.get("KS", "0,5,10,20,50").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
N_TREES = int(os.environ.get("N_TREES", "400"))

GATE_K = 20
GATE_SPEARMAN = 0.30
GATE_T = 2.0


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
    df = pd.read_csv(SRC)
    df = df[(df.is_kinase == True)                         # noqa: E712
            & ~(df.censored_wt | df.censored_mut)
            & (df.mut_parsed == True)                      # noqa: E712
            & df.smiles.notna() & df.ddG.notna()].copy().reset_index(drop=True)

    print("[1/3] features")
    X, ok, _ = build(df)
    df = df[ok].copy().reset_index(drop=True)
    print(f"  {len(df):,} jode, {df.gene.nunique()} genes, "
          f"{X.shape[1]:,} features")

    genes, need = eligible(df, KS)
    print(f"\n  POORA curve sirf un genes pe jinme {need}+ jode hain: "
          f"{len(genes)}")
    print(f"  ({', '.join(genes)})")
    if len(genes) < 6:
        raise RuntimeError(f"sirf {len(genes)} genes - curve bharosemand "
                           f"nahi hoga")

    print(f"\n[2/3] curve: k = {KS}, {len(SEEDS)} seeds")
    recs = []
    for gene in genes:
        gidx = df.index[df.gene == gene].to_numpy()
        other = df.index[df.gene != gene].to_numpy()
        for seed in SEEDS:
            pool, test = split_gene(gidx, seed)
            te = df.loc[test]
            y = te.ddG.to_numpy(dtype=float)
            for k in KS:
                shots = take_shots(pool, k, seed)
                tr_idx = np.concatenate([other, shots])
                tr = df.loc[tr_idx]
                sh = df.loc[shots]
                ytr = tr.ddG.to_numpy(dtype=float)

                preds = {
                    "0 bolo": np.zeros(len(te)),
                    "shots ka ausat": np.full(
                        len(te), shots_mean(sh.ddG.to_numpy(dtype=float),
                                            ytr)),
                    "shots se har dawa ka ausat": shots_drug_mean(
                        sh, te, df.loc[other]),
                    "XGB (+shots)": fit(X[tr_idx], ytr, X[test], seed),
                }
                for nm, p in preds.items():
                    recs.append({"gene": gene, "k": k, "seed": seed,
                                 "model": nm, **metrics(y, p)})
        print(f"  {gene:<10} n={len(gidx):>5,}  ({time.time()-t0:.0f}s)",
              flush=True)

    r = pd.DataFrame(recs)
    per = (r.groupby(["gene", "k", "model"])
             .agg(n=("n", "first"), mae=("mae", "mean"),
                  spearman=("spearman", "mean")).reset_index())
    per.to_csv(RES / "fewshot_curve.csv", index=False)

    print(f"\n{'=' * 72}\nCURVE ({len(genes)} genes, har gene ek ikai)\n"
          f"{'=' * 72}\n")
    piv_s = per.pivot_table(index="k", columns="model", values="spearman")
    piv_m = per.pivot_table(index="k", columns="model", values="mae")
    print("Spearman (gene ke andar ranking - yahi asli cheez hai):")
    print(piv_s.round(3).to_string())
    print("\nMAE:")
    print(piv_m.round(3).to_string())

    # Gate
    hard = "shots se har dawa ka ausat"
    gk = per[per.k == GATE_K]
    A = gk[gk.model == "XGB (+shots)"].set_index("gene")
    out = {}
    print(f"\n{'=' * 72}\nGATE (k={GATE_K})\n{'=' * 72}\n")
    sp = float(A.spearman.mean())
    print(f"  XGB ka Spearman k={GATE_K} pe: {sp:+.3f}  "
          f"(chahiye >= {GATE_SPEARMAN})")
    ok_sp = sp >= GATE_SPEARMAN
    ok_all = ok_sp
    for b in ("0 bolo", "shots ka ausat", hard):
        B = gk[gk.model == b].set_index("gene")
        c = A.index.intersection(B.index)
        t = paired_t((A.loc[c, "spearman"] - B.loc[c, "spearman"]).to_numpy())
        tm = paired_t((B.loc[c, "mae"] - A.loc[c, "mae"]).to_numpy())
        beat = t["t"] >= GATE_T if b == hard else t["mean"] > 0
        out[b] = {"spearman": t, "mae": tm, "beaten": bool(beat)}
        print(f"  vs {b:<28} dSpearman {t['mean']:+.3f} t={t['t']:+6.2f} "
              f"jeeta {t['wins']}/{t['n']}"
              + ("   <- sabse kathin" if b == hard else ""))
        if b == hard:
            ok_all = ok_all and beat
        else:
            ok_all = ok_all and beat

    # Zero-shot se kitna upar - yahi "shots ka faida" hai
    z = per[(per.k == 0) & (per.model == "XGB (+shots)")].set_index("gene")
    c = A.index.intersection(z.index)
    lift = paired_t((A.loc[c, "spearman"] - z.loc[c, "spearman"]).to_numpy())
    print(f"\n  k=0 se k={GATE_K} tak XGB ka sudhaar: "
          f"{lift['mean']:+.3f}  t={lift['t']:+6.2f}  "
          f"jeeta {lift['wins']}/{lift['n']}")

    print(f"\n{'=' * 72}\nFAISLA\n{'=' * 72}")
    if ok_all:
        print(f"  GATE PAAS. k={GATE_K} maap dene par model Spearman "
              f"{sp:.3f} deta hai aur teeno baseline se upar hai.")
        print(f"  Ye ek SAAF offer hai: 'apne target ke {GATE_K} maap do, "
              f"baaki hum batayenge.'")
    else:
        why = []
        if not ok_sp:
            why.append(f"Spearman {sp:.3f} < {GATE_SPEARMAN}")
        for b, v in out.items():
            if not v["beaten"]:
                why.append(f"'{b}' se nahi jeeta")
        print(f"  GATE FAIL: {'; '.join(why)}")
        print(f"  Agar model un k numbers se hi nahi jeeta, to faida "
              f"model ka nahi - un numbers ka hai.")

    (RES / "fewshot_report.json").write_text(json.dumps({
        "genes": genes, "need_rows": need, "ks": KS, "seeds": SEEDS,
        "gate": {"k": GATE_K, "spearman": GATE_SPEARMAN, "t": GATE_T},
        "spearman_curve": json.loads(piv_s.to_json()),
        "mae_curve": json.loads(piv_m.to_json()),
        "gate_checks": out, "zero_to_k_lift": lift,
        "spearman_at_gate_k": sp, "gate_passed": bool(ok_all)}, indent=2))
    print(f"\n  files: {RES}/fewshot_curve.csv, fewshot_report.json "
          f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
