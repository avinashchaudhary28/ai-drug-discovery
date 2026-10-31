"""Phase 3 kadam 6: top-3 hit rate - Spearman ko khareedaar ki bhasha me.

Spearman 0.34 se ye pata nahi chalta ki cheez bikne layak hai ya nahi.
Khareedaar ka sawaal hai:

    "Meri compound series hai. Ye resistance mutation clinic me aa gayi.
     Meri kaun si compound abhi bhi chalegi?"

Model ki top-3 compounds me asli sabse-tikne-wali kitni baar aati hai -
ye ek vaakya hai jo kisi ko samjhana nahi padta.

SAMOOH: (gene, mutation) - us ek mutation ke liye kai dawaiyan. Ye
framing data me TIKTA hai (75 samooh me 10+ dawa, 17 genes, 88% data).
Ulta framing nahi tikta: har (gene, dawa) me median SIRF EK mutation.

GATE PEHLE SE TAY, CODE ME LIKHA - RUN SE PEHLE COMMIT:

    paas = k=20 pe
      (a) hit@3 >= 0.60 (nirpeksh), AUR
      (b) `shots se har dawa ka ausat` se upar, paired t >= 2.0
          (gene hi ikai), AUR
      (c) jackknife-majboot: kisi EK gene ko hatane pe wo t 2.0 se
          neeche na gire

(b) ki wajah: wo baseline kehta hai "jo tumne apni har compound ka
naapa hai, usi ka ausat bol do". Agar model usse nahi jeeta to faida
model ka nahi, khareedaar ke apne numbers ka hai.

(c) ki wajah: 15-gene wale few-shot run me gate paas hua tha par ek
gene hatane pe gir jata tha. Ye haath se pakda gaya tha; ab khud chalta
hai.

`0 bolo` aur `shots ka ausat` sab dawaiyon ko EK HI number dete hain,
to wo kram hi nahi bana sakte - unka hit mauka (3/n) ke barabar hona
chahiye, aur wahi unka sahi sthan hai. Unhe gate me nahi rakha; unka
number sirf jaanch ke liye chhapa jata hai (agar wo mauke se bahut
upar nikle, to tie-breaking me bug hai).

Nateeja: results/topk_<TAG>.csv, results/topk_report<TAG>.json
"""
import json
import os
import pathlib
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build                                  # noqa: E402
from fewshot import (eligible, fit_shrink, shots_drug_mean,  # noqa: E402
                     shots_mean, split_gene, take_shots)
from logo import jackknife_t, paired_t                      # noqa: E402
from topk import MIN_GROUP, TOP, group_scores               # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
SRC = HERE.parent / "phase2" / "results" / "mutation_ddg_merged.csv"

KS = [int(x) for x in os.environ.get("KS", "0,5,10,20").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
N_TREES = int(os.environ.get("N_TREES", "400"))
TEST_FRAC = float(os.environ.get("TEST_FRAC", "0.5"))
OBJECTIVE = os.environ.get("OBJECTIVE", "reg:squarederror")
TAG = os.environ.get("TAG", "")

GATE_K = 20
GATE_HIT = 0.60
GATE_T = 2.0
HARD = "shots se har dawa ka ausat"
MODEL = "XGB (+shots)"
CONST = ("0 bolo", "shots ka ausat")


def stable_seed(*parts):
    """Python ka hash() strings pe HAR PROCESS me badalta hai
    (PYTHONHASHSEED), to usse tie-break ka seed banane se run dobara
    wahi nateeja nahi deta - aur ye chup-chaap hota hai. crc32 sthir
    hai."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2**32)


def dedupe(sub):
    """Ek dawa, ek pankti. 357/7722 samoohon me ek hi dawa ki kai rows
    hain (alag assay_type). Bina ise theek kiye wo dawa top-3 me kai
    seat le leti aur hit rate jhootha ho jata - bina kisi error ke."""
    return sub.groupby("drug_key", sort=True).agg(
        ddG=("ddG", "median"), pred=("pred", "mean"))


def _mk(seed):
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=N_TREES, max_depth=6, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
        objective=OBJECTIVE, tree_method="hist", max_bin=128,
        random_state=seed, n_jobs=0)


def main():
    t0 = time.time()
    if GATE_K not in KS:
        raise RuntimeError(f"gate k={GATE_K} KS me hona chahiye, mila {KS}")
    RES.mkdir(exist_ok=True)
    df = pd.read_csv(SRC)
    df = df[(df.is_kinase == True)                           # noqa: E712
            & ~(df.censored_wt | df.censored_mut)
            & (df.mut_parsed == True)                        # noqa: E712
            & df.smiles.notna() & df.ddG.notna()].copy().reset_index(drop=True)

    print("[1/3] features")
    X, ok, _ = build(df)
    df = df[ok].copy().reset_index(drop=True)
    genes, need = eligible(df, KS, test_frac=TEST_FRAC)
    print(f"  {len(df):,} jode | test_frac={TEST_FRAC}, max k={max(KS)} "
          f"-> {need}+ jode wale {len(genes)} genes")
    print(f"  ({', '.join(genes)})")

    print(f"\n[2/3] top-{TOP} hit rate: k = {KS}, {len(SEEDS)} seeds")
    recs = []
    for gene in genes:
        gidx = df.index[df.gene == gene].to_numpy()
        other = df.index[df.gene != gene].to_numpy()
        for seed in SEEDS:
            pool, test = split_gene(gidx, seed, test_frac=TEST_FRAC)
            te = df.loc[test]
            # Samooh: ek mutation, kai dawaiyan. Chhote samooh chhoot
            # jate hain - top-3 ka matlab tabhi hai jab samooh bada ho.
            grp = {m: g.index.to_numpy()
                   for m, g in te.groupby("mutation")
                   if g.drug_key.nunique() >= MIN_GROUP}
            if not grp:
                continue
            cal_m = _mk(seed + 777)
            cal_m.fit(X[other], df.loc[other].ddG.to_numpy(dtype=float))
            oos = dict(zip(pool.tolist(), cal_m.predict(X[pool]).tolist()))

            for k in KS:
                shots = take_shots(pool, k, seed)
                tr_idx = np.concatenate([other, shots])
                sh = df.loc[shots]
                ytr = df.loc[tr_idx].ddG.to_numpy(dtype=float)
                m = _mk(seed)
                m.fit(X[tr_idx], ytr)
                pm = dict(zip(test.tolist(), m.predict(X[test]).tolist()))
                a = (fit_shrink([oos[i] for i in shots.tolist()],
                                sh.ddG.to_numpy(dtype=float))
                     if len(shots) else 1.0)

                sm = shots_mean(sh.ddG.to_numpy(dtype=float), ytr)
                hard_p = dict(zip(test.tolist(),
                                  shots_drug_mean(sh, te,
                                                  df.loc[other]).tolist()))
                for mut, rows in grp.items():
                    base = df.loc[rows, ["drug_key", "ddG"]].copy()
                    mdl = np.array([pm[i] for i in rows])
                    preds = {
                        "0 bolo": np.zeros(len(rows)),
                        "shots ka ausat": np.full(len(rows), sm),
                        HARD: np.array([hard_p[i] for i in rows]),
                        MODEL: mdl,
                        "XGB (+shots, cal)": a * mdl,
                    }
                    # Tie-break ka rng samooh+seed se - har model ko
                    # WAHI tie-break mile, warna tulna galat hogi
                    sd = stable_seed(gene, mut, seed, k)
                    for nm, p in preds.items():
                        sub = base.assign(pred=p)
                        g1 = dedupe(sub)       # ek dawa, ek pankti
                        s = group_scores(g1.ddG.to_numpy(),
                                         g1.pred.to_numpy(),
                                         np.random.default_rng(sd))
                        if s is None:
                            continue
                        recs.append({"gene": gene, "mutation": mut, "k": k,
                                     "seed": seed, "model": nm, **s})
        print(f"  {gene:<10} ({time.time()-t0:.0f}s)", flush=True)

    r = pd.DataFrame(recs)
    if r.empty:
        raise RuntimeError("koi samooh nahi bana")
    r.to_csv(RES / f"topk{TAG}.csv", index=False)

    # Gene hi ikai: pehle gene ke andar ausat, phir genes pe
    per = (r.groupby(["gene", "k", "model"])
             .agg(hit=("hit", "mean"), overlap=("overlap", "mean"),
                  rnd=("random_hit", "mean"), groups=("hit", "size"))
             .reset_index())
    per.to_csv(RES / f"topk_per_gene{TAG}.csv", index=False)

    print(f"\n{'=' * 72}\nTOP-{TOP} HIT RATE ({len(genes)} genes, "
          f"har gene ek ikai)\n{'=' * 72}\n")
    piv = per.pivot_table(index="k", columns="model", values="hit")
    print(f"hit@{TOP} - asli sabse-tikne-wali dawa model ke top-{TOP} me:")
    print(piv.round(3).to_string())
    print(f"\n  mauke se ({TOP}/n): "
          f"{per[per.model == MODEL].rnd.mean():.3f}")
    print(f"\noverlap@{TOP}:")
    print(per.pivot_table(index="k", columns="model",
                          values="overlap").round(3).to_string())

    # Jaanch: ek-hi-number model ka hit mauke ke paas hona chahiye
    print("\nSEHAT JAANCH (ek-hi-number model mauke ke paas ho):")
    gk = per[per.k == GATE_K]
    rnd = float(gk[gk.model == MODEL].rnd.mean())
    for c in CONST:
        h = float(gk[gk.model == c].hit.mean())
        flag = "" if abs(h - rnd) < 0.12 else "   <- MAUKE SE DOOR, " \
                                              "tie-break me bug?"
        print(f"  {c:<28} hit={h:.3f}  (mauka {rnd:.3f}){flag}")

    A = gk[gk.model == MODEL].set_index("gene")
    B = gk[gk.model == HARD].set_index("gene")
    c = A.index.intersection(B.index)
    d = (A.loc[c, "hit"] - B.loc[c, "hit"]).to_numpy()
    t = paired_t(d)
    jk = jackknife_t(d, list(c), thresh=GATE_T)
    hit = float(A.hit.mean())

    print(f"\n{'=' * 72}\nGATE (k={GATE_K})\n{'=' * 72}\n")
    ok_hit = bool(hit >= GATE_HIT)
    print(f"  (a) hit@{TOP} = {hit:.3f}   (chahiye >= {GATE_HIT})"
          f"   {'PAAS' if ok_hit else 'FAIL'}")
    ok_t = bool(t["t"] >= GATE_T)
    print(f"  (b) vs {HARD}: {t['mean']:+.3f}  t={t['t']:+.2f}  "
          f"jeeta {t['wins']}/{t['n']}   {'PAAS' if ok_t else 'FAIL'}")
    ok_jk = bool(jk["min_t"] is not None and jk["min_t"] >= GATE_T)
    print(f"  (c) jackknife: {jk['worst_gene']} hatane pe t="
          f"{jk['min_t']:+.2f}   {'PAAS' if ok_jk else 'FAIL'}")

    passed = bool(ok_hit and ok_t and ok_jk)
    print(f"\n{'=' * 72}\nFAISLA\n{'=' * 72}")
    if passed:
        print(f"  PAAS. Model ki top-{TOP} compounds me asli sabse-tikne-"
              f"wali {hit:.0%} baar aati hai (mauka {rnd:.0%}),")
        print(f"  khareedaar ke apne numbers se upar, aur kisi ek gene pe "
              f"nahi tika.")
        print(f"  Bechne wala vaakya: 'apne kinase ke {GATE_K} maap do - "
              f"hum batayenge aapki kaun si {TOP} compounds tikengi.'")
    else:
        why = []
        if not ok_hit:
            why.append(f"hit@{TOP} {hit:.3f} < {GATE_HIT}")
        if not ok_t:
            why.append(f"baseline se t {t['t']:+.2f} < {GATE_T}")
        if not ok_jk:
            why.append(f"ek gene ({jk['worst_gene']}) hatane pe t "
                       f"{jk['min_t']:+.2f} < {GATE_T}")
        print(f"  FAIL: {'; '.join(why)}")
        print(f"  Is dave ko deck me nahi daalna. Dataset asli asset hai.")

    (RES / f"topk_report{TAG}.json").write_text(json.dumps({
        "genes": genes, "n_genes": len(genes), "ks": KS, "seeds": SEEDS,
        "test_frac": TEST_FRAC, "objective": OBJECTIVE, "top": TOP,
        "min_group": MIN_GROUP,
        "gate": {"k": GATE_K, "hit": GATE_HIT, "t": GATE_T,
                 "baseline": HARD, "model": MODEL},
        "hit_curve": json.loads(piv.to_json()),
        "hit_at_gate_k": hit, "random_hit": rnd,
        "vs_baseline": t, "jackknife": jk,
        # bool() zaroori: numpy ka bool_ json me nahi jata, aur ye
        # aakhri line pe phatta hai - poora run ho chuka hota hai
        "checks": {"hit": bool(ok_hit), "t": bool(ok_t),
                   "jackknife": bool(ok_jk)},
        "passed": bool(passed)}, indent=2))
    print(f"\n  files: {RES}/topk{TAG}.csv, topk_report{TAG}.json "
          f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
