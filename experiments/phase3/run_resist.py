"""Phase 3 koshish 6+7: resistance ka CLASSIFICATION + interaction features.

Paanch koshishein fail ho chuki hain. Ye unka dohraav NAHI hai - dono
badlaav seedhe `phase3_results.md` ke diagnosis se nikle hain.

BADLAAV 1 - sawaal badlo: regression -> classification
    `phase2_results.md` kehta hai label noise ~0.5 log units hai. Is
    modelling set ka MEDIAN |ddG| 0.492 hai. Yaani aadhe labels shor se
    alag hi nahi kiye ja sakte, aur model ko wahi fit karne ko kaha ja
    raha tha. Us doc me likha bhi tha ki |ddG| >= 1.0 wale subset pe
    naapna chahiye - aur wo aaj tak kiya nahi gaya.
    To ab sawaal: "kya ye mutation dawa ko TODTI hai (ddG <= -1)?"
    1,230 aise events hain (15.2%). AUROC shor ke prati majboot hai,
    aur yahi sawaal khareedaar bhi poochta hai.

BADLAAV 2 - interaction features (crossfeat.py)
    Diagnosis: "none describes the interaction geometry between them."
    Seedha drug x mutation cross nahi ban sakta (lakhon combinations,
    8,110 rows). To interaction ko aisi granularity pe laaya gaya hai
    jo seekhi ja sake aur jiske dono axes GENE SE AZAAD hon:
    scaffold-cluster x KLIFS-position. Isi liye KLIFS hai - uski
    numbering saare kinases me aligned hai.

DO GATE, DONO PEHLE SE TAY, CODE ME LIKHE, RUN SE PEHLE COMMIT:

    GATE 1 (sawaal badalne se faida hua?)
      ARM A (bina cross features): k=22 folds pe
        (a) ausat AUROC >= 0.65
        (b) `har dawa ka resistance rate` se paired t >= 2.0 (gene ikai)
        (c) jackknife-majboot

    GATE 2 (interaction features ne kuch joda?)
      ARM B (cross features ke saath):
        (a) ARM A se AUROC pe paired t >= 2.0
        (b) jackknife-majboot

Do alag sawaal, do alag jawab. Agar A paas ho aur B fail, to nateeja ye
hai ki "sawaal badalne se faida hua, interaction features se nahi" - aur
wahi likha jayega.

`har dawa ka resistance rate` hi sabse kathin baseline hai (pehle ke
saare runs me per-drug hi sabse majboot nikla), aur wo PEHLE SE tay hai,
number dekhkar nahi chuna gaya.

Nateeja: results/resist_<TAG>.csv, results/resist_report<TAG>.json
"""
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossfeat                                             # noqa: E402
from features import build, pocket_features, region_name     # noqa: E402
from logo import _sub, jackknife_t, paired_t                 # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
SRC = HERE.parent / "phase2" / "results" / "mutation_ddg_merged.csv"
MAPS = RES / "klifs_map.json"

THRESH = float(os.environ.get("THRESH", "-1.0"))   # resistance ki rekha
MIN_PAIRS = int(os.environ.get("MIN_PAIRS", "20"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
N_TREES = int(os.environ.get("N_TREES", "400"))
TAG = os.environ.get("TAG", "")

GATE_AUROC = 0.65
GATE_T = 2.0
HARD = "har dawa ka resistance rate"
ARM_A = "XGB (bina cross)"
ARM_B = "XGB (+cross)"


def auroc(y, p):
    """Gene ke andar AUROC. Ek hi class ho to NaN - 0.5 likhna jhooth
    hoga aur 0 likhna uससe bada jhooth."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2 or not np.isfinite(p).all():
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def auprc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2 or not np.isfinite(p).all():
        return float("nan")
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def clf(seed):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=N_TREES, max_depth=5, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", max_bin=128, random_state=seed, n_jobs=0)


# ------------------------------------------------------- bina model wale
def b_rate(tr, te, **_):
    """Sab ko ek hi number - AUROC 0.5 hoga, aur yahi uska sahi sthan."""
    return np.full(len(te), tr.y.mean())


def b_drug(tr, te, **_):
    m = tr.groupby("drug_key").y.mean()
    return te.drug_key.map(m).fillna(tr.y.mean()).to_numpy(dtype=float)


def b_sub(tr, te, **_):
    m = tr.assign(_s=tr.mutation.map(_sub)).groupby("_s").y.mean()
    return (te.mutation.map(_sub).map(m).fillna(tr.y.mean())
            .to_numpy(dtype=float))


def b_pos(tr, te, **_):
    m = tr.groupby("pos").y.mean()
    return te.pos.map(m).fillna(tr.y.mean()).to_numpy(dtype=float)


BASELINES = {"ek hi rate (farsh)": b_rate, HARD: b_drug,
             "AA badlav ka rate": b_sub, "pocket jagah ka rate": b_pos}


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    if not MAPS.exists():
        raise SystemExit(f"{MAPS} nahi mila - pehle run_klifs_map.py chalao")

    df = pd.read_csv(SRC)
    df = df[(df.is_kinase == True)                           # noqa: E712
            & ~(df.censored_wt | df.censored_mut)
            & (df.mut_parsed == True)                        # noqa: E712
            & df.smiles.notna() & df.ddG.notna()].reset_index(drop=True)

    print("[1/4] features")
    X, ok, names = build(df)
    df = df[ok].reset_index(drop=True)
    n_fp = sum(1 for n in names if n.startswith("fp"))

    maps = json.loads(MAPS.read_text())
    regions = sorted({region_name(h[1]) for info in maps.values()
                      for h in info["map"].values()})
    P, pos = [], []
    for g, mu in zip(df.gene, df.mutation):
        v, _ = pocket_features(g, mu, maps, regions)
        P.append(v)
        pos.append(int(v[1]) if v[0] > 0 else -1)
    X = np.hstack([X, np.vstack(P).astype(np.float32)])
    df["pos"] = pos
    df["y"] = (df.ddG <= THRESH).astype(int)

    vc = df.gene.value_counts()
    cand = sorted(vc[vc >= MIN_PAIRS].index)
    # Ek hi class wale gene pe AUROC banta hi nahi - wo fold chhoot jata
    # hai, aur ye PEHLE SE tay hai, nateeja dekhkar nahi.
    folds = [g for g in cand
             if 0 < int(df[df.gene == g].y.sum()) < int((df.gene == g).sum())]
    drop = sorted(set(cand) - set(folds))
    print(f"  {len(df):,} jode | {df.gene.nunique()} genes | "
          f"resistance (ddG <= {THRESH}): {int(df.y.sum())} "
          f"({100*df.y.mean():.1f}%)")
    print(f"  pocket position mili: {int((df.pos > 0).sum())}/{len(df)} "
          f"({100*(df.pos > 0).mean():.0f}%)")
    print(f"  folds: {len(folds)}  (ek-class hone se chhoote: "
          f"{', '.join(drop) if drop else 'koi nahi'})")
    if len(folds) < 8:
        raise SystemExit(f"sirf {len(folds)} folds - bharosemand nahi")

    print(f"\n[2/4] LOGO: {len(folds)} folds x {len(SEEDS)} seeds x 2 arms")
    recs = []
    for gene in folds:
        te_m = (df.gene == gene).to_numpy()
        tr_m = ~te_m
        tr, te = df[tr_m], df[te_m]
        ytr = tr.y.to_numpy(dtype=int)
        yte = te.y.to_numpy(dtype=int)

        for nm, fn in BASELINES.items():
            p = fn(tr, te)
            recs.append({"gene": gene, "seed": -1, "model": nm,
                         "auroc": auroc(yte, p), "auprc": auprc(yte, p),
                         "n": len(te), "pos": int(yte.sum())})

        for seed in SEEDS:
            # Cluster SIRF train pe fit - test ka dhaancha training me
            # nahi ghusna chahiye (ESM ke PCA wala sabak)
            km = crossfeat.fit_clusters(X[tr_m][:, :n_fp], seed=seed)
            cl_tr = crossfeat.assign(km, X[tr_m][:, :n_fp])
            cl_te = crossfeat.assign(km, X[te_m][:, :n_fp])
            Ctr, Cte = crossfeat.build(
                cl_tr, tr.pos.to_numpy(), tr.mutation.map(_sub).to_numpy(),
                ytr.astype(float),
                cl_te, te.pos.to_numpy(), te.mutation.map(_sub).to_numpy(),
                seed=seed)

            for arm, Xtr, Xte in ((ARM_A, X[tr_m], X[te_m]),
                                  (ARM_B, np.hstack([X[tr_m], Ctr]),
                                   np.hstack([X[te_m], Cte]))):
                m = clf(seed)
                m.fit(Xtr, ytr)
                p = m.predict_proba(Xte)[:, 1]
                recs.append({"gene": gene, "seed": seed, "model": arm,
                             "auroc": auroc(yte, p), "auprc": auprc(yte, p),
                             "n": len(te), "pos": int(yte.sum())})
        print(f"  {gene:<10} n={len(te):>5,} pos={int(yte.sum()):>4}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    r = pd.DataFrame(recs)
    r.to_csv(RES / f"resist{TAG}.csv", index=False)
    per = (r.groupby(["gene", "model"])
             .agg(auroc=("auroc", "mean"), auprc=("auprc", "mean"),
                  n=("n", "first"), pos=("pos", "first")).reset_index())

    print(f"\n{'=' * 72}\nRESISTANCE CLASSIFICATION ({len(folds)} folds, "
          f"gene hi ikai)\n{'=' * 72}\n")
    s = (per.groupby("model")[["auroc", "auprc"]].mean()
         .sort_values("auroc", ascending=False))
    print(s.round(3).to_string())
    print(f"\n  (base rate {df.y.mean():.3f}; AUROC 0.5 = mauka)")

    out = {}
    print(f"\n{'=' * 72}\nGATE 1 - sawaal badalne se faida? (ARM A)\n"
          f"{'=' * 72}\n")
    A = per[per.model == ARM_A].set_index("gene")
    a_auroc = float(A.auroc.mean())
    ok_a1 = bool(a_auroc >= GATE_AUROC)
    print(f"  (a) ausat AUROC {a_auroc:.3f}  (chahiye >= {GATE_AUROC})"
          f"   {'PAAS' if ok_a1 else 'FAIL'}")
    B0 = per[per.model == HARD].set_index("gene")
    c = A.index.intersection(B0.index)
    d = (A.loc[c, "auroc"] - B0.loc[c, "auroc"]).to_numpy()
    t1 = paired_t(d)
    jk1 = jackknife_t(d, list(c), thresh=GATE_T)
    ok_a2 = bool(t1["t"] >= GATE_T)
    ok_a3 = bool(np.isfinite(jk1["min_t"]) and jk1["min_t"] >= GATE_T)
    print(f"  (b) vs {HARD}: {t1['mean']:+.3f}  t={t1['t']:+.2f}  "
          f"jeeta {t1['wins']}/{t1['n']}   {'PAAS' if ok_a2 else 'FAIL'}")
    print(f"  (c) jackknife: {jk1['worst_gene']} hatane pe "
          f"t={jk1['min_t']:+.2f}   {'PAAS' if ok_a3 else 'FAIL'}")
    g1 = ok_a1 and ok_a2 and ok_a3
    out["gate1"] = {"auroc": a_auroc, "vs_hard": t1, "jackknife": jk1,
                    "checks": {"auroc": ok_a1, "t": ok_a2, "jk": ok_a3},
                    "passed": g1}

    print(f"\n{'=' * 72}\nGATE 2 - interaction features ne kuch joda? "
          f"(ARM B vs ARM A)\n{'=' * 72}\n")
    Bm = per[per.model == ARM_B].set_index("gene")
    c2 = Bm.index.intersection(A.index)
    d2 = (Bm.loc[c2, "auroc"] - A.loc[c2, "auroc"]).to_numpy()
    t2 = paired_t(d2)
    jk2 = jackknife_t(d2, list(c2), thresh=GATE_T)
    ok_b1 = bool(t2["t"] >= GATE_T)
    ok_b2 = bool(np.isfinite(jk2["min_t"]) and jk2["min_t"] >= GATE_T)
    print(f"  ARM B ka AUROC {Bm.auroc.mean():.3f} vs ARM A ka "
          f"{a_auroc:.3f}")
    print(f"  (a) dAUROC {t2['mean']:+.4f}  t={t2['t']:+.2f}  "
          f"jeeta {t2['wins']}/{t2['n']}   {'PAAS' if ok_b1 else 'FAIL'}")
    print(f"  (b) jackknife: {jk2['worst_gene']} hatane pe "
          f"t={jk2['min_t']:+.2f}   {'PAAS' if ok_b2 else 'FAIL'}")
    g2 = ok_b1 and ok_b2
    out["gate2"] = {"auroc_b": float(Bm.auroc.mean()), "vs_arm_a": t2,
                    "jackknife": jk2,
                    "checks": {"t": ok_b1, "jk": ok_b2}, "passed": g2}

    print(f"\n{'=' * 72}\nFAISLA\n{'=' * 72}")
    if g1 and g2:
        print("  DONO PAAS. Sawaal badalne se faida hua, AUR interaction "
              "features ne uske upar kuch joda.")
    elif g1:
        print("  GATE 1 PAAS, GATE 2 FAIL. Nateeja: regression chhodkar "
              "classification karne se faida hua;")
        print("  interaction features ne uske upar kuch NAHI joda - yaani "
              "us granularity pe drug x mutation")
        print("  interaction nahi mila. Ye ek asli jawab hai, shrug nahi.")
    elif g2:
        print("  GATE 1 FAIL, GATE 2 PAAS. Cross features ne ARM A se "
              "sudhaar diya, par ARM A khud baseline")
        print("  se upar nahi tha - to kul milakar bechne layak kuch nahi.")
    else:
        print("  DONO FAIL. Chhathi aur saatvi koshish bhi nahi chali.")
        print("  Diagnosis wahi rehta hai: interaction GEOMETRY chahiye, "
              "aur wo in features me nahi aati.")

    (RES / f"resist_report{TAG}.json").write_text(json.dumps({
        "folds": folds, "dropped_one_class": drop, "n_folds": len(folds),
        "threshold": THRESH, "seeds": SEEDS, "base_rate": float(df.y.mean()),
        "gate": {"auroc": GATE_AUROC, "t": GATE_T, "hard_baseline": HARD},
        "summary": json.loads(s.to_json()), **out,
        "both_passed": bool(g1 and g2)}, indent=2))
    print(f"\n  files: {RES}/resist{TAG}.csv, resist_report{TAG}.json "
          f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
