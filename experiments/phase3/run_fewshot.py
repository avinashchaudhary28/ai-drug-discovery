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
from fewshot import (A_FLOOR, eligible, fit_shrink,          # noqa: E402
                     shots_drug_mean, shots_mean, split_gene, take_shots)
from logo import metrics, paired_t                        # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
SRC = HERE.parent / "phase2" / "results" / "mutation_ddg_merged.csv"

KS = [int(x) for x in os.environ.get("KS", "0,5,10,20,50").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
N_TREES = int(os.environ.get("N_TREES", "400"))
# Gene ki ginti hi `t` ki seema hai. max(k) ghatao ya test_frac ghatao,
# to gene set BADA ho jata hai - par wo gene set ALAG bhi ho jata hai.
TEST_FRAC = float(os.environ.get("TEST_FRAC", "0.5"))
TAG = os.environ.get("TAG", "")  # files ko alag naam, purana nateeja bacha rahe

# Gate MAE naapta hai par model squared error pe seekhta tha - wo
# outliers ke peeche bhaagta hai. Jo naapa jaa raha hai, wahi seekho.
OBJECTIVE = os.environ.get("OBJECTIVE", "reg:absoluteerror")

GATE_K = 20
GATE_SPEARMAN = 0.30
GATE_T = 2.0
# Gate KIS model pe lagega - ye NUMBER DEKHNE SE PEHLE tay hota hai,
# launch ke waqt. Do model me se baad me behtar chun lena do mauke lena
# hai, ek nahi - aur khareedaar ke saamne wo nahi tikta. Dono ke number
# report me likhe jate hain taaki kuch chhupa na rahe.
GATE_MODEL = os.environ.get("GATE_MODEL", "XGB (+shots, cal)")

MAIN = "XGB (+shots)"
CAL = "XGB (+shots, cal)"


def _mk(seed):
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=N_TREES, max_depth=6, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
        objective=OBJECTIVE,
        tree_method="hist", max_bin=128, random_state=seed, n_jobs=0)


def fit(Xtr, ytr, Xte, seed):
    m = _mk(seed)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def main():
    t0 = time.time()
    # Pehle hi - warna 20 minute chalkar fail hoga
    if GATE_K not in KS:
        raise RuntimeError(f"gate k={GATE_K} KS me hona chahiye, mila {KS}")
    if 0 not in KS:
        raise RuntimeError(f"k=0 KS me hona chahiye (zero-shot ka aadhaar), "
                           f"mila {KS}")
    if not 0.0 < TEST_FRAC < 1.0:
        raise RuntimeError(f"TEST_FRAC 0 aur 1 ke beech ho, mila {TEST_FRAC}")
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

    genes, need = eligible(df, KS, test_frac=TEST_FRAC)
    print(f"\n  test_frac={TEST_FRAC}, max k={max(KS)}")
    print(f"  POORA curve sirf un genes pe jinme {need}+ jode hain: "
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
            pool, test = split_gene(gidx, seed, test_frac=TEST_FRAC)
            te = df.loc[test]
            y = te.ddG.to_numpy(dtype=float)

            # Calibration ka model: SIRF doosre genes pe. Isne is gene ki
            # ek bhi row nahi dekhi, isliye pool pe iski prediction sach
            # me bahar ki hai - aur usi se `a` nikalna imaandaar hai.
            # Ye k pe nirbhar nahi hai, to ek baar - saare k ke liye.
            cal_m = _mk(seed + 777)
            cal_m.fit(X[other], df.loc[other].ddG.to_numpy(dtype=float))
            pool_oos = cal_m.predict(X[pool])
            oos_of = dict(zip(pool.tolist(), pool_oos.tolist()))

            for k in KS:
                shots = take_shots(pool, k, seed)
                tr_idx = np.concatenate([other, shots])
                tr = df.loc[tr_idx]
                sh = df.loc[shots]
                ytr = tr.ddG.to_numpy(dtype=float)

                main = fit(X[tr_idx], ytr, X[test], seed)

                # Shots se shrink factor. k=0 pe shots nahi hain, to
                # a=1 - aur ye sahi hai: bina maap calibration ka koi
                # aadhaar nahi. Yahi kahani bhi hai: maap do, faida lo.
                if len(shots):
                    a = fit_shrink([oos_of[i] for i in shots.tolist()],
                                   sh.ddG.to_numpy(dtype=float))
                else:
                    a = 1.0

                preds = {
                    "0 bolo": np.zeros(len(te)),
                    "shots ka ausat": np.full(
                        len(te), shots_mean(sh.ddG.to_numpy(dtype=float),
                                            ytr)),
                    "shots se har dawa ka ausat": shots_drug_mean(
                        sh, te, df.loc[other]),
                    MAIN: main,
                    CAL: a * main,
                }
                for nm, p in preds.items():
                    recs.append({"gene": gene, "k": k, "seed": seed,
                                 "model": nm, "a": a if nm == CAL else np.nan,
                                 **metrics(y, p)})
        print(f"  {gene:<10} n={len(gidx):>5,}  ({time.time()-t0:.0f}s)",
              flush=True)

    r = pd.DataFrame(recs)
    per = (r.groupby(["gene", "k", "model"])
             .agg(n=("n", "first"), mae=("mae", "mean"),
                  spearman=("spearman", "mean"), a=("a", "mean"))
             .reset_index())
    per.to_csv(RES / f"fewshot_curve{TAG}.csv", index=False)

    print(f"\n{'=' * 72}\nCURVE ({len(genes)} genes, har gene ek ikai)\n"
          f"{'=' * 72}\n")
    piv_s = per.pivot_table(index="k", columns="model", values="spearman")
    piv_m = per.pivot_table(index="k", columns="model", values="mae")
    print("Spearman (gene ke andar ranking - yahi asli cheez hai):")
    print(piv_s.round(3).to_string())
    print("\nMAE:")
    print(piv_m.round(3).to_string())

    sh_a = (per[per.model == CAL].pivot_table(index="k", values="a")
            .round(3))
    print("\nShrink factor `a` (shots se nikla, 1.0 = koi badlaav nahi):")
    print(sh_a.to_string())
    ar = r[r.model == CAL]
    floor = int((ar.a <= A_FLOOR + 1e-9).sum())
    print(f"  `a` farsh ({A_FLOOR}) pe aaya: {floor}/{len(ar)} baar"
          + ("   <- itni baar model ke paas maatra ki jaankari nahi thi"
             if floor else ""))

    # Gate
    hard = "shots se har dawa ka ausat"
    gk = per[per.k == GATE_K]
    A = gk[gk.model == GATE_MODEL].set_index("gene")
    if A.empty:
        raise RuntimeError(f"GATE_MODEL '{GATE_MODEL}' nateeje me nahi mila")
    out = {}
    print(f"\n{'=' * 72}\nGATE (k={GATE_K}, model: {GATE_MODEL})\n{'=' * 72}\n")
    sp = float(A.spearman.mean())
    print(f"  Spearman k={GATE_K} pe: {sp:+.3f}  "
          f"(chahiye >= {GATE_SPEARMAN})")
    ok_sp = sp >= GATE_SPEARMAN
    ok_all = ok_sp
    for b in ("0 bolo", "shots ka ausat", hard):
        B = gk[gk.model == b].set_index("gene")
        c = A.index.intersection(B.index)
        t = paired_t((A.loc[c, "spearman"] - B.loc[c, "spearman"]).to_numpy())
        tm = paired_t((B.loc[c, "mae"] - A.loc[c, "mae"]).to_numpy())
        # "0 bolo" aur "shots ka ausat" EK HI number dete hain. Ek hi
        # number se ranking hoti hi nahi, isliye unka Spearman NaN hai -
        # aur ye sahi hai. Par pehle gate ne NaN ko "nahi jeeta" gin
        # liya tha, aur ek paas hone wale nateeje ko FAIL likh diya.
        # Aise baseline se tulna MAE pe honi chahiye, ranking pe nahi.
        const_pred = not np.isfinite(t["mean"])
        if const_pred:
            beat = tm["mean"] > 0 and tm["t"] >= GATE_T
            how = f"MAE {tm['mean']:+.3f} t={tm['t']:+6.2f} " \
                  f"jeeta {tm['wins']}/{tm['n']}  (ranking nahi - ye " \
                  f"baseline ek hi number deta hai)"
        else:
            beat = t["t"] >= GATE_T if b == hard else t["mean"] > 0
            how = f"dSpearman {t['mean']:+.3f} t={t['t']:+6.2f} " \
                  f"jeeta {t['wins']}/{t['n']}"
        out[b] = {"spearman": t, "mae": tm, "beaten": bool(beat),
                  "compared_on": "mae" if const_pred else "spearman"}
        print(f"  vs {b:<28} {how}"
              + ("   <- sabse kathin" if b == hard else ""))
        ok_all = ok_all and beat

    # Zero-shot se kitna upar - yahi "shots ka faida" hai
    z = per[(per.k == 0) & (per.model == GATE_MODEL)].set_index("gene")
    c = A.index.intersection(z.index)
    lift = paired_t((A.loc[c, "spearman"] - z.loc[c, "spearman"]).to_numpy())
    print(f"\n  k=0 se k={GATE_K} tak sudhaar: "
          f"{lift['mean']:+.3f}  t={lift['t']:+6.2f}  "
          f"jeeta {lift['wins']}/{lift['n']}")

    # Doosra model bhi likho - gate usse nahi juda, par chhupana nahi hai
    alt_nm = MAIN if GATE_MODEL == CAL else CAL
    alt = gk[gk.model == alt_nm].set_index("gene")
    alt_out = {}
    if not alt.empty:
        print(f"\n  (jaankari ke liye - gate ISPE NAHI hai) {alt_nm}: "
              f"Spearman {alt.spearman.mean():+.3f}")
        for b in ("0 bolo", "shots ka ausat", hard):
            B = gk[gk.model == b].set_index("gene")
            c2 = alt.index.intersection(B.index)
            tm2 = paired_t((B.loc[c2, "mae"] - alt.loc[c2, "mae"]).to_numpy())
            ts2 = paired_t((alt.loc[c2, "spearman"]
                            - B.loc[c2, "spearman"]).to_numpy())
            alt_out[b] = {"spearman": ts2, "mae": tm2}
            print(f"    vs {b:<28} MAE {tm2['mean']:+.3f} t={tm2['t']:+6.2f} "
                  f"jeeta {tm2['wins']}/{tm2['n']}")

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

    (RES / f"fewshot_report{TAG}.json").write_text(json.dumps({
        "genes": genes, "n_genes": len(genes), "need_rows": need,
        "ks": KS, "seeds": SEEDS, "test_frac": TEST_FRAC, "tag": TAG,
        "note": ("gene set max(ks) aur test_frac se banta hai; dono me se "
                 "kuch badla to doosre run se seedhi tulna galat hogi"),
        "gate": {"k": GATE_K, "spearman": GATE_SPEARMAN, "t": GATE_T,
                 "model": GATE_MODEL, "objective": OBJECTIVE},
        "alt_model": {"name": alt_nm, "checks": alt_out},
        "shrink_a": json.loads(sh_a.to_json()),
        "spearman_curve": json.loads(piv_s.to_json()),
        "mae_curve": json.loads(piv_m.to_json()),
        "gate_checks": out, "zero_to_k_lift": lift,
        "spearman_at_gate_k": sp, "gate_passed": bool(ok_all)}, indent=2))
    print(f"\n  files: {RES}/fewshot_curve{TAG}.csv, "
          f"fewshot_report{TAG}.json ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
