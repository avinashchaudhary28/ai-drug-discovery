"""run_topk ka poora raasta nakli xgboost/features se chalao.

Maqsad nateeja nahi - ye dekhna ki code TOOTTA nahi, aur gate ki teen
jaanch sach me bharti hain. Pichhle bugs 20+ minute Actions chalne ke
baad milte the.
"""
import os
import pathlib
import sys
import types

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(os.environ.get("DRYOUT", "/tmp/dryout_topk"))
OUT.mkdir(parents=True, exist_ok=True)


class FakeReg:
    def __init__(self, **kw):
        assert "objective" in kw
        self.rs = kw.get("random_state", 0)

    def fit(self, X, y):
        assert len(X) == len(y) and len(X) > 0
        self.n = X.shape[1]
        return self

    def predict(self, X):
        assert X.shape[1] == self.n
        rng = np.random.default_rng(self.rs)
        return (2.0 * X[:, 0] + rng.normal(0, 0.2, len(X))).astype(float)


xgb = types.ModuleType("xgboost")
xgb.XGBRegressor = FakeReg
sys.modules["xgboost"] = xgb

NF = 6


def fake_build(df):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (len(df), NF))
    X[:, 0] = df.ddG.to_numpy(dtype=float) * 0.5 + rng.normal(0, 0.3, len(df))
    return X, np.ones(len(df), dtype=bool), [f"f{i}" for i in range(NF)]


feat = types.ModuleType("features")
feat.build = fake_build
sys.modules["features"] = feat

# Nakla data: har gene me kai mutations, har mutation pe kai dawaiyan,
# aur jaan-boojh kar kuch (mutation, dawa) ki DO rows - dedupe jaanchne
# ke liye
rng = np.random.default_rng(1)
rows = []
for g, nmut in [("G1", 6), ("G2", 6), ("G3", 5), ("G4", 5), ("G5", 5),
                ("G6", 4), ("G7", 4), ("S1", 1)]:
    for mi in range(nmut):
        ndrug = int(rng.integers(6, 12))
        for di in range(ndrug):
            reps = 2 if (mi == 0 and di == 0) else 1
            for _ in range(reps):
                rows.append({"gene": g, "mutation": f"A{100+mi}V",
                             "drug_key": f"D{di}",
                             "ddG": float(rng.normal(0, 1)), "smiles": "CCO",
                             "is_kinase": True, "censored_wt": False,
                             "censored_mut": False, "mut_parsed": True,
                             "assay_type": "B"})
df = pd.DataFrame(rows)
csv = OUT / "fake.csv"
df.to_csv(csv, index=False)

sys.path.insert(0, str(HERE))
import run_topk as T                                          # noqa: E402

T.SRC = csv
T.RES = OUT
T.KS = [0, 20]
T.SEEDS = [0, 1]
T.N_TREES = 5
T.TEST_FRAC = 0.5
T.TAG = "_dry"
T.MIN_GROUP = 4          # nakle data me samooh chhote hain

rc = T.main()
print(f"\n--- dry run rc={rc}")

import json                                                   # noqa: E402

rep = json.loads((OUT / "topk_report_dry.json").read_text())
raw = pd.read_csv(OUT / "topk_dry.csv")

bad = []


def ck(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila: {got!r}\n      chahiye: {want!r}")
        bad.append(name)


print("\nreport")
ck("teen jaanch likhi", sorted(rep["checks"]), ["hit", "jackknife", "t"])
ck("passed teeno ka AND", rep["passed"],
   all(rep["checks"].values()))
ck("gate k=20", rep["gate"]["k"], 20)
ck("top=3", rep["top"], 3)
ck("jackknife bhara", "worst_gene" in rep["jackknife"], True)
ck("random_hit 0 se bada", bool(rep["random_hit"] > 0), True)

print("\nraw")
ck("chhe model (purana baseline bhi tulna ke liye)",
   sorted(raw.model.unique()),
   sorted(["0 bolo", "shots ka ausat", T.HARD, T.OLD_HARD, T.MODEL,
           "XGB (+shots, cal)"]))
ck("gate NAYE pooled baseline pe hai", T.HARD.startswith("pooled"), True)
ck("hit sirf 0 ya 1", sorted(raw.hit.unique()), [0.0, 1.0])
ck("overlap 0..1", bool(raw.overlap.between(0, 1).all()), True)
ck("n >= top+1 har jagah", bool((raw.n >= T.TOP + 1).all()), True)
ck("chhota gene chhoot gaya", "S1" in set(raw.gene), False)

# Sabse zaroori: ek-hi-number model ka hit mauke ke paas ho
for c in ("0 bolo", "shots ka ausat"):
    h = raw[(raw.model == c) & (raw.k == 20)].hit.mean()
    r = raw[(raw.model == c) & (raw.k == 20)].random_hit.mean()
    ck(f"{c}: hit ~ mauka ({h:.2f} vs {r:.2f})", bool(abs(h - r) < 0.2),
       True)

# dedupe sach me laga: ek samooh me ek dawa ek hi baar
g0 = raw[(raw.gene == "G1") & (raw.mutation == "A100V") & (raw.k == 20)
         & (raw.model == T.MODEL)]
src = df[(df.gene == "G1") & (df.mutation == "A100V")]
ck("us samooh me duplicate dawa thi",
   bool(src.groupby("drug_key").size().max() > 1), True)
ck("par score me dawa ek hi baar (n <= alag dawa)",
   bool(g0.n.max() <= src.drug_key.nunique()), True)

print()
if bad:
    print(f"✗ {len(bad)} jaanch fail: {bad}")
    sys.exit(1)
print("✓ dry run saaf")
