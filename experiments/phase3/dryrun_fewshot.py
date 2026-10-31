"""run_fewshot ka poora raasta nakli xgboost/features se chalao.

Maqsad model ka nateeja nahi - ye dekhna ki code TOOTTA nahi. Pichhli
baar jo bugs 25 minute baad Actions pe mile the (index, dict lookup,
column ka naam, gate ka lookup), wo yahan 5 second me milne chahiye.
"""
import os
import pathlib
import sys
import types

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(os.environ.get("DRYOUT", "/tmp/dryout_fewshot"))
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------- nakli xgboost
CALLS = {"fit": 0}


class FakeReg:
    def __init__(self, **kw):
        self.kw = kw
        assert "objective" in kw, "objective pass hi nahi hua"
        assert kw["objective"] in ("reg:absoluteerror", "reg:squarederror"), \
            kw["objective"]
        self.rs = kw.get("random_state", 0)

    def fit(self, X, y):
        CALLS["fit"] += 1
        assert len(X) == len(y), (len(X), len(y))
        assert len(X) > 0
        self.n = X.shape[1]
        self.scale = 1.0 + (self.rs % 3) * 0.5
        return self

    def predict(self, X):
        assert X.shape[1] == self.n
        # kuch badalta hua number, taaki ranking bane; 2x bada taaki
        # calibration ko sach me kaam mile
        rng = np.random.default_rng(self.rs)
        return (2.0 * self.scale * X[:, 0]
                + rng.normal(0, 0.1, len(X))).astype(float)


xgb = types.ModuleType("xgboost")
xgb.XGBRegressor = FakeReg
sys.modules["xgboost"] = xgb

# --------------------------------------------------------- nakli features
NF = 7


def fake_build(df):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (len(df), NF))
    # pehla feature ddG se juda - taaki model me kuch asli signal ho
    X[:, 0] = df.ddG.to_numpy(dtype=float) * 0.5 + rng.normal(0, 0.3, len(df))
    ok = np.ones(len(df), dtype=bool)
    return X, ok, [f"f{i}" for i in range(NF)]


feat = types.ModuleType("features")
feat.build = fake_build
sys.modules["features"] = feat

# ------------------------------------------------------------ nakla data
rng = np.random.default_rng(1)
rows = []
# 8 bade genes (POORA curve ban sake) + 3 chhote (chhoot jane chahiye)
for g, n in [("G1", 120), ("G2", 110), ("G3", 100), ("G4", 95), ("G5", 90),
             ("G6", 80), ("G7", 70), ("G8", 60), ("S1", 20), ("S2", 10),
             ("S3", 5)]:
    for i in range(n):
        rows.append({"gene": g, "drug_key": f"D{i % 12}",
                     "ddG": float(rng.normal(0, 1)), "smiles": "CCO",
                     "is_kinase": True, "censored_wt": False,
                     "censored_mut": False, "mut_parsed": True})
df = pd.DataFrame(rows)
csv = OUT / "fake_merged.csv"
df.to_csv(csv, index=False)

# ------------------------------------------------------------------- chalao
sys.path.insert(0, str(HERE))
import run_fewshot as R                                       # noqa: E402

R.SRC = csv
R.RES = OUT
R.KS = [0, 5, 10, 20]
R.SEEDS = [0, 1]
R.N_TREES = 10
R.TEST_FRAC = 0.5
R.TAG = "_dry"

rc = R.main()
print(f"\n--- dry run rc={rc}, fits={CALLS['fit']}")

import json                                                   # noqa: E402

rep = json.loads((OUT / "fewshot_report_dry.json").read_text())
cur = pd.read_csv(OUT / "fewshot_curve_dry.csv")

bad = []


def ck(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila: {got!r}\n      chahiye: {want!r}")
        bad.append(name)


print("\nreport ki jaanch")
ck("test_frac likha", rep["test_frac"], 0.5)
ck("n_genes likha", rep["n_genes"], len(rep["genes"]))
ck("chhote genes chhoot gaye", [g for g in rep["genes"] if g.startswith("S")],
   [])
ck("gate model likha", rep["gate"]["model"], R.GATE_MODEL)
ck("objective likha", rep["gate"]["objective"], R.OBJECTIVE)
ck("alt model doosra hai", rep["alt_model"]["name"] != R.GATE_MODEL, True)
ck("alt model ke checks bhare", len(rep["alt_model"]["checks"]), 3)
ck("shrink_a har k ke liye", sorted(rep["shrink_a"]["a"].keys()),
   ["0", "10", "20", "5"])
ck("k=0 pe a = 1 (shots nahi)", rep["shrink_a"]["a"]["0"], 1.0)
ck("gate_checks teen baseline", len(rep["gate_checks"]), 3)

print("\ncurve ki jaanch")
ck("paanch model", sorted(cur.model.unique()),
   sorted(["0 bolo", "shots ka ausat", "shots se har dawa ka ausat",
           R.MAIN, R.CAL]))
ck("a sirf cal model pe",
   bool(cur[cur.model != R.CAL].a.isna().all()), True)
ck("cal model pe a bhara", bool(cur[cur.model == R.CAL].a.notna().all()),
   True)
ck("a farsh se neeche nahi",
   bool((cur[cur.model == R.CAL].a >= R.A_FLOOR - 1e-9).all()), True)
ck("har gene har k", len(cur),
   len(rep["genes"]) * len(R.KS) * 5)

# Sabse zaroori: cal aur main ki RANKING ek jaisi honi chahiye (a>0)
s_main = cur[cur.model == R.MAIN].set_index(["gene", "k"]).spearman
s_cal = cur[cur.model == R.CAL].set_index(["gene", "k"]).spearman
same = np.allclose(s_main.to_numpy(), s_cal.loc[s_main.index].to_numpy(),
                   equal_nan=True)
ck("shrink se Spearman BILKUL nahi badla", same, True)

print()
if bad:
    print(f"✗ {len(bad)} jaanch fail: {bad}")
    sys.exit(1)
print("✓ dry run saaf")
