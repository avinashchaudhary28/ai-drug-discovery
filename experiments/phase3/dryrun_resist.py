"""run_resist ka poora raasta nakli xgboost/features/klifs se chalao.

Maqsad nateeja nahi - ye dekhna ki code TOOTTA nahi aur dono gate sach
me bharte hain. Pichhle bugs 20+ minute Actions chalne ke baad milte the.
"""
import json
import os
import pathlib
import sys
import types

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(os.environ.get("DRYOUT", "/tmp/dryout_resist"))
OUT.mkdir(parents=True, exist_ok=True)


class FakeClf:
    def __init__(self, **kw):
        assert kw.get("objective") == "binary:logistic", kw.get("objective")
        self.rs = kw.get("random_state", 0)

    def fit(self, X, y):
        assert len(X) == len(y) and len(X) > 0
        self.n = X.shape[1]
        # Nakli model: pehla feature y se juda hai, to usi ka istemaal
        return self

    def predict_proba(self, X):
        assert X.shape[1] == self.n
        rng = np.random.default_rng(self.rs)
        # AAKHRI feature bhi dekho. Warna nakli model sirf X[:,0] pe
        # chalta, aur ARM A/B bilkul ek jaise aate - to ye dry run ye
        # jaanch hi nahi pata ki cross features model TAK PAHUNCHE ya
        # nahi. Saare zero hote to bhi test paas ho jata.
        s = X[:, 0] + 0.8 * X[:, -1] + rng.normal(0, 0.4, len(X))
        p = 1 / (1 + np.exp(-s))
        return np.column_stack([1 - p, p])


xgb = types.ModuleType("xgboost")
xgb.XGBClassifier = FakeClf
sys.modules["xgboost"] = xgb

NF = 12          # 8 "fp" + 4 baaki


def fake_build(df):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (len(df), NF)).astype(np.float32)
    # pehla feature label se juda - taaki nakli model me signal ho
    X[:, 0] = (df.ddG.to_numpy(dtype=float) <= -1).astype(float) * 1.5 \
        + rng.normal(0, 0.9, len(df))
    names = [f"fp{i}" for i in range(8)] + ["d0", "d1", "m0", "m1"]
    return X, np.ones(len(df), dtype=bool), names


feat = types.ModuleType("features")
feat.build = fake_build
feat.region_name = lambda h: "R" + str(h)[:1]
feat.pocket_features = lambda g, mu, maps, regions: (
    np.array([1.0, (abs(hash(str(mu))) % 85) + 1, 0.5], dtype=np.float32)
    if (abs(hash(str(g) + str(mu))) % 10) < 6
    else np.array([0.0, 0.0, 0.0], dtype=np.float32), "Ra")
sys.modules["features"] = feat

# Nakla data: 12 bade genes, resistance rate alag-alag
rng = np.random.default_rng(1)
rows = []
for gi, (g, n, rate) in enumerate(
        [("G1", 300, 0.18), ("G2", 260, 0.22), ("G3", 200, 0.15),
         ("G4", 170, 0.30), ("G5", 150, 0.12), ("G6", 120, 0.25),
         ("G7", 100, 0.20), ("G8", 90, 0.28), ("G9", 70, 0.17),
         ("G10", 60, 0.33), ("G11", 40, 0.20), ("G12", 30, 0.25),
         ("ALLZERO", 40, 0.0), ("SMALL", 8, 0.3)]):
    for i in range(n):
        res = rng.random() < rate
        rows.append({"gene": g, "mutation": f"A{100+(i % 23)}V",
                     "drug_key": f"D{i % 17}", "smiles": "CCO",
                     "ddG": (-1.5 if res else -0.2) + rng.normal(0, 0.15),
                     "is_kinase": True, "censored_wt": False,
                     "censored_mut": False, "mut_parsed": True})
df = pd.DataFrame(rows)
csv = OUT / "fake.csv"
df.to_csv(csv, index=False)
(OUT / "klifs_map.json").write_text(json.dumps(
    {"G1": {"offset": 0, "map": {"100": [1, "I.1", "A"]}}}))

sys.path.insert(0, str(HERE))
import run_resist as R                                       # noqa: E402

R.SRC = csv
R.RES = OUT
R.MAPS = OUT / "klifs_map.json"
R.SEEDS = [0, 1]
R.N_TREES = 5
R.MIN_PAIRS = 20
R.TAG = "_dry"

rc = R.main()
print(f"\n--- dry run rc={rc}")

rep = json.loads((OUT / "resist_report_dry.json").read_text())
raw = pd.read_csv(OUT / "resist_dry.csv")

bad = []


def ck(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila: {got!r}\n      chahiye: {want!r}")
        bad.append(name)


print("\nfolds")
ck("ek-class wala gene chhoot gaya", "ALLZERO" in rep["folds"], False)
ck("  aur wo drop list me hai", "ALLZERO" in rep["dropped_one_class"], True)
ck("chhota gene chhoot gaya", "SMALL" in rep["folds"], False)
ck("12 folds bache", rep["n_folds"], 12)

print("\ngate dhaancha")
ck("gate1 ke teen check", sorted(rep["gate1"]["checks"]),
   ["auroc", "jk", "t"])
ck("gate2 ke do check", sorted(rep["gate2"]["checks"]), ["jk", "t"])
ck("gate1 passed teeno ka AND", rep["gate1"]["passed"],
   all(rep["gate1"]["checks"].values()))
ck("gate2 passed dono ka AND", rep["gate2"]["passed"],
   all(rep["gate2"]["checks"].values()))
ck("both_passed = dono", rep["both_passed"],
   rep["gate1"]["passed"] and rep["gate2"]["passed"])
ck("hard baseline pehle se tay", rep["gate"]["hard_baseline"], R.HARD)

print("\nmodels")
ck("chhe model", sorted(raw.model.unique()),
   sorted(list(R.BASELINES) + [R.ARM_A, R.ARM_B]))
ck("baseline seed = -1 (ek hi baar)",
   sorted(raw[raw.model == R.HARD].seed.unique()), [-1])
ck("arm ke paas saare seeds",
   sorted(raw[raw.model == R.ARM_A].seed.unique()), [0, 1])

print("\nek-hi-number baseline ka AUROC 0.5 hona chahiye")
f = raw[raw.model == "ek hi rate (farsh)"].auroc.dropna()
ck(f"farsh ka AUROC ~0.5 (mila {f.mean():.3f})",
   bool(abs(f.mean() - 0.5) < 1e-6), True)

print("\ncross features sach me model TAK pahunche?")
# ARM A aur ARM B ke feature count me farak crossfeat.NAMES ke barabar
import crossfeat                                             # noqa: E402
ck("ARM B me 6 extra feature", len(crossfeat.NAMES), 6)
# Aur unka asar dikhe: nakli model aakhri feature dekhta hai, to dono
# arm ka nateeja ALAG hona chahiye. Same aane ka matlab cross features
# model tak pahunche hi nahi (ya sab zero hain).
pa = raw[raw.model == R.ARM_A].set_index(["gene", "seed"]).auroc
pb = raw[raw.model == R.ARM_B].set_index(["gene", "seed"]).auroc
import numpy as _np
ck("ARM A aur ARM B ka nateeja alag hai",
   bool(not _np.allclose(pa.to_numpy(), pb.loc[pa.index].to_numpy(),
                         equal_nan=True)), True)

print("\nAUROC ki seema")
a = raw.auroc.dropna()
ck("AUROC 0..1 me", bool(((a >= 0) & (a <= 1)).all()), True)
ck("kuch AUPRC bhara", bool(raw.auprc.notna().any()), True)

print()
if bad:
    print(f"✗ {len(bad)} jaanch fail: {bad}")
    sys.exit(1)
print("✓ dry run saaf")
