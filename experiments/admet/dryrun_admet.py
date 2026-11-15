"""run_admet ka poora raasta nakli TDC/xgboost/features se chalao.

Ye file isliye hai kyunki maine ise PEHLE nahi banaya tha. ADMET ka
train_val run 27 minute chala, saare 22 datasets ke number nikal liye,
aur phir AAKHRI line pe `NameError: TAG is not defined` pe foot gaya -
artifacts likhe hi nahi gaye. Theek wahi kism ki galti jiske khilaf
maine run_topk ke commit me likha tha.

Har doosre run ke liye dryrun tha; ADMET ke liye nahi tha. Ab hai.
"""
import json
import os
import pathlib
import sys
import types

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(os.environ.get("DRYOUT", "/tmp/dryout_admet"))
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(0)
SM = ["CCO", "CCC", "CCN", "c1ccccc1", "CC(=O)O", "CCOCC", "CCCl", "CCBr"]


def frame(n, binary):
    d = RNG.integers(0, len(SM), n)
    y = (RNG.random(n) < 0.4).astype(float) if binary else RNG.normal(0, 1, n)
    return pd.DataFrame({"Drug": [SM[i] for i in d], "Y": y})


DATASETS = {"fake_cls": True, "fake_reg": False}


class FakeGroup:
    dataset_names = list(DATASETS)
    # metric_dict jaan-boojh kar NAHI - taaki fallback ka raasta bhi chale
    def __init__(self, path=None):
        self._c = {n: {"train_val": frame(60, b), "test": frame(20, b)}
                   for n, b in DATASETS.items()}

    def get(self, name):
        return self._c[name]

    def get_train_valid_split(self, benchmark, split_type, seed):
        tv = self._c[benchmark]["train_val"]
        return tv.iloc[:45], tv.iloc[45:]

    def evaluate_many(self, preds):
        assert isinstance(preds, list) and len(preds) > 0, type(preds)
        out = {}
        for n in DATASETS:
            vals = []
            for one in preds:
                assert n in one, f"{n} predictions me nahi"
                p = np.asarray(one[n], dtype=float)
                assert len(p) == len(self._c[n]["test"]), \
                    f"{n}: {len(p)} vs {len(self._c[n]['test'])}"
                vals.append(float(p.mean()))
            out[n] = [float(np.mean(vals)), float(np.std(vals))]
        return out


tdc = types.ModuleType("tdc")
bg = types.ModuleType("tdc.benchmark_group")
bg.admet_group = FakeGroup
tdc.benchmark_group = bg
sys.modules["tdc"] = tdc
sys.modules["tdc.benchmark_group"] = bg


class FakeXGB:
    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y):
        assert len(X) == len(y) and len(X) > 0
        assert np.isfinite(X).all(), "feature me NaN/inf reh gaya"
        self.n = X.shape[1]
        return self

    def predict(self, X):
        assert X.shape[1] == self.n
        return RNG.normal(0, 1, len(X))

    def predict_proba(self, X):
        assert X.shape[1] == self.n
        p = RNG.random(len(X))
        return np.column_stack([1 - p, p])


xgb = types.ModuleType("xgboost")
xgb.XGBRegressor = FakeXGB
xgb.XGBClassifier = FakeXGB
sys.modules["xgboost"] = xgb

sys.path.insert(0, str(HERE))
import run_admet as R                                        # noqa: E402

# ChemBERTa aur RDKit dono ko nakli karo - wo bhaari dependencies hain
# aur maqsad yahan FEATURES ki jaanch nahi, RAASTE ki jaanch hai
# (naam define hai ya nahi, file likhi ya nahi, alignment tooti ya nahi)
R.chemberta_block = lambda smiles, batch=0: RNG.normal(
    0, 1, (len(smiles), 8)).astype(np.float32)


def _fake_rdkit(smiles):
    X = RNG.normal(0, 1, (len(smiles), 6)).astype(np.float32)
    X[0, 0] = np.nan            # ek parse-fail jaisa row, fill_nan jaanchne ko
    return X, 1


R.rdkit_block = _fake_rdkit
R.RES = OUT
R.DATA = OUT / "tdc_data"
R.SEEDS = [1, 2]
R.N_TREES = 3

bad = []


def ck(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila: {got!r}\n      chahiye: {want!r}")
        bad.append(name)


for tag, train_on in (("_dry", "train"), ("_drytv", "train_val")):
    R.TAG = tag
    R.TRAIN_ON = train_on
    print(f"\n=== TRAIN_ON={train_on}, TAG={tag}")
    rc = R.main()
    ck(f"{train_on}: rc 0", rc, 0)
    ck(f"{train_on}: csv likhi gayi",
       (OUT / f"admet_official{tag}.csv").exists(), True)
    ck(f"{train_on}: json likhi gayi",
       (OUT / f"admet_official{tag}.json").exists(), True)
    rep = json.loads((OUT / f"admet_official{tag}.json").read_text())
    ck(f"{train_on}: trained_on report me sahi", rep["trained_on"],
       "train+valid" if train_on == "train_val" else "train split only")
    ck(f"{train_on}: dono dataset", sorted(rep["results"]),
       sorted(DATASETS))

print("\ndono TAG ne ALAG files banayi (ek doosre ko mitaya nahi)")
ck("train wali csv bachi", (OUT / "admet_official_dry.csv").exists(), True)
ck("train_val wali csv bani", (OUT / "admet_official_drytv.csv").exists(),
   True)

print("\ntrain_val me train split se ZYADA rows milni chahiye")
a = json.loads((OUT / "admet_official_dry.json").read_text())
b = json.loads((OUT / "admet_official_drytv.json").read_text())
ck("train_val ka n_train bada", b["per_dataset"]["fake_cls"]["n_train"]
   > a["per_dataset"]["fake_cls"]["n_train"], True)

print()
if bad:
    print(f"✗ {len(bad)} jaanch fail: {bad}")
    sys.exit(1)
print("✓ dry run saaf")
