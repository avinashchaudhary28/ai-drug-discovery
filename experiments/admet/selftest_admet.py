"""Bina network ke test: ADMET run ke wo hisse jo chup-chaap galat ho sakte hain.

TDC ka data yahan download nahi ho sakta, par jo cheezein galat nateeja
deti hain wo data pe nirbhar nahi hain:

  - classification ko regression samajh lena (ya ulta)
  - test row ka feature NaN rehna ya row drop ho jana -> alignment toot
  - classifier se 0/1 label lena, probability ki jagah

Teeno ki jaanch yahan seedhe hai.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_admet import (fill_nan, is_classification,  # noqa: E402
                       rows, tdc_metric)

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


print("is_classification: TDC ka metric pehle, andaza baad me")
check("roc-auc -> classification", is_classification([0.2, 0.9], "roc-auc"),
      True)
check("PR-AUC (bada-chhota akshar) -> classification",
      is_classification([0.2, 0.9], "PR-AUC"), True)
check("mae -> regression", is_classification([0.0, 1.0], "mae"), False)
check("spearman -> regression", is_classification([0.0, 1.0], "spearman"),
      False)

# Yahi wo jaal hai: regression target jiski values ittefaq se 0/1 hain.
# Metric ke bina ye classifier me chala jata; metric ke saath nahi jata.
check("0/1 values PAR metric mae -> REGRESSION hi rahe",
      is_classification([0.0, 1.0, 0.0, 1.0], "mae"), False)
check("  (metric na ho to hi andaza lage)",
      is_classification([0.0, 1.0, 0.0, 1.0], None), True)
check("lagataar values -> regression",
      is_classification([0.3, 1.7, -2.2], None), False)
check("teen class -> regression nahi, par classifier bhi nahi",
      is_classification([0.0, 1.0, 2.0], None), False)

print("\ntdc_metric: naam badalne pe bhi dhoondhe, warna None")
class G1:
    metric_dict = {"caco2_wang": "mae"}
class G2:
    metrics = {"herg": "roc-auc"}
class G3:
    pass
check("metric_dict se", tdc_metric(G1(), "caco2_wang"), "mae")
check("metrics se", tdc_metric(G2(), "herg"), "roc-auc")
check("kuch na mile -> None", tdc_metric(G3(), "ames"), None)
check("dataset list me na ho -> None", tdc_metric(G1(), "ames"), None)

print("\nfill_nan: test row KABHI drop na ho, NaN bachta na rahe")
Xtr = np.array([[1.0, 10.0], [3.0, 30.0], [np.nan, 50.0]], dtype=np.float32)
Xte = np.array([[np.nan, 20.0], [2.0, np.inf]], dtype=np.float32)
a, b = fill_nan(Xtr, Xte)
check("test ki ginti nahi badli", b.shape, (2, 2))
check("train ki ginti nahi badli", a.shape, (3, 2))
check("test me koi NaN/inf nahi", bool(np.isfinite(b).all()), True)
check("train me koi NaN/inf nahi", bool(np.isfinite(a).all()), True)
# column 0 ka training median (1,3 ka) = 2.0
check("NaN ki jagah TRAINING ka median", float(b[0, 0]), 2.0)
check("asli value chhui nahi gayi", float(b[1, 0]), 2.0)

# Poora column NaN ho to 0 - crash nahi
X2 = np.array([[np.nan, 1.0], [np.nan, 2.0]], dtype=np.float32)
(c,) = fill_nan(X2)
check("poora column NaN -> 0, crash nahi", bool(np.isfinite(c).all()), True)

print("\nrows: SMILES ka kram bilkul wahi rahe")
tbl = {"CCO": np.array([1.0, 2.0], np.float32),
       "CCC": np.array([3.0, 4.0], np.float32)}
r = rows(["CCC", "CCO", "CCC"], tbl, 2)
check("ginti", r.shape, (3, 2))
check("kram wahi", r[:, 0].tolist(), [3.0, 1.0, 3.0])

print("\nclassifier se PROBABILITY aani chahiye, 0/1 label nahi")
# Ye seedha xgboost pe - agar predict() lag jaye to values sirf 0/1
# hongi aur ye test pakad lega
try:
    import xgboost  # noqa: F401
    from run_admet import fit_predict
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (120, 4)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 0.3, 120) > 0).astype(float)
    p = fit_predict(X, y, X[:20], seed=0, clf=True)
    check("values 0..1 ke beech", bool(((p >= 0) & (p <= 1)).all()), True)
    check("sirf 0/1 NAHI (probability hai)",
          bool(set(np.unique(p).tolist()) <= {0.0, 1.0}), False)
    pr = fit_predict(X, X[:, 0], X[:20], seed=0, clf=False)
    check("regression 0..1 me bandha nahi", bool(pr.min() < 0), True)
except ImportError:
    print("  - xgboost nahi hai, ye hissa chhoda")

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas")
