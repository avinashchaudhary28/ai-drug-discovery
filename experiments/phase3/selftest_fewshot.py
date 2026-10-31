"""Bina network ke test: few-shot ka batwara.

Do cheezein galat hon to curve jhootha ho jata hai aur koi error nahi
aata:

  1. shots aur test me koi row SAAJHA ho -> model ne test dekh liya,
     nateeja shandaar aur bekaar
  2. test set k badalne pe BADAL jaye -> k=50 ka test alag (aur shayad
     aasan) hoga, aur hum sudhaar model ka samjhenge jab wo test ka hai

Dono ki jaanch yahan seedhe taur pe hai.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fewshot import (eligible, shots_drug_mean, shots_mean,   # noqa: E402
                     split_gene, take_shots)

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


IDX = np.arange(100)

print("split_gene")
pool, test = split_gene(IDX, seed=0)
check("pool + test = poora", len(pool) + len(test), 100)
check("50-50", len(test), 50)
check("koi row dono me nahi", len(set(pool) & set(test)), 0)
check("sab rows kahin na kahin", len(set(pool) | set(test)), 100)

p2, t2 = split_gene(IDX, seed=0)
check("wahi seed -> wahi batwara", (list(p2), list(t2)), (list(pool), list(test)))
p3, t3 = split_gene(IDX, seed=1)
check("alag seed -> alag batwara", list(t3) == list(test), False)

pool, test = split_gene(np.arange(3), seed=0, test_frac=0.5)
check("chhota gene: dono me kuch hai", min(len(pool), len(test)) >= 1, True)
pool, test = split_gene(np.arange(2), seed=0, test_frac=0.9)
check("bahut chhota: pool khali nahi", len(pool) >= 1, True)

print("\ntake_shots")
pool, test = split_gene(IDX, seed=0)
check("k=0 -> koi shot nahi", len(take_shots(pool, 0, 0)), 0)
check("k=5 -> 5", len(take_shots(pool, 5, 0)), 5)
check("k=20 -> 20", len(take_shots(pool, 20, 0)), 20)
check("k pool se bada -> poora pool", len(take_shots(pool, 999, 0)), len(pool))

print("\nSABSE ZAROORI: shots aur test kabhi saajha na hon")
for k in (0, 5, 10, 20, 50):
    s = take_shots(pool, k, 0)
    check(f"k={k}: shots test me nahi", len(set(s) & set(test)), 0)
    check(f"k={k}: shots pool me hi hain", set(s) <= set(pool), True)

print("\nSABSE ZAROORI: test set har k pe WAHI rehna chahiye")
tests = []
for k in (0, 5, 10, 20, 50):
    pool_k, test_k = split_gene(IDX, seed=0)
    _ = take_shots(pool_k, k, 0)
    tests.append(tuple(sorted(test_k)))
check("saare k ka test ek jaisa", len(set(tests)), 1)

print("\neligible: sirf wo genes jinpe POORA curve ban sake")
df = pd.DataFrame({"gene": (["BIG"] * 200 + ["MID"] * 100 + ["SMALL"] * 30
                            + ["TINY"] * 10)})
genes, need = eligible(df, [0, 5, 10, 20, 50])
check("need = 50 / 0.5 = 100", need, 100)
check("BIG aur MID", genes, ["BIG", "MID"])
genes, need = eligible(df, [0, 5, 10])
check("chhote k pe zyada genes", genes, ["BIG", "MID", "SMALL"])

print("\ntest_frac badalne pe: gene set BADA hona chahiye, k ghatane pe bhi")
# Yahi wo lever hai jisse t ki power badhti hai. Agar ye plumbing tooti
# ho to run chup-chaap purane 11 genes pe hi chalega aur hum samjhenge
# ki n badh gaya.
g50, n50 = eligible(df, [0, 20, 50], test_frac=0.5)
g20, n20 = eligible(df, [0, 20], test_frac=0.5)
check("max k 50 -> need 100", n50, 100)
check("max k 20 -> need 40", n20, 40)
check("k ghatao -> genes barabar ya zyada", len(g20) >= len(g50), True)
g33, n33 = eligible(df, [0, 20], test_frac=0.33)
check("test_frac 0.33, k=20 -> need 30", n33, 30)
check("test_frac ghatao -> genes barabar ya zyada", len(g33) >= len(g20), True)

print("\ntest_frac=0.33 pe pool me max k ke liye jagah bachni chahiye")
# need = ceil(k/(1-frac)) ka matlab yahi hai. Agar pool k se chhota ho
# jaye to take_shots poora pool de dega aur k ka matlab hi badal jayega.
for frac, ks in ((0.5, [0, 5, 10, 20]), (0.33, [0, 5, 10, 20])):
    mk = max(ks)
    need_f = int(np.ceil(mk / (1 - frac)))
    pl, ts = split_gene(np.arange(need_f), seed=0, test_frac=frac)
    check(f"frac={frac}: pool >= max k ({len(pl)} >= {mk})", len(pl) >= mk,
          True)
    check(f"frac={frac}: k={mk} pe poore {mk} shots mile",
          len(take_shots(pl, mk, 0)), mk)
    check(f"frac={frac}: test khali nahi", len(ts) >= 1, True)

print("\nshots_mean")
check("shots ka ausat", shots_mean(np.array([1.0, 3.0]), np.array([99.0])),
      2.0)
check("shots na hon -> training ka",
      shots_mean(np.array([]), np.array([5.0, 7.0])), 6.0)

print("\nshots_drug_mean - sabse kathin baseline")
shots = pd.DataFrame({"drug_key": ["D1", "D1", "D2"],
                      "ddG": [2.0, 4.0, -1.0]})
test = pd.DataFrame({"drug_key": ["D1", "D2", "D9"]})
train = pd.DataFrame({"drug_key": ["D9"], "ddG": [99.0]})
p = shots_drug_mean(shots, test, train)
check("D1 -> (2+4)/2 = 3", float(p[0]), 3.0)
check("D2 -> -1", float(p[1]), -1.0)
# D9 shots me nahi - shots ka ausat milna chahiye, training ka 99 NAHI
check("andekhi dawa -> shots ka ausat (99 nahi)",
      round(float(p[2]), 6), round((2 + 4 - 1) / 3, 6))

p = shots_drug_mean(pd.DataFrame({"drug_key": [], "ddG": []}), test, train)
check("shots na hon -> training se", float(p[0]), 99.0)

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas")
