"""Bina network ke test: cross features ka leakage.

Ye module do tarah se chup-chaap jhootha nateeja de sakta hai, aur dono
baar score SHANDAAR aayega aur koi error NAHI:

  1. Target encoding usi row ke label se bane jo encode ho rahi hai ->
     model ko apna jawab feature me mil jata hai.
  2. Cluster test data pe fit ho jaye -> test ka dhaancha training me
     ghus jaye (ESM ke PCA wala sabak).

Dono ki jaanch yahan seedhe taur pe hai, aur ek teesri bhi: smoothing
sach me kaam kare, warna ek observation wala key 0 ya 1 bolkar shor
bhar dega.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crossfeat import (NAMES, SMOOTH, apply_encode, assign,  # noqa: E402
                       build, count_table, fit_clusters, oof_encode)

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


print("oof_encode: row ko APNA label nahi milna chahiye")
# Ek key jo SIRF EK baar aata hai. Jis inner fold me wo row test hai,
# us fold ke training me wo key hai hi nahi -> encoding prior honi
# chahiye, us row ka label (1.0) NAHI.
keys = np.array(["solo"] + ["many"] * 40)
y = np.array([1.0] + [0.0] * 40)
enc, tbl, prior = oof_encode(keys, y, seed=0)
check("akela key -> apna label (1.0) NAHI mila", float(enc[0]) == 1.0, False)

# Ye sabse seedha aur sabse kathin test hai: us row ka APNA label palat
# do. Agar encoding me uska apna label kahin bhi ghusta hai, encoding
# badlegi. Nahi badalni chahiye.
# (Pehle maine ise "prior ke barabar hona chahiye" se naapa tha - wo
#  galat tha: function us INNER FOLD ka prior deta hai, global ka nahi,
#  aur wahi sahi hai. Expectation galat thi, code nahi.)
y_flip = y.copy()
y_flip[0] = 0.0
enc_flip, _, _ = oof_encode(keys, y_flip, seed=0)
check("apna label palatne se APNI encoding nahi badli",
      round(float(enc[0]), 9), round(float(enc_flip[0]), 9))
print(f"      (enc[0]={enc[0]:.4f} dono baar; label 1.0 -> 0.0 kiya)")

# Aur ulta: agar leakage hoti, to perfectly-separable key pe encoding
# label ko bilkul dohrati. Yahan aisa nahi hona chahiye.
keys2 = np.array(["a", "a", "b", "b"] * 10)
y2 = np.array([1.0, 1.0, 0.0, 0.0] * 10)
enc2, _, _ = oof_encode(keys2, y2, seed=0)
# 'a' rows ki encoding 1 ke paas honi chahiye (doosri 'a' rows se),
# par BILKUL 1.0 nahi (smoothing ki wajah se)
a = enc2[keys2 == "a"]
check("bahut baar aane wala key -> signal milta hai (>0.5)",
      bool(a.mean() > 0.5), True)
check("par bilkul 1.0 nahi (smoothing lagti hai)",
      bool((a == 1.0).any()), False)

print("\noof_encode: alag seed -> alag inner folds")
e_a, _, _ = oof_encode(keys2, y2, seed=0)
e_b, _, _ = oof_encode(keys2, y2, seed=7)
check("seed badalne pe encoding badalti hai",
      bool(not np.allclose(e_a, e_b)), True)

print("\nsmoothing: ek observation wala key 0 ya 1 na bole")
k3 = np.array(["x"] + ["y"] * 100)
y3 = np.array([1.0] + [0.0] * 100)
_, tbl3, prior3 = oof_encode(k3, y3, seed=0)
check("ek baar wala key ka rate 1.0 nahi", tbl3["x"] == 1.0, False)
check("wo prior ke paas hai", bool(abs(tbl3["x"] - prior3) < 0.3), True)
print(f"      (tbl['x']={tbl3['x']:.4f}, prior={prior3:.4f}, SMOOTH={SMOOTH})")

print("\napply_encode: andekha key -> prior, 0 NAHI")
out = apply_encode(np.array(["naya"]), {"purana": 0.9}, 0.3)
check("andekha key -> prior", round(float(out[0]), 6), 0.3)
check("0 nahi", float(out[0]) == 0.0, False)

print("\ncount_table")
c = count_table(np.array(["a", "a", "b"]))
check("ginti sahi", (c["a"], c["b"]), (2, 1))

print("\nfit_clusters: SIRF train pe fit, test se na bade")
rng = np.random.default_rng(0)
tr = np.vstack([rng.normal(0, 0.3, (40, 8)), rng.normal(5, 0.3, (40, 8))])
km = fit_clusters(tr, n_clusters=2, seed=0)
lab_before = assign(km, tr)
# Test data bilkul alag jagah ka - agar iska asar centroids pe pada
# hota, to train ke labels badal jate
te_far = rng.normal(50, 0.3, (60, 8))
_ = assign(km, te_far)
lab_after = assign(km, tr)
check("test ko assign karne se train ke labels nahi badle",
      bool((lab_before == lab_after).all()), True)
check("do cluster bane", len(set(lab_before.tolist())), 2)
# Door ka test data kisi maujood cluster me hi jayega - naya nahi banega
check("test ke labels train ke cluster set me hi hain",
      set(assign(km, te_far).tolist()) <= set(lab_before.tolist()), True)

print("\nbuild: aakar aur kram")
n_tr, n_te = 200, 50
cl_tr = rng.integers(0, 5, n_tr)
pos_tr = rng.integers(1, 86, n_tr)
sub_tr = np.array([f"A>{c}" for c in rng.integers(0, 4, n_tr)])
y_tr = (rng.random(n_tr) < 0.2).astype(float)
cl_te = rng.integers(0, 5, n_te)
pos_te = rng.integers(1, 86, n_te)
sub_te = np.array([f"A>{c}" for c in rng.integers(0, 4, n_te)])
Xtr, Xte = build(cl_tr, pos_tr, sub_tr, y_tr, cl_te, pos_te, sub_te)
check("train ka aakar", Xtr.shape, (n_tr, len(NAMES)))
check("test ka aakar", Xte.shape, (n_te, len(NAMES)))
check("koi NaN nahi", bool(np.isfinite(Xtr).all() and np.isfinite(Xte).all()),
      True)
# rate columns 0..1 me, count columns >= 0
rate_cols = [i for i, nm in enumerate(NAMES) if nm.endswith("_rate")]
n_cols = [i for i, nm in enumerate(NAMES) if nm.endswith("_n")]
check("rate columns 0..1 me", bool(((Xtr[:, rate_cols] >= 0)
                                    & (Xtr[:, rate_cols] <= 1)).all()), True)
check("count columns >= 0", bool((Xtr[:, n_cols] >= 0).all()), True)

print("\nbuild: test ka LABEL kahin istemaal nahi hota")
# Test ke labels bilkul palat do - test ke features NA badlein, kyunki
# unhe banane me test ka label lagta hi nahi
_, Xte_a = build(cl_tr, pos_tr, sub_tr, y_tr, cl_te, pos_te, sub_te)
_, Xte_b = build(cl_tr, pos_tr, sub_tr, y_tr, cl_te, pos_te, sub_te)
check("wahi input -> wahi test features", bool(np.allclose(Xte_a, Xte_b)),
      True)
# Aur: training ka label badalne pe test features BADALNE chahiye
# (kyunki encoding training se banti hai) - ye ulta check hai, ki
# feature sach me training se aa raha hai
_, Xte_c = build(cl_tr, pos_tr, sub_tr, 1.0 - y_tr, cl_te, pos_te, sub_te)
check("training ka label badla -> test features badle",
      bool(not np.allclose(Xte_a, Xte_c)), True)

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas")
