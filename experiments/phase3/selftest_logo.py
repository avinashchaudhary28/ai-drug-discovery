"""Bina network ke test: batwara aur farsh.

Is phase me sabse khatarnak galti leak hai. Leak hone par nateeja
shandaar dikhega, koi error nahi aayega, aur hum galat daawa leke
khareedaar ke saamne jayenge. Isliye leak ki jaanch alag se, seedhe
taur pe.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logo import (BASELINES, _sub, eligible_genes, logo_splits,  # noqa: E402
                  metrics, paired_t, predict_drug_mean,
                  predict_global_mean, predict_sub_mean, predict_zero)

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


def close(name, got, want, tol=1e-9):
    if got is not None and np.isfinite(got) and abs(got - want) < tol:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


def make(genes_counts, ddg=None):
    rows = []
    i = 0
    for g, n in genes_counts.items():
        for j in range(n):
            rows.append({"gene": g, "mutation": "T790M",
                         "drug_key": f"D{j % 3}",
                         "ddG": (ddg or (lambda a, b: 0.0))(g, j)})
            i += 1
    return pd.DataFrame(rows)


print("eligible_genes: chhote gene chhoot jate hain")
df = make({"EGFR": 50, "ABL1": 20, "TINY": 19, "SMALL": 3})
check("sirf >=20 wale", eligible_genes(df), ["ABL1", "EGFR"])
check("seema badal sakti hai", eligible_genes(df, min_pairs=3),
      ["ABL1", "EGFR", "SMALL", "TINY"])

print("\nlogo_splits: LEAK ki jaanch - sabse zaroori")
df = make({"EGFR": 30, "ABL1": 25, "KIT": 22})
splits = list(logo_splits(df))
check("teen batware bane", len(splits), 3)
for gene, train, test in splits:
    check(f"{gene}: test me sirf wahi gene",
          set(test.gene.unique()), {gene})
    check(f"{gene}: train me wo gene BILKUL nahi",
          gene in set(train.gene.unique()), False)
    check(f"{gene}: train + test = poora data",
          len(train) + len(test), len(df))
    # Index se bhi jaancho - koi row dono taraf na ho
    check(f"{gene}: ek bhi row dono taraf nahi",
          len(set(train.index) & set(test.index)), 0)

print("\nbaseline: 0 bolo")
_, train, test = splits[0]
p = predict_zero(train, test)
check("sab 0", float(np.abs(p).max()), 0.0)
check("lambai test jitni", len(p), len(test))

print("\nbaseline ka LEAK: ausat sirf TRAINING se nikalna chahiye")
# EGFR ke saare ddG 10, baki sab 0. EGFR hata kar training ka ausat 0
# hona chahiye - agar 10 ke aas-paas aaya to test ka data chhua gaya.
df = make({"EGFR": 30, "ABL1": 25, "KIT": 22},
          ddg=lambda g, j: 10.0 if g == "EGFR" else 0.0)
sp = {g: (tr, te) for g, tr, te in logo_splits(df)}
tr, te = sp["EGFR"]
p = predict_global_mean(tr, te)
close("EGFR hata kar ausat 0 hai (test ka data nahi chhua)",
      float(p[0]), 0.0)
check("poore data ka ausat (~5.06) NAHI aaya", float(p[0]) > 1.0, False)

tr, te = sp["ABL1"]
p = predict_global_mean(tr, te)
# EGFR 30 rows x 10 + KIT 22 rows x 0 = 300/52
close("ABL1 ke liye ausat training se bana", float(p[0]), 300.0 / 52.0,
      tol=1e-6)

print("\nbaseline: har dawa ka ausat")
df = pd.DataFrame([
    {"gene": "A", "mutation": "T790M", "drug_key": "D1", "ddG": 2.0},
    {"gene": "A", "mutation": "T790M", "drug_key": "D1", "ddG": 4.0},
    {"gene": "A", "mutation": "T790M", "drug_key": "D2", "ddG": -1.0},
    {"gene": "B", "mutation": "T790M", "drug_key": "D1", "ddG": 99.0},
    {"gene": "B", "mutation": "T790M", "drug_key": "D9", "ddG": 99.0},
])
tr = df[df.gene == "A"]
te = df[df.gene == "B"]
p = predict_drug_mean(tr, te)
close("D1 ka ausat training se (2+4)/2 = 3", float(p[0]), 3.0)
# D9 training me thi hi nahi -> training ka global ausat (2+4-1)/3
close("andekhi dawa -> training ka global ausat", float(p[1]), 5.0 / 3.0)
check("test ke 99 kahin nahi aaye", float(np.abs(p - 99).min()) > 1.0, True)

print("\n_sub: amino acid badlav")
check("T790M -> T>M", _sub("T790M"), "T>M")
check("multi-mutant ka pehla", _sub("L858R+T790M"), "L>R")
check("chhote akshar bhi", _sub("t790m"), "T>M")
# Ye wo bug tha jo test ne pakda: pehla aur aakhri akshar uthakar
# "RAW:ITD" se "R>D" ban raha tha - samajh na aaya naam ek asli AA
# badlav ke group me chala jata.
check("RAW:ITD -> ? (R>D nahi)", _sub("RAW:ITD"), "?")
check("deletion -> ?", _sub("E746_A750del"), "?")
check("khali -> ?", _sub(""), "?")
check("bakwaas -> ?", _sub("hello world"), "?")

print("\nbaseline: AA badlav ka ausat")
df = pd.DataFrame([
    {"gene": "A", "mutation": "T790M", "drug_key": "D1", "ddG": 2.0},
    {"gene": "A", "mutation": "T315M", "drug_key": "D2", "ddG": 4.0},
    {"gene": "A", "mutation": "L858R", "drug_key": "D3", "ddG": -2.0},
    {"gene": "B", "mutation": "T100M", "drug_key": "D4", "ddG": 99.0},
])
p = predict_sub_mean(df[df.gene == "A"], df[df.gene == "B"])
close("T>M ka ausat (2+4)/2 = 3", float(p[0]), 3.0)

print("\nmetrics")
m = metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
close("perfect -> mae 0", m["mae"], 0.0)
close("perfect -> pearson 1", m["pearson"], 1.0, tol=1e-9)
m = metrics([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
close("0 bolo -> mae 2", m["mae"], 2.0)
# Yahi asli baat hai: "0 bolo" ka correlation banta hi nahi
check("0 bolo -> pearson NaN (0 nahi)", bool(np.isnan(m["pearson"])), True)
check("0 bolo -> spearman NaN", bool(np.isnan(m["spearman"])), True)

print("\npaired_t: gene hi ikai hai")
t = paired_t([0.1, 0.1, 0.1, 0.1])
check("4 genes gine gaye", t["n"], 4)
check("chaaron me jeet", t["wins"], 4)
check("sab barabar -> t bahut bada", t["t"] > 1e6, True)
t = paired_t([0.5, -0.5, 0.4, -0.4])
close("aage-peechhe -> mean ~0", t["mean"], 0.0, tol=1e-9)
check("do me jeet", t["wins"], 2)
t = paired_t([float("nan"), 0.2, 0.2])
check("NaN hat jata hai", t["n"], 2)
t = paired_t([0.5])
check("ek hi gene -> t nahi banta", bool(np.isnan(t["t"])), True)

# SE = 0 ki haalat: saare delta bilkul ek jaise. Pehle yahan mean ka
# chinh dekhe BINA +inf lautta tha, to "koi farak nahi" aur "har gene
# pe haar" dono gate ka t >= 2.0 chup-chaap paas kar jate the.
print("\npaired_t: saare delta ek jaise (SE = 0) - chinh maayne rakhta hai")
t = paired_t([0.0, 0.0, 0.0, 0.0])
check("koi farak hi nahi -> t = NaN, +inf NAHI", bool(np.isnan(t["t"])), True)
check("koi farak nahi -> jeet 0", t["wins"], 0)
t = paired_t([-0.3, -0.3, -0.3])
check("har gene pe HAAR -> t = -inf", t["t"], float("-inf"))
check("haar me jeet 0", t["wins"], 0)
t = paired_t([0.3, 0.3, 0.3])
check("har gene pe JEET -> t = +inf", t["t"], float("inf"))
check("jeet me teeno", t["wins"], 3)
for d, nm in (([0.0] * 4, "sab shunya"), ([-0.3] * 3, "sab rinatmak")):
    check(f"{nm}: gate ka t >= 2.0 PAAS nahi hona chahiye",
          bool(paired_t(d)["t"] >= 2.0), False)

print("\nsaare baseline chalte hain")
df = make({"EGFR": 30, "ABL1": 25, "KIT": 22},
          ddg=lambda g, j: float(j % 5) - 2)
for name, fn in BASELINES.items():
    for gene, tr, te in logo_splits(df):
        p = fn(tr, te)
        if len(p) != len(te) or not np.isfinite(p).all():
            check(f"{name} ({gene})", "kharab", "theek")
            break
    else:
        print(f"  ✓ {name}")

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas")
