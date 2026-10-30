"""Bina network ke test: IFP padhna aur jodna.

Khatra yahan ye hai ki bits GALAT position se padh liye jayein. Lambai
595 hai (85 x 7); agar main 7 ki jagah kuch aur maan loon, to har
position ke bits khisak jayenge, feature jhootha ban jayega, aur koi
error kabhi nahi aayega. Isliye yahan sabse zyada zor isi pe hai.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from klifs_ifp import (IFP_LEN, IFP_NAMES, N_POS, N_TYPES,   # noqa: E402
                       aggregate, ifp_features, parse_ifp)

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


def make_ifp(hits):
    """hits: {position(1-85): [type indices 0-6]} -> 595 ka string"""
    bits = ["0"] * IFP_LEN
    for pos, types in hits.items():
        for t in types:
            bits[(pos - 1) * N_TYPES + t] = "1"
    return "".join(bits)


print("dhaancha")
check("lambai 595 = 85 x 7", IFP_LEN, 595)
check("85 positions", N_POS, 85)
check("7 types", N_TYPES, 7)

print("\nparse_ifp: bits sahi position se padhe jate hain")
s = make_ifp({1: [0]})
p = parse_ifp(s)
check("85 rows", len(p), 85)
check("har row me 7", len(p[0]), 7)
check("position 1, type 0 pe 1", p[0][0], 1)
check("position 1, type 1 pe 0", p[0][1], 0)
check("position 2 bilkul khali", sum(p[1]), 0)

# Asli jaanch: aakhri position pe bit, kahin khisak to nahi raha
p = parse_ifp(make_ifp({85: [6]}))
check("position 85, type 6 pe 1", p[84][6], 1)
check("position 84 khali", sum(p[83]), 0)

p = parse_ifp(make_ifp({45: [0, 3]}))
check("position 45 pe do type", [p[44][0], p[44][3]], [1, 1])
check("position 45 pe baki khali", sum(p[44]), 2)
check("position 44 aur 46 khali", sum(p[43]) + sum(p[45]), 0)

print("\nparse_ifp: galat roop pakda jana chahiye")
check("chhoti string -> None", parse_ifp("0101"), None)
check("lambi string -> None", parse_ifp("0" * 596), None)
check("594 (ek kam) -> None", parse_ifp("0" * 594), None)
check("0/1 ke alawa -> None", parse_ifp("2" * 595), None)
check("khali -> None", parse_ifp(""), None)
check("None -> None", parse_ifp(None), None)

print("\naggregate: contact frequency")
a = aggregate([make_ifp({10: [0]}), make_ifp({10: [0]}),
               make_ifp({10: [0]}), make_ifp({20: [0]})])
check("4 structures gini", a["n"], 4)
check("position 10 pe 3/4 = 0.75", round(a["any"][9], 6), 0.75)
check("position 20 pe 1/4 = 0.25", round(a["any"][19], 6), 0.25)
check("position 30 pe 0", a["any"][29], 0.0)
check("type-wise bhi sahi", round(a["by_type"][9][0], 6), 0.75)
check("doosre type pe 0", a["by_type"][9][1], 0.0)

print("\naggregate: kharab IFP chhoot jati hain, gintee girti nahi")
a = aggregate([make_ifp({10: [0]}), "kharab", None, make_ifp({10: [0]})])
check("sirf 2 gini", a["n"], 2)
check("position 10 pe 2/2 = 1.0", round(a["any"][9], 6), 1.0)

a = aggregate([])
check("khali list -> n=0", a["n"], 0)
check("khali list -> sab 0", sum(a["any"]), 0.0)
a = aggregate(["kharab", "bhi kharab"])
check("sab kharab -> n=0", a["n"], 0)

print("\nifp_features")
a = aggregate([make_ifp({45: [0]})] * 3 + [make_ifp({45: [0, 2]})])
f = ifp_features(a, 45)
check("lambai", len(f), len(IFP_NAMES))
check("contact_any = 1.0", round(float(f[0]), 6), 1.0)
check("apolar = 1.0", round(float(f[1]), 6), 1.0)
check("arom_e2f = 0.25", round(float(f[3]), 6), 0.25)

f = ifp_features(a, 10)
check("jahan contact nahi -> sab 0 (structures ki gintee chhod kar)",
      float(np.abs(f[:8]).sum()), 0.0)

print("\nifp_features: pocket se bahar wale")
check("index 0 -> sab 0", float(np.abs(ifp_features(a, 0)).sum()), 0.0)
check("index 86 -> sab 0", float(np.abs(ifp_features(a, 86)).sum()), 0.0)
check("index -1 -> sab 0", float(np.abs(ifp_features(a, -1)).sum()), 0.0)
check("khali agg -> sab 0",
      float(np.abs(ifp_features(aggregate([]), 45)).sum()), 0.0)
check("agg None -> sab 0", float(np.abs(ifp_features(None, 45)).sum()), 0.0)

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas")
