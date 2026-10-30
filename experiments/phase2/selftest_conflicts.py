"""Bina network ke test: matbhed ka niyam aur faisla ka niyam.

Yahan do cheezein jaanchi ja rahi hain, dono aise hain jo galat hone pe
koi error nahi dengi - bas galat kahani sunayengi:

  1. "ek jaisa farak" ka pata lagana. Agar ye bikhre hue shor ko bhi
     "ek jaisa" keh de, to hum kahenge "jad mil gayi" jab mili nahi.
  2. faisla. Agar ye ek hi paper ko do maan le, to hum Davis ko galat
     keh denge bina saboot ke.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conflicts import const_groups, overlap_table         # noqa: E402
from resolve_conflicts import verdict                     # noqa: E402

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


def row(source, gene, mut, dk, ddG, at="Kd"):
    return {"source": source, "gene": gene, "mutation": mut, "mut_note": "",
            "drug_key": dk, "assay_type": at, "ddG": ddG}


print("overlap_table")
df = pd.DataFrame([
    row("chembl", "ABL1", "T315I", "D1", -4.0),
    row("davis", "ABL1", "T315I", "D1", -3.8),
    row("chembl", "ABL1", "E255K", "D1", -1.0),      # davis me nahi
    row("davis", "KIT", "D816V", "D2", -2.0),        # chembl me nahi
])
both = overlap_table(df)
check("sirf saajha jodi aati hai", len(both), 1)

df1 = pd.DataFrame([row("chembl", "ABL1", "T315I", "D1", -4.0)])
check("ek hi source ho to khali", len(overlap_table(df1)), 0)

print("\nconst_groups: ek jaisa farak")
# Ek hi (gene, drug) pe 4 mutations, har jagah farak theek 1.0
rows = []
for i, m in enumerate(["T315I", "E255K", "M351T", "F317I"]):
    rows.append(row("chembl", "ABL1", m, "D1", -1.0 - i * 0.5))
    rows.append(row("davis", "ABL1", m, "D1", -2.0 - i * 0.5))
multi, const = const_groups(overlap_table(pd.DataFrame(rows)))
check("4 mutation wala group multi me hai", len(multi), 1)
check("ek jaisa farak pakda gaya", len(const), 1)
check("offset +1.0 nikla", round(float(const.off.iloc[0]), 6), 1.0)
check("spread lagbhag 0", float(const.spread.iloc[0]) < 1e-9, True)

# Bikhra hua farak - ise "ek jaisa" NAHI kehna chahiye
rows = []
for m, c, d in [("T315I", -1.0, -2.0), ("E255K", -1.0, -0.2),
                ("M351T", -1.0, -3.1), ("F317I", -1.0, 0.4)]:
    rows.append(row("chembl", "ABL1", m, "D1", c))
    rows.append(row("davis", "ABL1", m, "D1", d))
multi, const = const_groups(overlap_table(pd.DataFrame(rows)))
check("bikhra farak multi me hai", len(multi), 1)
check("bikhra farak 'ek jaisa' NAHI mana gaya", len(const), 0)

# Ek jaisa par bahut chhota farak - matlab nahi rakhta
rows = []
for i, m in enumerate(["T315I", "E255K", "M351T"]):
    rows.append(row("chembl", "ABL1", m, "D1", -1.0 - i))
    rows.append(row("davis", "ABL1", m, "D1", -1.05 - i))
multi, const = const_groups(overlap_table(pd.DataFrame(rows)))
check("chhota farak (0.05) nazarandaz hua", len(const), 0)

# Sirf 2 mutations - group banane ke liye kaafi nahi (MIN_N=3)
rows = []
for i, m in enumerate(["T315I", "E255K"]):
    rows.append(row("chembl", "ABL1", m, "D1", -1.0 - i))
    rows.append(row("davis", "ABL1", m, "D1", -2.0 - i))
multi, const = const_groups(overlap_table(pd.DataFrame(rows)))
check("2 mutation wala group nahi ginaa", len(multi), 0)
check("isliye const bhi khali", len(const), 0)

check("khali input se crash nahi", len(const_groups(None)[1]), 0)

print("\nverdict: faisla ka niyam")


def meas(pks, docs):
    return [{"pK": p, "nM": 1.0, "doc": d, "assay": "A", "year": None}
            for p, d in zip(pks, docs)]


v, docs, med = verdict(meas([9.0, 9.1, 8.95], ["D1", "D2", "D3"]), 8.0)
check("3 sehmat paper + Davis door -> ChEMBL behtar",
      v.startswith("ChEMBL zyada bharosemand"), True)
check("3 paper gine gaye", len(docs), 3)

v, _, _ = verdict(meas([9.0, 9.1, 8.95], ["D1", "D1", "D1"]), 8.0)
check("ek hi paper teen baar -> faisla nahi",
      v.startswith("faisla nahi"), True)

v, _, _ = verdict(meas([9.0], ["D1"]), 8.0)
check("ek maap -> faisla nahi", v.startswith("faisla nahi"), True)

v, _, _ = verdict(meas([9.0, 7.5, 8.2], ["D1", "D2", "D3"]), 8.0)
check("ChEMBL ke paper khud alag -> Davis ko dosh nahi",
      v.startswith("ChEMBL khud bata hua"), True)

v, _, _ = verdict(meas([9.0, 9.1, 8.95], ["D1", "D2", "D3"]), 9.05)
check("Davis bhi paas -> jhagda bada nahi",
      "jhagda bada nahi" in v, True)

v, _, _ = verdict([], 8.0)
check("koi maap nahi -> faisla nahi", v.startswith("faisla nahi"), True)

# Ye wo bug tha jo likhte waqt pakda: Davis ka number None ho to
# purana code f-string pe crash karta tha
v, _, _ = verdict(meas([9.0, 9.1, 8.95], ["D1", "D2", "D3"]), None)
check("Davis ka number None -> crash nahi, faisla nahi",
      v.startswith("faisla nahi"), True)

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas")
