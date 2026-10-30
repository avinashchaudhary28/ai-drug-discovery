"""Bina network ke test: KLIFS mapping aur uski jaanch.

Yahan galti ka nateeja sabse chup-chaap hota hai: agar numbering na mile
to har mutation GALAT pocket position pe baith jayega, model kuch na kuch
seekh hi lega, aur koi error kabhi nahi aayega. Phase 1 me ABL1 ke 19
residue ke offset ne theek yahi kiya tha.

Isliye yahan khaas taur pe ye jaancha jata hai ki galat mapping PAKDI
jaye, na ki ye ki sahi mapping chalti hai.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from klifs_map import (best_offset, build_map, parse_mut,      # noqa: E402
                       pick_structure, region_of, verify)

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


# Nakli kinase: pocket 10 akshar ka, residue 100..109
POCKET = "HKLTGGQYGE"
INFO = {
    "gene": "FAKE", "pocket": POCKET, "structure_pocket": POCKET,
    "residues": [{"index": i + 1, "xray": str(100 + i),
                  "klifs": f"I.{i + 1}" if i < 3 else f"g.l.{i + 1}"}
                 for i in range(10)],
}

print("build_map")
m = build_map(INFO)
check("10 residue mile", len(m), 10)
check("residue 100 -> index 1", m[100][0], 1)
check("residue 100 ka AA H", m[100][2], "H")
check("residue 103 ka AA T", m[103][2], "T")
check("residue 109 -> index 10", m[109][0], 10)
check("pocket se bahar wala nahi hai", 200 in m, False)

bad = dict(INFO)
bad["residues"] = INFO["residues"] + [{"index": 11, "xray": "_",
                                       "klifs": "x"}]
check("'_' wala xray chhoot jata hai", len(build_map(bad)), 10)

print("\nregion_of")
check("I.1 -> I", region_of("I.1"), "I")
check("g.l.4 -> g.l", region_of("g.l.4"), "g.l")
check("hinge.46 -> hinge", region_of("hinge.46"), "hinge")

print("\nparse_mut")
check("T790M", parse_mut("T790M"), ("T", 790, "M"))
check("multi ka pehla", parse_mut("L858R+T790M"), ("L", 858, "R"))
check("samajh na aaye -> None", parse_mut("RAW:ITD"), None)
check("deletion -> None", parse_mut("E746_A750del"), None)

print("\nbest_offset: offset 0 (numbering milti hai)")
# T103 pocket me hai (index 4, AA T)
off, n, tot = best_offset(m, [("T", 103), ("H", 100), ("G", 105)])
check("offset 0", off, 0)
check("teeno mile", n, 3)

print("\nbest_offset: ABL1 jaisa 19 ka farak")
# Mutation labels 19 zyada: T122 ka matlab structure ka 103
off, n, tot = best_offset(m, [("T", 122), ("H", 119), ("G", 124)])
check("offset -19 mila", off, -19)
check("teeno mile", n, 3)

print("\nverify: sahi mapping paas hoti hai")
v = verify("FAKE", m, ["T103I", "H100Y", "G105A"])
check("usable", v["usable"], True)
check("offset 0", v["offset"], 0)
check("teeno pocket me", v["n_in_pocket"], 3)
check("match 100%", round(v["match_rate"], 6), 1.0)

print("\nverify: offset wali mapping bhi paas (offset dhundh leta hai)")
v = verify("FAKE", m, ["T122I", "H119Y", "G124A"])
check("usable", v["usable"], True)
check("offset -19", v["offset"], -19)

print("\nverify: GALAT mapping PAKDI jani chahiye - yahi asli test hai")
# Position pocket me hai par wild-type AA bilkul nahi milta
v = verify("FAKE", m, ["W100A", "W101A", "W102A", "W103A", "W104A"])
check("usable FALSE", v["usable"], False)
check("wajah likhi gayi", "wild-type AA" in v["reason"], True)
check("match rate 0", round(v["match_rate"], 6), 0.0)

# Aadha-aadha: 80% ki seema paar nahi hoti
v = verify("FAKE", m, ["H100A", "K101A", "W102A", "W103A", "W104A"])
check("aadha match -> usable FALSE", v["usable"], False)

print("\nverify: pocket se bahar wale mutations")
v = verify("FAKE", m, ["A500V", "A501V"])
check("koi pocket me nahi -> usable FALSE", v["usable"], False)
check("wajah saaf", "pocket me nahi" in v["reason"], True)

print("\nverify: parse na hone wale")
v = verify("FAKE", m, ["RAW:ITD", "E746_A750del"])
check("usable FALSE", v["usable"], False)
check("n_mutations 0", v["n_mutations"], 0)

print("\npick_structure: adhoore residue wala nahi chunna chahiye")
rows = [
    {"structure_ID": 1, "missing_residues": 3, "quality_score": 10,
     "resolution": 1.0},
    {"structure_ID": 2, "missing_residues": 0, "quality_score": 8,
     "resolution": 2.4},
    {"structure_ID": 3, "missing_residues": 0, "quality_score": 8,
     "resolution": 1.9},
]
check("poora + behtar resolution chuna", pick_structure(rows)["structure_ID"],
      3)

# Yahi wo bug tha: `missing_residues or 99` me 0 falsy hai, to bina kisi
# missing residue wale structure ko 99 mil jata tha - sabse achha wala
# sabse kharab gina jata.
rows2 = [
    {"structure_ID": 7, "missing_residues": 0, "quality_score": 6,
     "resolution": 3.0},
    {"structure_ID": 8, "missing_residues": 5, "quality_score": 10,
     "resolution": 1.0},
]
check("missing_residues=0 ko 99 nahi mana jata",
      pick_structure(rows2)["structure_ID"], 7)

rows3 = [
    {"structure_ID": 9, "missing_residues": None, "quality_score": 10,
     "resolution": 1.0},
    {"structure_ID": 10, "missing_residues": 0, "quality_score": 5,
     "resolution": 3.5},
]
check("None wala peeche jata hai", pick_structure(rows3)["structure_ID"], 10)

rows4 = [{"structure_ID": 11}, {"structure_ID": 12, "missing_residues": 0,
                                "quality_score": 9, "resolution": 2.0}]
check("khali fields pe crash nahi", pick_structure(rows4)["structure_ID"], 12)

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas")
