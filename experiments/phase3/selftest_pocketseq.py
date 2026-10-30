"""Bina network ke test: mutated pocket sequence.

Khatra saaf hai: mutation galat index pe lag jaye. Tab sequence galat,
embedding galat, model kuch na kuch seekh hi lega, aur error kabhi nahi
aayega. Isliye yahan sabse zyada zor is baat pe hai ki GALAT jagah pe
mutation LAGE HI NA - chup-chaap lag jane se behtar hai jodi chhod dena.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pocket_seq import (N_POS, build_pair, mutate_pocket,   # noqa: E402
                        parse, wt_pocket)

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


# Nakli pocket: 85 akshar, position 45 pe T, position 1 pe H, 85 pe S
POCKET = ("H" + "A" * 43 + "T" + "A" * 39 + "S")
assert len(POCKET) == N_POS, len(POCKET)
INFO = {"consensus_pocket": POCKET}
# residue 315 -> index 45, residue 100 -> index 1, residue 400 -> index 85
MAPS = {"offset": 0,
        "map": {"315": [45, "GK.45", "T"], "100": [1, "I.1", "H"],
                "400": [85, "a.l.85", "S"]}}

print("parse")
check("T790M", parse("T790M"), [("T", 790, "M")])
check("multi", parse("L858R+T790M"),
      [("L", 858, "R"), ("T", 790, "M")])
check("samajh na aaye -> khali", parse("RAW:ITD"), [])

print("\nwt_pocket")
check("85 akshar -> wahi", wt_pocket(INFO), POCKET)
check("galat lambai -> None", wt_pocket({"consensus_pocket": "HHH"}), None)
check("kuch nahi -> None", wt_pocket({}), None)

print("\nmutate_pocket: sahi jagah pe lagta hai")
s, why = mutate_pocket(POCKET, 45, "T", "I")
check("kya hua", why, "ok")
check("position 45 pe ab I", s[44], "I")
check("baki sab waisa hi", s[:44] + s[45:], POCKET[:44] + POCKET[45:])
check("lambai nahi badli", len(s), N_POS)

s, _ = mutate_pocket(POCKET, 1, "H", "Y")
check("pehli position", s[0], "Y")
s, _ = mutate_pocket(POCKET, 85, "S", "G")
check("aakhri position", s[84], "G")

print("\nmutate_pocket: GALAT jagah pe LAGNA NAHI chahiye")
# Position 45 pe T hai, par mutation kehta hai wahan W hona chahiye
s, why = mutate_pocket(POCKET, 45, "W", "I")
check("wild-type match na ho -> None", s, None)
check("wajah likhi gayi", "T hai, W nahi" in why, True)

s, why = mutate_pocket(POCKET, 0, "H", "Y")
check("index 0 -> None", s, None)
s, why = mutate_pocket(POCKET, 86, "S", "G")
check("index 86 -> None", s, None)
s, why = mutate_pocket("HHH", 1, "H", "Y")
check("galat lambai ka pocket -> None", s, None)
s, why = mutate_pocket(None, 1, "H", "Y")
check("None pocket -> None", s, None)

print("\nbuild_pair: poora raasta")
wt, mut, why = build_pair("FAKE", "T315I", MAPS, INFO)
check("kya hua", why, "ok")
check("wild-type waisa hi", wt, POCKET)
check("mutant me position 45 badli", mut[44], "I")
check("sirf ek akshar ka farak",
      sum(1 for a, b in zip(wt, mut) if a != b), 1)

print("\nbuild_pair: multi-mutant, dono lagni chahiye")
wt, mut, why = build_pair("FAKE", "H100Y+T315I", MAPS, INFO)
check("kya hua", why, "ok")
check("dono akshar badle",
      sum(1 for a, b in zip(wt, mut) if a != b), 2)
check("position 1 pe Y", mut[0], "Y")
check("position 45 pe I", mut[44], "I")

print("\nbuild_pair: offset ke saath")
M2 = {"offset": -19, "map": MAPS["map"]}
# Label 334 kehta hai, offset -19 lagakar 315 banta hai -> index 45
wt, mut, why = build_pair("FAKE", "T334I", M2, INFO)
check("offset lagkar sahi jagah", why, "ok")
check("position 45 badli", mut[44], "I")

print("\nbuild_pair: jo nahi ban sakti")
wt, mut, why = build_pair("FAKE", "A999V", MAPS, INFO)
check("pocket me nahi -> None", mut, None)
check("wajah saaf", "koi mutation pocket me nahi" in why, True)

# Wild-type AA galat - yahi wo haalat hai jisme galat sequence ban sakta tha
wt, mut, why = build_pair("FAKE", "W315I", MAPS, INFO)
check("wild-type galat -> jodi CHHOD di gayi", mut, None)
check("wajah me residue likha", "T hai, W nahi" in why, True)

wt, mut, why = build_pair("FAKE", "RAW:ITD", MAPS, INFO)
check("parse na ho -> None", mut, None)
wt, mut, why = build_pair("FAKE", "T315I", MAPS, {"consensus_pocket": "HH"})
check("pocket kharab -> None", mut, None)

print("\nek multi-mutant jisme ek pocket me hai ek nahi")
wt, mut, why = build_pair("FAKE", "T315I+A999V", MAPS, INFO)
check("jo pocket me hai wo lagti hai", why, "ok")
check("sirf ek akshar badla",
      sum(1 for a, b in zip(wt, mut) if a != b), 1)

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas")
