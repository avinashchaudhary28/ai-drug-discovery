"""mutate.py ka test. Ye 2-ghante wale run se PEHLE chalta hai, taaki parser ki
galti pe poora run zaaya na ho."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutate import apply_mutations, split_edits

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


print("split_edits:")
check(split_edits("E255K") == ["E255K"], "point")
check(split_edits("V559D-T670I") == ["V559D", "T670I"], "dash se do mutation")
check(split_edits("L858R,T790M") == ["L858R", "T790M"], "comma se do mutation")
check(split_edits("L858RT790M") == ["L858R", "T790M"], "bina separator do mutation")
check(split_edits("E746A750del") == ["E746A750del"], "deletion")
check(split_edits("L747S752del,P753S") == ["L747S752del", "P753S"], "deletion + point")
check(split_edits("ITD") == [], "ITD skip")
check(split_edits("JH1domain-catalytic") == [], "domain naam skip")

SEQ = "MKKAVLLAGEVTSRPQWYFNDGHILKMCEE" + "A" * 40 + "T" * 30   # pos10 = E
print(f"\ntest sequence: len={len(SEQ)}, pos10={SEQ[9]}, pos71={SEQ[70]}")

print("\napply_mutations:")
s, st, _ = apply_mutations("ABL1", SEQ)
check(st == "wt" and s == SEQ, "bracket nahi -> sequence waisi hi")

s, st, _ = apply_mutations("ABL1(E10K)", SEQ)
check(st == "ok" and s[9] == "K" and len(s) == len(SEQ), "point mutation laga")

s, st, _ = apply_mutations("ABL1(E10K)p", SEQ)
check(st == "ok" and s[9] == "K", "phospho suffix ignore hua")

s, st, _ = apply_mutations("KIT(E10K-T71I)", SEQ)
check(st == "ok" and s[9] == "K" and s[70] == "I", "do mutation ek saath lage")

s, st, _ = apply_mutations("EGFR(E10A14del)", SEQ)
check(st == "ok" and len(s) == len(SEQ) - 5, "deletion ne 5 residues hataye")

s, st, _ = apply_mutations("FLT3(ITD)", SEQ)
check(st == "skip" and s == SEQ, "ITD skip, sequence nahi chhedi")

s, st, _ = apply_mutations("X(E10Z)", SEQ)
check(st == "skip" and s == SEQ, "Z amino acid nahi -> skip")

print("\nper-gene offset (sabse zaroori hissa):")
print("Ek gene me numbering ek hi tarah khiskti hai. Pehle har mutation apni alag")
print("window me dhundhta tha, jisse ABL1 ko +3/+2/+4/-1 jaise alag offset mile -")
print("yaani kuch mutations ittefaq se GALAT residue pe lag gaye the.")

import numpy as np
from mutate import gene_offset, build

rng = np.random.default_rng(0)
seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 600))
OFF = -19          # ABL1 jaisa: naam ki numbering sequence se 19 aage


def put(pos, aa):
    global seq
    i = pos - 1 + OFF
    seq = seq[:i] + aa + seq[i + 1:]


MUTS = [(252, "Q", "H"), (253, "Y", "F"), (255, "E", "K"), (315, "T", "I"),
        (317, "F", "I"), (351, "M", "T"), (396, "H", "P")]
for pos, frm, _ in MUTS:
    put(pos, frm)

off, n, tot = gene_offset(seq, [f"{f}{p}{t}" for p, f, t in MUTS])
check(off == OFF and n == tot, f"gene ka ek offset mila: {off} (asli {OFF}), match {n}/{tot}")

names = {"ABL1": seq}
for p, f, t in MUTS:
    names[f"ABL1({f}{p}{t})"] = seq
new, rows = build(names)
n_ok = sum(1 for r in rows if r["status"] == "ok")
check(n_ok == len(MUTS), f"saare {len(MUTS)} mutations lage (mile {n_ok})")
check(len(set(new.values())) == len(MUTS) + 1,
      f"har mutant ki apni sequence bani ({len(set(new.values()))} unique)")

bad = True
for p, f, t in MUTS:
    a, b = seq, new[f"ABL1({f}{p}{t})"]
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    if not (len(diff) == 1 and a[diff[0]] == f and b[diff[0]] == t):
        bad = False
check(bad, "har mutation ne theek ek position badli, aur wahan pehle sahi residue tha")

far = {"XYZ": seq, "XYZ(W999A)": seq}
_, rb = build(far)
st = [r for r in rb if r["name"] == "XYZ(W999A)"][0]
check(st["changed"] == 0 and st["status"] == "skip",
      "offset pe match na ho to SKIP, paas ka residue chup-chaap nahi badla")

print()
if fails:
    print(f"{len(fails)} test fail hue:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("saare test pass \u2713")
