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

print("\nsabse zaroori invariant: jo position badli, wahan PEHLE wahi amino acid")
print("hona chahiye jo naam kehta hai. Warna hum random jagah mutate kar rahe hain.")


def changed_positions(a, b):
    return [i for i in range(min(len(a), len(b))) if a[i] != b[i]]


for nm, frm, to in [("X(E10K)", "E", "K"), ("X(Q10K)", "Q", "K"),
                    ("X(T71I)", "T", "I"), ("X(W10F)", "W", "F")]:
    s, st, detail = apply_mutations(nm, SEQ)
    if st == "skip":
        check(s == SEQ, f"{nm}: skip hua, sequence nahi chhedi ({detail})")
        continue
    pos = changed_positions(SEQ, s)
    ok = len(pos) == 1 and SEQ[pos[0]] == frm and s[pos[0]] == to
    check(ok, f"{nm}: position {pos} pe '{frm}'->'{to}' ({st}, {detail})")

# window se bahar wala kuch nahi chhedna chahiye
far = "E" + "A" * 200          # 'E' sirf position 1 pe
s, st, detail = apply_mutations("X(E150K)", far)
check(st == "skip" and s == far,
      f"150 se 10 ke andar 'E' nahi -> skip ({st}, {detail})")

# sabse zaroori: mutation lagne ke baad sequence asal se ALAG honi chahiye
s, _, _ = apply_mutations("ABL1(E10K)", SEQ)
check(s != SEQ, "mutant sequence asal se alag hai (yahi poore experiment ki jaan hai)")

print()
if fails:
    print(f"{len(fails)} test fail hue:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("saare test pass ✓")
