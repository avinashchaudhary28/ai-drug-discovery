"""Merge se pehle ka test: mutation ke naam aur drug key.

Yahi jagah hai jahan galti chup-chaap nikalti hai. Agar "FLT3(ITD)"
khali mutation ban jaye to wo wild-type jaisa dikhega aur har doosri
entry se jud jayega - koi error nahi, bas galat dataset.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutnorm import drug_key, normalize                  # noqa: E402

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


def toks(raw):
    return normalize(raw)[0]


print("Davis ke naam")
check("ABL1(E255K)", toks("ABL1(E255K)"), ("E255K",))
check("ABL1(T315I)p -> token", toks("ABL1(T315I)p"), ("T315I",))
check("ABL1(T315I)p -> note", normalize("ABL1(T315I)p")[1], "p")
check("KIT(V559D-T670I) (kram tay)", toks("KIT(V559D-T670I)"),
      ("T670I", "V559D"))
check("EGFR(L858RT790M) (bina alag karne wale chihn)",
      toks("EGFR(L858RT790M)"), ("L858R", "T790M"))
check("EGFR(E746-A750del)", toks("EGFR(E746-A750del)"), ("E746_A750del",))

print("\nChEMBL ke naam")
check("T790M", toks("T790M"), ("T790M",))
check("L858R,T790M", toks("L858R,T790M"), ("L858R", "T790M"))
check("E746A750del (bina dash)", toks("E746A750del"), ("E746_A750del",))
check("chhote akshar t790m", toks("t790m"), ("T790M",))
check("gene ke saath: EGFR-T790M", toks("EGFR-T790M"), ("T790M",))

print("\nkram se farak nahi padna chahiye")
check("'L858R,T790M' == 'T790M L858R'",
      toks("L858R,T790M") == toks("T790M L858R"), True)
check("Davis aur ChEMBL ek hi token dete hain",
      toks("EGFR(T790M)") == toks("T790M"), True)

print("\nphosphorylated alag rehna chahiye")
check("note se farak padta hai",
      normalize("ABL1(T315I)p")[1] != normalize("ABL1(T315I)")[1], True)

print("\njo samajh na aaye - khali NAHI banna chahiye")
for raw in ["FLT3(ITD)", "JAK2(JH1domain-catalytic)",
            "RSK1(KinDom.1-N-terminal)", "D770_N771insNPG", "wild type",
            "", None]:
    t, _, parsed = normalize(raw)
    ok = (not parsed) and len(t) == 1 and t[0].startswith("RAW:")
    check(f"{raw!r} -> RAW nishaan, parsed False", ok, True)

# Asli khatra: do alag samajh-na-aane wale naam ek doosre se na judein
check("do alag RAW naam alag rehte hain",
      toks("FLT3(ITD)") != toks("JAK2(JH1domain-catalytic)"), True)
check("khali aur ITD alag rehte hain", toks("") != toks("FLT3(ITD)"), True)

print("\ndrug key (InChIKey skeleton)")
try:
    import rdkit                                         # noqa: F401
    HAVE_RDKIT = True
except ImportError:
    HAVE_RDKIT = False
    # Chup-chaap paas nahi dikhana - saaf likhna ki test chale hi nahi.
    print("  -- rdkit nahi hai, ye test NAHI chale (Actions pe chalenge) --")
    SKIPPED = True

ASP = "CC(=O)Oc1ccccc1C(=O)O"
if not HAVE_RDKIT:
    print()
    if FAIL:
        print(f"✗ {len(FAIL)} test fail: {FAIL}")
        sys.exit(1)
    print("✓ mutation ke test paas (drug key ke test chhoote)")
    sys.exit(0)
check("aspirin se key banti hai", isinstance(drug_key(ASP), str), True)
check("key 14 akshar ki hai", len(drug_key(ASP) or ""), 14)
check("wahi SMILES -> wahi key", drug_key(ASP) == drug_key(ASP), True)
# Namak wala roop aur bina namak - ek hi dawa maani jani chahiye
check("sodium salt == free acid",
      drug_key("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]") == drug_key(ASP), True)
check("alag dawa -> alag key", drug_key("CCO") != drug_key(ASP), True)
check("bekaar SMILES -> None", drug_key("not_a_smiles"), None)
check("khali -> None", drug_key(""), None)
check("None -> None", drug_key(None), None)

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas")
