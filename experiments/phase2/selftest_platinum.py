"""Bina network ke test: Platinum ka ddG hisaab aur sign.

Sabse khatarnak galti yahan sign ki hai. Agar ddG ka sign ulta ho jaye
to resistance (dawa kaam nahi karti) sudhaar (dawa behtar kaam karti)
jaisa dikhega - aur wo galti kabhi error nahi degi, bas model ko ulta
seekha degi.

Platinum ke asli data se naapi gayi convention:
    delta_k = k_wt - k_mt,  fold_change = +/- (k_mt / k_wt)
    => ddG = log10(k_wt / k_mt)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvest_platinum import compute, num, sign_check     # noqa: E402

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


def close(name, got, want, tol=1e-6):
    if got is not None and abs(got - want) < tol:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


def row(**kw):
    base = {
        "mutation": "T790M",
        "affin.k_wt": "1", "affin.k_mt": "3.2",
        "affin.delta_k": "-2.2", "affin.fold_change": "-3.2",
        "affin.unit": "Ki", "affin.exptal_method": "kinetic",
        "affin.lig_name": "X", "affin.pdb_id": "1ABC",
        "lig.canonical_smiles": "CCO",
        "mut.uniprot": "P00533", "mut.is_single_point": "YES",
        "mut.in_binding_site": "YES", "mut.distance_to_lig": "3.2",
        "mut.wt_pdb": "1ABC", "mut.mt_pdb": "1ABD",
        "mut.pmid": "1", "mut.doi": "d",
        "prot.molecule_name": "EGFR", "prot.protein_class": "Protein kinase",
        "prot.organism": "HUMAN",
    }
    base.update(kw)
    return base


print("ddG ka hisaab (asli Platinum row se)")
# Asli row: k_wt=1, k_mt=3.2 -> dawa mutant pe 3.2 guna KAMZOR
d, _ = compute(pd.DataFrame([row()]))
check("ek row bachi", len(d), 1)
close("ddG = log10(1/3.2) = -0.505", float(d.ddG.iloc[0]), np.log10(1 / 3.2))
check("ddG negative (kamzor binding)", float(d.ddG.iloc[0]) < 0, True)

# Ulta: mutant pe dawa BEHTAR lagti hai
d, _ = compute(pd.DataFrame([row(**{"affin.k_wt": "10", "affin.k_mt": "1",
                                    "affin.fold_change": "0.1"})]))
close("ddG = log10(10/1) = +1.0", float(d.ddG.iloc[0]), 1.0)
check("ddG positive (behtar binding)", float(d.ddG.iloc[0]) > 0, True)

# Koi farak nahi
d, _ = compute(pd.DataFrame([row(**{"affin.k_wt": "5", "affin.k_mt": "5",
                                    "affin.fold_change": "1"})]))
close("barabar -> ddG 0", float(d.ddG.iloc[0]), 0.0)

print("\nkooda hatana")
for name, kw in [
    ("k_wt nahi", {"affin.k_wt": "NA"}),
    ("k_mt nahi", {"affin.k_mt": "NR"}),
    ("k_wt zero", {"affin.k_wt": "0"}),
    ("k_mt negative", {"affin.k_mt": "-1"}),
    ("SMILES nahi", {"lig.canonical_smiles": "NA"}),
    ("SMILES khali", {"lig.canonical_smiles": ""}),
    ("mutation samajh na aaye", {"mutation": "deletion of loop"}),
]:
    d, _ = compute(pd.DataFrame([row(**kw)]))
    check(f"{name} -> row hat jati hai", len(d), 0)

print("\nmulti-mutant")
d, _ = compute(pd.DataFrame([row(mutation="D30N/N88D")]))
check("D30N/N88D parse hua", d.mutation_norm.iloc[0], "D30N+N88D")

print("\nkinase ka nishaan - yahan maine pehle galti ki thi")
# Platinum PDB ki vocabulary deta hai: kinase "TRANSFERASE" ke neeche
# aati hai aur class me "kinase" shabd KABHI nahi hota. Pehle maine sirf
# class me dhundha tha aur "0 kinase" nikla - jo galat tha.
d, _ = compute(pd.DataFrame([row(**{
    "mut.uniprot": "P00519",
    "prot.molecule_name": "PROTO-ONCOGENE TYROSINE-PROTEIN KINASE ABL1",
    "prot.protein_class": "TRANSFERASE"})]))
check("ABL1 (class TRANSFERASE) protein kinase mana gaya",
      bool(d.is_protein_kinase.iloc[0]), True)
check("naam me kinase bhi mila", bool(d.name_has_kinase.iloc[0]), True)

# Aur "kinase" naam hona kaafi nahi: thymidine kinase METABOLIC kinase
# hai, protein kinase nahi - uska KLIFS pocket nahi hota
d, _ = compute(pd.DataFrame([row(**{
    "mut.uniprot": "P03176",
    "prot.molecule_name": "THYMIDINE KINASE",
    "prot.protein_class": "TRANSFERASE"})]))
check("thymidine kinase: naam me kinase hai",
      bool(d.name_has_kinase.iloc[0]), True)
check("par protein kinase NAHI mana gaya",
      bool(d.is_protein_kinase.iloc[0]), False)

d, _ = compute(pd.DataFrame([row(**{
    "mut.uniprot": "P12345",
    "prot.molecule_name": "Aldose reductase",
    "prot.protein_class": "OXIDOREDUCTASE"})]))
check("aldose reductase: dono nishaan nahi",
      bool(d.name_has_kinase.iloc[0]) or bool(d.is_protein_kinase.iloc[0]),
      False)

print("\nsign ki jaanch - yahi asli pehredaar hai")
# Sab theek: ddG aur fold_change ek disha me
good = pd.DataFrame([row(), row(**{"affin.k_wt": "10", "affin.k_mt": "1",
                                   "affin.fold_change": "0.1"})])
d, _ = compute(good)
# fold_change -3.2 (kamzor) aur ddG -0.5 -> dono negative, ek disha
sc = sign_check(d[d["affin.fold_change"].astype(str) == "-3.2"])
check("ek disha wali row pe 100% agree", sc.get("agree_frac"), 1.0)

# Jaan-boojh kar ULTA data: ddG negative par fold_change positive bada
bad = pd.DataFrame([row(**{"affin.k_wt": "1", "affin.k_mt": "100",
                           "affin.fold_change": "100"}) for _ in range(5)])
d, _ = compute(bad)
sc = sign_check(d)
check("ulti disha pakdi gayi", sc.get("disagree"), 5)
check("agree_frac 0", sc.get("agree_frac"), 0.0)

print("\nnum(): Platinum ka text")
check("'NA' -> NaN", bool(pd.isna(num(pd.Series(["NA"])).iloc[0])), True)
check("'NR' -> NaN", bool(pd.isna(num(pd.Series(["NR"])).iloc[0])), True)
close("'3.2' -> 3.2", float(num(pd.Series(["3.2"])).iloc[0]), 3.2)

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas")
