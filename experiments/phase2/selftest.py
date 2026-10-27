"""Bina network ke test - lamba job shuru hone se PEHLE chalta hai.

Kal ki sabse badi seekh: jo bug error nahi deta, wahi mehnga padta hai.
Isliye yahan har wo jagah check ho rahi hai jahan galat jawab chup-chaap
nikal sakta hai: censored value ko maap samajh lena, Kd ko IC50 se jod
dena, ya median galat utha lena.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvest_chembl import clean, make_pairs, median_by, pK   # noqa: E402

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


def row(**kw):
    base = dict(activity_id=1, molecule_chembl_id="CHEMBL1",
                canonical_smiles="CCO", target_chembl_id="CHEMBL203",
                assay_chembl_id="A1", document_chembl_id="D1",
                assay_variant_mutation="T790M", assay_variant_accession="P1",
                standard_type="Kd", standard_relation="=",
                standard_value="1.0", standard_units="nM",
                data_validity_comment=None, assay_type="B")
    base.update(kw)
    return base


print("pK badalna")
check("1 nM -> 9.0", round(pK(1.0), 6), 9.0)
check("1000 nM -> 6.0", round(pK(1000.0), 6), 6.0)
check("10 nM -> 8.0", round(pK(10.0), 6), 8.0)

print("\nclean(): kooda hatana")
keep, drop = clean([row()], True)
check("saaf row bachti hai", len(keep), 1)

# Yahi Davis wali galti thi: '>' ka matlab "isse kam nahi naapa gaya",
# maap nahi. Ise value maan lena jhoota ddG banata hai.
keep, drop = clean([row(standard_relation=">")], True)
check("relation '>' hat jata hai", len(keep), 0)
keep, drop = clean([row(standard_relation="<")], True)
check("relation '<' hat jata hai", len(keep), 0)

keep, _ = clean([row(data_validity_comment="Potential transcription error")],
                True)
check("validity comment wali row hat jati hai", len(keep), 0)
keep, _ = clean([row(standard_units="uM")], True)
check("nM ke alawa units hat jate hain", len(keep), 0)
keep, _ = clean([row(standard_value="0")], True)
check("value 0 hat jati hai", len(keep), 0)
keep, _ = clean([row(standard_value=None)], True)
check("value None hat jati hai", len(keep), 0)
keep, _ = clean([row(assay_variant_mutation="")], True)
check("mutant chahte hue khali mutation hat jata hai", len(keep), 0)
keep, _ = clean([row(assay_variant_mutation="T790M")], False)
check("WT chahte hue mutation wali row hat jati hai", len(keep), 0)
keep, _ = clean([row(assay_variant_mutation=None)], False)
check("WT me khali mutation bachti hai", len(keep), 1)

print("\nmedian_by(): kai maapon me se median")
rows, _ = clean([row(standard_value="1.0", document_chembl_id="D1"),
                 row(standard_value="10.0", document_chembl_id="D2"),
                 row(standard_value="100.0", document_chembl_id="D3")], True)
M = median_by(rows, ["mol", "target", "type", "mut"])
k = ("CHEMBL1", "CHEMBL203", "Kd", "T790M")
check("ek hi bucket bana", list(M), [k])
check("n_meas = 3", M[k]["n_meas"], 3)
check("median pK = 8.0 (10 nM)", round(M[k]["pK"], 6), 8.0)
check("teeno docs rakhe", M[k]["docs"], ["D1", "D2", "D3"])

print("\nmake_pairs(): jodi banana")
mut, _ = clean([row(standard_value="100.0", assay_variant_mutation="T790M")],
               True)
wt, _ = clean([row(standard_value="1.0", assay_variant_mutation=None)], False)
M = median_by(mut, ["mol", "target", "type", "mut"])
W = median_by(wt, ["mol", "target", "type"])
pairs, no_wt = make_pairs(M, W, {"CHEMBL203": "EGFR"})
check("ek jodi bani", len(pairs), 1)
check("gene naam laga", pairs[0]["gene"], "EGFR")
# 1 nM -> pK 9, 100 nM -> pK 7, mutation ne 100x kamzor kiya => -2
check("ddG = -2.0 (100x kamzor)", pairs[0]["ddG"], -2.0)
check("same_doc True (dono D1 se)", pairs[0]["same_doc"], True)

# Asli test: Kd mutant ko IC50 WT se JODNA NAHI chahiye
mut, _ = clean([row(standard_type="Kd", standard_value="100.0")], True)
wt, _ = clean([row(standard_type="IC50", standard_value="1.0",
                   assay_variant_mutation=None)], False)
M = median_by(mut, ["mol", "target", "type", "mut"])
W = median_by(wt, ["mol", "target", "type"])
pairs, no_wt = make_pairs(M, W, {})
check("Kd x IC50 ki jodi NAHI banti", len(pairs), 0)
check("aur wo unpaired me gina gaya", no_wt, 1)

# Alag paper se aaye to jodi bane par nishaan lage
mut, _ = clean([row(standard_value="100.0", document_chembl_id="D9")], True)
wt, _ = clean([row(standard_value="1.0", document_chembl_id="D1",
                   assay_variant_mutation=None)], False)
M = median_by(mut, ["mol", "target", "type", "mut"])
W = median_by(wt, ["mol", "target", "type"])
pairs, _ = make_pairs(M, W, {})
check("alag paper -> jodi banti hai", len(pairs), 1)
check("alag paper -> same_doc False", pairs[0]["same_doc"], False)

# Alag drug ki jodi nahi banni chahiye
mut, _ = clean([row(molecule_chembl_id="CHEMBL1")], True)
wt, _ = clean([row(molecule_chembl_id="CHEMBL2",
                   assay_variant_mutation=None)], False)
M = median_by(mut, ["mol", "target", "type", "mut"])
W = median_by(wt, ["mol", "target", "type"])
pairs, _ = make_pairs(M, W, {})
check("alag drug ki jodi NAHI banti", len(pairs), 0)

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas - lamba job chala sakte hain")
