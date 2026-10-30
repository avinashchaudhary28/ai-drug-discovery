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


def pair(mut_rows, wt_rows, genes=None):
    """Wahi raasta jo asli script leta hai - doc-aware."""
    mut, _ = clean(mut_rows, True)
    wt, _ = clean(wt_rows, False)
    M = median_by(mut, ["mol", "target", "type", "mut", "doc"])
    W_doc = median_by(wt, ["mol", "target", "type", "doc"])
    W_all = median_by(wt, ["mol", "target", "type"])
    return make_pairs(M, W_doc, W_all, genes or {})


pairs, no_wt = pair(
    [row(standard_value="100.0", assay_variant_mutation="T790M")],
    [row(standard_value="1.0", assay_variant_mutation=None)],
    {"CHEMBL203": "EGFR"})
check("ek jodi bani", len(pairs), 1)
check("gene naam laga", pairs[0]["gene"], "EGFR")
# 1 nM -> pK 9, 100 nM -> pK 7, mutation ne 100x kamzor kiya => -2
check("ddG = -2.0 (100x kamzor)", pairs[0]["ddG"], -2.0)
check("same_doc True (dono D1 se)", pairs[0]["same_doc"], True)

# Asli test: Kd mutant ko IC50 WT se JODNA NAHI chahiye
pairs, no_wt = pair(
    [row(standard_type="Kd", standard_value="100.0")],
    [row(standard_type="IC50", standard_value="1.0",
         assay_variant_mutation=None)])
check("Kd x IC50 ki jodi NAHI banti", len(pairs), 0)
check("aur wo unpaired me gina gaya", no_wt, 1)

# Alag drug ki jodi nahi banni chahiye
pairs, _ = pair(
    [row(molecule_chembl_id="CHEMBL1")],
    [row(molecule_chembl_id="CHEMBL2", assay_variant_mutation=None)])
check("alag drug ki jodi NAHI banti", len(pairs), 0)

print("\nUSI PAPER ka wild-type pehle (yahi asli sudhaar hai)")
# Mutant D1 me hai. WT dono me: D1 me 1 nM (pK 9), D2 me 1000 nM (pK 6).
# Usi paper ka WT (D1) lena chahiye, dono ka median nahi.
pairs, _ = pair(
    [row(standard_value="100.0", document_chembl_id="D1")],
    [row(standard_value="1.0", document_chembl_id="D1",
         assay_variant_mutation=None),
     row(standard_value="1000.0", document_chembl_id="D2",
         assay_variant_mutation=None)])
check("ek jodi bani", len(pairs), 1)
check("usi paper ka WT liya (ddG -2.0, median wala -0.5 nahi)",
      pairs[0]["ddG"], -2.0)
check("same_doc True", pairs[0]["same_doc"], True)

# Ek hi paper me kai WT maap - unka median theek hai, kyunki ek hi lab
pairs, _ = pair(
    [row(standard_value="100.0", document_chembl_id="D1")],
    [row(standard_value="1.0", document_chembl_id="D1",
         assay_variant_mutation=None),
     row(standard_value="10.0", document_chembl_id="D1",
         assay_variant_mutation=None),
     row(standard_value="100.0", document_chembl_id="D1",
         assay_variant_mutation=None)])
check("ek hi paper ke 3 maapon ka median (pK 8) -> ddG -1.0",
      pairs[0]["ddG"], -1.0)

print("\nusi paper ka WT na ho to nishaan ke saath doosre paper se")
pairs, _ = pair(
    [row(standard_value="100.0", document_chembl_id="D9")],
    [row(standard_value="1.0", document_chembl_id="D1",
         assay_variant_mutation=None)])
check("jodi banti hai", len(pairs), 1)
check("same_doc False (nishaan lagta hai)", pairs[0]["same_doc"], False)
check("ddG -2.0", pairs[0]["ddG"], -2.0)

print("\ndo paper = do SWATANTRA maap, ek nahi")
pairs, _ = pair(
    [row(standard_value="100.0", document_chembl_id="D1"),
     row(standard_value="10.0", document_chembl_id="D2")],
    [row(standard_value="1.0", document_chembl_id="D1",
         assay_variant_mutation=None),
     row(standard_value="1.0", document_chembl_id="D2",
         assay_variant_mutation=None)])
check("do alag jode bane", len(pairs), 2)
check("dono same_doc", all(p["same_doc"] for p in pairs), True)
check("ddG alag-alag (-2.0 aur -1.0)",
      sorted(p["ddG"] for p in pairs), [-2.0, -1.0])
check("doc column darj hua", sorted(p["doc"] for p in pairs), ["D1", "D2"])

print("\ncache ka resume (adhoora download poora na dikhe)")
import json as _json                                        # noqa: E402
import pathlib as _pl                                       # noqa: E402
import tempfile                                             # noqa: E402

import harvest_chembl as H                                  # noqa: E402

_tmp = _pl.Path(tempfile.mkdtemp())
H.WORK = _tmp
H.TYPES = ["Kd"]


def _fake_fetch(path, params, splitters, on_batch, label=""):
    """Ek nakli row deta hai, taaki asli network na lage."""
    tid = params.get("target_chembl_id", "CHEMBL203")
    on_batch([row(target_chembl_id=tid, assay_variant_mutation=None)])
    return 1


H.fetch_all = _fake_fetch

# Bina 'done' nishaan wali adhoori file -> dobara download honi chahiye
(_tmp / "variant_activities.jsonl").write_text(_json.dumps(row()) + "\n")
got = H.harvest_mutants()
check("done nishaan ke bina file dobara utarti hai", len(got), 1)
check("nishaan ab ban gaya", (_tmp / "variant.done").exists(), True)

# Ab nishaan hai -> cache se hi uthna chahiye, download nahi
H.fetch_all = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("nishaan hone ke baad bhi download hua"))
got = H.harvest_mutants()
check("nishaan hone par cache se uthta hai", len(got), 1)

# WT resume: do targets me se ek pehle ho chuka hai
H.fetch_all = _fake_fetch
(_tmp / "wt_done.json").write_text(_json.dumps(["CHEMBL1"]))
(_tmp / "wt_activities.jsonl").write_text(
    _json.dumps(row(target_chembl_id="CHEMBL1",
                    assay_variant_mutation=None)) + "\n")
got = H.harvest_wt(["CHEMBL1", "CHEMBL2"])
check("WT resume: purana + naya = 2 rows", len(got), 2)
check("WT resume: purana target dobara nahi utra",
      sum(1 for r in got if r["target_chembl_id"] == "CHEMBL1"), 1)
check("WT resume: dono targets done me darj",
      _json.loads((_tmp / "wt_done.json").read_text()),
      ["CHEMBL1", "CHEMBL2"])

print("\nbatware ki jaanch (tukdon ka jod parent se milna chahiye)")
import chembl_api as CA                                     # noqa: E402


def _fake_api(counts):
    """counts: assay_type -> kitni rows. None key = bina tod wala parent."""
    def fake_get(path, params=None, tries=5):
        params = params or {}
        n = counts.get(params.get("assay_type"), 0)
        if params.get("limit") == 1:
            return {"page_meta": {"total_count": n}, "activities": []}
        off = params.get("offset", 0)
        lim = params.get("limit", 1000)
        rows = [{"activity_id": off + i} for i in range(max(0, min(lim, n - off)))]
        nxt = "more" if off + lim < n else None
        return {"page_meta": {"total_count": n, "next": nxt},
                "activities": rows}
    return fake_get


# Tod sirf tab hota hai jab parent MAX_OFFSET se bada ho, isliye test ke
# liye limit ghata rahe hain: parent 30 > 25, par har tukda 25 se chhota.
CA.MAX_OFFSET = 25

# Theek batwara: 20 + 10 = 30 = parent
CA.get = _fake_api({None: 30, "B": 20, "F": 10})
CA.SHORTFALL.clear()
got = CA.fetch_all("/activity.json", {}, [ASSAY := [{"assay_type": "B"},
                                                    {"assay_type": "F"}]],
                   lambda b: None, label="test-ok")
check("poora batwara: 30 rows aayin", got, 30)
check("poora batwara: koi shikayat nahi", len(CA.SHORTFALL), 0)

# Adhoora batwara: 20 + 0 = 20 par parent 30 -> 10 rows ka hisaab nahi
CA.get = _fake_api({None: 30, "B": 20, "F": 0})
CA.SHORTFALL.clear()
got = CA.fetch_all("/activity.json", {}, [ASSAY], lambda b: None,
                   label="test-gap")
check("adhoora batwara pakda gaya", len(CA.SHORTFALL), 1)
check("adhoori gintee sahi darj hui",
      CA.SHORTFALL[0]["missing"] if CA.SHORTFALL else None, 10)

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas - lamba job chala sakte hain")
