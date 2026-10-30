"""Phase 2, kaam #2a: har target ko ChEMBL ki protein classification se chhanno.

Kyun: ChEMBL harvest me 217 genes aaye, par wo sab kinase nahi hain. CXCR4
aur MC4R GPCR hain, AR aur ESR1 nuclear receptor, BRD4/EZH2/HDAC8
epigenetic. Phase 3 ka pocket-aware tareeka (KLIFS ke 85 pocket residues)
SIRF kinase ka hai, isliye batwara zaroori hai.

Faisla gene ke naam se NAHI hota - "LRRK2 kinase lagta hai" wala tareeka
kuch genes pe chup-chaap galat hoga. ChEMBL khud classification deti hai.

ChEMBL ka asli dhaancha (probe_api.py se naapa gaya, andaza nahi):
    /target/{id}            -> target_components[].component_id
    /target_component/{cid} -> protein_classifications[].protein_classification_id
    /protein_classification -> protein_class_desc (poora path ek string me)
Pehli koshish me maine socha tha ki classification /target me hi milegi
aur usme l1..l8 fields honge - dono galat the.

Nateeja: results/target_classes.json
"""
import json
import os
import pathlib
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chembl_api import PAGE, get                         # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "work_cls"   # harvest ke 200MB cache se alag
RES = HERE / "results"
WORK.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# Jaanch ke liye: do pakke kinase, ek pakka non-kinase.
# CHEMBL1856 pehle galti se "GNRHR" samjha gaya tha; probe ne pref_name
# chhapa to pata chala ki wo SRD5A2 hai - ek reductase, kinase nahi.
# Negative control ke liye wo theek hai, bas naam sahi hona chahiye.
KINASE_OK = [("EGFR", "CHEMBL203"), ("ABL1", "CHEMBL1862")]
KINASE_NO = [("SRD5A2 (reductase)", "CHEMBL1856")]


def all_classes():
    """Saari 905 protein classifications ek baar me - id -> desc."""
    cache = WORK / "class_desc.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        print(f"  classifications cache se: {len(d)}")
        return d
    out, offset = {}, 0
    while True:
        d = get("/protein_classification.json",
                {"limit": PAGE, "offset": offset})
        rows = d.get("protein_classifications") or []
        if not rows:
            break
        for r in rows:
            out[str(r["protein_class_id"])] = {
                "desc": r.get("protein_class_desc") or "",
                "pref_name": r.get("pref_name"),
                "level": r.get("class_level"),
            }
        total = d["page_meta"].get("total_count") or 0
        offset += PAGE
        if offset >= total:
            break
    if not out:
        raise RuntimeError("ek bhi protein_classification nahi aayi")
    cache.write_text(json.dumps(out))
    print(f"  classifications: {len(out)}")
    return out


def component_classes(tid, classes, comp_cache):
    """-> (gene, [desc, ...], n_components)"""
    d = get(f"/target/{tid}.json")
    gene, cids = None, []
    for comp in d.get("target_components") or []:
        for syn in comp.get("target_component_synonyms") or []:
            if syn.get("syn_type") == "GENE_SYMBOL" and not gene:
                gene = syn.get("component_synonym")
        if comp.get("component_id") is not None:
            cids.append(str(comp["component_id"]))

    descs = []
    for cid in cids:
        if cid not in comp_cache:
            try:
                tc = get(f"/target_component/{cid}.json")
                comp_cache[cid] = [
                    str(pc["protein_classification_id"])
                    for pc in (tc.get("protein_classifications") or [])
                    if pc.get("protein_classification_id") is not None]
            except Exception as e:                        # noqa: BLE001
                print(f"      component {cid} fail: {e}")
                comp_cache[cid] = []
        for pid in comp_cache[cid]:
            if pid in classes:
                descs.append(classes[pid]["desc"])
    return gene, descs, len(cids)


def is_kinase(descs):
    """ChEMBL ki classification me kinase 'enzyme/kinase/protein kinase'
    ke neeche aati hai. Naam se andaza nahi - classification ke text se."""
    return any("kinase" in (d or "").lower() for d in descs)


def probe(classes):
    """Poora raasta dono tarah ke target pe chala kar dekho.

    Sirf kinase pe chal jana kaafi nahi: agar tareeka har cheez ko kinase
    keh de, to wo bhi paas ho jayega. Isliye ek non-kinase bhi.
    """
    comp_cache = {}
    for name, tid in KINASE_OK:
        gene, descs, _ = component_classes(tid, classes, comp_cache)
        if not descs:
            raise RuntimeError(f"{name} ({tid}): koi classification nahi aayi")
        if not is_kinase(descs):
            raise RuntimeError(
                f"{name} ({tid}) kinase nahi nikla: {descs}. Yaani humara "
                f"pehchaan ka tareeka galat hai - ruk rahe hain.")
        print(f"  ✓ {name} -> kinase  ({descs[0][:70]})")
    for name, tid in KINASE_NO:
        gene, descs, _ = component_classes(tid, classes, comp_cache)
        if not descs:
            raise RuntimeError(f"{name} ({tid}): koi classification nahi aayi")
        if is_kinase(descs):
            raise RuntimeError(
                f"{name} ({tid}) ko kinase bata diya: {descs}. Tareeka "
                f"har cheez ko kinase keh raha hai - ruk rahe hain.")
        print(f"  ✓ {name} -> kinase nahi  ({descs[0][:70]})")
    return comp_cache


def main():
    print("[1/3] ChEMBL ki classification list")
    classes = all_classes()

    print("\n[2/3] tareeke ki jaanch (kinase aur non-kinase, dono pe)")
    comp_cache = probe(classes)

    src = RES / "chembl_ddg.csv"
    if not src.exists():
        raise RuntimeError(f"{src} nahi mila - pehle harvest chalao")
    df = pd.read_csv(src)
    tids = sorted(df.target_chembl_id.unique())

    print(f"\n[3/3] {len(tids)} targets chhanna")
    cache = WORK / "classes.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    cc_path = WORK / "components.json"
    if cc_path.exists():
        comp_cache.update(json.loads(cc_path.read_text()))
    todo = [t for t in tids if t not in known]
    print(f"  {len(known)} cache me, {len(todo)} baaki")

    for i, tid in enumerate(todo, 1):
        try:
            gene, descs, ncomp = component_classes(tid, classes, comp_cache)
        except Exception as e:                            # noqa: BLE001
            # Chup-chaap 'kinase nahi' likhna galat hoga - alag nishaan
            known[tid] = {"error": repr(e)}
            print(f"    {tid}: {e}")
            continue
        known[tid] = {"gene": gene, "n_components": ncomp,
                      "classes": sorted(set(descs)),
                      "is_kinase": is_kinase(descs)}
        if i % 25 == 0 or i == len(todo):
            cache.write_text(json.dumps(known))
            cc_path.write_text(json.dumps(comp_cache))
            print(f"    {i}/{len(todo)}", flush=True)
    cache.write_text(json.dumps(known))
    cc_path.write_text(json.dumps(comp_cache))
    (RES / "target_classes.json").write_text(json.dumps(known, indent=2))

    err = [t for t, v in known.items() if v.get("error")]
    unk = [t for t, v in known.items()
           if not v.get("error") and not v.get("classes")]
    kin = {t for t, v in known.items() if v.get("is_kinase")}

    df["is_kinase"] = df.target_chembl_id.map(
        lambda t: bool(known.get(t, {}).get("is_kinase")))
    df["class_known"] = df.target_chembl_id.map(
        lambda t: bool(known.get(t, {}).get("classes")))
    k, nk = df[df.is_kinase], df[~df.is_kinase]

    print(f"\n{'=' * 68}\nNATEEJA\n{'=' * 68}\n")
    print(f"  targets {len(tids)}  |  kinase {len(kin)}  |  "
          f"kinase nahi {len(tids) - len(kin) - len(err) - len(unk)}  |  "
          f"classification hi nahi {len(unk)}  |  lookup fail {len(err)}")
    if unk:
        print(f"  !! {len(unk)} targets ki classification nahi aayi. Inhe "
              f"'kinase nahi' gina gaya, jo galat ho sakta hai: {unk[:10]}")
    if err:
        print(f"  !! {len(err)} targets ka lookup fail hua, wo bhi 'kinase "
              f"nahi' me gine gaye: {err[:10]}")

    for name, d_ in (("KINASE", k), ("KINASE NAHI", nk)):
        if d_.empty:
            print(f"\n  {name}: kuch nahi")
            continue
        big = d_[d_.ddG.abs() >= 1.0]
        kk = d_[d_.assay_type != "IC50"]
        print(f"\n  {name}")
        print(f"    jode       : {len(d_):,}")
        print(f"    genes      : {d_.gene.nunique()}")
        print(f"    mutations  : {d_.mutation.nunique()}")
        print(f"    drugs      : {d_.mol_chembl_id.nunique()}")
        print(f"    |ddG|>=1.0 : {len(big):,} ({len(big) / len(d_):.0%}), "
              f"{big.gene.nunique()} genes")
        print(f"    Kd/Ki      : {len(kk):,} ({len(kk) / len(d_):.0%})")
        print(f"    top genes  : " + ", ".join(
            f"{g} {n}" for g, n in d_.gene.value_counts().head(10).items()))

    summary = {
        "targets": len(tids),
        "kinase_targets": len(kin),
        "lookup_failed": err,
        "no_classification": unk,
        "kinase_pairs": int(len(k)), "non_kinase_pairs": int(len(nk)),
        "kinase_genes": int(k.gene.nunique()) if not k.empty else 0,
        "non_kinase_genes": int(nk.gene.nunique()) if not nk.empty else 0,
        "kinase_pairs_ddG_ge_1": int((k.ddG.abs() >= 1).sum()) if not k.empty else 0,
        "kinase_genes_ddG_ge_1": int(k[k.ddG.abs() >= 1].gene.nunique()) if not k.empty else 0,
    }
    (RES / "classify_report.json").write_text(json.dumps(summary, indent=2))
    df.to_csv(RES / "chembl_ddg_classified.csv", index=False)
    print(f"\n  files: {RES}/target_classes.json, chembl_ddg_classified.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
