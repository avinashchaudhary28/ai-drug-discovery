"""Phase 2, kaam #2a: har target ko ChEMBL ki protein classification se chhanno.

Kyun: ChEMBL harvest me 217 genes aaye, par wo sab kinase nahi hain. GNRHR
aur CXCR4 GPCR hain, AR aur ESR1 nuclear receptor, BRD4/EZH2/HDAC8
epigenetic. Phase 3 ka pocket-aware tareeka (KLIFS ke 85 pocket residues)
SIRF kinase ka hai.

Ye faisla gene ke naam se andaza lagakar nahi karna hai - "LRRK2 kinase
lagta hai" wala tareeka kuch genes pe chup-chaap galat hoga. ChEMBL khud
protein classification deta hai; wahi uthate hain.

Nateeja: results/target_classes.json  { target_chembl_id: {gene, l1..l3,
is_kinase} }
"""
import json
import os
import pathlib
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chembl_api import get                               # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "work_cls"   # harvest ke 200MB cache se alag
RES = HERE / "results"
WORK.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)


def probe():
    """Structure check. Agar ChEMBL ka schema badla to yahin ruk jayein -
    warna sab targets 'kinase nahi' nikal aayenge aur koi error nahi aayega.
    """
    d = get("/target/CHEMBL203.json")        # EGFR - pakka kinase hai
    comps = d.get("target_components") or []
    if not comps:
        raise RuntimeError("CHEMBL203 me target_components nahi mile")
    pcs = comps[0].get("protein_classifications")
    if pcs is None:
        raise RuntimeError(
            "target_component me 'protein_classifications' nahi hai. "
            f"maujood: {sorted(comps[0])}")
    if not pcs:
        raise RuntimeError("EGFR ki koi protein_classification nahi aayi")
    cid = pcs[0].get("protein_classification_id")
    if cid is None:
        raise RuntimeError(f"classification entry me id nahi: {pcs[0]}")
    c = get(f"/protein_classification/{cid}.json")
    for f in ("l1", "l2", "l3"):
        if f not in c:
            raise RuntimeError(f"protein_classification me '{f}' nahi hai. "
                               f"maujood: {sorted(c)}")
    # EGFR kinase nikalna hi chahiye - ye poore tareeke ka asli test hai
    labels = [str(c.get(f"l{i}") or "") for i in range(1, 4)]
    if not any("kinase" in x.lower() for x in labels):
        raise RuntimeError(
            f"EGFR ki classification me 'kinase' nahi mila: {labels}. "
            "Yaani humara kinase pakadne ka tareeka galat hai - ruk rahe hain.")
    print(f"✓ schema theek hai; EGFR -> {' / '.join(x for x in labels if x)}")


def classify(target_ids):
    cache = WORK / "classes.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [t for t in target_ids if t not in known]
    cls_cache = WORK / "class_labels.json"
    labels = json.loads(cls_cache.read_text()) if cls_cache.exists() else {}
    print(f"  {len(known)} cache me, {len(todo)} baaki")

    for i, tid in enumerate(todo, 1):
        try:
            d = get(f"/target/{tid}.json")
        except Exception as e:                            # noqa: BLE001
            # Chup-chaap 'kinase nahi' likhna galat hoga - alag nishaan
            known[tid] = {"error": repr(e)}
            print(f"    {tid}: {e}")
            continue

        gene, paths = None, []
        for comp in d.get("target_components") or []:
            for syn in comp.get("target_component_synonyms") or []:
                if syn.get("syn_type") == "GENE_SYMBOL" and not gene:
                    gene = syn.get("component_synonym")
            for pc in comp.get("protein_classifications") or []:
                cid = str(pc.get("protein_classification_id"))
                if cid not in labels:
                    try:
                        c = get(f"/protein_classification/{cid}.json")
                    except Exception:                     # noqa: BLE001
                        continue
                    labels[cid] = [c.get(f"l{n}") for n in range(1, 9)]
                paths.append(labels[cid])

        flat = " / ".join(str(x) for p in paths for x in p if x).lower()
        known[tid] = {
            "gene": gene,
            "target_type": d.get("target_type"),
            "pref_name": d.get("pref_name"),
            "paths": paths,
            # ChEMBL ki classification me kinase "Enzyme / Kinase /
            # Protein Kinase" ke neeche aati hai. Naam se andaza nahi -
            # classification ke text se.
            "is_kinase": "kinase" in flat,
        }
        if i % 25 == 0 or i == len(todo):
            cache.write_text(json.dumps(known))
            cls_cache.write_text(json.dumps(labels))
            print(f"    {i}/{len(todo)}", flush=True)

    cache.write_text(json.dumps(known))
    cls_cache.write_text(json.dumps(labels))
    return known


def main():
    probe()
    src = RES / "chembl_ddg.csv"
    if not src.exists():
        raise RuntimeError(f"{src} nahi mila - pehle harvest chalao")
    df = pd.read_csv(src)
    tids = sorted(df.target_chembl_id.unique())
    print(f"\n{len(tids)} targets chhanne hain")
    known = classify(tids)

    (RES / "target_classes.json").write_text(json.dumps(known, indent=2))

    err = [t for t, v in known.items() if v.get("error")]
    kin = {t for t, v in known.items() if v.get("is_kinase")}
    unk = [t for t, v in known.items()
           if not v.get("error") and not v.get("paths")]

    df["is_kinase"] = df.target_chembl_id.map(
        lambda t: known.get(t, {}).get("is_kinase", False))
    k, nk = df[df.is_kinase], df[~df.is_kinase]

    print(f"\n{'=' * 68}\nNATEEJA\n{'=' * 68}\n")
    print(f"  targets: {len(tids)}  |  kinase: {len(kin)}  |  "
          f"kinase nahi: {len(tids) - len(kin) - len(err)}  |  "
          f"lookup fail: {len(err)}")
    if unk:
        print(f"  !! {len(unk)} targets ki koi classification hi nahi aayi - "
              f"unhe 'kinase nahi' mana gaya, jo galat ho sakta hai: "
              f"{unk[:10]}")
    if err:
        print(f"  !! {len(err)} targets ka lookup fail hua, unhe bhi "
              f"'kinase nahi' mana gaya: {err[:10]}")

    for name, d_ in (("KINASE", k), ("KINASE NAHI", nk)):
        if d_.empty:
            print(f"\n  {name}: kuch nahi")
            continue
        big = d_[d_.ddG.abs() >= 1.0]
        print(f"\n  {name}")
        print(f"    jode        : {len(d_):,}")
        print(f"    genes       : {d_.gene.nunique()}")
        print(f"    mutations   : {d_.mutation.nunique()}")
        print(f"    drugs       : {d_.mol_chembl_id.nunique()}")
        print(f"    |ddG|>=1.0  : {len(big):,} ({len(big) / len(d_):.0%}), "
              f"{big.gene.nunique()} genes")
        kk = d_[d_.assay_type != "IC50"]
        print(f"    Kd/Ki hissa : {len(kk):,} ({len(kk) / len(d_):.0%})")
        print(f"    top genes   : "
              f"{', '.join(f'{g} {n}' for g, n in d_.gene.value_counts().head(8).items())}")

    summary = {
        "targets": len(tids),
        "kinase_targets": len(kin),
        "lookup_failed": err,
        "no_classification": unk,
        "kinase_pairs": int(len(k)),
        "non_kinase_pairs": int(len(nk)),
        "kinase_genes": int(k.gene.nunique()) if not k.empty else 0,
        "non_kinase_genes": int(nk.gene.nunique()) if not nk.empty else 0,
        "kinase_pairs_ddG_ge_1": int((k.ddG.abs() >= 1).sum()) if not k.empty else 0,
    }
    (RES / "classify_report.json").write_text(json.dumps(summary, indent=2))
    df.to_csv(RES / "chembl_ddg_classified.csv", index=False)
    print(f"\n  files: {RES}/target_classes.json, chembl_ddg_classified.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
