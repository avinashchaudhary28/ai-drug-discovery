"""Phase 2, kaam #2c: wo 9 wild-type jhagde jaanchna.

Milaan me nikla tha ki 1,108 saajha jodon me se jo 32 alag hain, unki jad
mutant ka maap nahi - 9 (gene, drug) jodiyon ka WILD-TYPE number dono
source me alag hai, aur wo ek number poore block ko ek saath khiska deta
hai.

Yahan koi naya download nahi hai. ChEMBL khud batata hai ki har maap kis
paper se aaya - wahi poochte hain:

  us gene x drug ke wild-type ke KITNE swatantra paper hain, aur wo apas
  me sehmat hain ya nahi?

Faisla ka tareeka, pehle se tay (baad me number dekh kar badla nahi
jayega):
  - 2+ swatantra paper, apas me sehmat (spread < 0.3), aur Davis unse
    door  -> ChEMBL ka number zyada bharosemand
  - sirf 1 paper                                 -> faisla nahi ho sakta
  - ChEMBL ke paper khud apas me alag (spread >= 0.3) -> ChEMBL hi
    doubtful, Davis ko dosh dena galat hoga

Jo "faisla nahi ho sakta" hai use waisa hi likha jayega. Teen me se do
ka jawab nikal aana bhi kaam ka hai; andaza lagakar nau bhar dena nahi.

Nateeja: results/wt_conflicts.json
"""
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chembl_api import fetch_all                          # noqa: E402
from conflicts import const_groups, overlap_table         # noqa: E402
from merge_davis import load_chembl, load_davis           # noqa: E402
from mutnorm import drug_key, normalize                   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"

AGREE_SPREAD = 0.3     # itne ke andar ho to paper apas me sehmat maane jayenge


def build():
    """Merge wala hi dataframe, wahi tareeka - alag copy nahi."""
    df = pd.concat([load_chembl(), load_davis()], ignore_index=True)
    norm = df.mutation_raw.map(normalize)
    df["mutation"] = ["+".join(t) for t, _, _ in norm]
    df["mut_note"] = [n for _, n, _ in norm]
    df["mut_parsed"] = [p for _, _, p in norm]
    df = df[df.mut_parsed].copy()
    keys = {s: drug_key(s) for s in df.smiles.dropna().unique()}
    df["drug_key"] = df.smiles.map(keys)
    return df[df.drug_key.notna()].copy()


def wt_measurements(target_id, mol_id, assay_type):
    """Us target x drug ke SAARE wild-type maap, paper ke saath."""
    rows = []
    fetch_all("/activity.json", {
        "assay_variant_mutation__isnull": "true",
        "target_chembl_id": target_id,
        "molecule_chembl_id": mol_id,
        "standard_type": assay_type,
        "standard_units": "nM",
        "standard_relation": "=",
    }, splitters=[], on_batch=rows.extend, label=f"{target_id}/{mol_id}")

    out = []
    for r in rows:
        if r.get("data_validity_comment"):
            continue
        try:
            v = float(r["standard_value"])
        except (TypeError, ValueError):
            continue
        if not (v > 0):
            continue
        out.append({
            "pK": round(9.0 - np.log10(v), 4),
            "nM": v,
            "doc": r.get("document_chembl_id"),
            "assay": r.get("assay_chembl_id"),
            "year": r.get("document_year"),
        })
    return out


def verdict(meas, davis_wt):
    """Pehle se tay niyam. Number dekh kar niyam nahi badlega."""
    docs = sorted({m["doc"] for m in meas if m["doc"]})
    pks = [m["pK"] for m in meas]
    if not pks:
        return "faisla nahi - ChEMBL me wild-type maap hi nahi mila", docs, None
    spread = float(np.std(pks)) if len(pks) > 1 else 0.0
    med = float(np.median(pks))
    gap = abs(med - davis_wt) if davis_wt is not None else None

    if len(docs) < 2:
        return (f"faisla nahi - sirf {len(docs)} paper", docs, med)
    if spread >= AGREE_SPREAD:
        return (f"ChEMBL khud bata hua (paperon ka spread {spread:.2f}) - "
                f"Davis ko galat kehna theek nahi", docs, med)
    if gap is None:
        # Davis ka wild-type number hi nahi mila - tulna ka aadhar nahi
        return (f"faisla nahi - Davis ka wild-type number nahi mila", docs, med)
    if gap >= 0.5:
        return (f"ChEMBL zyada bharosemand - {len(docs)} paper sehmat "
                f"(spread {spread:.2f}), Davis {gap:.2f} door", docs, med)
    return (f"{len(docs)} paper sehmat, par Davis bhi paas hai "
            f"({gap:.2f}) - jhagda bada nahi", docs, med)


def main():
    print("[1/3] dataset banao (merge wala hi tareeka)")
    df = build()
    both = overlap_table(df)
    if not len(both):
        raise RuntimeError("koi saajha jodi nahi mili - aage kuch nahi")
    multi, const = const_groups(both)
    print(f"  saajha jode {len(both):,}  |  3+ mutation wale groups "
          f"{len(multi)}  |  ek jaisa farak wale {len(const)}")
    if not len(const):
        print("  ek bhi ek-jaisa-farak group nahi - jaanchne ko kuch nahi")
        (RES / "wt_conflicts.json").write_text(json.dumps(
            {"groups": 0, "note": "koi const-offset group nahi"}, indent=2))
        return 0

    print("\n[2/3] har jhagde ka wild-type ChEMBL se poocho")
    ch = df[df.source == "chembl"]
    dv = df[df.source == "davis"]
    out = []
    for (gene, dk, at), row in const.sort_values(
            "off", key=abs, ascending=False).iterrows():
        sub = ch[(ch.gene == gene) & (ch.drug_key == dk)
                 & (ch.assay_type == at)]
        if sub.empty:
            print(f"  {gene}/{dk}: ChEMBL rows nahi mile - chhod rahe hain")
            continue
        tid = sub.target_id.iloc[0]
        mols = sorted(sub.drug_id.unique())
        dsub = dv[(dv.gene == gene) & (dv.drug_key == dk)]
        davis_wt = float(dsub.pK_wt.median()) if not dsub.empty else None
        chembl_wt = float(sub.pK_wt.median())

        meas = []
        for mol in mols[:3]:          # ek skeleton ke kai salt forms ho sakte
            meas += wt_measurements(tid, mol, at)
        v, docs, med = verdict(meas, davis_wt)

        print(f"\n  {gene}  {at}  offset {row.off:+.3f}  "
              f"({int(row.n)} mutations)")
        print(f"    wild-type pK: chembl {chembl_wt:.2f}  |  "
              f"davis {davis_wt if davis_wt is None else round(davis_wt, 2)}")
        print(f"    ChEMBL maap: {len(meas)}, paper: {len(docs)}"
              + (f", median pK {med:.2f}" if med is not None else ""))
        if meas:
            for m in sorted(meas, key=lambda x: x["pK"])[:6]:
                print(f"        pK {m['pK']:>7.2f}  {m['nM']:>12,.1f} nM  "
                      f"{m['doc']}  {m['assay']}")
        print(f"    >>> {v}")

        out.append({
            "gene": gene, "assay_type": at, "drug_key": dk,
            "target_chembl_id": tid, "molecules": mols,
            "n_mutations_affected": int(row.n),
            "offset": float(row.off), "offset_spread": float(row.spread or 0),
            "wt_chembl": chembl_wt, "wt_davis": davis_wt,
            "chembl_n_measurements": len(meas),
            "chembl_n_documents": len(docs),
            "chembl_documents": docs,
            "chembl_wt_median": med,
            "verdict": v,
        })

    print(f"\n[3/3] nateeja")
    if not out:
        print("  ek bhi jhagda jaancha nahi ja saka")
        (RES / "wt_conflicts.json").write_text(json.dumps(
            {"groups": 0, "note": "const-offset group mile par ChEMBL rows "
                                  "nahi mile"}, indent=2))
        return 0
    res = pd.DataFrame(out)
    # Pehle yahan .str.split(" (") tha. pandas use regex samajhta hai aur
    # "(" se pattern toot jata hai - run yahin gir gayi thi.
    tag = (res.verdict.str.split(" - ").str[0]
              .str.replace(r"\s*\(.*$", "", regex=True))
    print(f"  {len(res)} jhagde jaanche gaye, "
          f"{int(res.n_mutations_affected.sum())} jode inse jude hain")
    for k, n in tag.value_counts().items():
        print(f"    {n:>2}  {k}")

    decided = res[res.verdict.str.startswith("ChEMBL zyada")]
    print(f"\n  faisla ho gaya: {len(decided)} / {len(res)}  "
          f"({int(decided.n_mutations_affected.sum())} jode)")
    if len(decided):
        print(decided[["gene", "assay_type", "wt_chembl", "wt_davis",
                       "chembl_n_documents", "n_mutations_affected"]]
              .to_string(index=False))

    (RES / "wt_conflicts.json").write_text(json.dumps(
        {"groups": len(res),
         "pairs_affected": int(res.n_mutations_affected.sum()),
         "resolved_chembl_better": int(len(decided)),
         "pairs_resolved": int(decided.n_mutations_affected.sum()),
         "detail": out}, indent=2))
    res.to_csv(RES / "wt_conflicts.csv", index=False)
    print(f"\n  files: {RES}/wt_conflicts.json, wt_conflicts.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
