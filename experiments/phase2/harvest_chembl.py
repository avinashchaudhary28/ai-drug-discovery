"""Phase 2, kaam #1: ChEMBL se mutation wale assays nikalna.

Kyun: Davis se humein 1,480 saaf ddG jode mile, par sirf **10 genes** se,
aur unme se 70% bas ABL1+EGFR+KIT. Model ko aise data pe train karke
"ye mutation ka asar batata hai" kehna bemaani hai - wo sirf teen genes
ka asar seekhega.

ChEMBL me mutation assay pe hi dargaj hota hai: activity record me
`assay_variant_mutation` field hota hai (jaise "T790M"). Wahi se
wild-type vs mutant jode banenge.

Jo is script me JAAN-BOOJH kar nahi kiya gaya:
  - do alag assay type (Kd aur IC50) ke beech jodi nahi banegi. Dono
    alag cheez naapte hain; unko mila dena mithe-mithe numbers deta hai
    aur nateeja kooda ho jata hai.
  - jahan relation '=' nahi hai (jaise ">10000") wahan se value nahi li
    jayegi. Davis me yahi galti chhupi thi: 5.0 ek chhat thi, maap nahi.
  - WT aur mutant alag-alag paper se aaye hain to jodi banegi par
    `same_doc=False` likha jayega, taaki baad me chhaan sakein.

Nateeja: results/chembl_ddg.csv aur results/chembl_harvest.json
"""
import collections
import json
import math
import os
import pathlib
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chembl_api import count, fetch_all, get          # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "work"
RES = HERE / "results"
WORK.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

TYPES = ["Kd", "Ki", "IC50"]
ORGANISM = "Homo sapiens"

# Tulna ke liye maujooda aankde
PLATINUM = 796
DAVIS_CLEAN = 1480


# ---------------------------------------------------------------- probe
def probe():
    """Pehla kadam: check karo ki field maujood hai.

    Agar ChEMBL ne field ka naam badal diya aur hum bina check aage badhein
    to zero rows milenge aur koi error nahi - "kuch nahi mila" jaisa
    dikhega, jo jhoot hai.
    """
    d = get("/activity.json", {
        "assay_variant_mutation__isnull": "false",
        "limit": 1, "offset": 0})
    rows = d.get("activities") or []
    if not rows:
        raise RuntimeError(
            "assay_variant_mutation filter se ek bhi row nahi mili. "
            "ChEMBL ka schema badla ho sakta hai - aage badhna bekaar hai.")
    r = rows[0]
    for f in ("assay_variant_mutation", "target_chembl_id",
              "molecule_chembl_id", "standard_type", "standard_value",
              "standard_units", "standard_relation", "document_chembl_id"):
        if f not in r:
            raise RuntimeError(f"activity record me '{f}' field nahi hai. "
                               f"maujood: {sorted(r)}")
    total = count("/activity.json",
                  {"assay_variant_mutation__isnull": "false"})
    print(f"✓ schema theek hai. ChEMBL me kul variant activities: {total:,}")
    return total


# -------------------------------------------------------------- harvest
FIELDS = ["activity_id", "molecule_chembl_id", "canonical_smiles",
          "target_chembl_id", "assay_chembl_id", "document_chembl_id",
          "assay_variant_mutation", "assay_variant_accession",
          "standard_type", "standard_relation", "standard_value",
          "standard_units", "data_validity_comment", "assay_type"]


def _slim(rows):
    return [{k: r.get(k) for k in FIELDS} for r in rows]


def _cached(cache_name):
    path = WORK / cache_name
    if path.exists():
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        print(f"  cache se: {len(rows):,} rows ({cache_name})")
        return rows
    return None


def harvest_mutants():
    """Saare mutation-annotated assays (poori ChEMBL me)."""
    cache_name = "variant_activities.jsonl"
    hit = _cached(cache_name)
    if hit is not None:
        return hit
    out = []
    with (WORK / cache_name).open("w") as fh:
        def on_batch(batch):
            for r in _slim(batch):
                fh.write(json.dumps(r) + "\n")
                out.append(r)
            fh.flush()

        for t in TYPES:
            fetch_all("/activity.json", {
                "assay_variant_mutation__isnull": "false",
                "target_organism": ORGANISM,
                "standard_type": t,
                "standard_units": "nM",
                "standard_relation": "=",
            }, split_on=[("assay_type", ["B", "F", "A", "T", "P", "U"])],
                on_batch=on_batch, label=f"MUT {t}")
    print(f"  kul: {len(out):,} rows -> {cache_name}")
    return out


def harvest_wt(target_ids):
    """Wild-type assays - SIRF un targets ke liye jinpe mutation data mila.

    Poori ChEMBL ka WT data lakhon rows ka hai aur ChEMBL ki paging limit
    se bada; target-wise maangne se wo bhi theek ho jata hai aur download
    bhi sau guna chhota.
    """
    cache_name = "wt_activities.jsonl"
    hit = _cached(cache_name)
    if hit is not None:
        return hit
    out = []
    done = 0
    with (WORK / cache_name).open("w") as fh:
        def on_batch(batch):
            for r in _slim(batch):
                fh.write(json.dumps(r) + "\n")
                out.append(r)
            fh.flush()

        for tid in target_ids:
            for t in TYPES:
                fetch_all("/activity.json", {
                    "assay_variant_mutation__isnull": "true",
                    "target_chembl_id": tid,
                    "standard_type": t,
                    "standard_units": "nM",
                    "standard_relation": "=",
                }, split_on=[("assay_type", ["B", "F", "A", "T", "P", "U"])],
                    on_batch=on_batch, label=f"WT {tid} {t}")
            done += 1
            if done % 10 == 0:
                print(f"  --- {done}/{len(target_ids)} targets, "
                      f"{len(out):,} rows tak", flush=True)
    print(f"  kul: {len(out):,} rows -> {cache_name}")
    return out


# ------------------------------------------------------------ gene names
def gene_names(target_ids):
    path = WORK / "genes.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    todo = [t for t in target_ids if t not in known]
    print(f"  gene symbol: {len(known)} cache me, {len(todo)} baaki")
    for i, tid in enumerate(todo, 1):
        try:
            d = get(f"/target/{tid}.json")
        except Exception as e:                        # noqa: BLE001
            known[tid] = None
            print(f"    {tid}: {e}")
            continue
        gene = None
        for comp in d.get("target_components") or []:
            for syn in comp.get("target_component_synonyms") or []:
                if syn.get("syn_type") == "GENE_SYMBOL":
                    gene = syn.get("component_synonym")
                    break
            if gene:
                break
        known[tid] = gene
        if i % 50 == 0:
            path.write_text(json.dumps(known))
            print(f"    {i}/{len(todo)}")
    path.write_text(json.dumps(known))
    return known


# ----------------------------------------------------------------- pair
def pK(nm):
    return 9.0 - math.log10(nm)


def clean(rows, want_variant):
    """Rows ko bharosemand maapon tak seemit karo."""
    drop = collections.Counter()
    keep = []
    for r in rows:
        if r.get("data_validity_comment"):
            drop[f"validity: {r['data_validity_comment']}"] += 1
            continue
        if r.get("standard_relation") != "=":
            drop[f"relation {r.get('standard_relation')}"] += 1
            continue
        if r.get("standard_units") != "nM":
            drop["units != nM"] += 1
            continue
        try:
            v = float(r["standard_value"])
        except (TypeError, ValueError):
            drop["value nahi padha"] += 1
            continue
        if not (v > 0) or v > 1e7:
            drop["value range se bahar"] += 1
            continue
        mut = (r.get("assay_variant_mutation") or "").strip()
        if want_variant and not mut:
            drop["mutation khali"] += 1
            continue
        if not want_variant and mut:
            drop["WT me mutation mila"] += 1
            continue
        if not r.get("molecule_chembl_id") or not r.get("target_chembl_id"):
            drop["id missing"] += 1
            continue
        keep.append({
            "mol": r["molecule_chembl_id"],
            "smiles": r.get("canonical_smiles"),
            "target": r["target_chembl_id"],
            "doc": r.get("document_chembl_id"),
            "assay": r.get("assay_chembl_id"),
            "type": r["standard_type"],
            "mut": mut,
            "pK": pK(v),
            "nM": v,
        })
    return keep, drop


def median_by(rows, keys):
    """Ek hi (mol, target, type[, mut]) ke kai maap - median lo."""
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in keys)].append(r)
    out = {}
    for k, rs in buckets.items():
        rs = sorted(rs, key=lambda r: r["pK"])
        mid = rs[len(rs) // 2]
        out[k] = {**mid, "n_meas": len(rs),
                  "docs": sorted({r["doc"] for r in rs if r["doc"]})}
    return out


def make_pairs(M, W, genes):
    """Mutant maap ko usi (drug, target, assay-type) ke WT maap se jodo.

    Jodi SIRF ek hi assay type ke andar banti hai - Kd ko IC50 se jodna
    numbers ko mithe dikhata hai aur nateeja bekaar kar deta hai.
    """
    pairs = []
    no_wt = 0
    for (mol, tgt, typ, mu), m in M.items():
        w = W.get((mol, tgt, typ))
        if w is None:
            no_wt += 1
            continue
        pairs.append({
            "gene": genes.get(tgt) or tgt,
            "target_chembl_id": tgt,
            "mutation": mu,
            "mol_chembl_id": mol,
            "smiles": m["smiles"] or w["smiles"],
            "assay_type": typ,
            "pK_wt": round(w["pK"], 4),
            "pK_mut": round(m["pK"], 4),
            "ddG": round(m["pK"] - w["pK"], 4),
            "n_meas_wt": w["n_meas"],
            "n_meas_mut": m["n_meas"],
            "same_doc": bool(set(m["docs"]) & set(w["docs"])),
        })
    return pairs, no_wt


def main():
    probe()

    print("\n[1/5] mutation wale assays")
    mut_raw = harvest_mutants()
    mut, mut_drop = clean(mut_raw, True)
    print(f"  mutation rows: {len(mut_raw):,} -> saaf {len(mut):,}")
    for k, v in mut_drop.most_common(8):
        print(f"      hataya {v:>7,}  {k}")
    if not mut:
        raise RuntimeError("saaf karne ke baad ek bhi mutation row nahi bachi")

    tids = sorted({r["target"] for r in mut})
    print(f"\n[2/5] wild-type assays - {len(tids)} targets ke liye")
    wt_raw = harvest_wt(tids)
    wt, wt_drop = clean(wt_raw, False)
    print(f"  wild-type rows: {len(wt_raw):,} -> saaf {len(wt):,}")
    for k, v in wt_drop.most_common(8):
        print(f"      hataya {v:>7,}  {k}")

    print("\n[3/5] gene symbol")
    genes = gene_names(tids)

    print("\n[4/5] jodi banana")
    M = median_by(mut, ["mol", "target", "type", "mut"])
    W = median_by(wt, ["mol", "target", "type"])

    pairs, no_wt = make_pairs(M, W, genes)

    df = pd.DataFrame(pairs)
    if df.empty:
        print("!! ek bhi jodi nahi bani - aage kuch nahi kar sakte")
        (RES / "chembl_harvest.json").write_text(json.dumps(
            {"pairs": 0, "verdict": "KOI JODI NAHI"}, indent=2))
        return 1

    df = df.sort_values("ddG")
    df.to_csv(RES / "chembl_ddg.csv", index=False)

    same = df[df.same_doc]
    g = (df.groupby("gene")
           .agg(mutations=("mutation", "nunique"),
                drugs=("mol_chembl_id", "nunique"),
                pairs=("ddG", "size"),
                same_doc=("same_doc", "sum"))
           .sort_values("pairs", ascending=False))

    print(f"\n{'=' * 68}\nNATEEJA\n{'=' * 68}\n")
    print(f"  ddG jode                     : {len(df):,}")
    print(f"  ... jahan WT aur mutant ek hi paper se: {len(same):,}"
          f"  ({len(same) / len(df):.0%})")
    print(f"  alag genes                   : {df.gene.nunique()}")
    print(f"  alag mutations               : {df.mutation.nunique()}")
    print(f"  alag drugs                   : {df.mol_chembl_id.nunique()}")
    print(f"  jin mutant maapon ka WT nahi mila: {no_wt:,} (jodi nahi ban saki)")
    print(f"\n  tulna:")
    print(f"    Platinum (field ka dataset) : {PLATINUM:,} - genes: ~100+")
    print(f"    Davis (humara Phase 1)      : {DAVIS_CLEAN:,} - genes: 10")
    print(f"    ChEMBL (ye)                 : {len(df):,} - "
          f"genes: {df.gene.nunique()}")

    print(f"\n  assay type ke hisaab se:")
    for t, n in df.assay_type.value_counts().items():
        print(f"    {t:<6} {n:>7,}")

    print(f"\n  ddG ka faila: min {df.ddG.min():.2f}  max {df.ddG.max():.2f}  "
          f"median {df.ddG.median():.2f}  std {df.ddG.std():.2f}")
    for thr in (0.5, 1.0, 2.0):
        n = int((df.ddG.abs() >= thr).sum())
        print(f"    |ddG| >= {thr}: {n:,} ({n / len(df):.0%})")

    print(f"\n  top 25 genes:")
    print(g.head(25).to_string())

    print(f"\n  sabse bade asar:")
    cols = ["gene", "mutation", "assay_type", "pK_wt", "pK_mut", "ddG",
            "same_doc"]
    print(df.head(10)[cols].to_string(index=False))

    # Phase 2 ka gate: gene coverage. Yahi Phase 1 ki sabse badi kami thi.
    ng = int(df.gene.nunique())
    ok = ng >= 40 and len(df) >= DAVIS_CLEAN
    print(f"\n  >>> {'PHASE 2 GATE PAAS' if ok else 'GATE FAIL'} "
          f"(chahiye: >=40 genes aur >={DAVIS_CLEAN:,} jode; "
          f"mila: {ng} genes, {len(df):,} jode)")

    summary = {
        "pairs": len(df),
        "pairs_same_doc": len(same),
        "genes": ng,
        "mutations": int(df.mutation.nunique()),
        "drugs": int(df.mol_chembl_id.nunique()),
        "unpaired_mutant_measurements": no_wt,
        "by_assay_type": {k: int(v) for k, v in
                          df.assay_type.value_counts().items()},
        "ddG_abs_ge_1": int((df.ddG.abs() >= 1).sum()),
        "per_gene": json.loads(g.to_json(orient="index")),
        "compare": {"platinum": PLATINUM, "davis_clean": DAVIS_CLEAN},
        "gate_pass": bool(ok),
    }
    (RES / "chembl_harvest.json").write_text(json.dumps(summary, indent=2))

    md = [f"**ddG jode: {len(df):,}** ({df.gene.nunique()} genes, "
          f"{df.mutation.nunique()} mutations, "
          f"{df.mol_chembl_id.nunique()} drugs)", "",
          f"- ek hi paper se WT+mutant: {len(same):,}",
          f"- |ddG| >= 1.0: {int((df.ddG.abs() >= 1).sum()):,}",
          f"- Davis (Phase 1) se tulna: {DAVIS_CLEAN:,} jode / 10 genes", "",
          "Top genes:", "", "```", g.head(20).to_string(), "```"]
    (RES / "RESULT.md").write_text("\n".join(md))
    print(f"\n  files: {RES}/chembl_ddg.csv, chembl_harvest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
