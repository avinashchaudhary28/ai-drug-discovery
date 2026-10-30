"""Har gene ke liye KLIFS mapping banao aur JAANCHO.

Jaan-boojh kar yahan sirf mapping banti hai, feature nahi. Pehle ye pata
hona chahiye ki kitne genes ki numbering sach me milti hai. Agar bahut
kam genes verify huye to poora plan badalta hai, aur wo baat features
banane se PEHLE pata honi chahiye - baad me nahi, jab nateeja aa chuka ho
aur usko bachane ka mann kare.

Nateeja: results/klifs_map.json, results/klifs_verify.csv
"""
import json
import os
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from klifs_map import build_map, fetch_gene, parse_mut, verify  # noqa: E402
from logo import eligible_genes                           # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
SRC = HERE.parent / "phase2" / "results" / "mutation_ddg_merged.csv"
MIN_PAIRS = int(os.environ.get("MIN_PAIRS", "20"))


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    df = pd.read_csv(SRC)
    df = df[(df.is_kinase == True)                         # noqa: E712
            & ~(df.censored_wt | df.censored_mut)
            & (df.mut_parsed == True)]                     # noqa: E712
    genes = sorted(df.gene.unique())
    folds = set(eligible_genes(df, MIN_PAIRS))
    print(f"  {len(df):,} jode, {len(genes)} genes "
          f"({len(folds)} me {MIN_PAIRS}+ jode)")

    rows, maps = [], {}
    for i, gene in enumerate(genes, 1):
        info = fetch_gene(gene)
        if info.get("error"):
            rows.append({"gene": gene, "usable": False,
                         "reason": info["error"], "is_fold": gene in folds,
                         "n_pairs": int((df.gene == gene).sum())})
            print(f"  {gene:<10} ✗ {info['error']}")
            continue
        resmap = build_map(info)
        muts = df[df.gene == gene].mutation.unique().tolist()
        v = verify(gene, resmap, muts)
        v["is_fold"] = gene in folds
        v["n_pairs"] = int((df.gene == gene).sum())
        v["pdb"] = info.get("pdb")
        v["uniprot"] = info.get("uniprot")
        v["pocket_len"] = len(info.get("structure_pocket") or "")
        rows.append(v)
        if v["usable"]:
            maps[gene] = {"offset": v["offset"], "pdb": info.get("pdb"),
                          "structure_ID": info.get("structure_ID"),
                          "map": {str(k): list(val)
                                  for k, val in resmap.items()}}
        mark = "✓" if v["usable"] else "✗"
        print(f"  {gene:<10} {mark} offset {v.get('offset', 0):+3d}  "
              f"pocket me {v.get('n_in_pocket', 0):>3}/{v.get('n_mutations', 0):<3}  "
              f"wt match {v.get('match_rate', 0):.0%}  "
              f"({v.get('reason')})", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(RES / "klifs_verify.csv", index=False)
    (RES / "klifs_map.json").write_text(json.dumps(maps))

    ok = r[r.usable == True]                               # noqa: E712
    fold_ok = ok[ok.is_fold == True]                       # noqa: E712
    folds_df = r[r.is_fold == True]                        # noqa: E712

    print(f"\n{'=' * 70}\nNATEEJA\n{'=' * 70}\n")
    print(f"  genes dekhe        : {len(r)}")
    print(f"  mapping bani aur jaanch paas: {len(ok)}")
    print(f"  LOGO ke folds ({len(folds_df)}) me se paas: {len(fold_ok)}")
    if len(ok):
        print(f"\n  offset 0 wale (numbering seedha milti hai): "
              f"{int((ok.offset == 0).sum())}/{len(ok)}")
        nz = ok[ok.offset != 0]
        if len(nz):
            print(f"  jinme offset laga (Phase 1 jaisa farak):")
            for _, x in nz.iterrows():
                print(f"      {x.gene:<10} offset {int(x.offset):+4d}  "
                      f"wt match {x.match_rate:.0%}  ({x.pdb})")
        tot_m = int(ok.n_mutations.sum())
        tot_p = int(ok.n_in_pocket.sum())
        print(f"\n  in genes ke mutations: {tot_m:,}, "
              f"pocket ke andar {tot_p:,} ({tot_p / max(tot_m, 1):.0%})")

    bad = r[r.usable == False]                             # noqa: E712
    if len(bad):
        print(f"\n  jo istemaal NAHI honge ({len(bad)}):")
        for _, x in bad.sort_values("n_pairs", ascending=False).head(15).iterrows():
            print(f"      {x.gene:<10} {int(x.n_pairs):>5,} jode  "
                  f"{'(LOGO fold)' if x.is_fold else ''}  {x.reason}")

    pairs_ok = int(df[df.gene.isin(ok.gene)].shape[0]) if len(ok) else 0
    print(f"\n  jode jo aage ja sakte hain: {pairs_ok:,} / {len(df):,} "
          f"({pairs_ok / max(len(df), 1):.0%})")

    rep = {"genes": len(r), "usable": len(ok),
           "fold_genes": len(folds_df), "fold_usable": len(fold_ok),
           "pairs_total": int(len(df)), "pairs_usable": pairs_ok,
           "offset_zero": int((ok.offset == 0).sum()) if len(ok) else 0,
           "per_gene": json.loads(r.to_json(orient="records"))}
    (RES / "klifs_report.json").write_text(json.dumps(rep, indent=2))

    print(f"\n{'=' * 70}\nFAISLA\n{'=' * 70}")
    if len(fold_ok) >= 10 and pairs_ok >= 0.5 * len(df):
        print(f"  Aage badh sakte hain: {len(fold_ok)} folds aur "
              f"{pairs_ok:,} jode. Ab pocket position ko feature banakar "
              f"dekhna hai ki farsh toota ya nahi.")
    else:
        print(f"  RUKO: sirf {len(fold_ok)} folds aur {pairs_ok:,} jode "
              f"bache. Itne pe LOGO ka nateeja bharosemand nahi hoga - "
              f"pehle mapping theek karni padegi, feature banane se "
              f"pehle.")
    print(f"\n  files: {RES}/klifs_map.json, klifs_verify.csv ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
