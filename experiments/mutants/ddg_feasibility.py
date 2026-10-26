"""
Feasibility: Davis se kitna mutation-effect (ddG) data nikal sakta hai?

Kyun ye sawaal: field ka asli bottleneck mutation data hai, aur ye naapa hua hai.
2025 ka ek recent structure-based paper (Transfer learning across molecular
graphs, bioRxiv 2025.07.19.665665) mutation-effect prediction pe ye reporting
karta hai:
    Platinum dataset, 10-fold CV : Pearson r = 0.20
    TKI-ABL1 test (8 inhibitors) : Spearman 0.39
                                   (FEP+ 0.37, Prime 0.38, PremPLI 0.32)
Unka poora mutation dataset Platinum hai: n = 796.

Davis me 45 mutant-labelled kinases hain, aur unke wild-type bhi, SAME 68 drugs
ke against. To ddG seedha nikal sakta hai:
    ddG(drug, mutant) = pKd(drug, mutant) - pKd(drug, wild-type)
Ye data ab tak istemaal nahi hua kyunki Davis har mutant ko wild-type ki hubahu
SEQUENCE deta hai (humne ye naapa: embeddings bit-identical). Labels sahi the -
sirf input galat tha. Sequence theek karne ke baad ye ek asli ddG dataset hai.

Ye script sirf GINTI karta hai - koi model nahi. Agar ginti Platinum (796) se
kaafi badi nikli, to aage ka plan tikta hai; nahi nikli to plan badalna padega.
"""
import json, os, pickle, re, shutil, sys, time, urllib.request
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
WORK = os.path.join(HERE, "work")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
PLATINUM_N = 796          # tulna ke liye: field ka maujooda mutation dataset

BASES = ["https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/",
         "https://raw.githubusercontent.com/thinng/GraphDTA/master/data/davis/"]


def davis_file(name, min_bytes=1000):
    p = os.path.join(WORK, f"davis_{name}")
    if os.path.exists(p) and os.path.getsize(p) >= min_bytes:
        print(f"   cache: davis_{name}")
        return p
    errs = []
    for b in BASES:
        for _ in range(2):
            try:
                req = urllib.request.Request(b + name,
                                             headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=600) as r, open(p, "wb") as f:
                    shutil.copyfileobj(r, f)
                if os.path.getsize(p) >= min_bytes:
                    print(f"   downloaded davis_{name}")
                    return p
            except Exception as e:
                errs.append(f"{type(e).__name__}: {e}")
            time.sleep(3)
    raise RuntimeError(f"davis/{name} nahi mila: {errs}")


print("=" * 70)
print("Davis se ddG (mutation effect) data ki ginti")
print("=" * 70)

lig = json.load(open(davis_file("ligands_can.txt")), object_pairs_hook=OrderedDict)
pro = json.load(open(davis_file("proteins.txt")), object_pairs_hook=OrderedDict)
Y = np.asarray(pickle.load(open(davis_file("Y", 10000), "rb"), encoding="latin1"),
               dtype=float)
Y = -np.log10(Y / 1e9)                        # Kd (nM) -> pKd
NAMES = list(pro)
SMILES = list(lig.values())
print(f"\nDavis: {len(SMILES)} drugs x {len(NAMES)} proteins · "
      f"{int((~np.isnan(Y)).sum()):,} measured pairs")

MUT_RE = re.compile(r"\(([A-Z]\d+[A-Z][^)]*|[A-Z]\d+[A-Z]?\d*del[^)]*)\)")


def base_of(n):
    return n.split("(")[0].replace("-nonphosphorylated", "") \
            .replace("-phosphorylated", "").strip()


def is_mutant(n):
    return bool(MUT_RE.search(n))


# wild-type kaun: bracket nahi, ya sirf phospho/domain label
groups = {}
for i, n in enumerate(NAMES):
    groups.setdefault(base_of(n), []).append(i)

rows = []
per_gene = []
for gene, idx in groups.items():
    wt = [i for i in idx if not is_mutant(NAMES[i])]
    mu = [i for i in idx if is_mutant(NAMES[i])]
    if not wt or not mu:
        continue
    # wild-type reference: sabse chhota naam (bina suffix wala)
    w = sorted(wt, key=lambda i: len(NAMES[i]))[0]
    n_pairs = 0
    for m in mu:
        both = ~np.isnan(Y[:, w]) & ~np.isnan(Y[:, m])
        for d in np.where(both)[0]:
            rows.append({"gene": gene, "wt": NAMES[w], "mutant": NAMES[m],
                         "drug_idx": int(d), "smiles": SMILES[d],
                         "pKd_wt": round(float(Y[d, w]), 4),
                         "pKd_mut": round(float(Y[d, m]), 4),
                         "ddG": round(float(Y[d, m] - Y[d, w]), 4)})
            n_pairs += 1
    per_gene.append({"gene": gene, "n_mutants": len(mu), "wt_used": NAMES[w],
                     "ddG_pairs": n_pairs})

df = pd.DataFrame(rows)
pg = pd.DataFrame(per_gene).sort_values("ddG_pairs", ascending=False)

print(f"\n{'='*70}\nNATEEJA\n{'='*70}")
print(f"\n  ddG jode (drug, wild-type, mutant): {len(df):,}")
print(f"  alag mutants                       : {df.mutant.nunique() if len(df) else 0}")
print(f"  alag genes                         : {df.gene.nunique() if len(df) else 0}")
print(f"  alag drugs                         : {df.drug_idx.nunique() if len(df) else 0}")
print(f"\n  tulna - Platinum (field ka maujooda mutation dataset): {PLATINUM_N}")
if len(df):
    print(f"  Davis se mila                                        : {len(df):,} "
          f"({len(df)/PLATINUM_N:.1f}x)")

print("\n  Gene ke hisaab se:")
print(pg.to_string(index=False))

if len(df):
    print(f"\n  ddG ka faila: min {df.ddG.min():.2f} max {df.ddG.max():.2f} "
          f"median {df.ddG.median():.2f} · std {df.ddG.std():.2f}")
    big = int((df.ddG.abs() >= 1.0).sum())
    print(f"  |ddG| >= 1.0 (10x affinity farak): {big:,} jode ({100*big/len(df):.0f}%)")
    print(f"  |ddG| >= 0.5                      : {int((df.ddG.abs()>=0.5).sum()):,}")
    print(f"  |ddG| ~ 0 (farak nahi)            : {int((df.ddG.abs()<0.2).sum()):,}")
    print("\n  (agar zyadatar ddG ~0 hai to signal kam hai - ye zaroori check hai)")

    print("\n  Sabse bade effects (resistance jaise):")
    top = df.reindex(df.ddG.abs().sort_values(ascending=False).index).head(10)
    print(top[["gene", "mutant", "pKd_wt", "pKd_mut", "ddG"]].to_string(index=False))

    df.to_csv(os.path.join(OUT, "davis_ddg.csv"), index=False)
    pg.to_csv(os.path.join(OUT, "davis_ddg_per_gene.csv"), index=False)

verdict = ("PLAN TIKTA HAI" if len(df) >= 3 * PLATINUM_N else
           "PLAN KAMZOR - data Platinum se kaafi bada nahi")
print(f"\n  >>> {verdict}")

json.dump({"ddg_pairs": int(len(df)),
           "distinct_mutants": int(df.mutant.nunique()) if len(df) else 0,
           "distinct_genes": int(df.gene.nunique()) if len(df) else 0,
           "distinct_drugs": int(df.drug_idx.nunique()) if len(df) else 0,
           "platinum_n": PLATINUM_N,
           "ratio_vs_platinum": round(len(df) / PLATINUM_N, 2) if len(df) else 0,
           "abs_ddg_ge_1": int((df.ddG.abs() >= 1.0).sum()) if len(df) else 0,
           "abs_ddg_lt_02": int((df.ddG.abs() < 0.2).sum()) if len(df) else 0,
           "per_gene": per_gene, "verdict": verdict},
          open(os.path.join(OUT, "ddg_feasibility.json"), "w"), indent=2)
print(f"\n  files: {OUT}/davis_ddg.csv, ddg_feasibility.json")
