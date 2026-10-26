"""
Jeetne wali setting se final table banata hai — score matrix se, offline.

Setting: **dawa ka mean ghatao + MW 250–700 ka filter**

Kyun yahi (aur sirf top-5 ka aankda dekh ke nahi chuna):

  bar      raw   MW akela   MW + dawa ka mean ghata
  top-5     23      45              43
  top-10    24      46              52
  top-20    24      49              56
  top-50    41      58              59

top-5 pe MW akela 2 target aage hai, par top-10/20/50 pe combined saaf jeetta
hai. Aur aankh se dekhne pe combined bahut behtar nikla:
  ABL1  : Dasatinib #3 -> **#1**
  MET   : Crizotinib top-8 me aaya (MW akele me nahi tha)
  ALK   : Conteltinib top-8 me aaya (MW akele me kuch known nahi tha)
Isliye combined ship kar rahe hain.

Ek zaroori baat jo display pe asar daalti hai: mean ghatane ke baad score ab
**pKd nahi** rehta - wo "ye dawa is target pe apne aam level se kitna upar hai"
ban jata hai. Isliye table me DONO rakhte hain: asli pKd (padhne ke liye) aur
adj (ranking ke liye). App me saaf likha hota hai ki kis se sort hua hai.
"""
import json, os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO, "data")
TOP_N = int(os.environ.get("TOP_N", 50))
MW_LO, MW_HI = 250.0, 700.0


def gene_of(n):
    return (str(n).split("(")[0].replace("-nonphosphorylated", "")
            .replace("-phosphorylated", "").strip())


z = np.load(os.path.join(DATA, "score_matrix.npz"), allow_pickle=True)
known = json.load(open(os.path.join(DATA, "known_drugs.json")))
S = np.asarray(z["scores"], dtype=np.float32)
targets = [str(t) for t in z["targets"]]
allids = [str(a).split("|") for a in z["all_ids"]]
mw = np.asarray(z["mw"], dtype=np.float32)
smiles = [str(x) for x in z["smiles"]]
dnames = [str(x) for x in z["drug_names"]]
dids = [str(x) for x in z["drug_ids"]]
phase = np.asarray(z["max_phase"], dtype=np.float32)

ADJ = S - S.mean(1, keepdims=True)          # promiscuity correction
keep = (mw >= MW_LO) & (mw <= MW_HI)
print(f"matrix {S.shape} · MW {MW_LO:.0f}-{MW_HI:.0f} filter: "
      f"{int(keep.sum()):,}/{len(mw):,} molecules")

rows = []
for j, t in enumerate(targets):
    kn = set(known.get(gene_of(t), []))
    col = np.where(keep, ADJ[:, j], -np.inf)
    order = np.argsort(-col)[:TOP_N]
    for rank, i in enumerate(order, 1):
        rows.append({
            "target": t, "rank": rank,
            "drug": dnames[i] or dids[i],
            "chembl_id": dids[i],
            "max_phase": phase[i],
            "pKd": round(float(S[i, j]), 3),
            "adj": round(float(ADJ[i, j]), 3),
            "mw": round(float(mw[i]), 1),
            "known_for_target": bool(kn & set(allids[i])),
            "smiles": smiles[i],
        })

tbl = pd.DataFrame(rows)
out = os.path.join(DATA, "repurposing_top_corrected.csv")
tbl.to_csv(out, index=False)
print(f"{len(tbl):,} rows -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")

resolved = {g for g, v in known.items() if v}
sub = tbl[tbl.target.map(gene_of).isin(resolved)]
per = sub.groupby("target").known_for_target.sum()
print(f"\ncheckable targets {per.size} · koi known top-{TOP_N} me: {int((per>0).sum())}")
for N in (5, 10, 20, 50):
    s2 = sub[sub["rank"] <= N].groupby("target").known_for_target.sum()
    print(f"  top-{N:<3}: {int((s2>0).sum())}/{per.size}")
print(f"\nmedian MW {tbl.mw.median():.0f} Da · rank-1 known {int(tbl[tbl['rank']==1].known_for_target.sum())}/{len(targets)}")

meta_p = os.path.join(DATA, "repurposing_meta.json")
meta = json.load(open(meta_p))
meta["corrected"] = {
    "transform": "dawa ka mean ghatao (promiscuity correction)",
    "filter": f"MW {MW_LO:.0f}-{MW_HI:.0f} Da",
    "top5": f"{int((sub[sub['rank']<=5].groupby('target').known_for_target.sum()>0).sum())}/{per.size}",
    "top10": f"{int((sub[sub['rank']<=10].groupby('target').known_for_target.sum()>0).sum())}/{per.size}",
    "baseline_top5": "23/134",
    "note": ("adj = pKd minus us dawa ka saare targets pe average. Ranking isi se "
             "hoti hai. pKd column asli value hai, padhne ke liye."),
}
json.dump(meta, open(meta_p, "w"), indent=2)
print("meta update ✅")
