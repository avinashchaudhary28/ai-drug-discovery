"""
Kinases "badhane" ki koshish — bina retraining.

Diagnosis: model ke top-50 me wahi kuch molecules 442 me se 250-290 targets pe
baithe hain. Ye ek jaani-pehchani bimari hai: model ne DAWA KA APNA AVERAGE
seekh liya, jodi ka interaction nahi. Davis me kuch dawaein bahut saare kinases
se chipakti hain. MSE ghatane ka sabse aasan raasta yahi hai - "ye dawa aam
taur pe strong hai", "ye target aam taur pe weak hai" - aur jodi ka asli farak
seekhne ki zaroorat nahi padti.

Agar ye sach hai to kuch saadhe transforms isse theek kar dene chahiye, bina
model ko chhede. Yahan unki poori battery chalti hai aur ek hi paimane pe
naapi jaati hai:

  kitne checkable targets pe koi known dawa top-N me aati hai

Paimana yahi rakha hai kyunki ranking (CI) nahi, "kaam ki list" yahi banati hai.
Sab kuch poore score matrix se offline chalta hai - koi Actions minute nahi lagta.
"""
import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO, "data")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

TOPS = (5, 10, 20, 50)


def gene_of(n):
    return (str(n).split("(")[0].replace("-nonphosphorylated", "")
            .replace("-phosphorylated", "").strip())


def load():
    z = np.load(os.path.join(DATA, "score_matrix.npz"), allow_pickle=True)
    known = json.load(open(os.path.join(DATA, "known_drugs.json")))
    S = np.asarray(z["scores"], dtype=np.float32)       # (drugs, targets)
    targets = [str(t) for t in z["targets"]]
    allids = [str(a).split("|") for a in z["all_ids"]]
    mw = np.asarray(z["mw"], dtype=np.float32)
    smiles = [str(x) for x in z["smiles"]]
    names = [str(x) for x in z["drug_names"]]
    resolved = {g for g, v in known.items() if v}
    # har target ke liye: kaun-kaun drug row known hai
    known_mask = np.zeros(S.shape, dtype=bool)
    checkable = np.zeros(len(targets), dtype=bool)
    for j, t in enumerate(targets):
        g = gene_of(t)
        if g not in resolved:
            continue
        checkable[j] = True
        kn = set(known[g])
        for i, ids in enumerate(allids):
            if kn & set(ids):
                known_mask[i, j] = True
    return S, targets, mw, smiles, names, known_mask, checkable


def recall(S, known_mask, checkable, valid=None, tops=TOPS):
    """valid: (drugs,) bool mask — jo molecules list me aa sakte hain."""
    res = {}
    n_chk = int(checkable.sum())
    idx = np.where(checkable)[0]
    for N in tops:
        hit = 0
        for j in idx:
            col = S[:, j].copy()
            if valid is not None:
                col = np.where(valid, col, -np.inf)
            k = min(N, int(np.isfinite(col).sum()))
            if k <= 0:
                continue
            top = np.argpartition(-col, k - 1)[:k]
            if known_mask[top, j].any():
                hit += 1
        res[f"top{N}"] = f"{hit}/{n_chk}"
        res[f"top{N}_n"] = hit
    return res


def main():
    S, targets, mw, smiles, names, known_mask, checkable = load()
    print(f"matrix {S.shape} · checkable targets {int(checkable.sum())} · "
          f"known jode {int(known_mask.sum()):,}\n", flush=True)

    # ---- transforms (ranking badalte hain, model nahi) ----
    drug_mean = S.mean(1, keepdims=True)
    drug_std = S.std(1, keepdims=True) + 1e-6
    tgt_mean = S.mean(0, keepdims=True)
    tgt_std = S.std(0, keepdims=True) + 1e-6

    TRANSFORMS = {
        "raw (jaisa hai)": S,
        "dawa ka mean ghatao": S - drug_mean,
        "dawa ka z-score": (S - drug_mean) / drug_std,
        "target ka mean ghatao": S - tgt_mean,
        "target ka z-score": (S - tgt_mean) / tgt_std,
        "dono mean ghatao": S - drug_mean - tgt_mean + S.mean(),
        "dawa z + target z": ((S - drug_mean) / drug_std
                              + (S - tgt_mean) / tgt_std) / 2,
    }

    # ---- molecule filters ----
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, Crippen, Lipinski
    RDLogger.DisableLog("rdApp.*")

    cache = os.path.join(HERE, "work", "filters.npz")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache):
        f = np.load(cache)
        druglike, lipinski, nopains = f["druglike"], f["lipinski"], f["nopains"]
        print("filters cache se load\n", flush=True)
    else:
        from rdkit.Chem import FilterCatalog
        pp = FilterCatalog.FilterCatalogParams()
        pp.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        cat = FilterCatalog.FilterCatalog(pp)
        druglike = (mw >= 250) & (mw <= 700)
        lipinski = np.zeros(len(smiles), bool)
        nopains = np.zeros(len(smiles), bool)
        for i, s in enumerate(smiles):
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            try:
                ok = (Descriptors.MolWt(m) <= 700 and Crippen.MolLogP(m) <= 5.5
                      and Lipinski.NumHDonors(m) <= 5
                      and Lipinski.NumHAcceptors(m) <= 10
                      and Lipinski.NumRotatableBonds(m) <= 12
                      and m.GetRingInfo().NumRings() >= 1)
                lipinski[i] = ok
                nopains[i] = not cat.HasMatch(m)
            except Exception:
                pass
            if (i + 1) % 2000 == 0:
                print(f"  filters {i+1:,}/{len(smiles):,}", flush=True)
        np.savez_compressed(cache, druglike=druglike, lipinski=lipinski,
                            nopains=nopains)

    FILTERS = {
        "koi nahi": None,
        "MW 250-700": druglike,
        "Lipinski-ish": lipinski,
        "Lipinski + no PAINS": lipinski & nopains,
    }
    for k, v in FILTERS.items():
        if v is not None:
            print(f"  filter '{k}': {int(v.sum()):,}/{len(mw):,} molecules bachte hain")
    print(flush=True)

    rows = []
    for tname, M in TRANSFORMS.items():
        for fname, fmask in FILTERS.items():
            r = recall(M, known_mask, checkable, fmask)
            rows.append({"transform": tname, "filter": fname, **r})
            print(f"  {tname:24s} | {fname:20s} "
                  + "  ".join(f"top{N}={r[f'top{N}']}" for N in TOPS), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "corrections.csv"), index=False)

    base = df[(df.transform == "raw (jaisa hai)") & (df["filter"] == "koi nahi")].iloc[0]
    best = df.sort_values(["top5_n", "top10_n"], ascending=False).iloc[0]
    print("\n" + "=" * 70)
    print(f"  shuruaat : {base.transform} + {base['filter']:20s} top5={base.top5}")
    print(f"  sabse achha: {best.transform} + {best['filter']:20s} top5={best.top5}")
    gain = int(best.top5_n) - int(base.top5_n)
    print(f"  faida    : +{gain} targets ({base.top5_n} -> {best.top5_n}), "
          f"bina model ko chhede")
    print("=" * 70, flush=True)

    piv = df.pivot(index="transform", columns="filter", values="top5_n")
    with open(os.path.join(OUT, "CORRECTIONS.md"), "w") as f:
        f.write("# Bina retraining kitne kinases 'badhaye' ja sakte hain?\n\n")
        f.write("Paimana: **kitne checkable targets (146) pe koi jaani-pehchani "
                "dawa top-5 me aati hai.** Model, features, training — kuch nahi "
                "badla; sirf ranking ka transform aur molecule filter badla.\n\n")
        f.write("## top-5 recall (jitna zyada utna behtar)\n\n")
        f.write(piv.to_markdown())
        f.write("\n\n## Poora table\n\n")
        f.write(df[["transform", "filter"] + [f"top{N}" for N in TOPS]]
                .to_markdown(index=False))
        f.write(f"\n\n**Shuruaat:** {base.top5} · **Sabse achha:** {best.top5} "
                f"({best.transform} + {best['filter']})\n")
    json.dump({"baseline_top5": base.top5, "best_top5": best.top5,
               "best_transform": best.transform, "best_filter": best["filter"],
               "gain_targets": gain, "n_checkable": int(checkable.sum())},
              open(os.path.join(OUT, "corrections_summary.json"), "w"), indent=2)
    print(f"\nFiles: {OUT}/CORRECTIONS.md, corrections.csv")


if __name__ == "__main__":
    main()
