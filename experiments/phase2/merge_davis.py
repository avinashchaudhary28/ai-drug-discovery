"""Phase 2, kaam #2b: Davis aur ChEMBL ke ddG ko ek dataset me jodna.

Do kaam, aur doosra zyada keemti hai:

1. Jodna: dono source ek shakal me, dohrai hataakar.
2. **Milaan**: jahan dono source ne EK HI (gene, mutation, drug) naapa
   hai, wahan dono ke ddG ki tulna. Ye audit hai - aur yahi wo cheez hai
   jo koi apne tool ke baare me chhapta nahi. Agar do swatantra source ek
   hi jagah alag jawab dete hain, to kam se kam ek galat hai, aur model
   ko dono khilana usse jhooth sikhana hai.

Davis ki ek baat yaad rakhni hai: wo pKd ko 5.0 pe kaat deta hai. 5.0 ka
matlab "isse kamzor, kitna pata nahi" - maap nahi. Un jodon ko number ki
tarah nahi, nishaan (censored) ke saath rakha jata hai.

Nateeja: results/mutation_ddg_merged.csv, results/merge_report.json
"""
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutnorm import drug_key, normalize                  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
DAVIS = HERE.parent / "mutants" / "results" / "davis_ddg.csv"
DAVIS_FLOOR = 5.0            # Davis is se neeche naapta nahi


def load_chembl():
    p = RES / "chembl_ddg_classified.csv"
    if not p.exists():
        p = RES / "chembl_ddg.csv"
        print(f"  !! classified file nahi mili, {p.name} se chala rahe hain - "
              f"kinase ka batwara nahi hoga")
    df = pd.read_csv(p)
    if "is_kinase" not in df:
        df["is_kinase"] = None
    out = pd.DataFrame({
        "source": "chembl",
        "gene": df.gene,
        "mutation_raw": df.mutation,
        "drug_id": df.mol_chembl_id,
        "smiles": df.smiles,
        "assay_type": df.assay_type,
        "pK_wt": df.pK_wt,
        "pK_mut": df.pK_mut,
        "ddG": df.ddG,
        "same_doc": df.same_doc,
        "is_kinase": df.is_kinase,
        # ChEMBL me humne relation '=' hi liya tha, isliye koi chhat nahi
        "censored_wt": False,
        "censored_mut": False,
    })
    return out


def load_davis():
    if not DAVIS.exists():
        raise RuntimeError(f"{DAVIS} nahi mila")
    df = pd.read_csv(DAVIS)
    return pd.DataFrame({
        "source": "davis",
        "gene": df.gene,
        "mutation_raw": df.mutant,
        "drug_id": "davis_" + df.drug_idx.astype(str),
        "smiles": df.smiles,
        "assay_type": "Kd",                  # Davis poora Kd hai
        "pK_wt": df.pKd_wt,
        "pK_mut": df.pKd_mut,
        "ddG": df.ddG,
        "same_doc": True,                    # ek hi source, ek hi protocol
        "is_kinase": True,                   # Davis ka poora panel kinase hai
        "censored_wt": np.isclose(df.pKd_wt, DAVIS_FLOOR),
        "censored_mut": np.isclose(df.pKd_mut, DAVIS_FLOOR),
    })


def main():
    print("[1/4] dono source padho")
    ch, dv = load_chembl(), load_davis()
    print(f"  chembl: {len(ch):,} jode  |  davis: {len(dv):,} jode")

    df = pd.concat([ch, dv], ignore_index=True)

    print("\n[2/4] mutation aur drug ko ek shakal do")
    norm = df.mutation_raw.map(normalize)
    df["mutation"] = ["+".join(t) for t, _, _ in norm]
    df["mut_note"] = [n for _, n, _ in norm]
    df["mut_parsed"] = [p for _, _, p in norm]

    bad = df[~df.mut_parsed]
    print(f"  mutation parse: {int(df.mut_parsed.sum()):,} theek, "
          f"{len(bad):,} samajh nahi aaye")
    if len(bad):
        ex = bad.mutation_raw.value_counts().head(10)
        print("    (inhe 'RAW:' nishaan mila - ye kisi doosri entry se "
              "nahi judenge)")
        for k, v in ex.items():
            print(f"      {v:>5,}  {k}")

    uniq = df.smiles.dropna().unique()
    print(f"  drug key: {len(uniq):,} alag SMILES -> InChIKey skeleton")
    keys = {s: drug_key(s) for s in uniq}
    nk = sum(1 for v in keys.values() if v is None)
    if nk:
        print(f"    !! {nk} SMILES se key nahi bani - wo jode gire jayenge")
    df["drug_key"] = df.smiles.map(keys)
    before = len(df)
    df = df[df.drug_key.notna()].copy()
    if len(df) != before:
        print(f"    {before - len(df)} jode gire (drug key nahi bani)")

    # Dohrai ka key. mut_note isme jaan-boojh kar hai: Davis ka
    # "ABL1(T315I)p" phosphorylated protein hai, "ABL1(T315I)" nahi -
    # dono ek maan lena do alag cheezon ko mila dena hota.
    KEY = ["gene", "mutation", "mut_note", "drug_key", "assay_type"]

    print("\n[3/4] dono source ka milaan (yahi asli audit hai)")
    piv = (df.groupby(KEY + ["source"]).ddG.median().unstack("source"))
    both = piv.dropna(subset=[c for c in ("chembl", "davis") if c in piv])
    both = both[both.index.map(lambda _: True)]
    if "chembl" in piv and "davis" in piv:
        both = piv[piv.chembl.notna() & piv.davis.notna()]
    else:
        both = piv.iloc[0:0]

    overlap = {}
    if len(both):
        d = (both.chembl - both.davis).abs()
        overlap = {
            "n": int(len(both)),
            "pearson": float(both.chembl.corr(both.davis)),
            "spearman": float(both.chembl.corr(both.davis, method="spearman")),
            "mean_abs_diff": float(d.mean()),
            "median_abs_diff": float(d.median()),
            "disagree_gt_1": int((d > 1.0).sum()),
            "disagree_gt_2": int((d > 2.0).sum()),
        }
        print(f"  dono me maujood: {overlap['n']:,} (gene, mutation, drug, "
              f"assay) jode")
        print(f"    Pearson  : {overlap['pearson']:.3f}")
        print(f"    Spearman : {overlap['spearman']:.3f}")
        print(f"    |farak|  : mean {overlap['mean_abs_diff']:.2f}, "
              f"median {overlap['median_abs_diff']:.2f}")
        print(f"    1.0 se zyada farak: {overlap['disagree_gt_1']:,} "
              f"({overlap['disagree_gt_1'] / len(both):.0%})")
        print(f"    2.0 se zyada farak: {overlap['disagree_gt_2']:,}")
        worst = (both.assign(diff=(both.chembl - both.davis))
                     .reindex(d.sort_values(ascending=False).index).head(10))
        print("\n    sabse bada matbhed:")
        print(worst.to_string())
    else:
        print("  dono source me ek bhi saajha jodi nahi mili - "
              "yaani koi tulna nahi ho sakti")

    print("\n[4/4] final dataset")
    # Overlap pe dono ke number ka ausat NAHI lenge. Do labs ke numbers
    # ghata/jod dena wahi galti hai jisse Davis me gadbad hui thi.
    # ChEMBL ko rakhte hain (relation '=' saaf hai, 90% ek hi paper se)
    # aur Davis ka number alag column me audit ke liye.
    df["pref"] = (df.source == "chembl").astype(int)
    df = df.sort_values(["pref", "same_doc"], ascending=False)
    merged = df.drop_duplicates(subset=KEY, keep="first").copy()
    merged["in_both_sources"] = merged.set_index(KEY).index.isin(both.index)
    merged = merged.drop(columns=["pref"])

    print(f"  {len(df):,} jode -> {len(merged):,} (dohrai hatane ke baad)")
    print(f"  source: " + ", ".join(
        f"{k} {v:,}" for k, v in merged.source.value_counts().items()))
    print(f"  genes {merged.gene.nunique()}  |  "
          f"mutations {merged.mutation.nunique()}  |  "
          f"drugs {merged.drug_key.nunique()}")

    cen = merged[merged.censored_wt | merged.censored_mut]
    print(f"  censored (Davis ki 5.0 chhat): {len(cen):,} - ye number ki "
          f"tarah nahi, nishaan ke saath rakhe hain")
    clean = merged[~(merged.censored_wt | merged.censored_mut)]
    big = clean[clean.ddG.abs() >= 1.0]
    print(f"  saaf jode: {len(clean):,}  |  |ddG|>=1.0: {len(big):,} "
          f"({big.gene.nunique()} genes)")

    if merged.is_kinase.notna().any():
        k = merged[merged.is_kinase == True]              # noqa: E712
        print(f"  kinase: {len(k):,} jode, {k.gene.nunique()} genes  |  "
              f"baki: {len(merged) - len(k):,}")

    cols = ["source", "gene", "mutation", "mut_note", "mutation_raw",
            "drug_key", "drug_id", "smiles", "assay_type", "pK_wt", "pK_mut",
            "ddG", "censored_wt", "censored_mut", "same_doc", "is_kinase",
            "mut_parsed", "in_both_sources"]
    merged[cols].sort_values("ddG").to_csv(
        RES / "mutation_ddg_merged.csv", index=False)

    report = {
        "chembl_pairs": int(len(ch)),
        "davis_pairs": int(len(dv)),
        "merged_pairs": int(len(merged)),
        "genes": int(merged.gene.nunique()),
        "mutations": int(merged.mutation.nunique()),
        "drugs": int(merged.drug_key.nunique()),
        "mutation_unparsed": int((~df.mut_parsed).sum()),
        "censored_pairs": int(len(cen)),
        "clean_pairs": int(len(clean)),
        "clean_ddG_ge_1": int(len(big)),
        "overlap": overlap,
    }
    if merged.is_kinase.notna().any():
        k = merged[merged.is_kinase == True]              # noqa: E712
        report["kinase_pairs"] = int(len(k))
        report["kinase_genes"] = int(k.gene.nunique())
    (RES / "merge_report.json").write_text(json.dumps(report, indent=2))

    md = [f"**Merged: {len(merged):,} ddG jode** "
          f"({merged.gene.nunique()} genes, {merged.mutation.nunique()} "
          f"mutations, {merged.drug_key.nunique()} drugs)", "",
          f"- chembl {len(ch):,} + davis {len(dv):,} -> {len(merged):,}",
          f"- saaf (censored nahi): {len(clean):,}; |ddG|>=1.0: {len(big):,}"]
    if overlap:
        md.append(f"- dono source me saajha: {overlap['n']:,} jode, "
                  f"Pearson {overlap['pearson']:.3f}, "
                  f"|farak| median {overlap['median_abs_diff']:.2f}")
    (RES / "RESULT.md").write_text("\n".join(md))
    print(f"\n  files: {RES}/mutation_ddg_merged.csv, merge_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
