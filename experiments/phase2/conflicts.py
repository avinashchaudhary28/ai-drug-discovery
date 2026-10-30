"""Dono source ke matbhed dhundhne ka ek hi niyam, ek hi jagah.

Ye module jaan-boojh kar alag hai. Yahi niyam merge_davis.py aur
resolve_conflicts.py dono istemaal karte hain. Agar do jagah do copy
hoti, to ek din ek badal jati aur doosri nahi - aur donon alag jawab
dene lagte, bina kisi error ke. Wahi chup-chaap wali galti hai jo is
project me bar bar nikli hai.
"""
import pandas as pd

# Matbhed ko "ek jaisa khisakna" kab maana jaye
MIN_N = 3          # kam se kam itne mutations dono source me hon
MAX_SPREAD = 0.1   # farak ka bikherav isse kam ho (yaani ek jaisa farak)
MIN_OFFSET = 0.5   # aur farak itna bada ho ki matlab rakhta ho


def overlap_table(df):
    """-> wo (gene, mutation, mut_note, drug_key, assay_type) jinke
    liye DONO source me maap hai, dono ke ddG saath me."""
    key = ["gene", "mutation", "mut_note", "drug_key", "assay_type"]
    piv = df.groupby(key + ["source"]).ddG.median().unstack("source")
    if "chembl" not in piv or "davis" not in piv:
        return piv.iloc[0:0]
    return piv[piv.chembl.notna() & piv.davis.notna()]


def const_groups(both, min_n=MIN_N, max_spread=MAX_SPREAD,
                 min_off=MIN_OFFSET):
    """Wo (gene, drug, assay) jinme farak HAR mutation pe ek jaisa hai.

    ddG = pK_mut - pK_wt. Isliye agar ek hi (gene, drug) ka WILD-TYPE
    number dono source me alag hai, to us drug ke saare mutations ka ddG
    bilkul utna hi khisak jata hai - ek jaisa. Ulta bhi sach hai: ek
    jaisa farak dikhe to jad mutant me nahi, wild-type me hai.

    -> (saare multi-mutation groups, sirf ek-jaise-farak wale groups)
    """
    if both is None or len(both) == 0:
        empty = pd.DataFrame(columns=["n", "spread", "off"])
        return empty, empty
    od = both.assign(diff=both.chembl - both.davis).reset_index()
    g = (od.groupby(["gene", "drug_key", "assay_type"])
           .agg(n=("diff", "size"), spread=("diff", "std"),
                off=("diff", "mean")))
    multi = g[g.n >= min_n].copy()
    # std ek hi row pe NaN deta hai; min_n>=2 se wo bachta hai, par
    # fillna(0) rakha hai taaki niyam saaf rahe.
    const = multi[(multi.spread.fillna(0) < max_spread)
                  & (multi.off.abs() >= min_off)]
    return multi, const
