"""Few-shot batwara: naye gene ke k maap dekar baaki ki bhavishyvani.

Zero-shot (bilkul andekha gene) chaar koshishon me fail ho chuka. Par
koi company kabhi aise nahi aati - uske paas apne target ke kuch maap
hote hi hain. Asli sawal:

    "Humare paas apne kinase pe 20 maap hain. Baaki ke liye bata do."

DESIGN KA SABSE ZAROORI HISSA: test set har k pe EK JAISA rehna chahiye.

Seedha tareeka ye hota ki gene ke pehle k rows shots banein aur baaki
test. Par tab k badhne pe test set CHHOTA aur ALAG ho jata, aur curve
padhna bemaani - sudhaar model ka hai ya aasan test set ka, pata hi
nahi chalta.

Isliye: gene ko pehle do hisson me baanto - pool aur test. Shots HAMESHA
pool se aate hain, test HAMESHA wahi rehta hai. Tab k=0 se k=50 tak ka
curve ek hi paimane pe hai.
"""
import numpy as np


def split_gene(idx, seed, test_frac=0.5):
    """-> (pool, test). Gene ke rows ko do hisson me baanto.

    Test har k pe wahi rahega; shots sirf pool se niklenge.
    """
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    perm = rng.permutation(len(idx))
    n_test = int(round(len(idx) * test_frac))
    n_test = max(1, min(n_test, len(idx) - 1))
    return idx[perm[n_test:]], idx[perm[:n_test]]


def take_shots(pool, k, seed):
    """Pool me se k shots. k=0 pe khali - yahi zero-shot wali haalat hai."""
    if k <= 0:
        return np.array([], dtype=pool.dtype)
    rng = np.random.default_rng(seed + 10_000)
    pool = np.asarray(pool)
    if k >= len(pool):
        return pool.copy()
    return pool[rng.permutation(len(pool))[:k]]


def eligible(df, ks, test_frac=0.5):
    """Wo genes jinme har k ke liye kaafi rows hain.

    Agar alag-alag k pe alag genes holte, to curve me sudhaar genes
    badalne ka bhi ho sakta hai - aur wo model ka sudhaar jaisa dikhta.
    Isliye sirf wahi genes jinpe POORA curve ban sakta hai.

    SAWDHANI: ye shart max(ks) aur test_frac SE BANTI hai. Dono me se
    kuch bhi badla to GENE SET badal jata hai, aur tab do run ke number
    seedhe tulnaa ke laayak NAHI rehte. Isliye run_fewshot gene set aur
    `need` dono report me likhta hai.
    """
    need = int(np.ceil(max(ks) / (1 - test_frac)))
    vc = df.gene.value_counts()
    return sorted(vc[vc >= need].index), need


# ------------------------------------------------- shots wale baseline
def shots_mean(shots_y, train_y):
    """Shots ka ausat. Shots na hon to training ka."""
    if len(shots_y):
        return float(np.mean(shots_y))
    return float(np.mean(train_y))


def shots_drug_mean(shots_df, test_df, train_df):
    """Har dawa ka ausat, SIRF shots se.

    Yahi sabse kathin baseline hai: agar model isse nahi jeeta, to faida
    model ka nahi, un k numbers ka hai - aur tab bechne ko kuch nahi.
    """
    if len(shots_df):
        m = shots_df.groupby("drug_key").ddG.mean()
        fallback = float(shots_df.ddG.mean())
    else:
        m = train_df.groupby("drug_key").ddG.mean()
        fallback = float(train_df.ddG.mean())
    return test_df.drug_key.map(m).fillna(fallback).to_numpy(dtype=float)
