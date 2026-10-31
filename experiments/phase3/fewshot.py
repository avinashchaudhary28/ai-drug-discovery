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


# ------------------------------------------------------------ calibration
# Model ranking me jeet raha hai par MAE pe nahi - matlab kram sahi hai,
# MAATRA bada-chhota hai. To prediction ko 0 ki taraf ek factor `a` se
# sikodo. Ranking bilkul nahi badalti (a > 0 se kram wahi), MAE girta hai.
#
# JAAL YAHAN HAI: shots training ke andar hote hain, to main model ki
# unpe prediction pehle se overfit hai - usse `a` fit karein to `a ≈ 1`
# aayega aur calibration ka koi faida nahi, jabki code chalta rahega aur
# koi error nahi aayega.
#
# Isliye `a` ek ALAG model se nikalta hai jo shots ko DEKHA HI NAHI -
# sirf doosre genes pe fit. Uski shots pe prediction sach me bahar ki
# hai. Test set dono haalat me nahi chhua jata.
#
# Grid 0 se NAHI shuru hota, 0.05 se. Wajah: a=0 pe prediction bilkul
# `0 bolo` baseline ban jati hai - tab "baseline se jeeta" ka koi matlab
# nahi rehta, aur ek hi number hone se Spearman undefined ho jata hai.
# Ye chup-chaap hota aur mean Spearman kharab kar deta. a ka farsh pe
# aa jana KHUD ek nateeja hai - ki model ke paas maatra ki jaankari nahi -
# aur run_fewshot ginta hai kitni baar aisa hua.
A_FLOOR = 0.05
A_GRID = np.round(np.arange(A_FLOOR, 1.31, 0.05), 3)


def fit_shrink(pred_oos, y_true, grid=A_GRID):
    """-> a: wo factor jo `a * pred` ka MAE sabse kam kare.

    MAE ka closed form nahi hota, isliye grid. pred_oos wo prediction
    hai jo shots ko dekhe BINA bani ho - warna a bemaani hai.
    """
    pred_oos = np.asarray(pred_oos, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) == 0 or not np.isfinite(pred_oos).all():
        return 1.0
    mae = [np.mean(np.abs(y_true - a * pred_oos)) for a in grid]
    return float(grid[int(np.argmin(mae))])


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
