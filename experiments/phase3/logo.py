"""Leave-one-gene-out ka batwara, aur wo farsh jisse model ko jeetna hai.

Phase 3 ka pehla sawal model ka nahi, farsh ka hai: kya koi bhi cheez
"hamesha 0 bolo" se behtar hai? Kaafi published mutation models us farsh
se bas thoda hi upar hain, aur kuch to uske neeche.

Yahan DO cheezein aisi hain jo galat hon to nateeja shandaar dikhega aur
poora jhooth hoga, bina kisi error ke:

  1. Batware me leak - held-out gene ki rows training me bhi chali jayein
  2. Baseline ka leak - ausat POORE data se nikala jaye (test samet)
     bajaye sirf training ke

Isliye dono pure functions hain aur selftest_logo.py inhe seedha jaanchta
hai. Yahi is module ka poora maqsad hai.
"""
import numpy as np
import pandas as pd

MIN_PAIRS = 20          # itne se kam jode ho to gene pe test bemaani hai


def eligible_genes(df, min_pairs=MIN_PAIRS):
    vc = df.gene.value_counts()
    return sorted(vc[vc >= min_pairs].index)


def logo_splits(df, min_pairs=MIN_PAIRS):
    """Har gene ke liye: (gene, train_df, test_df).

    Test me SIRF wahi gene, train me baki SAB. Ek hi gene dono taraf
    kabhi nahi - yahi is poore phase ki sabse zaroori shart hai, kyunki
    asli sawal yahi hai ki model ANDEKHE gene pe chalta hai ya nahi.
    """
    for gene in eligible_genes(df, min_pairs):
        test = df[df.gene == gene]
        train = df[df.gene != gene]
        if train.empty or test.empty:
            continue
        yield gene, train, test


# ---------------------------------------------------------------- farsh
def predict_zero(train, test):
    """Farsh: har jagah 0. Mutation ka koi asar nahi."""
    return np.zeros(len(test))


def predict_global_mean(train, test):
    """Sirf TRAINING ka ausat. Test ka data chhuaa bhi nahi jata."""
    return np.full(len(test), float(train.ddG.mean()))


def predict_drug_mean(train, test):
    """Har dawa ka apna ausat - training se. Jo dawa training me nahi
    thi uske liye training ka global ausat."""
    m = train.groupby("drug_key").ddG.mean()
    g = float(train.ddG.mean())
    return test.drug_key.map(m).fillna(g).to_numpy(dtype=float)


AA = "ACDEFGHIKLMNPQRSTVWY"
# Token POORA mutation jaisa hona chahiye. Pehle sirf pehla aur aakhri
# akshar dekha jata tha, to "RAW:ITD" se "R>D" ban jata - yaani samajh
# na aaya naam ek asli AA badlav ke group me chala jata. Test ne pakda.
_SUB_RE = __import__("re").compile(rf"^([{AA}])(\d{{1,4}})([{AA}])$")


def _sub(mutation):
    """'T790M' -> 'T>M'. Multi-mutant ka pehla token."""
    tok = str(mutation).split("+")[0].strip().upper()
    m = _SUB_RE.match(tok)
    return f"{m.group(1)}>{m.group(3)}" if m else "?"


def predict_sub_mean(train, test):
    """Amino acid badlav (jaise T>M) ka ausat - training se."""
    tr = train.assign(_s=train.mutation.map(_sub))
    te = test.assign(_s=test.mutation.map(_sub))
    m = tr.groupby("_s").ddG.mean()
    g = float(train.ddG.mean())
    return te._s.map(m).fillna(g).to_numpy(dtype=float)


BASELINES = {
    "0 bolo (farsh)": predict_zero,
    "training ka ausat": predict_global_mean,
    "har dawa ka ausat": predict_drug_mean,
    "AA badlav ka ausat": predict_sub_mean,
}


# ------------------------------------------------------------- naap-tol
def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = {"n": int(len(y)),
           "mae": float(np.mean(np.abs(y - p))),
           "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}
    # Agar prediction ya sach me koi farak hi nahi to correlation
    # banta hi nahi - 0 likh dena jhooth hoga, NaN likhna sach hai.
    if len(y) < 3 or np.std(p) < 1e-12 or np.std(y) < 1e-12:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")
    else:
        out["pearson"] = float(np.corrcoef(y, p)[0, 1])
        out["spearman"] = float(pd.Series(y).corr(pd.Series(p),
                                                  method="spearman"))
    return out


def paired_t(deltas):
    """Har gene pe ek delta -> t = mean/SE. Gene hi ikai hai, row nahi:
    EGFR ki 2,376 rows ko 2,376 swatantra naap maan lena galat hoga."""
    d = np.asarray([x for x in deltas if np.isfinite(x)], dtype=float)
    if len(d) < 2:
        return {"n": int(len(d)), "mean": float("nan"), "t": float("nan"),
                "wins": 0}
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"n": int(len(d)), "mean": float(d.mean()),
            "t": float(d.mean() / se) if se > 0 else float("inf"),
            "wins": int((d > 0).sum())}
