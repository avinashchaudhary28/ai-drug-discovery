"""Top-k hit rate: Spearman ko khareedaar ki bhasha me badalna.

Spearman 0.34 ek abstract number hai. Khareedaar jo poochta hai wo ye hai:

    "Meri compound series hai. Ye resistance mutation clinic me aa gayi.
     Meri kaun si compound abhi bhi chalegi?"

Ye sawaal data me TIKTA hai: 75 samooh (gene x mutation) me 10+ dawaiyan
hain, saare 17 genes, 7,192 jode - data ka 88%. Ulta sawaal ("ek dawa,
kai mutations") data me nahi hai: har (gene, dawa) me median SIRF EK
mutation hai.

DISHA - yahan galti karna sabse aasan hai aur sabse mehnga:

    ddG = pK_mut - pK_wt

    ddG bahut RINATMAK  -> mutant se dawa ki pakad TOOT gayi -> dawa FAIL
    ddG ~0 ya DHANATMAK -> pakad bani rahi               -> dawa TIKI

Asli data pe jaanch: ABL1 ki sabse rinatmak mutations G250E, F317I,
T315I, E255K, Y253F, F317L nikli - theek wahi jo imatinib-resistance ke
liye jaani jaati hain. Disha sahi hai.

To "kaun si compound tikegi" = ddG ke UTARTE kram me (sabse dhanatmak
pehle). Agar disha ulti lag jaye to model ki sabse BEKAAR compounds
"sabse achhi" dikhengi, nateeja plausible hoga, aur koi error nahi
aayega. Isliye niche test seedhe disha jaanchte hain.
"""
import numpy as np

TOP = 3
MIN_GROUP = 5          # top-3 ka matlab tabhi hai jab samooh isse bada ho


def _rank_desc(v, rng):
    """Utarte kram me index. Barabar values ka kram RANDOM, pehle-aaya
    nahi - warna jo model sab ko ek hi number de (jaise `0 bolo`) usko
    data ka apna kram muft me mil jata aur wo jhoothi jeet dikhati."""
    v = np.asarray(v, dtype=float)
    jitter = rng.random(len(v))
    return np.lexsort((jitter, -v))


def group_scores(y_true, y_pred, rng, top=TOP):
    """Ek samooh (ek gene, ek mutation, kai dawaiyan) ke score.

    -> {"hit": asli sabse-tikne-wali dawa model ke top-`top` me hai?,
        "overlap": model ke top-`top` me se kitne asli top-`top` me hain,
        "n": samooh ka aakar, "random_hit": itna hi mauka se milta}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n < top + 1 or not np.isfinite(y_pred).all():
        return None
    true_order = _rank_desc(y_true, rng)
    pred_order = _rank_desc(y_pred, rng)
    best_true = int(true_order[0])
    pred_top = set(int(i) for i in pred_order[:top])
    true_top = set(int(i) for i in true_order[:top])
    return {"hit": float(best_true in pred_top),
            "overlap": len(pred_top & true_top) / top,
            "n": n, "random_hit": top / n}
