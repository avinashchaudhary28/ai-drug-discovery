"""Leave-one-gene-out benchmark for kinase mutation-effect (ddG) models.

Standalone: needs only pandas, numpy and scipy. Run it to reproduce the
baseline numbers, or import `evaluate` and pass your own model.

    python logo_benchmark.py --data ../mutation_ddg_v1.0.csv

WHY LEAVE-ONE-GENE-OUT, AND NOT A RANDOM SPLIT
----------------------------------------------
A random split lets the same gene appear on both sides. On this dataset
that scores Spearman ~0.75 and means nothing: the model is recognising
the gene, not learning mutation effects. Remove the gene entirely and
the same model drops to ~0.067.

That 0.752-vs-0.067 gap is the single most useful number here. If your
method reports a number far above the baselines below, check how your
split was made before believing it.

THE UNIT IS THE GENE, NOT THE ROW
---------------------------------
EGFR contributes 2,376 of 8,110 pairs. Treating rows as independent
observations manufactures confidence. Every statistic here is computed
per gene and then aggregated across genes.

THE NOISE FLOOR
---------------
Within one publication, ChEMBL holds four wild-type measurements for one
compound and target spanning 0.8 log units. Roughly 0.5 log units of
irreducible noise sits in these labels. A reported MAE far below that is
evidence of leakage, not skill.
"""
import argparse
import sys

import numpy as np
import pandas as pd

MIN_PAIRS = 20          # genes with fewer pairs make unstable folds


# --------------------------------------------------------------- loading
def load(path, uncensored=True, min_pairs=MIN_PAIRS):
    """-> (dataframe, list of fold genes). Applies the documented filter."""
    d = pd.read_csv(path)
    d = d[d.is_kinase & d.mut_parsed & d.smiles.notna() & d.ddG.notna()]
    if uncensored:
        d = d[~(d.censored_wt | d.censored_mut)]
    d = d.reset_index(drop=True)
    vc = d.gene.value_counts()
    return d, sorted(vc[vc >= min_pairs].index)


def sanity_check(d, verbose=True):
    """Sign convention: ddG = pK_mut - pK_wt, so NEGATIVE = resistance.

    Inverting this is the easiest mistake to make with this dataset and
    it raises no error - it simply reports the least-affected compounds
    as the worst. So check it against biology that is not in dispute:
    ABL1's most negative mutations must be the known imatinib-resistance
    set.
    """
    known = {"T315I", "E255K", "Y253F", "F317I", "F317L", "G250E"}
    a = d[d.gene == "ABL1"]
    if a.empty:
        if verbose:
            print("  (ABL1 absent - sign check skipped)")
        return None
    m = a.groupby("mutation").ddG.agg(["mean", "size"])
    m = m[m["size"] >= 10].sort_values("mean")
    if len(m) < 6:
        if verbose:
            print("  (too few ABL1 mutations - sign check skipped)")
        return None
    worst6 = set(m.head(6).index)
    hits = len(worst6 & known)
    if verbose:
        print(f"  sign check: {hits}/6 of ABL1's most-negative mutations are "
              f"known resistance mutations")
        print(f"    most negative: {', '.join(m.head(6).index)}")
    if hits < 3:
        raise SystemExit(
            "SIGN CHECK FAILED. ABL1's most negative mutations are not the "
            "known resistance set, so ddG's sign is probably inverted "
            "somewhere. Expected: negative ddG = mutation weakens binding.")
    return hits


# ------------------------------------------------------------- statistics
def metrics(y, p):
    """MAE and within-gene Spearman. Spearman is NaN for a constant
    prediction - that is correct, not zero: one number cannot rank."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = {"n": len(y), "mae": float(np.mean(np.abs(y - p))),
           "spearman": float("nan")}
    if len(y) > 2 and np.ptp(p) > 0 and np.ptp(y) > 0:
        from scipy.stats import spearmanr
        out["spearman"] = float(spearmanr(y, p).statistic)
    return out


def paired_t(deltas):
    """Paired t over GENES. se == 0 returns +-inf by the sign of the mean,
    never a bare +inf - all-zero or all-negative differences must not
    read as a win."""
    d = np.asarray([x for x in deltas if np.isfinite(x)], dtype=float)
    if len(d) < 2:
        return {"n": len(d), "mean": float(d.mean()) if len(d) else float("nan"),
                "t": float("nan"), "wins": int((d > 0).sum())}
    se = d.std(ddof=1) / np.sqrt(len(d))
    m = float(d.mean())
    if se > 0:
        t = m / se
    elif m > 0:
        t = float("inf")
    elif m < 0:
        t = float("-inf")
    else:
        t = float("nan")
    return {"n": int(len(d)), "mean": m, "t": t, "wins": int((d > 0).sum())}


def jackknife_t(deltas, labels):
    """Drop each gene in turn and recompute t. A result that collapses
    when one gene is removed rests on that gene."""
    pairs = [(l, x) for l, x in zip(labels, deltas) if np.isfinite(x)]
    if len(pairs) < 3:
        return {"min_t": float("nan"), "worst_gene": None}
    per = {}
    for i, (lb, _) in enumerate(pairs):
        per[lb] = paired_t([x for j, (_, x) in enumerate(pairs) if j != i])["t"]
    worst = min(per, key=lambda g: per[g])
    return {"min_t": float(per[worst]), "worst_gene": worst}


# -------------------------------------------------------------- baselines
def b_zero(tr, te):
    return np.zeros(len(te))


def b_train_mean(tr, te):
    return np.full(len(te), tr.ddG.mean())


def b_drug_mean(tr, te):
    m = tr.groupby("drug_key").ddG.mean()
    return te.drug_key.map(m).fillna(tr.ddG.mean()).to_numpy(dtype=float)


_SUB_RX = __import__("re").compile(
    r"^([ACDEFGHIKLMNPQRSTVWY])(\d{1,4})([ACDEFGHIKLMNPQRSTVWY])$")


def _sub_class(mutation):
    """'T790M' -> 'T>M'. For a multi-mutant, the FIRST substitution.

    The strict regex matters: an unparseable name must not be sliced
    into a letter pair and dropped into a real substitution class. It
    goes to '?' instead, which is its own group.
    """
    tok = str(mutation).split("+")[0].strip().upper()
    m = _SUB_RX.match(tok)
    return f"{m.group(1)}>{m.group(3)}" if m else "?"


def b_sub_mean(tr, te):
    """Mean by amino-acid substitution class (T>M etc.)."""
    g = tr.assign(_s=tr.mutation.map(_sub_class))
    m = g.groupby("_s").ddG.mean()
    return (te.mutation.map(_sub_class).map(m)
            .fillna(float(tr.ddG.mean())).to_numpy(dtype=float))


BASELINES = {
    "predict 0 (the floor)": b_zero,
    "training mean": b_train_mean,
    "per-drug mean": b_drug_mean,
    "per-substitution mean": b_sub_mean,
}


# ------------------------------------------------------------- evaluation
def evaluate(d, folds, predict, name="model"):
    """`predict(train_df, test_df) -> array of ddG predictions`.

    -> per-gene DataFrame. One row per fold gene.
    """
    rows = []
    for g in folds:
        te = d[d.gene == g]
        tr = d[d.gene != g]
        p = np.asarray(predict(tr, te), dtype=float)
        if len(p) != len(te):
            raise ValueError(f"{name}: {len(p)} predictions for {len(te)} "
                             f"test rows on gene {g}")
        rows.append({"gene": g, **metrics(te.ddG.to_numpy(dtype=float), p)})
    return pd.DataFrame(rows).set_index("gene")


def compare(a, b, col="mae", lower_is_better=True):
    """Paired comparison of two per-gene tables, gene as the unit."""
    idx = a.index.intersection(b.index)
    d = ((b.loc[idx, col] - a.loc[idx, col]) if lower_is_better
         else (a.loc[idx, col] - b.loc[idx, col]))
    t = paired_t(d.to_numpy())
    t.update(jackknife_t(d.to_numpy(), list(idx)))
    return t


# -------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="../mutation_ddg_v1.0.csv")
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    ap.add_argument("--keep-censored", action="store_true",
                    help="include '>' / '<' measurements (not recommended)")
    a = ap.parse_args(argv)

    d, folds = load(a.data, uncensored=not a.keep_censored,
                    min_pairs=a.min_pairs)
    print(f"{len(d):,} pairs | {d.gene.nunique()} genes | "
          f"{len(folds)} leave-one-gene-out folds (>= {a.min_pairs} pairs)")
    sanity_check(d)

    print("\nBASELINES (gene is the unit; Spearman is within-gene)\n")
    tables = {}
    for nm, fn in BASELINES.items():
        tables[nm] = evaluate(d, folds, fn, nm)
    summary = pd.DataFrame({
        nm: {"MAE": t.mae.mean(), "Spearman": t.spearman.mean()}
        for nm, t in tables.items()}).T.sort_values("MAE")
    print(summary.round(3).to_string())

    floor = tables["predict 0 (the floor)"]
    print("\nAgainst the floor (positive MAE = better than predicting 0):\n")
    for nm, t in tables.items():
        if nm == "predict 0 (the floor)":
            continue
        c = compare(t, floor)
        print(f"  {nm:<24} MAE {c['mean']:+.4f}  t={c['t']:+6.2f}  "
              f"wins {c['wins']}/{c['n']}  | jackknife min t "
              f"{c['min_t']:+.2f} ({c['worst_gene']})")

    print("""
HOW TO EVALUATE YOUR OWN MODEL

    from logo_benchmark import load, evaluate, compare, BASELINES
    d, folds = load("../mutation_ddg_v1.0.csv")
    mine  = evaluate(d, folds, my_predict_fn)
    floor = evaluate(d, folds, BASELINES["predict 0 (the floor)"])
    print(compare(mine, floor))

Report, at minimum: mean MAE and mean within-gene Spearman over folds;
the paired t against `predict 0` AND against `per-drug mean`; how many
folds you win; and the jackknife, so a reader can see whether the result
rests on one gene. See REFERENCE_RESULTS.md for numbers to beat.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
