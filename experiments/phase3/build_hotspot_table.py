"""KLIFS position-level resistance propensity - ek TABLE, model nahi.

Koshish 6/7 me ek cheez mili: per-KLIFS-position resistance rate (AUROC
0.583) per-drug reasoning (0.556) se AAGE hai. Yaani signal POSITION me
hai, dawa me nahi.

Maine isse ek naya model banane ka sochha tha - "is kinase me kaun si
positions hotspot hain". Phir ginti ki:

    kul (gene, position) jodi : 85
    n >= 5 pe                 : 44 jodi, 20 genes, 5 hotspot (11%)
    jin genes me DONO class   : 1

Leave-one-gene-out ke liye AUROC chahiye, aur 19 me se 20 genes ke paas
ek hi class hai. To wo task naapa hi nahi ja sakta - banane ka matlab
hi nahi. Wajah biological hai: har gene me sirf 1-13 positions pe
measurements hain, kyunki log wahi jaani-maani sites naapte hain.

To finding ka sahi roop ek LOOKUP TABLE hai: har KLIFS position pe
resistance kitni baar hoti hai, kitne data pe, kitne genes se, aur
kitne bharose ke saath. Aur uski upyogita leave-one-gene-out se naapi
jaati hai - table doosre genes se banti hai aur held-out gene pe
jaanchi jaati hai. Yahi 0.583 ka imaandaar roop hai.

Nateeja: results/hotspot_table.csv, results/hotspot_table.md
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from features import pocket_features, region_name              # noqa: E402
from logo import jackknife_t, paired_t                         # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
SRC = HERE.parent / "phase2" / "results" / "mutation_ddg_merged.csv"
MAPS = RES / "klifs_map.json"
THRESH = -1.0
SMOOTH = 10.0      # pooled rate ki taraf sikudan, kam data wali jagah ke liye


def wilson(k, n, z=1.96):
    """Binomial ka Wilson interval. Normal approximation 0 ya 1 ke paas
    toot jata hai, aur yahan bahut si positions pe k=0 hai."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load():
    maps = json.loads(MAPS.read_text())
    d = pd.read_csv(SRC)
    d = d[(d.is_kinase == True)                                # noqa: E712
          & ~(d.censored_wt | d.censored_mut)
          & (d.mut_parsed == True)                             # noqa: E712
          & d.smiles.notna() & d.ddG.notna()].reset_index(drop=True)
    regions = sorted({region_name(h[1]) for i in maps.values()
                      for h in i["map"].values()})
    pos, reg = [], []
    for g, mu in zip(d.gene, d.mutation):
        v, r = pocket_features(g, mu, maps, regions)
        pos.append(int(v[1]) if v[0] > 0 else -1)
        reg.append(r if v[0] > 0 else "")
    d["pos"] = pos
    d["region"] = reg
    d["y"] = (d.ddG <= THRESH).astype(int)
    # Har (gene, position) ka wild-type residue, KLIFS map se
    wt = {}
    for g, info in maps.items():
        for hit in info["map"].values():
            wt[(g, int(hit[0]))] = hit[2]
    d["wt_aa"] = [wt.get((g, p), "") for g, p in zip(d.gene, d.pos)]
    return d[d.pos > 0].reset_index(drop=True), maps


def table(df, smooth=SMOOTH):
    """Pooled per-position resistance rate, saare genes milakar."""
    prior = float(df.y.mean())
    rows = []
    for p, g in df.groupby("pos"):
        k, n = int(g.y.sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({
            "klifs_pos": p,
            "region": g.region.mode().iat[0] if len(g.region.mode()) else "",
            "n_measurements": n,
            "n_genes": g.gene.nunique(),
            "n_mutations": g.mutation.nunique(),
            "resistance_events": k,
            "rate_raw": round(k / n, 4),
            "rate_smoothed": round((k + smooth * prior) / (n + smooth), 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "wt_residues": "".join(sorted(set(g.wt_aa) - {""})),
            "genes": ",".join(sorted(g.gene.unique())[:8]),
        })
    t = pd.DataFrame(rows).sort_values("rate_smoothed", ascending=False)
    return t.reset_index(drop=True), prior


def validate(df, smooth=SMOOTH):
    """Leave-one-gene-out: table DOOSRE genes se bane, held-out gene pe
    jaanchi jaaye. Yahi batata hai ki table transfer hoti hai ya nahi."""
    from sklearn.metrics import roc_auc_score
    out = []
    for gene in sorted(df.gene.unique()):
        te = df[df.gene == gene]
        tr = df[df.gene != gene]
        if te.y.nunique() < 2 or len(tr) == 0:
            continue
        prior = float(tr.y.mean())
        rate = tr.groupby("pos").y.agg(["sum", "size"])
        sm = ((rate["sum"] + smooth * prior) / (rate["size"] + smooth))
        p_tbl = te.pos.map(sm).fillna(prior).to_numpy(dtype=float)
        p_base = np.full(len(te), prior)
        out.append({
            "gene": gene, "n": len(te), "pos_events": int(te.y.sum()),
            "auroc_table": float(roc_auc_score(te.y, p_tbl)),
            "auroc_flat": float(roc_auc_score(te.y, p_base))
            if te.y.nunique() > 1 and len(set(p_base)) > 1 else 0.5,
        })
    return pd.DataFrame(out)


def main():
    RES.mkdir(exist_ok=True)
    df, maps = load()
    print(f"pocket me rows: {len(df):,} | genes {df.gene.nunique()} | "
          f"alag positions {df.pos.nunique()}")
    print(f"resistance (ddG <= {THRESH}): {int(df.y.sum())} "
          f"({100*df.y.mean():.1f}%)")

    t, prior = table(df)
    t.to_csv(RES / "hotspot_table.csv", index=False)
    print(f"\ntable: {len(t)} positions  (pooled base rate {prior:.3f})")
    show = ["klifs_pos", "region", "n_measurements", "n_genes",
            "resistance_events", "rate_raw", "ci95_low", "ci95_high",
            "wt_residues"]
    print("\nTOP 12 (rate_smoothed se):")
    print(t.head(12)[show].to_string(index=False))

    v = validate(df)
    d = (v.auroc_table - 0.5).to_numpy()
    tt = paired_t(d)
    jk = jackknife_t(d, list(v.gene), thresh=2.0)
    print(f"\n{'=' * 70}\nLEAVE-ONE-GENE-OUT: table transfer hoti hai?\n"
          f"{'=' * 70}")
    print(v.round(3).to_string(index=False))
    print(f"\n  ausat AUROC {v.auroc_table.mean():.3f}  "
          f"(mauka 0.500, {len(v)} genes)")
    print(f"  mauke se upar: {tt['mean']:+.3f}  t={tt['t']:+.2f}  "
          f"jeeta {tt['wins']}/{tt['n']}")
    print(f"  jackknife: {jk['worst_gene']} hatane pe t={jk['min_t']:+.2f}")

    md = [
        "# KLIFS position-level resistance propensity",
        "",
        "Which aligned kinase pocket positions produce drug resistance when",
        f"mutated. Built from {len(df):,} measured (drug, mutation) pairs over",
        f"{df.gene.nunique()} genes; a resistance event is ΔΔG ≤ {THRESH}",
        f"(the drug loses ≥10× affinity). Pooled base rate "
        f"**{prior:.3f}**.",
        "",
        "**This is a lookup table, not a model, and that is deliberate.**",
        "A model was considered and abandoned after counting: there are only",
        "**85 (gene, position) pairs**, and at n ≥ 5 just **1 of 20 genes** has",
        "both hotspot and non-hotspot positions — so leave-one-gene-out AUROC",
        "is undefined almost everywhere and the task cannot be evaluated",
        "honestly. The cause is biological: each gene has measurements at only",
        "1–13 positions, because the field measures the resistance sites it",
        "already knows about.",
        "",
        "## Does it transfer to an unseen gene?",
        "",
        "Leave-one-gene-out: the table is built from the other genes and scored",
        f"on the held-out one. Mean AUROC **{v.auroc_table.mean():.3f}** over",
        f"{len(v)} genes (chance 0.500), beating chance by {tt['mean']:+.3f} at",
        f"t = {tt['t']:+.2f}, winning {tt['wins']}/{tt['n']} genes; removing the",
        f"most influential gene ({jk['worst_gene']}) leaves t = "
        f"{jk['min_t']:+.2f}.",
        "",
        "## Where the signal actually comes from — a correction",
        "",
        "The per-position rate scored **0.583** in the Step 7 classification",
        "run (17 folds, all rows), beating a per-drug rate (0.556) and nearly",
        "matching a trained model (0.596). I described that as *position",
        "identity beating drug identity*. Decomposing it shows that reading was",
        "too generous:",
        "",
        "| Predictor | genes | AUROC | t vs chance | jackknife |",
        "|---|---|---|---|---|",
        "| per-position, with *not-in-pocket* as its own category | 17 | **0.583** | +3.02 | +2.67 |",
        "| **just the one bit: is the mutation inside the pocket?** | 17 | **0.545** | **+2.51** | **+2.18** |",
        "| per-position, restricted to in-pocket rows | 12 | 0.555 | +1.70 | +1.28 |",
        "",
        "So **most of that 0.583 is a single binary fact** — whether the",
        "mutation falls inside the ATP pocket at all — which is real,",
        "jackknife-robust, and needs no model. The fine-grained 85-position",
        "resolution that this table provides is **not established**: with the",
        "in/out bit held constant it reaches t = 1.70 on 12 genes.",
        "",
        "That is the honest status of this table. It is a usable prior for",
        "ranking positions, not a validated predictor.",
        "",
        "## Columns",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| `klifs_pos` | KLIFS pocket position, 1–85, aligned across the kinome |",
        "| `region` | structural region of that position |",
        "| `n_measurements` / `n_genes` / `n_mutations` | how much evidence |",
        "| `resistance_events` | measurements with ΔΔG ≤ −1 |",
        "| `rate_raw` | events / measurements |",
        "| `rate_smoothed` | shrunk toward the pooled rate; use this for ranking |",
        "| `ci95_low` / `ci95_high` | Wilson 95% interval on `rate_raw` |",
        "| `wt_residues` | wild-type residues seen at this position |",
        "| `genes` | contributing genes (up to 8) |",
        "",
        "**Read the interval, not the rate.** Many positions rest on a handful",
        "of measurements, where `rate_raw` is almost unconstrained — a position",
        "at 1.00 on 2 measurements has a 95% interval running down past 0.3.",
        "",
        "## Top positions by smoothed rate",
        "",
        t.head(15)[show].to_markdown(index=False),
    ]
    (RES / "hotspot_table.md").write_text("\n".join(md) + "\n")
    print(f"\n  files: {RES}/hotspot_table.csv, hotspot_table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
