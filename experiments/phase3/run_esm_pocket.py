"""Phase 3 kadam 4 (aakhri): mutated pocket ka ESM-2 embedding.

Pehle teen koshishon me ek baat saanjhi thi: dawa aur mutation ALAG-ALAG
describe hote the, aur pocket ka sirf ek number milta tha. Model ko
kabhi ye nahi pata chala ki pocket me AAS-PAAS kya hai.

Ab wo diya ja raha hai: poora 85-residue pocket, mutation lagi hui,
ESM-2 se embedding - aur khaas taur pe mutant aur wild-type ke
embedding ka ANTAR, kyunki asli sawal "is mutation se kya badla" hai.

GATE PEHLE SE TAY (code me likha hua, taaki baad me khisak na sake):
    paas = farsh se MAE pe t >= 2.0 AUR 14 me se 9+ genes pe jeet
         , YA  'XGB bina pocket' se Spearman pe t >= 2.0
    isse kam kuch bhi = FAIL.

Do ESM arm hain, jaan-boojh kar:
  - poora 1280-dim antar
  - aur PCA se 32 dim (PCA sirf TRAINING folds pe fit hoti hai)
Agar poora arm fail ho aur chhota chale, to matlab jaankari thi par
1280 shor-bhare dims me dab gayi. Agar dono fail hon, to jaankari hai
hi nahi - aur ye farak bina doosre arm ke pata nahi chalta.

Nateeja: results/esm_logo.csv, results/esm_report.json
"""
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_with_pocket                    # noqa: E402
from klifs_map import fetch_gene                          # noqa: E402
from logo import (BASELINES, eligible_genes, logo_splits,  # noqa: E402
                  metrics, paired_t)
from pocket_seq import build_pair                         # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
WORK = HERE / "work_klifs"
SRC = HERE.parent / "phase2" / "results" / "mutation_ddg_merged.csv"
MAPS = RES / "klifs_map.json"

MIN_PAIRS = int(os.environ.get("MIN_PAIRS", "20"))
N_TREES = int(os.environ.get("N_TREES", "400"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
ESM_NAME = os.environ.get("ESM_NAME", "facebook/esm2_t33_650M_UR50D")
PCA_DIM = int(os.environ.get("PCA_DIM", "32"))

# Gate - yahan likha hai, nateeja dekh kar nahi badlega
GATE_T = 2.0
GATE_WIN_FRAC = 0.6


def embed(seqs, batch=8):
    """-> {seq: 1280-dim vector}. Label-free, isliye leak ka sawal nahi."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ESM_NAME)
    model = AutoModel.from_pretrained(ESM_NAME)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    print(f"  {ESM_NAME} on {dev}, {len(seqs):,} sequences")
    out = {}
    uniq = sorted(seqs)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(uniq), batch):
            chunk = uniq[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True)
            enc = {k: v.to(dev) for k, v in enc.items()}
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            for s, e in zip(chunk, emb.cpu().numpy()):
                out[s] = e.astype(np.float32)
            if (i // batch) % 10 == 0:
                print(f"    {i + len(chunk)}/{len(uniq)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return out


def fit(Xtr, ytr, Xte, seed):
    import xgboost as xgb
    m = xgb.XGBRegressor(
        n_estimators=N_TREES, max_depth=6, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
        tree_method="hist", max_bin=128, random_state=seed, n_jobs=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    maps = json.loads(MAPS.read_text())
    df = pd.read_csv(SRC)
    df = df[(df.is_kinase == True)                         # noqa: E712
            & ~(df.censored_wt | df.censored_mut)
            & (df.mut_parsed == True)                      # noqa: E712
            & df.smiles.notna() & df.ddG.notna()]
    df = df[df.gene.isin(maps)].copy().reset_index(drop=True)
    print(f"[1/5] {len(df):,} jode, "
          f"{len(eligible_genes(df, MIN_PAIRS))} folds")

    print("\n[2/5] mutated pocket sequences")
    infos = {}
    for g in sorted(df.gene.unique()):
        infos[g] = fetch_gene(g)
    pairs, why_counts = [], {}
    for g, mu in zip(df.gene, df.mutation):
        wt, mt, why = build_pair(g, mu, maps[g], infos[g])
        pairs.append((wt, mt))
        why_counts[why] = why_counts.get(why, 0) + 1
    have = np.array([p[1] is not None for p in pairs])
    print(f"  ban payi: {int(have.sum()):,}/{len(df):,} "
          f"({have.mean():.0%})")
    for k, v in sorted(why_counts.items(), key=lambda x: -x[1])[:6]:
        print(f"      {v:>5,}  {k}")
    if have.mean() < 0.3:
        print(f"  !! sirf {have.mean():.0%} - ESM ke paas kehne ko kuch "
              f"hai hi nahi, nateeja uske hisaab se padhna")

    print("\n[3/5] ESM-2 embeddings")
    uniq = {s for p in pairs for s in p if s}
    emb = embed(uniq)
    D = len(next(iter(emb.values())))
    DELTA = np.zeros((len(df), D), dtype=np.float32)
    for i, (wt, mt) in enumerate(pairs):
        if wt and mt:
            DELTA[i] = emb[mt] - emb[wt]
    nz = float((np.abs(DELTA).sum(axis=1) > 0).mean())
    print(f"  antar ka vector: {D} dims, "
          f"{nz:.0%} jodon pe kuch hai")

    print("\n[4/5] baki features")
    X0, X1, ok, _ = build_with_pocket(df, maps)
    df = df[ok].copy().reset_index(drop=True)
    DELTA = DELTA[ok]
    X_esm = np.hstack([X1, DELTA]).astype(np.float32)
    print(f"  bina pocket {X0.shape[1]} | pocket {X1.shape[1]} | "
          f"+ESM {X_esm.shape[1]}")

    print(f"\n[5/5] leave-one-gene-out, {len(SEEDS)} seeds")
    from sklearn.decomposition import PCA
    recs = []
    for gene, train, test in logo_splits(df, MIN_PAIRS):
        itr, ite = train.index.to_numpy(), test.index.to_numpy()
        y = test.ddG.to_numpy(dtype=float)
        ytr = train.ddG.to_numpy(dtype=float)

        # PCA SIRF training pe fit - warna test ka data PCA ke zariye
        # leak ho jata, aur nateeja jhootha hota
        k = min(PCA_DIM, len(itr), DELTA.shape[1])
        pca = PCA(n_components=k, random_state=0).fit(DELTA[itr])
        Xp_tr = np.hstack([X1[itr], pca.transform(DELTA[itr])])
        Xp_te = np.hstack([X1[ite], pca.transform(DELTA[ite])])

        for name, fn in BASELINES.items():
            recs.append({"gene": gene, "model": name, "seed": -1,
                         **metrics(y, fn(train, test))})
        for seed in SEEDS:
            for nm, Xtr_, Xte_ in (
                    ("XGB bina pocket", X0[itr], X0[ite]),
                    ("XGB + pocket", X1[itr], X1[ite]),
                    ("XGB + pocket + ESM", X_esm[itr], X_esm[ite]),
                    (f"XGB + pocket + ESM-PCA{k}", Xp_tr, Xp_te)):
                recs.append({"gene": gene, "model": nm, "seed": seed,
                             **metrics(y, fit(Xtr_, ytr, Xte_, seed))})
        print(f"  {gene:<10} n={len(test):>5,}  ({time.time()-t0:.0f}s)",
              flush=True)

    r = pd.DataFrame(recs)
    per_gene = (r.groupby(["gene", "model"])
                 .agg(n=("n", "first"), mae=("mae", "mean"),
                      rmse=("rmse", "mean"), spearman=("spearman", "mean"))
                 .reset_index())
    per_gene.to_csv(RES / "esm_logo.csv", index=False)
    summ = (per_gene.groupby("model")
            .agg(genes=("gene", "nunique"), mae=("mae", "mean"),
                 rmse=("rmse", "mean"), spearman=("spearman", "mean"))
            .sort_values("mae"))
    print(f"\n{'=' * 72}\nNATEEJA\n{'=' * 72}\n")
    print(summ.to_string())

    def cmp(a, b):
        A = per_gene[per_gene.model == a].set_index("gene")
        B = per_gene[per_gene.model == b].set_index("gene")
        c = A.index.intersection(B.index)
        return (paired_t((B.loc[c, "mae"] - A.loc[c, "mae"]).to_numpy()),
                paired_t((A.loc[c, "spearman"]
                          - B.loc[c, "spearman"]).to_numpy()))

    esm_models = [m for m in summ.index if "ESM" in m]
    print(f"\n  GATE: farsh se MAE t>={GATE_T} aur "
          f"{GATE_WIN_FRAC:.0%}+ genes pe jeet, YA bina-pocket se "
          f"Spearman t>={GATE_T}\n")
    out, passed = {}, []
    for m in esm_models:
        fm, _ = cmp(m, "0 bolo (farsh)")
        _, ns = cmp(m, "XGB bina pocket")
        ok_ = ((fm["t"] >= GATE_T and fm["wins"] >= GATE_WIN_FRAC * fm["n"])
               or ns["t"] >= GATE_T)
        out[m] = {"vs_floor_mae": fm, "vs_nopocket_spearman": ns,
                  "passed": bool(ok_)}
        if ok_:
            passed.append(m)
        print(f"    {m}")
        print(f"      farsh se MAE   {fm['mean']:+.4f}  t={fm['t']:+6.2f}  "
              f"jeeta {fm['wins']}/{fm['n']}")
        print(f"      Spearman vs bina-pocket {ns['mean']:+.4f}  "
              f"t={ns['t']:+6.2f}  jeeta {ns['wins']}/{ns['n']}")
        print(f"      >>> {'GATE PAAS' if ok_ else 'gate fail'}")

    print(f"\n{'=' * 72}\nFAISLA\n{'=' * 72}")
    if passed:
        print(f"  GATE PAAS: {', '.join(passed)}")
        print(f"  Pehli baar kuch andekhe gene pe farsh se jeeta hai. "
              f"Ab isko TKI-ABL1 setting pe jaanchna hai.")
    else:
        print(f"  GATE FAIL. Ye chauthi koshish thi:")
        print(f"    1. mutation position + AA properties")
        print(f"    2. + KLIFS aligned pocket position")
        print(f"    3. + ligand contact frequency (IFP)")
        print(f"    4. + mutated pocket ka ESM-2 embedding (poora aur PCA)")
        print(f"  Chaaron andekhe gene pe '0 bolo' se nahi jeet paye.")
        print(f"  Dono ESM arm fail hone ka matlab: jaankari 1280 dims me "
              f"dabi nahi thi - wo thi hi nahi.")

    (RES / "esm_report.json").write_text(json.dumps({
        "pairs": int(len(df)), "esm_model": ESM_NAME,
        "pocket_seq_coverage": float(nz),
        "gate": {"t": GATE_T, "win_frac": GATE_WIN_FRAC},
        "summary": json.loads(summ.reset_index().to_json(orient="records")),
        "arms": out, "passed": passed,
        "gate_passed": bool(passed)}, indent=2))
    print(f"\n  files: {RES}/esm_logo.csv, esm_report.json "
          f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
