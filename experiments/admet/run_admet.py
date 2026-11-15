"""ADMET Benchmark Group - TDC ke APNE evaluator se.

Kyun dobara: `results_log.md` wale ADMET numbers humne khud sklearn/scipy
se nikale the (roc_auc_score, average_precision_score, mean_absolute_error,
spearmanr). TDC ki guideline saaf kehti hai ki `group.evaluate_many(...)`
use karo. Values shayad bahut paas hon - metric ki paribhasha wahi hai -
par "shayad paas" leaderboard entry ka aadhaar nahi hota.

Recipe wahi rakhi hai jo notebook me thi, taaki ye re-run purane numbers
ko JAANCHE bhi: Morgan count FP 2048 + MACCS 167 + RDKit 217 descriptors
+ ChemBERTa-77M-MTR 384 -> XGBoost (600 trees, depth 6, lr 0.05), har
dataset ke liye alag model, 5 seeds.

CHAAR CHEEZEIN JO CHUP-CHAAP GALAT NATEEJA DETI HAIN:

1. Classification pe `predict()` - wo 0/1 label deta hai, probability
   nahi. AUROC labels pe bahut kam aata hai par ERROR kabhi nahi. Isliye
   classifier pe hamesha predict_proba[:, 1].
2. Dataset ka naam galat - evaluate_many dict ki keys TDC ke naamon se
   bilkul milni chahiye, warna wo chup-chaap kuch aur naap lega.
3. Jo SMILES RDKit se parse na ho, use CHHOD dena - tab predictions ki
   ginti test rows se kam ho jati hai aur alignment toot jata hai. Isliye
   kabhi drop nahi, hamesha training ka median bhara jata hai, aur ginti
   report hoti hai.
4. Regression/classification ka hardcoded list - ek naam galat likhne se
   poora dataset ulta chala jayega. Isliye labels se KHUD pata lagaya
   jata hai: labels {0,1} me hain -> classifier.

Training SIRF `train` split pe hoti hai (train+valid pe nahi) - wahi
notebook me tha aur wo conservative hai.

Nateeja: results/admet_official.json, results/admet_official.csv
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
DATA = HERE / "tdc_data"
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1,2,3,4,5").split(",")]
N_TREES = int(os.environ.get("N_TREES", "600"))
ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]
# "train" = sirf train split (conservative). "train_val" = train+valid,
# jo notebook use karta tha. DONO vaajib hain - par ek chalakar, number
# dekhkar, phir doosra chun lena goalpost khiskana hai. Isliye dono
# chalte hain aur DONO report hote hain, alag TAG ke saath.
TRAIN_ON = os.environ.get("TRAIN_ON", "train")
assert TRAIN_ON in ("train", "train_val"), TRAIN_ON
CHEM_NAME = "DeepChem/ChemBERTa-77M-MTR"
CHEM_BATCH = int(os.environ.get("CHEM_BATCH", "64"))


# ------------------------------------------------------------- features
def rdkit_block(smiles):
    """Morgan count FP 2048 + MACCS 167 + 217 descriptors."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    names = [n for n, _ in Descriptors._descList]
    calc = [f for _, f in Descriptors._descList]
    width = 2048 + 167 + len(names)
    out = np.zeros((len(smiles), width), dtype=np.float32)
    bad = 0
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            out[i] = np.nan          # baad me training median se bharenge
            bad += 1
            continue
        fp = np.array(gen.GetCountFingerprintAsNumPy(m), dtype=np.float32)
        mac = np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32)
        d = np.empty(len(calc), dtype=np.float32)
        for j, f in enumerate(calc):
            try:
                v = f(m)
            except Exception:
                v = 0.0
            d[j] = v if np.isfinite(v) else 0.0
        out[i] = np.concatenate([fp, mac, d])
    out = np.clip(out, -1e6, 1e6)
    return out, bad


def chemberta_block(smiles, batch=CHEM_BATCH):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CHEM_NAME)
    mdl = AutoModel.from_pretrained(CHEM_NAME).eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch):
            chunk = [s if isinstance(s, str) else "" for s in smiles[i:i + batch]]
            enc = tok(chunk, padding=True, truncation=True, max_length=256,
                      return_tensors="pt")
            h = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            vecs.append(v.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)


def featurize(all_smiles):
    """Har ALAG SMILES ke liye ek baar - 22 datasets me molecules
    dohrate hain."""
    uniq = sorted(set(all_smiles))
    print(f"  {len(uniq):,} alag SMILES featurize kar raha hoon")
    t0 = time.time()
    rd, bad = rdkit_block(uniq)
    print(f"  RDKit done ({time.time()-t0:.0f}s), parse fail: {bad}")
    t0 = time.time()
    cb = chemberta_block(uniq)
    print(f"  ChemBERTa done ({time.time()-t0:.0f}s)")
    X = np.hstack([rd, cb])
    return {s: X[i] for i, s in enumerate(uniq)}, bad


def rows(smiles, table, dim):
    out = np.empty((len(smiles), dim), dtype=np.float32)
    for i, s in enumerate(smiles):
        out[i] = table[s]
    return out


def fill_nan(Xtr, *others):
    """Training ka median - test rows KABHI drop nahi hoti."""
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(Xtr, axis=0)   # poora column NaN -> niche 0
    med = np.where(np.isfinite(med), med, 0.0)
    def fix(A):
        m = ~np.isfinite(A)
        if m.any():
            A = A.copy()
            A[m] = np.take(med, np.where(m)[1])
        return A
    return (fix(Xtr),) + tuple(fix(o) for o in others)


CLS_METRICS = {"roc-auc", "auroc", "pr-auc", "auprc", "accuracy", "f1"}


def is_classification(y, metric=None):
    """Pehli pasand TDC ka APNA metric - wahi authority hai.

    Labels dekhkar andaza lagana bhi khatra hai: ek regression target
    jiski saari values ittefaq se 0/1 nikal aayein, wo classifier me
    chala jata, predict_proba [0,1] deta, aur MAE usi pe nap jata -
    bina kisi error ke. Isliye metric se tay karo; metric na mile
    tabhi labels pe giro, aur kaun sa tareeka laga wo CHHAAPO.
    """
    if metric:
        return str(metric).strip().lower() in CLS_METRICS
    u = np.unique(np.asarray(y, dtype=float))
    return len(u) <= 2 and set(u.tolist()) <= {0.0, 1.0}


def tdc_metric(group, name):
    """TDC jis naam se metric rakhta hai wo version ke saath badla hai,
    isliye dhoond kar nikalo - na mile to None."""
    for attr in ("metric_dict", "metrics", "dataset_metric"):
        d = getattr(group, attr, None)
        if isinstance(d, dict) and name in d:
            return d[name]
    return None


def fit_predict(Xtr, ytr, Xte, seed, clf):
    import xgboost as xgb
    common = dict(n_estimators=N_TREES, learning_rate=0.05, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  tree_method="hist", random_state=seed, n_jobs=-1)
    if clf:
        m = xgb.XGBClassifier(eval_metric="logloss", **common)
        m.fit(Xtr, ytr.astype(int))
        # predict() 0/1 deta hai - AUROC uspe girta hai aur error nahi aata
        return m.predict_proba(Xte)[:, 1]
    m = xgb.XGBRegressor(**common)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def main():
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    from tdc.benchmark_group import admet_group
    group = admet_group(path=str(DATA))
    names = list(group.dataset_names)
    if ONLY:
        names = [n for n in names if n in ONLY]
    print(f"[1/3] {len(names)} datasets | training data: {TRAIN_ON}")
    print(f"      {', '.join(names)}")

    # saare SMILES ek saath, taaki featurization ek hi baar ho
    pool, kinds, how = [], {}, {}
    for n in names:
        b = group.get(n)
        pool += list(b["train_val"]["Drug"]) + list(b["test"]["Drug"])
        met = tdc_metric(group, n)
        kinds[n] = is_classification(b["train_val"]["Y"].to_numpy(), met)
        how[n] = f"metric={met}" if met else "labels se andaza"
    n_guess = sum(1 for v in how.values() if "andaza" in v)
    print(f"  classification/regression: {len(names)-n_guess} TDC ke metric "
          f"se, {n_guess} labels se andaza")
    for n in names:
        print(f"    {n:<34} {'CLS' if kinds[n] else 'REG'}  ({how[n]})")
    print(f"\n[2/3] features")
    table, bad = featurize(pool)
    dim = len(next(iter(table.values())))
    print(f"  feature width {dim}")

    print(f"\n[3/3] {len(SEEDS)} seeds x {len(names)} datasets")
    preds_per_seed, notes = [], {}
    for seed in SEEDS:
        one = {}
        for n in names:
            b = group.get(n)
            if TRAIN_ON == "train_val":
                tr = b["train_val"]
            else:
                tr, _ = group.get_train_valid_split(benchmark=n,
                                                    split_type="default",
                                                    seed=seed)
            te = b["test"]
            Xtr = rows(list(tr["Drug"]), table, dim)
            Xte = rows(list(te["Drug"]), table, dim)
            Xtr, Xte = fill_nan(Xtr, Xte)
            p = fit_predict(Xtr, tr["Y"].to_numpy(dtype=float), Xte, seed,
                            kinds[n])
            assert len(p) == len(te), (n, len(p), len(te))
            one[n] = p
            notes[n] = {"n_train": int(len(tr)), "n_test": int(len(te)),
                        "classification": bool(kinds[n])}
        preds_per_seed.append(one)
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    # YAHI wo call hai jiske liye ye re-run hua
    official = group.evaluate_many(preds_per_seed)

    print(f"\n{'=' * 64}\nTDC KE APNE EVALUATOR SE ({len(SEEDS)} seeds)\n"
          f"{'=' * 64}\n")
    import pandas as pd
    t = pd.DataFrame(
        [{"dataset": k, "metric_mean": v[0], "metric_std": v[1],
          "classification": notes[k]["classification"],
          "n_train": notes[k]["n_train"], "n_test": notes[k]["n_test"]}
         for k, v in official.items()]).sort_values("dataset")
    print(t.to_string(index=False))
    t.to_csv(RES / f"admet_official{TAG}.csv", index=False)
    (RES / f"admet_official{TAG}.json").write_text(json.dumps({
        "evaluator": "tdc.benchmark_group.admet_group.evaluate_many",
        "seeds": SEEDS, "n_trees": N_TREES,
        "features": "Morgan count 2048 + MACCS 167 + RDKit descriptors "
                    "+ ChemBERTa-77M-MTR",
        "trained_on": ("train+valid" if TRAIN_ON == "train_val"
                       else "train split only"),
        "unparseable_smiles": int(bad),
        "results": {k: [float(v[0]), float(v[1])]
                    for k, v in official.items()},
        "per_dataset": notes}, indent=2))
    print(f"\n  parse na hone wale SMILES: {bad} "
          f"(training median se bhare, drop nahi kiye)")
    print(f"  files: {RES}/admet_official.json, admet_official.csv "
          f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
