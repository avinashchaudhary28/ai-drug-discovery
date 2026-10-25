"""
Kya ye experiment poochh raha hai?
----------------------------------
v2 ki sabse badi kamzori: nayi dawa pe CI 0.675 (Davis cold-drug).
Sawaal: **zyada data** se ye sudhrega, ya problem protein representation me hai?

Isko theek se naapne ke liye model badalte NAHI hain — sirf data badalte hain:

  A (v2 jaisa)  : XGBoost, training data = Davis train pairs
  B (v3 / zyada): XGBoost, training data = Davis train pairs + BindingDB pairs

Dono ko **ek hi test set** pe naapte hain. Features, model, hyperparameters — sab same.
Agar B jeeta -> data hi bottleneck tha. Agar nahi -> representation bottleneck hai.

GPU ki zaroorat nahi: Davis ke 442 proteins ki ESM-2 650M embeddings repo me
pehle se padi hain (models/davis_esm.npy), aur BindingDB ke proteins ko unhi se
sequence-match karke wahi vectors use karte hain.
"""
import gzip, json, os, pickle, shutil, sys, time, urllib.request
from collections import OrderedDict

import numpy as np
import pandas as pd

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS = os.path.join(REPO, "models")
WORK = os.path.join(HERE, "work")
OUT = os.path.join(HERE, "results")
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

MAX_TOTAL_PAIRS = int(os.environ.get("MAX_TOTAL_PAIRS", 120000))
N_TREES = int(os.environ.get("N_TREES", 400))
N_SEEDS = int(os.environ.get("N_SEEDS", 5))
SEED = 0
rng = np.random.default_rng(SEED)


def step(msg):
    print(f"\n{'='*70}\n[{(time.time()-T_START)/60:5.1f} min] {msg}\n{'='*70}", flush=True)


def info(msg):
    print(f"   {msg}", flush=True)


# ----------------------------------------------------------------- download
def download(url, path, min_bytes=1000):
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        info(f"cache: {os.path.basename(path)}")
        return path
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=900) as r, open(path, "wb") as f:
                shutil.copyfileobj(r, f)
            if os.path.getsize(path) >= min_bytes:
                info(f"downloaded {os.path.basename(path)} "
                     f"({os.path.getsize(path)/1e6:.1f} MB)")
                return path
            last = f"file too small ({os.path.getsize(path)} bytes)"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        info(f"  try {attempt+1}/3 failed - {last}")
        time.sleep(5)
    raise RuntimeError(f"download failed: {url} ({last})")


step("1. Davis dataset (DeepDTA) laate hain")
DAVIS_BASES = [
    "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/",
    "https://raw.githubusercontent.com/thinng/GraphDTA/master/data/davis/",
]


def davis_file(name, min_bytes=1000):
    path = os.path.join(WORK, f"davis_{name}")
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        info(f"cache: davis_{name}")
        return path
    errs = []
    for base in DAVIS_BASES:
        try:
            return download(base + name, path, min_bytes)
        except Exception as e:
            errs.append(str(e))
    raise RuntimeError(f"davis/{name} kahin se nahi mila: {errs}")


lig = json.load(open(davis_file("ligands_can.txt")), object_pairs_hook=OrderedDict)
pro = json.load(open(davis_file("proteins.txt")), object_pairs_hook=OrderedDict)
Y = np.asarray(pickle.load(open(davis_file("Y", 10000), "rb"), encoding="latin1"), dtype=float)
Y = -np.log10(Y / 1e9)  # Kd (nM) -> pKd

DAVIS_SMILES = list(lig.values())
DAVIS_PROT_IDS = list(pro)
DAVIS_SEQS = [str(s).upper().strip() for s in pro.values()]
ii, jj = np.where(~np.isnan(Y))
davis_pairs = pd.DataFrame({"d": ii, "p": jj, "y": Y[ii, jj].astype(np.float32)})
info(f"{len(DAVIS_SMILES)} dawaiyan x {len(DAVIS_SEQS)} proteins = {len(davis_pairs):,} jode")
assert Y.shape == (len(DAVIS_SMILES), len(DAVIS_SEQS)), f"Y shape {Y.shape} mel nahi khata"


step("2. Repo se Davis ki ESM-2 650M embeddings")
ESM = np.load(os.path.join(MODELS, "davis_esm.npy"))
esm_names = json.load(open(os.path.join(MODELS, "davis_proteins.json")))
info(f"ESM matrix {ESM.shape} - {len(esm_names)} protein naam")
assert ESM.shape[0] == len(esm_names) == len(DAVIS_PROT_IDS), \
    f"ESM {ESM.shape[0]} / naam {len(esm_names)} / davis {len(DAVIS_PROT_IDS)} mel nahi khate"
if esm_names != DAVIS_PROT_IDS:
    order = {n: k for k, n in enumerate(esm_names)}
    missing = [n for n in DAVIS_PROT_IDS if n not in order]
    assert not missing, f"{len(missing)} protein ESM me nahi: {missing[:5]}"
    ESM = ESM[[order[n] for n in DAVIS_PROT_IDS]]
    info("ESM rows ko Davis ke order me laga diya")
ESM = ESM.astype(np.float32)


step("3. BindingDB (Kd + Ki) laate hain")
DV = "https://dataverse.harvard.edu/api/access/datafile/"
BDB_FILES = {"Kd": 4291555, "Ki": 4291556}


def read_bdb(tag, fid):
    path = download(DV + str(fid), os.path.join(WORK, f"bindingdb_{tag}.tab"), 100000)
    with open(path, "r", errors="replace") as fh:
        head = fh.readline()
    sep = "\t" if head.count("\t") >= 2 else ","
    info(f"{tag}: separator {'TAB' if sep == chr(9) else 'COMMA'}")
    df = pd.read_csv(path, sep=sep, low_memory=False)
    cmap = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("x1", "drug", "smiles"):
            cmap[c] = "smiles"
        elif cl in ("x2", "target", "sequence", "seq"):
            cmap[c] = "seq"
        elif cl == "y":
            cmap[c] = "Y"
    df = df.rename(columns=cmap)
    miss = {"smiles", "seq", "Y"} - set(df.columns)
    if miss:
        raise RuntimeError(f"{tag}: column {miss} nahi mile; file me hain {list(df.columns)[:12]}")
    df = df[["smiles", "seq", "Y"]].copy()
    df["assay"] = tag
    info(f"{tag}: {len(df):,} raw rows")
    return df


bdb = pd.concat([read_bdb(t, f) for t, f in BDB_FILES.items()], ignore_index=True)
bdb["Y"] = pd.to_numeric(bdb["Y"], errors="coerce")
bdb = bdb[(bdb.Y > 0) & bdb.smiles.notna() & bdb.seq.notna()]
bdb["pK"] = 9.0 - np.log10(bdb.Y.astype(float))
bdb = bdb[(bdb.pK > 2) & (bdb.pK < 12)]
bdb["seq"] = bdb.seq.astype(str).str.upper().str.strip()
info(f"saaf karne ke baad: {len(bdb):,} rows, {bdb.seq.nunique():,} unique proteins")


step("4. BindingDB ke proteins ko Davis se match karte hain")
# Exact match pehle, phir containment (isoform / tag ki wajah se thoda farak hota hai).
davis_by_seq = {s: k for k, s in enumerate(DAVIS_SEQS)}
uniq_seqs = bdb.seq.unique()
info(f"{len(uniq_seqs):,} unique BindingDB sequences ko match kar rahe hain...")

seq2idx = {}
for s in uniq_seqs:
    k = davis_by_seq.get(s)
    if k is not None:
        seq2idx[s] = k
info(f"exact match: {len(seq2idx):,}")

unmatched = [s for s in uniq_seqs if s not in seq2idx]
# containment: sirf lambe sequences pe, aur sabse lamba overlap chunte hain
davis_sorted = sorted(range(len(DAVIS_SEQS)), key=lambda k: -len(DAVIS_SEQS[k]))
for s in unmatched:
    if len(s) < 100:
        continue
    for k in davis_sorted:
        d = DAVIS_SEQS[k]
        if len(d) < 100:
            break
        if d in s or s in d:
            seq2idx[s] = k
            break
info(f"containment ke baad total: {len(seq2idx):,} / {len(uniq_seqs):,} sequences matched")

bdb["p"] = bdb.seq.map(seq2idx)
bdb = bdb[bdb.p.notna()].copy()
bdb["p"] = bdb.p.astype(int)
info(f"istemaal layak BindingDB rows: {len(bdb):,} "
     f"({bdb.p.nunique()} Davis proteins pe, {bdb.smiles.nunique():,} molecules)")
if bdb.p.nunique() < 20:
    print("\n!! Bahut kam proteins match hue - experiment ka matlab nahi banega.", flush=True)

# ek hi (molecule, protein, assay) ke kai measurements -> median
# assay ko rakhte hain: Kd aur Ki mila-jula data shor badha sakta hai, ye test karna hai
bdb = (bdb.groupby(["smiles", "p", "assay"], as_index=False)
          .agg(pK=("pK", "median"), n=("pK", "size")))
info(f"dedupe ke baad: {len(bdb):,} unique jode")
info("assay ke hisaab se: " + ", ".join(
    f"{k}={v:,}" for k, v in bdb.assay.value_counts().items()))


step("5. Molecule features (Morgan FP + RDKit descriptors + ChemBERTa)")
# Features seed se aazad hain, isliye ek hi baar banate hain aur sab seeds me reuse.
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
fp1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
DESC = Descriptors._descList
info(f"RDKit {len(DESC)} descriptors")


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m is not None else None


DAVIS_CANON = [canon(s) for s in DAVIS_SMILES]
bdb["canon"] = [canon(s) for s in bdb.smiles]
bdb = bdb[bdb.canon.notna()].reset_index(drop=True)

ALL_SMILES = list(dict.fromkeys([str(s) for s in DAVIS_SMILES] + list(bdb.smiles)))
info(f"{len(ALL_SMILES):,} unique molecules ke features banenge")


def rdkit_feats(smis):
    out, bad = [], 0
    width = 1024 + len(DESC)
    for k, s in enumerate(smis):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            out.append(np.zeros(width, np.float32)); bad += 1; continue
        d = []
        for _, fn in DESC:
            try:
                d.append(float(fn(m)))
            except Exception:
                d.append(np.nan)
        d = np.nan_to_num(np.clip(np.array(d, np.float64), -1e6, 1e6)).astype(np.float32)
        out.append(np.concatenate([fp1024.GetFingerprintAsNumPy(m).astype(np.float32), d]))
        if (k + 1) % 10000 == 0:
            print(f"   RDKit {k+1:,}/{len(smis):,}", flush=True)
    if bad:
        info(f"{bad} molecules padhe nahi gaye (zero kar diye)")
    return np.stack(out)


def chemberta(smis, batch=128):
    import torch
    from transformers import AutoTokenizer, AutoModel
    name = json.load(open(os.path.join(MODELS, "config.json")))["chem_name"]
    info(f"ChemBERTa: {name}")
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smis), batch):
            t = tok([str(x) for x in smis[i:i+batch]], return_tensors="pt",
                    padding=True, truncation=True, max_length=256)
            h = mdl(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
            if (i // batch) % 50 == 0:
                print(f"   ChemBERTa {i:,}/{len(smis):,}", flush=True)
    return np.concatenate(out).astype(np.float32)


def cached_npy(name, fn):
    p = os.path.join(WORK, name + ".npy")
    if os.path.exists(p):
        a = np.load(p)
        if len(a) == len(ALL_SMILES):
            info(f"cache: {name} {a.shape}")
            return a
        info(f"cache {name} purana hai ({len(a)} vs {len(ALL_SMILES)}) - dobara banega")
    t0 = time.time()
    a = fn()
    np.save(p, a)
    info(f"bana: {name} {a.shape} ({(time.time()-t0)/60:.1f} min)")
    return a


RD = cached_npy("rdkit_feats", lambda: rdkit_feats(ALL_SMILES))
CB = cached_npy("chemberta", lambda: chemberta(ALL_SMILES))
DRUG_X = np.hstack([RD, CB]).astype(np.float32)
del RD, CB
SMI2ROW = {s: k for k, s in enumerate(ALL_SMILES)}
info(f"DRUG_X {DRUG_X.shape} ({DRUG_X.nbytes/1e9:.2f} GB)")

davis_rows_all = np.array([SMI2ROW[str(s)] for s in DAVIS_SMILES])
bdb_rows_all = np.array([SMI2ROW[str(s)] for s in bdb.smiles])


step("6. Tools: CI aur training")
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.model_selection import GroupShuffleSplit


def concordance_index(y_true, y_pred, chunk=2000):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    num = den = 0.0
    for s in range(0, len(y_true), chunk):
        yt, yp = y_true[s:s+chunk, None], y_pred[s:s+chunk, None]
        mask = yt > y_true[None, :]
        diff = yp - y_pred[None, :]
        num += ((diff > 0) & mask).sum() + 0.5 * ((diff == 0) & mask).sum()
        den += mask.sum()
    return num / den if den else float("nan")


def build_xy(drug_rows, prot_idx, y, is_ki=None, with_assay=False):
    parts = [DRUG_X[drug_rows], ESM[prot_idx]]
    if with_assay:
        # 1 extra column: 0 = Kd jaisa saaf naap (Davis bhi Kd hai), 1 = Ki
        col = np.zeros((len(drug_rows), 1), np.float32) if is_ki is None \
            else np.asarray(is_ki, np.float32).reshape(-1, 1)
        parts.append(col)
    X = np.hstack(parts).astype(np.float32)
    return X, np.asarray(y, dtype=np.float32)


# Kya-kya aazma rahe hain. Sab me model/features/settings same; sirf data badalta hai.
VARIANTS = [
    ("B  all assays",   None,   False),   # jo pehle chalaya tha
    ("B2 sirf Kd",      "Kd",   False),   # ek hi tarah ka saaf naap
    ("B3 all + assay",  None,   True),    # mila-jula, par model ko bata do kaunsa naap hai
    ("B4 Kd + assay",   "Kd",   True),
]


def fit_predict(X, y, X_test):
    # CPU pe chalana hai, isliye v2 se halka. Dono models me BILKUL yahi settings.
    m = xgb.XGBRegressor(n_estimators=N_TREES, max_depth=8, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.5, tree_method="hist",
                         max_bin=128, n_jobs=os.cpu_count(), random_state=0)
    m.fit(X, y)
    return m.predict(X_test)


def metrics(y_test, p):
    return {"CI": float(concordance_index(y_test, p)),
            "MSE": float(np.mean((y_test - p) ** 2)),
            "Pearson": float(pearsonr(y_test, p)[0])}


step(f"7. {N_SEEDS} seeds pe chalate hain (ek split ka result tukka ho sakta hai)")
rows = []
for seed in range(N_SEEDS):
    t_seed = time.time()
    print(f"\n--- seed {seed} " + "-" * 50, flush=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(gss.split(davis_pairs, groups=davis_pairs.d))
    train_pairs = davis_pairs.iloc[tr_idx]
    test_pairs = davis_pairs.iloc[te_idx]
    test_drugs = sorted(set(test_pairs.d) - set(train_pairs.d))
    test_canon = {DAVIS_CANON[d] for d in test_drugs if DAVIS_CANON[d]}

    # leakage: test dawaiyan BindingDB se bhi hatao
    keep = ~bdb.canon.isin(test_canon)
    # Davis train ke saath duplicate jode bhi hatao
    dtk = set(zip([DAVIS_CANON[d] for d in train_pairs.d], train_pairs.p))
    keep &= np.array([(c, p) not in dtk for c, p in zip(bdb.canon, bdb.p)])
    sub = bdb[keep]
    budget = MAX_TOTAL_PAIRS - len(train_pairs)
    if len(sub) > budget:
        sub = sub.sample(budget, random_state=seed)
    info(f"test: {len(test_pairs):,} jode / {len(test_drugs)} nayi dawaiyan · "
         f"BindingDB se jude: {len(sub):,}")

    for with_assay in (False, True):
        tag_a = "A  Davis sirf" + (" + assay" if with_assay else "")
        X_test, y_test = build_xy(davis_rows_all[test_pairs.d.values],
                                  test_pairs.p.values, test_pairs.y.values,
                                  None, with_assay)
        X_a, y_a = build_xy(davis_rows_all[train_pairs.d.values],
                            train_pairs.p.values, train_pairs.y.values,
                            None, with_assay)
        r = metrics(y_test, fit_predict(X_a, y_a, X_test))
        r.update(seed=seed, model=tag_a, n_train=int(len(y_a)))
        rows.append(r)
        print(f"   {tag_a:18s} n={len(y_a):>6,}  CI={r['CI']:.4f} "
              f"MSE={r['MSE']:.4f} Pearson={r['Pearson']:.4f}", flush=True)

        for tag, only_assay, wa in VARIANTS:
            if wa != with_assay:
                continue
            s2 = sub if only_assay is None else sub[sub.assay == only_assay]
            if len(s2) == 0:
                print(f"   {tag}: koi jodi nahi - skip", flush=True)
                continue
            X_extra, y_extra = build_xy(bdb_rows_all[s2.index.values], s2.p.values,
                                        s2.pK.values, (s2.assay == "Ki").values, wa)
            X_b = np.vstack([X_a, X_extra]); y_b = np.concatenate([y_a, y_extra])
            del X_extra
            r = metrics(y_test, fit_predict(X_b, y_b, X_test))
            r.update(seed=seed, model=tag, n_train=int(len(y_b)))
            rows.append(r)
            print(f"   {tag:18s} n={len(y_b):>6,}  CI={r['CI']:.4f} "
                  f"MSE={r['MSE']:.4f} Pearson={r['Pearson']:.4f}", flush=True)
            del X_b
        del X_a, X_test
    print(f"   seed {seed} done ({(time.time()-t_seed)/60:.1f} min)", flush=True)

    # har seed ke baad save - run adhoora toote to bhi kaam na jaye
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "per_seed.csv"), index=False)


step("8. NATEEJA")
df = pd.DataFrame(rows)
BASE_PLAIN = "A  Davis sirf"
BASE_ASSAY = "A  Davis sirf + assay"

agg = (df.groupby("model")[["CI", "MSE", "Pearson"]].agg(["mean", "std"]).round(4))
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.join(df.groupby("model")["n_train"].max().rename("n_train"))
print(agg.to_string(), flush=True)

piv = df.pivot(index="seed", columns="model", values="CI")
print("\n   Har seed ka CI:")
print(piv.round(4).to_string(), flush=True)

# Har variant ko apne sahi baseline se tolo
comp = []
for tag, only_assay, wa in VARIANTS:
    base = BASE_ASSAY if wa else BASE_PLAIN
    if tag not in piv.columns or base not in piv.columns:
        continue
    d = (piv[tag] - piv[base]).dropna()
    if len(d) == 0:
        continue
    comp.append({
        "variant": tag,
        "baseline": base,
        "seeds": len(d),
        "jeete": int((d > 0).sum()),
        "CI_farak": round(float(d.mean()), 4),
        "std": round(float(d.std()), 4),
    })
cmp_df = pd.DataFrame(comp)
print("\n   Baseline se tulna (CI_farak me + = data se faida):")
print(cmp_df.to_string(index=False), flush=True)

# Sabse achha variant chuno, par sirf tab jab farak shor se bada ho
best, verdict = None, None
if len(cmp_df):
    cand = cmp_df.sort_values("CI_farak", ascending=False).iloc[0]
    solid = (cand.CI_farak > max(cand["std"], 0.005)
             and cand.jeete >= max(4, cand.seeds - 1))
    if solid:
        best = cand.variant
        verdict = (f"DATA se faida hua - '{cand.variant}' ne {cand.jeete}/{cand.seeds} "
                   f"seeds jeete, CI {cand.CI_farak:+.4f} (±{cand['std']:.4f}). "
                   f"Yahi tareeka aage badhane layak hai.")
    else:
        verdict = (f"KISI BHI tarah ke data se pakka faida nahi. Sabse achha variant "
                   f"'{cand.variant}' bhi sirf {cand.jeete}/{cand.seeds} seeds jeeta "
                   f"(CI {cand.CI_farak:+.4f} ±{cand['std']:.4f}) - farak shor ke andar hai. "
                   f"Matlab mila-jula assay data wajah nahi thi; bottleneck PROTEIN "
                   f"REPRESENTATION hai (pocket-aware raasta).")
print(f"\n   >>> {verdict}", flush=True)

summary = {
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "total_minutes": round((time.time() - T_START) / 60, 1),
    "design": ("Model/features/hyperparameters sab variants me same; sirf training "
               "data badla. Baseline A = sirf Davis."),
    "test_set": "Davis cold-drug (GroupShuffleSplit on drug, 20%), per seed",
    "n_seeds": N_SEEDS,
    "bindingdb_proteins_matched": int(bdb.p.nunique()),
    "assay_counts": {str(k): int(v) for k, v in bdb.assay.value_counts().items()},
    "comparisons": comp,
    "best_variant": best,
    "verdict": verdict,
    "per_seed": rows,
}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
df.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)

with open(os.path.join(OUT, "RESULT.md"), "w") as f:
    f.write("# v3 experiment: kya zyada data se binding model behtar hota hai?\n\n")
    f.write("**Design:** model, features, hyperparameters sab variants me bilkul same. "
            "Sirf training data badla.\n\n")
    f.write(f"- Baseline **A** = sirf Davis train pairs\n")
    f.write("- **B** = A + BindingDB (Kd+Ki) · **B2** = A + sirf Kd · "
            "**B3/B4** = wahi, par model ko assay type bata ke\n")
    f.write(f"- Test: Davis **cold-drug** split, {N_SEEDS} seeds (mean ± std)\n")
    f.write(f"- BindingDB se match hue proteins: {bdb.p.nunique()}\n")
    f.write("- Leakage hataya: test dawaiyon ke canonical SMILES, aur Davis ke "
            "saath duplicate jode\n\n")
    f.write("## Har variant, average\n\n")
    f.write(agg.to_markdown())
    f.write("\n\n## Baseline se tulna\n\n")
    f.write(cmp_df.to_markdown(index=False))
    f.write("\n\n## Har seed ka CI\n\n")
    f.write(piv.round(4).to_markdown())
    f.write(f"\n\n**{verdict}**\n")

print(f"\n   Files: {OUT}/RESULT.md, summary.json, per_seed.csv")
print(f"   Total: {(time.time()-T_START)/60:.1f} min", flush=True)
