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

# ek hi (molecule, protein) ke kai measurements -> median
bdb = (bdb.groupby(["smiles", "p"], as_index=False)
          .agg(pK=("pK", "median"), n=("pK", "size")))
info(f"dedupe ke baad: {len(bdb):,} unique jode")


step("5. Test split - dono models ke liye bilkul ek jaisa")
# Cold-drug: test ki dawaiyan training me kabhi nahi aayengi (na Davis se, na BindingDB se).
# Yahi v2 ki sabse kamzor jagah thi (CI 0.675).
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(davis_pairs, groups=davis_pairs.d))
train_pairs = davis_pairs.iloc[tr_idx].reset_index(drop=True)
test_pairs = davis_pairs.iloc[te_idx].reset_index(drop=True)
TEST_DRUG_IDS = sorted(set(test_pairs.d) - set(train_pairs.d))
info(f"train {len(train_pairs):,} jode / test {len(test_pairs):,} jode")
info(f"test ki dawaiyan: {len(TEST_DRUG_IDS)} (training me ek bhi nahi)")

# leakage band karo: test dawaiyon ke SMILES BindingDB se bhi hatao
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m is not None else None


DAVIS_CANON = [canon(s) for s in DAVIS_SMILES]   # 68 dawaiyan, ek hi baar
test_canon = {DAVIS_CANON[d] for d in TEST_DRUG_IDS if DAVIS_CANON[d]}
info(f"{len(test_canon)} test SMILES canonical form me")

bdb["canon"] = [canon(s) for s in bdb.smiles]
before = len(bdb)
bdb = bdb[bdb.canon.notna() & ~bdb.canon.isin(test_canon)].reset_index(drop=True)
info(f"leakage hatane se {before - len(bdb):,} BindingDB rows gaye -> {len(bdb):,} bache")

# Davis train pairs bhi BindingDB me ho sakte hain - unko BindingDB side se hatao
davis_train_keys = set(zip([DAVIS_CANON[d] for d in train_pairs.d], train_pairs.p))
mask = [(c, p) not in davis_train_keys for c, p in zip(bdb.canon, bdb.p)]
dup = len(bdb) - sum(mask)
bdb = bdb[mask].reset_index(drop=True)
info(f"Davis ke saath duplicate {dup:,} jode hataye -> BindingDB se {len(bdb):,} naye jode")

budget = MAX_TOTAL_PAIRS - len(train_pairs)
if len(bdb) > budget:
    bdb = bdb.sample(budget, random_state=SEED).reset_index(drop=True)
    info(f"memory ke liye {budget:,} jode tak seemit kiya")


step("6. Molecule features (Morgan FP + RDKit descriptors + ChemBERTa)")
from rdkit.Chem import Descriptors, rdFingerprintGenerator

fp1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
DESC = Descriptors._descList
info(f"RDKit {len(DESC)} descriptors")

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
        info(f"cache: {name} {a.shape}")
        return a
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


step("7. Training matrices banate hain")


def build_xy(drug_rows, prot_idx, y):
    X = np.hstack([DRUG_X[drug_rows], ESM[prot_idx]]).astype(np.float32)
    return X, np.asarray(y, dtype=np.float32)


davis_rows = np.array([SMI2ROW[str(DAVIS_SMILES[d])] for d in davis_pairs.d])
tr_rows = davis_rows[tr_idx]
te_rows = davis_rows[te_idx]

X_test, y_test = build_xy(te_rows, test_pairs.p.values, test_pairs.y.values)
X_a, y_a = build_xy(tr_rows, train_pairs.p.values, train_pairs.y.values)
info(f"A (Davis sirf)   : {X_a.shape}")

bdb_rows = np.array([SMI2ROW[str(s)] for s in bdb.smiles])
X_b_extra, y_b_extra = build_xy(bdb_rows, bdb.p.values, bdb.pK.values)
X_b = np.vstack([X_a, X_b_extra])
y_b = np.concatenate([y_a, y_b_extra])
del X_b_extra
info(f"B (Davis+BindingDB): {X_b.shape}  (+{len(y_b)-len(y_a):,} jode = "
     f"{len(y_b)/max(len(y_a),1):.1f}x data)")
info(f"test             : {X_test.shape}")


step("8. Dono models train karte hain (bilkul same settings)")
import xgboost as xgb


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


def train_eval(tag, X, y):
    t0 = time.time()
    # CPU pe chalana hai, isliye v2 se halka: 400 trees, max_bin 128.
    # Dono models me BILKUL yahi settings hain, to muqabla jaayaz rehta hai.
    m = xgb.XGBRegressor(n_estimators=N_TREES, max_depth=8, learning_rate=0.08,
                         subsample=0.8, colsample_bytree=0.5, tree_method="hist",
                         max_bin=128, n_jobs=os.cpu_count(), random_state=SEED)
    m.fit(X, y)
    p = m.predict(X_test)
    from scipy.stats import pearsonr
    r = {
        "model": tag,
        "n_train": int(len(y)),
        "CI": float(concordance_index(y_test, p)),
        "MSE": float(np.mean((y_test - p) ** 2)),
        "Pearson": float(pearsonr(y_test, p)[0]),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    print(f"   {tag:22s} CI={r['CI']:.4f}  MSE={r['MSE']:.4f}  "
          f"Pearson={r['Pearson']:.4f}  ({r['minutes']} min)", flush=True)
    return r


rows = [train_eval("A: Davis sirf", X_a, y_a)]
del X_a
rows.append(train_eval("B: Davis+BindingDB", X_b, y_b))
del X_b


step("9. NATEEJA")
res = pd.DataFrame(rows)
ci_a, ci_b = res.CI[0], res.CI[1]
delta = ci_b - ci_a
res_tbl = res[["model", "n_train", "CI", "MSE", "Pearson"]].round(4)
print(res_tbl.to_string(index=False), flush=True)
print(f"\n   Davis cold-drug test set, {len(y_test):,} jode, {len(TEST_DRUG_IDS)} nayi dawaiyan")
print(f"   CI me farak: {delta:+.4f}", flush=True)
if delta > 0.01:
    verdict = "DATA se faida hua - BindingDB jodne se model behtar hua."
elif delta < -0.01:
    verdict = "DATA se nuksan hua - zyada data ne model kharab kiya."
else:
    verdict = ("DATA se koi farak nahi - bottleneck data nahi, "
               "protein representation hai (pocket-aware wala raasta).")
print(f"\n   >>> {verdict}", flush=True)

summary = {
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "total_minutes": round((time.time() - T_START) / 60, 1),
    "test_set": "Davis cold-drug (GroupShuffleSplit on drug, seed 0, 20%)",
    "test_pairs": int(len(y_test)),
    "test_drugs": len(TEST_DRUG_IDS),
    "bindingdb_proteins_matched": int(bdb.p.nunique()),
    "bindingdb_pairs_added": int(len(y_b) - len(y_a)),
    "delta_CI": round(float(delta), 4),
    "verdict": verdict,
    "results": rows,
}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
res.to_csv(os.path.join(OUT, "results.csv"), index=False)

with open(os.path.join(OUT, "RESULT.md"), "w") as f:
    f.write("# v3 experiment: kya zyada data se binding model behtar hota hai?\n\n")
    f.write(f"- Test set: **Davis cold-drug** ({len(y_test):,} jode, "
            f"{len(TEST_DRUG_IDS)} nayi dawaiyan jo training me nahi thi)\n")
    f.write(f"- BindingDB se jude jode: **{len(y_b)-len(y_a):,}** "
            f"({bdb.p.nunique()} proteins pe)\n")
    f.write(f"- Model, features, hyperparameters: **dono me same** (sirf data badla)\n\n")
    f.write(res_tbl.to_markdown(index=False))
    f.write(f"\n\n**CI farak: {delta:+.4f}**\n\n**{verdict}**\n")

print(f"\n   Files: {OUT}/summary.json, results.csv, RESULT.md")
print(f"   Total time: {(time.time()-T_START)/60:.1f} min", flush=True)
