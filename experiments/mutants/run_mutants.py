"""
Sawaal: Davis ke mutants ko unki ASLI sequence dene se model behtar hota hai?

Background: Davis me 442 protein naam hain par sirf 379 unique sequences. Saare
45 mutation-labelled proteins (ABL1(E255K), EGFR(L858R), ...) ko wild-type ki
hubahu sequence di gayi hai. Matlab model ko ek hi input mil raha hai aur
alag-alag jawab maanga ja raha hai. Ye model ki kami nahi, data ka contradiction
hai - aur ye poore model ki accuracy ki chhat neeche kheenchta hai.

Isliye naam se mutation parse karke asli sequence banate hain (mutate.py), aur
do taraf se naapte hain:

  WT   = jaisa Davis deta hai (mutants ko wild-type sequence)
  MUT  = sudhra hua (mutants ko unki asli sequence)

Dono me model, features, hyperparameters, splits - sab same. Sirf protein
sequence badalti hai, isliye jo farak aayega wo isi sudhaar ka hoga.

ESM ki ek ehtiyaat: repo me padi davis_esm.npy GPU pe half precision me bani
thi. Yahan CPU pe float32 me banegi. Dono ko mila dena ek chhupa hua farak
paida karta (WT purani, MUT nayi) - isliye SAARI embeddings yahin dobara
banate hain, dono taraf ke liye ek hi tareeke se.
"""
import json, os, pickle, shutil, sys, time, urllib.request
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutate import build as build_mutants

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS = os.path.join(REPO, "models")
WORK = os.path.join(HERE, "work")
OUT = os.path.join(HERE, "results")
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

N_SEEDS = int(os.environ.get("N_SEEDS", 5))
N_TREES = int(os.environ.get("N_TREES", 400))
ESM_NAME = os.environ.get("ESM_NAME", "facebook/esm2_t33_650M_UR50D")


def step(m):
    print(f"\n{'='*70}\n[{(time.time()-T_START)/60:5.1f} min] {m}\n{'='*70}", flush=True)


def info(m):
    print(f"   {m}", flush=True)


# ------------------------------------------------------------------ Davis
step("1. Davis dataset")
BASES = ["https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/",
         "https://raw.githubusercontent.com/thinng/GraphDTA/master/data/davis/"]


def davis_file(name, min_bytes=1000):
    path = os.path.join(WORK, f"davis_{name}")
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        info(f"cache: davis_{name}")
        return path
    errs = []
    for base in BASES:
        for attempt in range(2):
            try:
                req = urllib.request.Request(base + name,
                                             headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
                    shutil.copyfileobj(r, f)
                if os.path.getsize(path) >= min_bytes:
                    info(f"downloaded davis_{name}")
                    return path
                errs.append("file too small")
            except Exception as e:
                errs.append(f"{type(e).__name__}: {e}")
            time.sleep(3)
    raise RuntimeError(f"davis/{name} nahi mila: {errs}")


lig = json.load(open(davis_file("ligands_can.txt")), object_pairs_hook=OrderedDict)
pro = json.load(open(davis_file("proteins.txt")), object_pairs_hook=OrderedDict)
Y = np.asarray(pickle.load(open(davis_file("Y", 10000), "rb"), encoding="latin1"),
               dtype=float)
Y = -np.log10(Y / 1e9)
SMILES = list(lig.values())
NAMES = list(pro)
SEQ_WT = [str(s).upper().strip() for s in pro.values()]
ii, jj = np.where(~np.isnan(Y))
pairs = pd.DataFrame({"d": ii, "p": jj, "y": Y[ii, jj].astype(np.float32)})
info(f"{len(SMILES)} dawaiyan x {len(NAMES)} proteins = {len(pairs):,} jode")


# ------------------------------------------------------------- mutations
step("2. Naam se mutation parse karke asli sequence banate hain")
new_pro, report = build_mutants(OrderedDict(zip(NAMES, SEQ_WT)))
SEQ_MUT = [new_pro[n] for n in NAMES]
rep = pd.DataFrame(report)
info("status: " + str(dict(Counter(rep.status))))
info(f"sequences badli: {int(rep.changed.sum())}")
info(f"unique sequences: {len(set(SEQ_WT))} -> {len(set(SEQ_MUT))}")
for _, r in rep[rep.status.isin(["partial", "skip"])].iterrows():
    info(f"  {r['status']:8s} {r['name']:34s} {r['detail']}")
rep.to_csv(os.path.join(OUT, "mutation_report.csv"), index=False)


# -------------------------------------------------- contradictory labels
step("3. Contradictory labels naapte hain (ek hi input, alag jawab)")


def contradictions(seqs, tag):
    key = [(int(d), seqs[int(p)]) for d, p in zip(pairs.d, pairs.p)]
    g = pd.DataFrame({"k": key, "y": pairs.y.values}).groupby("k")["y"]
    n_multi = int((g.nunique() > 1).sum())
    affected = int(g.size()[g.nunique() > 1].sum())
    spread = float(g.std()[g.nunique() > 1].mean())
    rng = float(g.max()[g.nunique() > 1].sub(g.min()[g.nunique() > 1]).mean())
    info(f"{tag:5s}: {n_multi:,} aise (dawa, sequence) jode jinke 2+ alag jawab hain · "
         f"{affected:,} rows ({100*affected/len(pairs):.1f}%) · "
         f"andar ka std {spread:.3f} · average range {rng:.3f} pKd")
    return {"tag": tag, "groups": n_multi, "rows": affected,
            "pct_rows": round(100 * affected / len(pairs), 2),
            "mean_std": round(spread, 4), "mean_range": round(rng, 4)}


contra = [contradictions(SEQ_WT, "WT"), contradictions(SEQ_MUT, "MUT")]
json.dump(contra, open(os.path.join(OUT, "contradictions.json"), "w"), indent=2)


# ------------------------------------------------------------------- ESM
step("4. ESM-2 embeddings (dono taraf ke liye ek hi tareeke se)")
ALL_SEQ = sorted(set(SEQ_WT) | set(SEQ_MUT))
info(f"{len(ALL_SEQ)} unique sequences embed karni hain ({ESM_NAME})")

ESM_CACHE = os.path.join(WORK, "esm_" + ESM_NAME.split("/")[-1] + ".npz")
if os.path.exists(ESM_CACHE):
    z = np.load(ESM_CACHE, allow_pickle=True)
    cached = {s: v for s, v in zip(list(z["seqs"]), z["emb"])}
    info(f"cache me {len(cached)} embeddings")
else:
    cached = {}

todo = [s for s in ALL_SEQ if s not in cached]
if todo:
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(ESM_NAME)
    mdl = AutoModel.from_pretrained(ESM_NAME).eval()
    info(f"{len(todo)} nayi embeddings banani hain")
    t0 = time.time()
    with torch.no_grad():
        for k, s in enumerate(todo):
            chunks = [s[i:i+1000] for i in range(0, len(s), 1000)] or [s]
            vs = []
            for c in chunks:
                t = tok(c, return_tensors="pt")
                vs.append(mdl(**t).last_hidden_state[0, 1:-1].float().mean(0).numpy())
            cached[s] = np.mean(vs, 0).astype(np.float32)
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                el = (time.time() - t0) / 60
                left = el / (k + 1) * (len(todo) - k - 1)
                print(f"   ESM {k+1}/{len(todo)}  ({el:.1f} min beeta · "
                      f"~{left:.0f} min baaki)", flush=True)
                seqs = list(cached)
                np.savez_compressed(ESM_CACHE, seqs=np.array(seqs, dtype=object),
                                    emb=np.stack([cached[x] for x in seqs]))
    del mdl

ESM = {"WT": np.stack([cached[s] for s in SEQ_WT]).astype(np.float32),
       "MUT": np.stack([cached[s] for s in SEQ_MUT]).astype(np.float32)}
info(f"ESM WT {ESM['WT'].shape} · MUT {ESM['MUT'].shape}")
same = int(sum(1 for a, b in zip(SEQ_WT, SEQ_MUT) if a == b))
info(f"{442-same if len(NAMES)==442 else len(NAMES)-same} proteins ki embedding badli")


# -------------------------------------------------------- drug features
step("5. Molecule features")
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
fp1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
DESC = Descriptors._descList
DRUG_CACHE = os.path.join(WORK, "drug_feats.npy")
if os.path.exists(DRUG_CACHE):
    DRUG_X = np.load(DRUG_CACHE)
    info(f"cache: drug feats {DRUG_X.shape}")
else:
    rows_ = []
    for s in SMILES:
        m = Chem.MolFromSmiles(str(s))
        d = []
        for _, fn in DESC:
            try:
                d.append(float(fn(m)))
            except Exception:
                d.append(np.nan)
        d = np.nan_to_num(np.clip(np.array(d, np.float64), -1e6, 1e6)).astype(np.float32)
        rows_.append(np.concatenate(
            [fp1024.GetFingerprintAsNumPy(m).astype(np.float32), d]))
    RD = np.stack(rows_)
    import torch
    from transformers import AutoTokenizer, AutoModel
    cname = json.load(open(os.path.join(MODELS, "config.json")))["chem_name"]
    tok = AutoTokenizer.from_pretrained(cname)
    cm = AutoModel.from_pretrained(cname).eval()
    with torch.no_grad():
        t = tok([str(x) for x in SMILES], return_tensors="pt", padding=True,
                truncation=True, max_length=256)
        h = cm(**t).last_hidden_state
        msk = t["attention_mask"].unsqueeze(-1).float()
        CB = ((h * msk).sum(1) / msk.sum(1)).numpy().astype(np.float32)
    DRUG_X = np.hstack([RD, CB]).astype(np.float32)
    np.save(DRUG_CACHE, DRUG_X)
    info(f"bana: drug feats {DRUG_X.shape}")


# ------------------------------------------------------------- training
step(f"6. Training - {N_SEEDS} seeds x 2 splits x WT/MUT")
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.model_selection import GroupShuffleSplit


def ci(y_true, y_pred, chunk=2000):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    num = den = 0.0
    for s in range(0, len(y_true), chunk):
        yt, yp = y_true[s:s+chunk, None], y_pred[s:s+chunk, None]
        mask = yt > y_true[None, :]
        diff = yp - y_pred[None, :]
        num += ((diff > 0) & mask).sum() + 0.5 * ((diff == 0) & mask).sum()
        den += mask.sum()
    return num / den if den else float("nan")


MUT_P = {k for k, n in enumerate(NAMES) if "(" in n and new_pro[n] != SEQ_WT[k]}
info(f"{len(MUT_P)} proteins jinki sequence sudhri - unke paire alag se bhi naapenge")

rows = []
for seed in range(N_SEEDS):
    for split in ("random", "cold-target"):
        if split == "random":
            rs = np.random.default_rng(seed)
            perm = rs.permutation(len(pairs))
            cut = int(0.8 * len(pairs))
            tr, te = perm[:cut], perm[cut:]
        else:
            gss = GroupShuffleSplit(1, test_size=0.2, random_state=seed)
            tr, te = next(gss.split(pairs, groups=pairs.p))
        trp, tep = pairs.iloc[tr], pairs.iloc[te]
        mut_mask = tep.p.isin(MUT_P).values

        for arm in ("WT", "MUT"):
            E = ESM[arm]
            Xtr = np.hstack([DRUG_X[trp.d.values], E[trp.p.values]]).astype(np.float32)
            Xte = np.hstack([DRUG_X[tep.d.values], E[tep.p.values]]).astype(np.float32)
            m = xgb.XGBRegressor(n_estimators=N_TREES, max_depth=8, learning_rate=0.08,
                                 subsample=0.8, colsample_bytree=0.5,
                                 tree_method="hist", max_bin=128,
                                 n_jobs=os.cpu_count(), random_state=0)
            m.fit(Xtr, trp.y.values)
            p = m.predict(Xte)
            r = {"seed": seed, "split": split, "arm": arm,
                 "CI": float(ci(tep.y.values, p)),
                 "MSE": float(np.mean((tep.y.values - p) ** 2)),
                 "Pearson": float(pearsonr(tep.y.values, p)[0])}
            if mut_mask.sum() > 50:
                r["CI_mutants"] = float(ci(tep.y.values[mut_mask], p[mut_mask]))
                r["MSE_mutants"] = float(np.mean(
                    (tep.y.values[mut_mask] - p[mut_mask]) ** 2))
            rows.append(r)
            print(f"   seed {seed} {split:12s} {arm:3s}  CI={r['CI']:.4f} "
                  f"MSE={r['MSE']:.4f} Pearson={r['Pearson']:.4f}"
                  + (f"  [mutants CI={r['CI_mutants']:.4f}]"
                     if "CI_mutants" in r else ""), flush=True)
            del Xtr, Xte
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "per_seed.csv"), index=False)


# -------------------------------------------------------------- verdict
step("7. NATEEJA")
df = pd.DataFrame(rows)
lines = []
for split in df.split.unique():
    sub = df[df.split == split]
    piv = sub.pivot(index="seed", columns="arm", values="CI")
    d = (piv["MUT"] - piv["WT"]).dropna()
    se = d.std() / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    lines.append({"split": split, "metric": "CI", "WT": round(piv["WT"].mean(), 4),
                  "MUT": round(piv["MUT"].mean(), 4), "delta": round(d.mean(), 4),
                  "jeete": f"{int((d>0).sum())}/{len(d)}",
                  "t": round(d.mean()/se, 2) if se else None})
    if "CI_mutants" in sub.columns and sub.CI_mutants.notna().any():
        pm = sub.pivot(index="seed", columns="arm", values="CI_mutants")
        dm = (pm["MUT"] - pm["WT"]).dropna()
        sem = dm.std() / np.sqrt(len(dm)) if len(dm) > 1 else float("nan")
        lines.append({"split": split, "metric": "CI (sirf mutant proteins)",
                      "WT": round(pm["WT"].mean(), 4), "MUT": round(pm["MUT"].mean(), 4),
                      "delta": round(dm.mean(), 4),
                      "jeete": f"{int((dm>0).sum())}/{len(dm)}",
                      "t": round(dm.mean()/sem, 2) if sem else None})
summ = pd.DataFrame(lines)
print(summ.to_string(index=False), flush=True)

key = summ[(summ.split == "random") & (summ.metric.str.contains("mutant"))]
best = key.iloc[0] if len(key) else summ.iloc[0]
solid = best["t"] is not None and abs(best["t"]) >= 2 and best["delta"] > 0
verdict = (f"SUDHAAR ASLI HAI - mutant proteins pe CI {best['delta']:+.4f} "
           f"(t={best['t']}), {best['jeete']} seeds. Asli mutant sequences dena "
           f"faydemand hai." if solid else
           f"Sudhaar shor ke andar hai (CI {best['delta']:+.4f}, t={best['t']}, "
           f"{best['jeete']}). Contradictory labels hat gaye, par CI pe pakka "
           f"farak nahi dikha is paimane pe.")
print(f"\n   >>> {verdict}", flush=True)

json.dump({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "total_minutes": round((time.time()-T_START)/60, 1),
           "n_seeds": N_SEEDS, "esm": ESM_NAME,
           "unique_seqs_before": len(set(SEQ_WT)), "unique_seqs_after": len(set(SEQ_MUT)),
           "sequences_changed": int(rep.changed.sum()),
           "mutation_status": dict(Counter(rep.status)),
           "contradictions": contra, "summary": lines, "verdict": verdict,
           "per_seed": rows},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)

with open(os.path.join(OUT, "RESULT.md"), "w") as f:
    f.write("# Asli mutant sequences dene se model behtar hota hai?\n\n")
    f.write("Davis mutants ko unke wild-type ki hubahu sequence deta hai. "
            "Naam se mutation parse karke asli sequence banayi, phir dono taraf "
            "ek hi model/features/splits pe naapa.\n\n")
    f.write(f"- unique sequences: **{len(set(SEQ_WT))} -> {len(set(SEQ_MUT))}** "
            f"({int(rep.changed.sum())} badli)\n")
    f.write(f"- mutation parse status: {dict(Counter(rep.status))}\n")
    f.write(f"- ESM: {ESM_NAME}, dono taraf CPU float32 me (purani GPU-half "
            f"embeddings mix nahi ki)\n")
    f.write(f"- {N_SEEDS} seeds, splits: random aur cold-target\n\n")
    f.write("## Contradictory labels (ek hi input, alag jawab)\n\n")
    f.write(pd.DataFrame(contra).to_markdown(index=False))
    f.write("\n\n## Nateeja (t = mean/SE, |t|>=2 = shor se bada)\n\n")
    f.write(summ.to_markdown(index=False))
    f.write(f"\n\n**{verdict}**\n")

print(f"\n   Files: {OUT}/RESULT.md, summary.json, per_seed.csv, mutation_report.csv")
print(f"   Total: {(time.time()-T_START)/60:.1f} min", flush=True)
