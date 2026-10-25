"""
Repurposing table dobara banata hai, taaki app me dikh sake.

Pehle ye Colab me bana tha aur nateeja Google Drive me pada hai - repo me nahi.
Isliye yahan ChEMBL se dawaiyon ki library laate hain, features banate hain, aur
repo ke v2 binding model se saare Davis targets pe score karte hain.

Output: ek chhoti CSV (har target ke top N) jo repo me commit hoti hai aur app
seedha padh leti hai. Poori features (10k x 1600 floats) commit NAHI karte - wo
80+ MB ki hoti hain aur app ki 1 GB RAM me nahi khulti.

Imaandaari ki baat jo output me bhi jaati hai: "known_for_target" column batata
hai ki wo dawa us target ke liye pehle se jaani-pehchani hai (ChEMBL mechanism
data se). Jo top pe aaye aur known ho - wo model ka sahi hona dikhata hai, nayi
khoj nahi. Pehle is farak ko na dikhane se hi "naye candidates" ka bharam bana tha.
"""
import json, os, time, urllib.parse, urllib.request

import numpy as np
import pandas as pd

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS = os.path.join(REPO, "models")
WORK = os.path.join(HERE, "work")
OUT = os.path.join(REPO, "data")
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

MIN_PHASE = int(os.environ.get("MIN_PHASE", 1))
TOP_N = int(os.environ.get("TOP_N", 50))
API = "https://www.ebi.ac.uk/chembl/api/data"


def step(m):
    print(f"\n{'='*70}\n[{(time.time()-T0)/60:5.1f} min] {m}\n{'='*70}", flush=True)


def info(m):
    print(f"   {m}", flush=True)


def get_json(url, tries=4):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"ChEMBL se jawab nahi mila: {url} ({last})")


# ------------------------------------------------------------- 1. library
step("1. ChEMBL se insaan pe test ho chuki dawaiyan")
RAW = os.path.join(WORK, "drug_library_raw.csv")
if os.path.exists(RAW) and len(pd.read_csv(RAW)) > 2000:
    raw = pd.read_csv(RAW)
    info(f"cache: {len(raw):,} rows")
else:
    rows, offset = [], 0
    while True:
        page = get_json(f"{API}/molecule.json?max_phase__gte={MIN_PHASE}"
                        f"&limit=1000&offset={offset}")
        got = page.get("molecules", [])
        if not got:
            break
        for m in got:
            smi = (m.get("molecule_structures") or {}).get("canonical_smiles")
            if not smi:
                continue
            rows.append({"chembl_id": m.get("molecule_chembl_id"),
                         "name": (m.get("pref_name") or "").title(),
                         "smiles": smi, "max_phase": m.get("max_phase")})
        offset += 1000
        print(f"   {len(rows):,} molecules...", flush=True)
        if offset > 60000:
            break
    raw = pd.DataFrame(rows)
    raw.to_csv(RAW, index=False)
assert len(raw) > 2000, f"sirf {len(raw)} mile - ChEMBL check karo"
info(f"{len(raw):,} molecules (phase {MIN_PHASE}+)")


# ---------------------------------------------------------- 2. salt forms
step("2. Salt forms jodte hain (Erlotinib + Erlotinib HCl = ek dawa)")
from rdkit import Chem, RDLogger, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
CLEAN = os.path.join(WORK, "drug_library_clean.csv")
if os.path.exists(CLEAN) and len(pd.read_csv(CLEAN)) > 1000:
    drugs = pd.read_csv(CLEAN)
    info(f"cache: {len(drugs):,} dawaiyan")
else:
    chooser = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()
    keys, parents = [], []
    for i, s in enumerate(raw.smiles):
        try:
            m = Chem.MolFromSmiles(str(s))
            if m is None:
                keys.append(None); parents.append(None); continue
            m = uncharger.uncharge(chooser.choose(m))
            keys.append(inchi.MolToInchiKey(m)[:14])
            parents.append(Chem.MolToSmiles(m))
        except Exception:
            keys.append(None); parents.append(None)
        if (i + 1) % 2000 == 0:
            print(f"   {i+1:,}/{len(raw):,}", flush=True)
    t = raw.copy(); t["key"] = keys; t["parent_smiles"] = parents
    t = t.dropna(subset=["key", "parent_smiles"])
    t["namelen"] = t.name.fillna("").str.len()
    t["has_name"] = (t.namelen > 0).astype(int)
    t = t.sort_values(["key", "has_name", "max_phase", "namelen"],
                      ascending=[True, False, False, True])
    drugs = (t.groupby("key", as_index=False).first()
              [["chembl_id", "name", "parent_smiles", "max_phase"]]
              .rename(columns={"parent_smiles": "smiles"}).reset_index(drop=True))
    drugs.to_csv(CLEAN, index=False)
info(f"{len(raw):,} entries -> {len(drugs):,} alag-alag dawaiyan")


# ------------------------------------------------- 3. known drugs per target
step("3. Har target ke liye pehle se jaani dawaiyan (ChEMBL mechanism)")
names = json.load(open(os.path.join(MODELS, "davis_proteins.json")))
GENES = sorted({n.split("(")[0].replace("-nonphosphorylated", "")
                   .replace("-phosphorylated", "").strip() for n in names})
info(f"{len(GENES)} alag gene naam Davis me")

KNOWN_F = os.path.join(WORK, "known_drugs.json")
if os.path.exists(KNOWN_F):
    KNOWN = json.load(open(KNOWN_F))
    info(f"cache: {len(KNOWN)} targets")
else:
    KNOWN = {}
    for k, g in enumerate(GENES):
        try:
            s = get_json(f"{API}/target.json?pref_name__icontains={urllib.parse.quote(g)}"
                         f"&target_type=SINGLE%20PROTEIN&limit=5")
            tid = None
            for t_ in s.get("targets", []):
                for c in (t_.get("target_components") or []):
                    for syn in (c.get("target_component_synonyms") or []):
                        if str(syn.get("component_synonym", "")).upper() == g.upper():
                            tid = t_.get("target_chembl_id")
                if tid:
                    break
            ids = set()
            if tid:
                off = 0
                while True:
                    p = get_json(f"{API}/mechanism.json?target_chembl_id={tid}"
                                 f"&limit=1000&offset={off}")
                    got = p.get("mechanisms", [])
                    if not got:
                        break
                    for m in got:
                        if m.get("molecule_chembl_id"):
                            ids.add(m["molecule_chembl_id"])
                    off += 1000
            KNOWN[g] = sorted(ids)
        except Exception as e:
            info(f"  {g}: {type(e).__name__} - khaali chhoda")
            KNOWN[g] = []
        if (k + 1) % 25 == 0:
            print(f"   {k+1}/{len(GENES)} genes", flush=True)
            json.dump(KNOWN, open(KNOWN_F, "w"))
    json.dump(KNOWN, open(KNOWN_F, "w"))
info(f"known drugs mile: {sum(1 for v in KNOWN.values() if v)} targets pe, "
     f"total {sum(len(v) for v in KNOWN.values()):,} entries")


# ------------------------------------------------------------ 4. features
step("4. Features (v2 jaise hi: Morgan FP + RDKit descriptors + ChemBERTa)")
from rdkit.Chem import Descriptors, rdFingerprintGenerator

fp1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
DESC = Descriptors._descList
FEAT = os.path.join(WORK, "feats.npy")
if os.path.exists(FEAT):
    X = np.load(FEAT)
    info(f"cache: {X.shape}")
    assert len(X) == len(drugs), f"cache {len(X)} vs drugs {len(drugs)} - WORK saaf karo"
else:
    good, rd_rows = [], []
    for i, s in enumerate(drugs.smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        d = []
        for _, fn in DESC:
            try:
                d.append(float(fn(m)))
            except Exception:
                d.append(np.nan)
        d = np.nan_to_num(np.clip(np.array(d, np.float64), -1e6, 1e6)).astype(np.float32)
        rd_rows.append(np.concatenate(
            [fp1024.GetFingerprintAsNumPy(m).astype(np.float32), d]))
        good.append(i)
        if (i + 1) % 2000 == 0:
            print(f"   RDKit {i+1:,}/{len(drugs):,}", flush=True)
    drugs = drugs.iloc[good].reset_index(drop=True)
    RD = np.stack(rd_rows)
    import torch
    from transformers import AutoTokenizer, AutoModel
    cname = json.load(open(os.path.join(MODELS, "config.json")))["chem_name"]
    tok = AutoTokenizer.from_pretrained(cname)
    cm = AutoModel.from_pretrained(cname).eval()
    outs = []
    with torch.no_grad():
        B = 128
        for i in range(0, len(drugs), B):
            t = tok([str(x) for x in drugs.smiles[i:i+B]], return_tensors="pt",
                    padding=True, truncation=True, max_length=256)
            h = cm(**t).last_hidden_state
            msk = t["attention_mask"].unsqueeze(-1).float()
            outs.append(((h * msk).sum(1) / msk.sum(1)).numpy())
            if (i // B) % 20 == 0:
                print(f"   ChemBERTa {i:,}/{len(drugs):,}", flush=True)
    X = np.hstack([RD, np.concatenate(outs)]).astype(np.float32)
    np.save(FEAT, X)
    drugs.to_csv(CLEAN, index=False)
    info(f"bana: {X.shape}")


# ------------------------------------------------------------- 5. scoring
step("5. v2 model se saare targets pe scoring")
import gzip, shutil, xgboost as xgb

ESM = np.load(os.path.join(MODELS, "davis_esm.npy")).astype(np.float32)
mp = os.path.join(WORK, "binding_davis.ubj")
if not os.path.exists(mp):
    with gzip.open(os.path.join(MODELS, "binding_davis.ubj.gz"), "rb") as f, \
         open(mp, "wb") as o:
        shutil.copyfileobj(f, o)
model = xgb.XGBRegressor()
model.load_model(mp)
need = model.get_booster().num_features()
have = X.shape[1] + ESM.shape[1]
assert need == have, (f"feature mismatch: model {need} chahta hai, hum {have} de rahe "
                      f"({X.shape[1]} drug + {ESM.shape[1]} esm). RDKit version badla hoga.")
info(f"model ready · {need} features · {len(names)} targets · {len(drugs):,} dawaiyan")


def gene_of(n):
    return n.split("(")[0].replace("-nonphosphorylated", "") \
            .replace("-phosphorylated", "").strip()


all_rows = []
for pi, pname in enumerate(names):
    Xf = np.hstack([X, np.repeat(ESM[pi][None, :], len(X), 0)]).astype(np.float32)
    pred = model.predict(Xf)
    del Xf
    ordr = np.argsort(-pred)[:TOP_N]
    kn = set(KNOWN.get(gene_of(pname), []))
    for rank, j in enumerate(ordr, 1):
        all_rows.append({
            "target": pname, "rank": rank,
            "drug": drugs.name.iloc[j] or drugs.chembl_id.iloc[j],
            "chembl_id": drugs.chembl_id.iloc[j],
            "max_phase": drugs.max_phase.iloc[j],
            "pKd": round(float(pred[j]), 3),
            "known_for_target": bool(drugs.chembl_id.iloc[j] in kn),
            "smiles": drugs.smiles.iloc[j],
        })
    if (pi + 1) % 50 == 0:
        print(f"   {pi+1}/{len(names)} targets", flush=True)

tbl = pd.DataFrame(all_rows)
tbl.to_csv(os.path.join(OUT, "repurposing_top.csv"), index=False)
info(f"{len(tbl):,} rows likhe -> data/repurposing_top.csv "
     f"({os.path.getsize(os.path.join(OUT,'repurposing_top.csv'))/1e6:.1f} MB)")

# chhota meta jo app dikhati hai
kn_tot = int(tbl.known_for_target.sum())
meta = {
    "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "n_drugs": int(len(drugs)), "n_targets": int(len(names)), "top_n": TOP_N,
    "min_phase": MIN_PHASE,
    "known_in_top": kn_tot,
    "known_pct": round(100 * kn_tot / len(tbl), 1),
    "model": "binding v2 (Davis) - XGBoost on Morgan FP + RDKit desc + ChemBERTa + ESM-2 650M",
    "caveat": ("Ye computer ka andaza hai, lab ka nateeja nahi. 'known_for_target' "
               "sach wali dawaiyan model ke sahi hone ka suboot hain, nayi khoj nahi."),
}
json.dump(meta, open(os.path.join(OUT, "repurposing_meta.json"), "w"), indent=2)
print(f"\n   top {TOP_N} me se {meta['known_pct']}% pehle se jaani-pehchani dawaiyan hain")
print(f"   Total: {(time.time()-T0)/60:.1f} min", flush=True)
