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
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi          # rdkit.inchi nahi - rdkit.Chem.inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
CLEAN = os.path.join(WORK, "drug_library_clean.csv")
drugs = None
if os.path.exists(CLEAN):
    c = pd.read_csv(CLEAN)
    # all_ids chahiye: purana cache usi ke bina bana tha, aur usse known_for_target
    # galat (kam) nikalta tha. Chup-chaap use kar lena fix ko bekaar kar deta.
    if len(c) > 1000 and "all_ids" in c.columns:
        drugs = c
        info(f"cache: {len(drugs):,} dawaiyan")
    else:
        info(f"cache phenk rahe hain (all_ids nahi hai) - dobara banega")
if drugs is None:
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
    # Group ke SAARE ChEMBL ids rakhna zaroori hai. Pehle sirf representative id
    # rakhi thi, aur usi se "known_for_target" check hota tha - jiski wajah se
    # Dasatinib (table me CHEMBL5416410, ChEMBL ke mechanism data me CHEMBL1421)
    # 'known' me nahi gina gaya. Yaani naap model ko asli se KHARAB dikha rahi thi.
    allids = (t.groupby("key").chembl_id
               .apply(lambda s: "|".join(sorted(set(s.astype(str)))))
               .rename("all_ids"))
    drugs = (t.groupby("key", as_index=False).first()
              [["key", "chembl_id", "name", "parent_smiles", "max_phase"]]
              .merge(allids, on="key")
              .rename(columns={"parent_smiles": "smiles"})
              .drop(columns=["key"]).reset_index(drop=True))
    drugs.to_csv(CLEAN, index=False)
info(f"{len(raw):,} entries -> {len(drugs):,} alag-alag dawaiyan")


# ------------------------------------------------- 3. known drugs per target
step("3. Har target ke liye pehle se jaani dawaiyan (ChEMBL mechanism)")
names = json.load(open(os.path.join(MODELS, "davis_proteins.json")))
GENES = sorted({n.split("(")[0].replace("-nonphosphorylated", "")
                   .replace("-phosphorylated", "").strip() for n in names})
info(f"{len(GENES)} alag gene naam Davis me")

def chembl_target_id(gene):
    """
    Gene symbol -> ChEMBL target id.

    Full-text search + GENE_SYMBOL synonym pe match. Pehle maine
    `pref_name__icontains=EGFR` try kiya tha - wo fail hota hai, kyunki ChEMBL
    ka pref_name "Epidermal growth factor receptor erbB1" jaisa hota hai; gene
    symbol sirf component synonyms me milta hai. Usi galti se 381 me se sirf
    23 genes resolve hue the aur poori table 'novel' dikhne lagi thi.
    """
    try:
        res = get_json(f"{API}/target/search.json?q={urllib.parse.quote(gene)}"
                       f"&limit=25").get("targets", [])
    except Exception:
        return None
    for t_ in res:
        if t_.get("target_type") != "SINGLE PROTEIN":
            continue
        if t_.get("organism") != "Homo sapiens":
            continue
        syms = {str(s.get("component_synonym", "")).upper()
                for c in (t_.get("target_components") or [])
                for s in (c.get("target_component_synonyms") or [])
                if s.get("syn_type") == "GENE_SYMBOL"}
        if gene.upper() in syms:
            return t_.get("target_chembl_id")
    return None


def mechanism_drugs(target_id):
    out, off = set(), 0
    while True:
        p = get_json(f"{API}/mechanism.json?target_chembl_id={target_id}"
                     f"&limit=1000&offset={off}")
        got = p.get("mechanisms", [])
        if not got:
            break
        for m in got:
            if m.get("molecule_chembl_id"):
                out.add(m["molecule_chembl_id"])
        off += 1000
    return out


KNOWN_F = os.path.join(WORK, "known_drugs.json")
KNOWN = None
if os.path.exists(KNOWN_F):
    try:
        c = json.load(open(KNOWN_F))
        cov = 100 * sum(1 for v in c.values() if v) / max(len(c), 1)
        if cov >= 40 and len(c) >= len(GENES) * 0.9:
            KNOWN = c
            info(f"cache: {len(c)} targets, {cov:.0f}% resolve hue")
        else:
            # Purana cache toote hue lookup se bana tha (381 me se 23 genes).
            # Use chup-chaap use kar lena fix ko bekaar kar deta hai.
            info(f"cache phenk rahe hain: sirf {cov:.0f}% genes resolve "
                 f"({len(c)} entries) - dobara banega")
    except Exception as e:
        info(f"cache padha nahi gaya ({type(e).__name__}) - dobara banega")
if KNOWN is None:
    KNOWN = {}
    for k, g in enumerate(GENES):
        try:
            tid = chembl_target_id(g)
            KNOWN[g] = sorted(mechanism_drugs(tid)) if tid else []
        except Exception as e:
            info(f"  {g}: {type(e).__name__} - khaali chhoda")
            KNOWN[g] = []
        if (k + 1) % 25 == 0:
            print(f"   {k+1}/{len(GENES)} genes · ab tak "
                  f"{sum(1 for v in KNOWN.values() if v)} resolve hue", flush=True)
            json.dump(KNOWN, open(KNOWN_F, "w"))
    json.dump(KNOWN, open(KNOWN_F, "w"))

n_res = sum(1 for v in KNOWN.values() if v)
cover = 100 * n_res / max(len(GENES), 1)
info(f"known drugs mile: {n_res}/{len(GENES)} genes ({cover:.0f}%), "
     f"total {sum(len(v) for v in KNOWN.values()):,} entries")
for g in ("EGFR", "ABL1", "ALK", "KIT", "BRAF", "SRC"):
    if g in KNOWN:
        info(f"  {g:7s} {len(KNOWN[g]):4d} known dawaein")

# Agar lookup toot gaya to poori table 'novel' dikhegi - yahi wo bharam hai jo
# pichhli baar bana tha. Isliye chup-chaap aage nahi badhte.
KNOWN_OK = cover >= 40
if not KNOWN_OK:
    print(f"\n   !! known-drug lookup bharose layak nahi ({cover:.0f}% genes). "
          f"Table banegi par 'known' column par bharosa na karein - app usko "
          f"chhupa degi.", flush=True)


# ------------------------------------------------------------ 4. features
step("4. Features (v2 jaise hi: Morgan FP + RDKit descriptors + ChemBERTa)")
from rdkit.Chem import Descriptors, rdFingerprintGenerator

fp1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
DESC = Descriptors._descList
FEAT = os.path.join(WORK, "feats.npy")
FEAT_IDS = os.path.join(WORK, "feats_ids.json")
X = None
if os.path.exists(FEAT) and os.path.exists(FEAT_IDS):
    cand = np.load(FEAT)
    ids = json.load(open(FEAT_IDS))
    # Sirf length check kaafi nahi: order badal jaaye to features galat molecule
    # se jud jaate hain aur koi error nahi aata - chup-chaap galat nateeja.
    if len(cand) == len(drugs) and ids == list(drugs.chembl_id.astype(str)):
        X = cand
        info(f"cache: {X.shape} (order verify ho gaya)")
    else:
        info(f"features ka cache mel nahi khata ({len(cand)} vs {len(drugs)}) - dobara banega")
if X is None:
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
    json.dump(list(drugs.chembl_id.astype(str)), open(FEAT_IDS, "w"))
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


# Molecular weight: "model bade ajeeb molecules ko upar rakhta hai" ko aankde me
# badalne ke liye. Kinase inhibitor aam taur pe 300-600 Da ka hota hai; 1000+
# wale peptides kinase pocket me baithte hi nahi.
MW = np.array([Descriptors.MolWt(Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s))
               else 0.0 for s in drugs.smiles], dtype=np.float32)
info(f"library ka MW: median {np.median(MW):.0f} Da · "
     f"{int((MW > 1000).sum()):,} molecules 1000 Da se bade")


all_rows, dl_rows = [], []
# POORA score matrix bhi save karte hain (10,465 x 442 float16 = ~9 MB).
# Kyun: abhi sirf top-50 bachte hain, aur unse corrections test nahi ho sakte.
# Poora matrix ek baar nikalne ke baad "dawa ka mean ghatao", "per-drug z-score",
# "target-centering" jaisi dozens cheezein BINA Actions minutes kharch kiye
# offline naapi ja sakti hain.
SCORES = np.zeros((len(drugs), len(names)), dtype=np.float16)
for pi, pname in enumerate(names):
    Xf = np.hstack([X, np.repeat(ESM[pi][None, :], len(X), 0)]).astype(np.float32)
    pred = model.predict(Xf)
    del Xf
    SCORES[:, pi] = pred.astype(np.float16)
    kn = set(KNOWN.get(gene_of(pname), []))
    # Do list: ek seedhi top-N, aur ek sirf dawa-jaise size wale molecules me se.
    # Sawaal: model ka bada-molecule jhukav hi poori problem hai, ya ranking khud
    # tooti hui hai? Agar size filter karne se known dawaein upar aa jaati hain to
    # ek saadha post-filter bahut kuch sudhar deta hai; na aayein to ranking ki
    # kami size se alag hai. Dono ka recall meta me likhte hain.
    druglike = (MW >= 250) & (MW <= 700)
    ordr = np.argsort(-pred)[:TOP_N]
    dl_idx = np.where(druglike)[0]
    ordr_dl = dl_idx[np.argsort(-pred[dl_idx])][:TOP_N] if len(dl_idx) else np.array([], int)
    for rank, j in enumerate(ordr_dl, 1):
        ids_dl = str(drugs.all_ids.iloc[j] if "all_ids" in drugs.columns
                     else drugs.chembl_id.iloc[j]).split("|")
        dl_rows.append({"target": pname, "rank": rank,
                        "drug": drugs.name.iloc[j] or drugs.chembl_id.iloc[j],
                        "chembl_id": drugs.chembl_id.iloc[j],
                        "max_phase": drugs.max_phase.iloc[j],
                        "pKd": round(float(pred[j]), 3),
                        "known_for_target": bool(kn & set(ids_dl)),
                        "mw": round(float(MW[j]), 1),
                        "smiles": drugs.smiles.iloc[j]})
    for rank, j in enumerate(ordr, 1):
        ids = str(drugs.all_ids.iloc[j] if "all_ids" in drugs.columns
                  else drugs.chembl_id.iloc[j]).split("|")
        all_rows.append({
            "target": pname, "rank": rank,
            "drug": drugs.name.iloc[j] or drugs.chembl_id.iloc[j],
            "chembl_id": drugs.chembl_id.iloc[j],
            "max_phase": drugs.max_phase.iloc[j],
            "pKd": round(float(pred[j]), 3),
            # group ke kisi bhi id ka match kaafi hai (salt forms jud chuke hain)
            "known_for_target": bool(kn & set(ids)),
            "mw": round(float(MW[j]), 1),
            "smiles": drugs.smiles.iloc[j],
        })
    if (pi + 1) % 50 == 0:
        print(f"   {pi+1}/{len(names)} targets", flush=True)

np.savez_compressed(
    os.path.join(OUT, "score_matrix.npz"),
    scores=SCORES,
    drug_ids=np.array(drugs.chembl_id.astype(str), dtype=object),
    all_ids=np.array(drugs.all_ids.astype(str) if "all_ids" in drugs.columns
                     else drugs.chembl_id.astype(str), dtype=object),
    drug_names=np.array(drugs.name.fillna("").astype(str), dtype=object),
    max_phase=np.array(drugs.max_phase, dtype=np.float32),
    smiles=np.array(drugs.smiles.astype(str), dtype=object),
    mw=MW,
    targets=np.array(names, dtype=object))
# known lists bhi commit karte hain: inke bina offline recall nahi naapa ja sakta
json.dump(KNOWN, open(os.path.join(OUT, "known_drugs.json"), "w"))
info(f"known_drugs.json saved ({sum(len(v) for v in KNOWN.values()):,} entries)")
info(f"score matrix saved {SCORES.shape} "
     f"({os.path.getsize(os.path.join(OUT,'score_matrix.npz'))/1e6:.1f} MB)")

tbl = pd.DataFrame(all_rows)
dl = pd.DataFrame(dl_rows)
tbl.to_csv(os.path.join(OUT, "repurposing_top.csv"), index=False)
dl.to_csv(os.path.join(OUT, "repurposing_top_druglike.csv"), index=False)


def _recall(df):
    """Kitne checkable targets pe top-N me koi known dawa aayi."""
    if not len(df):
        return {}
    d = df.copy()
    d["gene"] = d.target.map(gene_of)
    resolved = {g for g, v in KNOWN.items() if v}
    sub = d[d.gene.isin(resolved)]
    if not len(sub):
        return {}
    per = sub.groupby("target").known_for_target.sum()
    return {"checkable_targets": int(per.size),
            "with_any_known": int((per > 0).sum()),
            "zero_known": int((per == 0).sum()),
            "median_mw": round(float(d.mw.median()), 1)}


_rp, _rd = _recall(tbl), _recall(dl)
print(f"\n   seedhi top-{TOP_N}      : {_rp.get('with_any_known')}/{_rp.get('checkable_targets')} "
      f"targets pe known mila · median MW {_rp.get('median_mw')} Da")
print(f"   dawa-jaise size se : {_rd.get('with_any_known')}/{_rd.get('checkable_targets')} "
      f"targets pe known mila · median MW {_rd.get('median_mw')} Da", flush=True)
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
    "known_lookup_ok": bool(KNOWN_OK),
    "known_genes_resolved": f"{n_res}/{len(GENES)}",
    # Poore column ko chhupana zyada sakhti hai: coverage 24% hai par bade
    # kinases theek resolve hue (EGFR 76, KIT 34, ABL1 18 known dawaein).
    # Isliye app ko per-target batate hain ki kis target ka known-list hai.
    # Jis gene pe ek bhi known dawa nahi mili use conservative tareeke se
    # "pata nahi" maanenge - "koi known nahi" keh dena jhooth hoga.
    "resolved_genes": sorted(g for g, v in KNOWN.items() if v),
    # Kinase inhibitor aam taur pe 300-600 Da. Top-50 ka median library se
    # bahut zyada ho to model bade molecules ki taraf jhuka hua hai.
    "library_mw_median": round(float(np.median(MW)), 1),
    "top_mw_median": round(float(tbl.mw.median()), 1),
    "top_pct_over_1000da": round(100 * float((tbl.mw > 1000).mean()), 1),
    "druglike_filter": "MW 250-700 Da",
    "recall_plain": _recall(tbl),
    "recall_druglike": _recall(dl),
    "model": "binding v2 (Davis) - XGBoost on Morgan FP + RDKit desc + ChemBERTa + ESM-2 650M",
    "caveat": ("Ye computer ka andaza hai, lab ka nateeja nahi. 'known_for_target' "
               "sach wali dawaiyan model ke sahi hone ka suboot hain, nayi khoj nahi."),
}
json.dump(meta, open(os.path.join(OUT, "repurposing_meta.json"), "w"), indent=2)
print(f"\n   top {TOP_N} me se {meta['known_pct']}% pehle se jaani-pehchani dawaiyan hain")
print(f"   Total: {(time.time()-T0)/60:.1f} min", flush=True)
