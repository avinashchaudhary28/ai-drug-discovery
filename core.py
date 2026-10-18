import os, json, gzip
import numpy as np, pandas as pd
import torch
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

try:  # molecule drawing needs system libs (libXrender); app works without it
    from rdkit.Chem import Draw
except Exception:
    Draw = None
from transformers import AutoTokenizer, AutoModel

RDLogger.DisableLog("rdApp.*")
M = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CFG = json.load(open(os.path.join(M, "config.json")))


def _cpu(m):
    try:
        m.set_params(device="cpu")
    except Exception:
        pass
    return m


def _raw(name):
    with gzip.open(os.path.join(M, name), "rb") as f:
        return bytearray(f.read())


def load_reg(name):
    m = xgb.XGBRegressor()
    m.load_model(_raw(name))
    return _cpu(m)


def load_clf(name):
    m = xgb.XGBClassifier()
    m.load_model(_raw(name))
    return _cpu(m)


# ---------- models ----------
BIND = {}
for ds in CFG["datasets"]:
    BIND[ds] = {
        "model": load_reg(f"binding_{ds}.ubj.gz"),
        "prot_ids": json.load(open(os.path.join(M, f"{ds}_proteins.json"))),
        "esm": np.load(os.path.join(M, f"{ds}_esm.npy")),
        "ref": np.load(os.path.join(M, f"{ds}_ref.npy")),
    }
TASKS = CFG["tasks"]
TOX = {t: load_clf(f"tox_{t}.ubj.gz") for t in TASKS}
SOL = load_reg("solubility.ubj.gz")

chem_tok = AutoTokenizer.from_pretrained(CFG["chem_name"])
chem_model = AutoModel.from_pretrained(CFG["chem_name"]).eval()

# ---------- features (exactly as in training) ----------
fp1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
fp2048 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@torch.no_grad()
def chemberta(smiles):
    t = chem_tok([smiles], return_tensors="pt", padding=True, truncation=True, max_length=256)
    h = chem_model(**t).last_hidden_state
    m = t["attention_mask"].unsqueeze(-1).float()
    return ((h * m).sum(1) / m.sum(1)).cpu().numpy()[0].astype(np.float32)


def descriptors(mol):
    d = []
    for _, fn in Descriptors._descList:
        try:
            d.append(float(fn(mol)))
        except Exception:
            d.append(np.nan)
    return np.nan_to_num(np.clip(np.array(d, np.float64), -1e6, 1e6)).astype(np.float32)


def to_mol(s):
    s = str(s or "").strip()
    return Chem.MolFromSmiles(s) if s else None


def drug_vec(mol, smiles, cb=None):
    cb = chemberta(smiles) if cb is None else cb
    return np.concatenate([fp1024.GetFingerprintAsNumPy(mol).astype(np.float32), descriptors(mol), cb])


def tox_vec(mol):
    return np.concatenate([fp2048.GetFingerprintAsNumPy(mol).astype(np.float32), descriptors(mol)])


# safety check: feature count must match the trained models
_n = len(drug_vec(Chem.MolFromSmiles("CCO"), "CCO"))
for ds, B in BIND.items():
    try:
        want = B["model"].get_booster().num_features()
    except Exception:
        want = None
    assert want in (None, _n + B["esm"].shape[1]), f"{ds}: feature mismatch ({want} vs {_n + B['esm'].shape[1]}) - RDKit version check karo"

# ---------- ADMET v2 (22 TDC properties) ----------
AD = os.path.join(M, "admet")
ADMET = None
if os.path.exists(os.path.join(AD, "config.json")):
    ACFG = json.load(open(os.path.join(AD, "config.json")))
    ADMET = {n: (load_clf if ACFG["metric"][n] in ("AUROC", "AUPRC") else load_reg)(
        os.path.join("admet", n + ".ubj.gz")) for n in ACFG["names"]}
    ADMET_LOG = set(ACFG["log_target"])
    fpcount = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def admet_vec(mol, cb):
    return np.concatenate([fpcount.GetCountFingerprintAsNumPy(mol).astype(np.float32),
                           np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.float32),
                           descriptors(mol), cb]).astype(np.float32)


# group, readable name, how to read it
ADMET_INFO = {
    "hia_hou": ("A · Absorption", "Aant se absorption (HIA)", "good"),
    "bioavailability_ma": ("A · Absorption", "Oral bioavailability", "good"),
    "caco2_wang": ("A · Absorption", "Caco-2 permeability (log cm/s)", "val"),
    "pgp_broccatelli": ("A · Absorption", "P-gp inhibitor (dawa-interaction)", "bad"),
    "solubility_aqsoldb": ("A · Absorption", "Solubility (log mol/L)", "val"),
    "lipophilicity_astrazeneca": ("A · Absorption", "Lipophilicity (logD)", "val"),
    "bbb_martins": ("D · Distribution", "Dimaag tak pahunch (BBB)", "info"),
    "ppbr_az": ("D · Distribution", "Plasma protein binding (%)", "val"),
    "vdss_lombardo": ("D · Distribution", "Volume of distribution (L/kg)", "val"),
    "cyp2c9_veith": ("M · Metabolism", "CYP2C9 inhibitor", "bad"),
    "cyp2d6_veith": ("M · Metabolism", "CYP2D6 inhibitor", "bad"),
    "cyp3a4_veith": ("M · Metabolism", "CYP3A4 inhibitor", "bad"),
    "cyp2c9_substrate_carbonmangels": ("M · Metabolism", "CYP2C9 substrate", "info"),
    "cyp2d6_substrate_carbonmangels": ("M · Metabolism", "CYP2D6 substrate", "info"),
    "cyp3a4_substrate_carbonmangels": ("M · Metabolism", "CYP3A4 substrate", "info"),
    "half_life_obach": ("E · Excretion", "Half-life (ghante)", "val"),
    "clearance_hepatocyte_az": ("E · Excretion", "Clearance – hepatocyte", "val"),
    "clearance_microsome_az": ("E · Excretion", "Clearance – microsome", "val"),
    "herg": ("T · Toxicity", "Dil pe asar (hERG block)", "bad"),
    "ames": ("T · Toxicity", "DNA damage (AMES)", "bad"),
    "dili": ("T · Toxicity", "Liver damage (DILI)", "bad"),
    "ld50_zhu": ("T · Toxicity", "Zehar LD50 (zyada = zyada zehreela)", "val"),
}


def _note(name, v):
    if name == "caco2_wang":
        return "achhi" if v > -5.15 else "kam"
    if name == "solubility_aqsoldb":
        return "achhi" if v > -2 else ("theek" if v > -4 else "kam")
    if name == "lipophilicity_astrazeneca":
        return "ideal (1-3)" if 1 <= v <= 3 else "range ke bahar"
    if name == "ppbr_az":
        return "bahut zyada" if v > 90 else "theek"
    if name == "half_life_obach":
        return "chhota" if v < 3 else ("lamba" if v > 12 else "medium")
    if name == "ld50_zhu":
        return "zyada zehreela" if v > 3.5 else "theek"
    return ""


def admet_report(mol, cb):
    """Returns (table, list of warnings). None if ADMET models not installed."""
    if ADMET is None:
        return None, []
    x = admet_vec(mol, cb)[None, :]
    rows, warn = [], []
    for name in ACFG["names"]:
        group, label, kind = ADMET_INFO.get(name, ("Other", name, "info"))
        if kind in ("good", "bad", "info"):
            p = float(ADMET[name].predict_proba(x)[0, 1])
            value = f"{round(100 * p)}%"
            if kind == "bad" and p >= 0.5:
                note = "⚠️ risk"
                warn.append(label)
            elif kind == "bad":
                note = "ok"
            elif kind == "good":
                note = "achhi" if p >= 0.5 else "kam"
            else:
                note = "haan" if p >= 0.5 else "nahi"
        else:
            v = float(ADMET[name].predict(x)[0])
            if name in ADMET_LOG:
                v = float(np.expm1(v))
            value, note = f"{v:.2f}", _note(name, v)
        rows.append({"Group": group, "Property": label, "Value": value, "Note": note})
    order = ["A · Absorption", "D · Distribution", "M · Metabolism", "E · Excretion", "T · Toxicity"]
    df = pd.DataFrame(rows)
    df["_o"] = df["Group"].map({g: i for i, g in enumerate(order)}).fillna(99)
    return df.sort_values("_o").drop(columns="_o").reset_index(drop=True), warn


# ---------- scoring ----------
TOX_NAMES = {
    "NR-AR": "Androgen receptor (male hormone)", "NR-AR-LBD": "Androgen receptor binding",
    "NR-AhR": "Aryl hydrocarbon receptor", "NR-Aromatase": "Aromatase (hormone enzyme)",
    "NR-ER": "Estrogen receptor", "NR-ER-LBD": "Estrogen receptor binding",
    "NR-PPAR-gamma": "PPAR-gamma (metabolism)", "SR-ARE": "Oxidative stress",
    "SR-ATAD5": "DNA damage (ATAD5)", "SR-HSE": "Heat shock stress",
    "SR-MMP": "Mitochondria damage", "SR-p53": "DNA damage (p53)",
}


def top_percent(ds, score):
    ref = BIND[ds]["ref"]
    return 100.0 * (1 - np.searchsorted(ref, score, side="right") / len(ref))


def strength(ds, score):
    # Davis pKd >= 7 strong, >= 6 medium; KIBA >= 12.1 = active (literature threshold)
    if ds == "davis":
        return "Strong 💪" if score >= 7 else ("Medium" if score >= 6 else "Weak")
    return "Strong 💪" if score >= 12.1 else "Weak"


CHOICES = [f"{p}  [{ds}]" for ds, B in BIND.items() for p in B["prot_ids"]]


def default_choice():
    for test in (lambda c: c.upper().startswith("ABL1-NONPHOSPH"),
                 lambda c: c.upper().startswith("ABL1") and "(" not in c.split("  [")[0],
                 lambda c: c.upper().startswith("ABL1")):
        hit = [c for c in CHOICES if test(c)]
        if hit:
            return hit[0]
    return CHOICES[0]


def screen_all(vec):
    rows = []
    for ds, B in BIND.items():
        X = np.hstack([np.repeat(vec[None, :], len(B["prot_ids"]), 0), B["esm"]]).astype(np.float32)
        for name, s in zip(B["prot_ids"], B["model"].predict(X)):
            s = float(s)
            rows.append({"Protein": name, "ds": ds, "Score": round(s, 2),
                         "Top %": round(top_percent(ds, s), 1), "Binding": strength(ds, s)})
    return pd.DataFrame(rows).sort_values("Top %").reset_index(drop=True)


def analyze(smiles, choice):
    mol = to_mol(smiles)
    if mol is None:
        return ("### ❌ SMILES samajh nahi aaya\nPubChem se *Canonical SMILES* copy karke dobara daalo.",
                None, None, None, None)
    smiles = str(smiles).strip()
    choice = choice or default_choice()
    name, ds = choice.rsplit("  [", 1)[0], choice.rsplit("[", 1)[-1].rstrip("]")

    cb = chemberta(smiles)
    allp = screen_all(drug_vec(mol, smiles, cb))
    row = allp[(allp.Protein == name) & (allp.ds == ds)].iloc[0]
    admet, admet_warn = admet_report(mol, cb)

    f = tox_vec(mol)[None, :]
    probs = {t: float(TOX[t].predict_proba(f)[0, 1]) for t in TASKS}
    sol = float(SOL.predict(f)[0])
    flags = [TOX_NAMES.get(t, t) for t in TASKS if probs[t] >= 0.5]
    sol_label = "Achhi" if sol > -2 else ("Theek-thaak" if sol > -4 else "Kam")

    good_bind = row["Binding"] != "Weak"
    all_flags = flags + admet_warn
    if good_bind and not all_flags and sol > -4:
        verdict = "🟢 Promising candidate"
        why = "Binding achhi, koi safety flag nahi, solubility theek."
    elif all_flags:
        verdict = f"⚠️ Safety flags: {len(all_flags)}"
        why = "Dhyan do — iska matlab 'bekaar' nahi; cancer jaisi dawaon me kuch risk sweekar hota hai."
    elif not good_bind:
        verdict = "🟡 Is protein se binding kamzor"
        why = "Neeche Top 10 list dekho — ye kis protein ke liye behtar hai."
    else:
        verdict = "🟡 Solubility kam"
        why = "Dawa banane ke liye structure sudharna padega."

    summary = f"""## {verdict}
{why}

- **Binding — {name}** ({ds}): score **{row['Score']}** · **Top {row['Top %']}%** · {row['Binding']}
- **Toxicity flags:** **{len(flags)} / {len(TASKS)}**{' — ' + ', '.join(flags) if flags else ''}
- **ADMET warnings:** {('**' + str(len(admet_warn)) + '** — ' + ', '.join(admet_warn)) if admet_warn else ('**0**' if admet is not None else 'models load nahi hue')}
- **Solubility:** **{sol:.2f}** log mol/L · {sol_label}
- **Molecule weight:** {Descriptors.MolWt(mol):.1f} g/mol

<sub>Top X% = predicted binding saare known jodon me kitni upar hai (kam % = zyada strong). Research estimate hai, lab test nahi.</sub>"""

    top10 = allp.head(10)[["Protein", "Top %", "Binding"]]
    tox = pd.DataFrame({"Risk %": [round(100 * probs[t]) for t in TASKS],
                        "Test": [TOX_NAMES.get(t, t) for t in TASKS]}).sort_values("Risk %", ascending=False)
    img = None
    if Draw is not None:
        try:
            img = Draw.MolToImage(mol, size=(400, 300))
        except Exception:
            img = None
    return summary, img, top10, tox, admet


