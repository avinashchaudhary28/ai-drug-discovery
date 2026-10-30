"""Features - bina protein sequence ke.

Jaan-boojh kar yahan protein ka koi model nahi hai. Pehla sawal yahi hai:
sirf DAWA aur MUTATION ke roop se, bina ye jane ki protein kaisa dikhta
hai, kya kuch seekha ja sakta hai?

Ye farsh ke upar ka pehla paaya hai. Agar ye "0 bolo" se behtar nikla, to
baad me pocket-aware representation ko ISSE jeetna padega - warna pocket
ka faida sabit nahi hota, sirf maan liya jata.

Gene ki pehchaan feature me NAHI hai, aur ye zaroori hai: leave-one-gene-
out me gene hi hataya jata hai, to uski pehchaan dena bemaani hota.
"""
import numpy as np

# Amino acid ke gun. Kyte-Doolittle hydropathy, aayatan (A^3), charge,
# polar hai ya nahi, aromatic hai ya nahi, aur bhaar.
AA_PROPS = {
    "A": (1.8, 88.6, 0, 0, 0, 89.1),
    "R": (-4.5, 173.4, 1, 1, 0, 174.2),
    "N": (-3.5, 114.1, 0, 1, 0, 132.1),
    "D": (-3.5, 111.1, -1, 1, 0, 133.1),
    "C": (2.5, 108.5, 0, 0, 0, 121.2),
    "Q": (-3.5, 143.8, 0, 1, 0, 146.2),
    "E": (-3.5, 138.4, -1, 1, 0, 147.1),
    "G": (-0.4, 60.1, 0, 0, 0, 75.1),
    "H": (-3.2, 153.2, 0, 1, 1, 155.2),
    "I": (4.5, 166.7, 0, 0, 0, 131.2),
    "L": (3.8, 166.7, 0, 0, 0, 131.2),
    "K": (-3.9, 168.6, 1, 1, 0, 146.2),
    "M": (1.9, 162.9, 0, 0, 0, 149.2),
    "F": (2.8, 189.9, 0, 0, 1, 165.2),
    "P": (-1.6, 112.7, 0, 0, 0, 115.1),
    "S": (-0.8, 89.0, 0, 1, 0, 105.1),
    "T": (-0.7, 116.1, 0, 1, 0, 119.1),
    "W": (-0.9, 227.8, 0, 0, 1, 204.2),
    "Y": (-1.3, 193.6, 0, 1, 1, 181.2),
    "V": (4.2, 140.0, 0, 0, 0, 117.1),
}
N_PROP = 6
AA = "".join(sorted(AA_PROPS))

MUT_NAMES = (
    ["n_mutations", "position", "is_multi", "is_del"]
    + [f"wt_{p}" for p in ("hydro", "vol", "charge", "polar", "arom", "mw")]
    + [f"mt_{p}" for p in ("hydro", "vol", "charge", "polar", "arom", "mw")]
    + [f"d_{p}" for p in ("hydro", "vol", "charge", "polar", "arom", "mw")]
)


def mutation_features(mutation):
    """-> numpy array, lambai len(MUT_NAMES)."""
    import re
    s = str(mutation)
    toks = [t for t in s.split("+") if t]
    m = re.match(rf"^([{AA}])(\d{{1,4}})([{AA}])$", toks[0].strip().upper()
                 ) if toks else None
    is_del = 1.0 if "del" in s.lower() else 0.0
    if m is None:
        # Samajh na aaye to gun shunya, par NISHAAN sach - jhootha
        # mutation mat gadho
        return np.array([len(toks), 0.0, float(len(toks) > 1), is_del]
                        + [0.0] * (3 * N_PROP), dtype=np.float32)
    wt = np.array(AA_PROPS[m.group(1)], dtype=np.float32)
    mt = np.array(AA_PROPS[m.group(3)], dtype=np.float32)
    return np.concatenate([
        np.array([len(toks), float(m.group(2)), float(len(toks) > 1), is_del],
                 dtype=np.float32),
        wt, mt, mt - wt]).astype(np.float32)


DESC_NAMES = ["mw", "logp", "tpsa", "hbd", "hba", "rotb", "rings", "heavy"]


def drug_features(smiles, n_bits=1024, radius=2):
    """Morgan fingerprint + thode descriptors. None agar SMILES kharab."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius,
                                                    fpSize=n_bits)
    fp = np.frombuffer(
        bytes(gen.GetFingerprintAsNumPy(mol).astype(np.uint8)),
        dtype=np.uint8).astype(np.float32)
    d = np.array([
        Descriptors.MolWt(mol), Crippen.MolLogP(mol),
        Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol), mol.GetNumHeavyAtoms(),
    ], dtype=np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.concatenate([fp, d]).astype(np.float32)


def build(df, n_bits=1024):
    """-> (X, ok_mask, feature_names). Jin rows ki SMILES nahi padhi
    gayi unka mask False - unhe chup-chaap shunya se nahi bhara jata."""
    uniq = {s: drug_features(s, n_bits) for s in df.smiles.astype(str).unique()}
    bad = {s for s, v in uniq.items() if v is None}
    ok = ~df.smiles.astype(str).isin(bad).to_numpy()
    rows = []
    for sm, mu in zip(df.smiles.astype(str)[ok], df.mutation[ok]):
        rows.append(np.concatenate([uniq[sm], mutation_features(mu)]))
    X = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, 1),
                                                                 np.float32)
    names = ([f"fp{i}" for i in range(n_bits)] + DESC_NAMES + MUT_NAMES)
    return X, ok, names
