"""Mutation ke naam aur drug ko ek jaisa roop dene ke liye.

Zaroori kyun: Davis "ABL1(T315I)" likhta hai, ChEMBL "T315I". Jodne ke
liye dono ko ek shakal me laana padega.

Sabse bada khatra: jo naam samajh na aaye use KHALI maan lena. Khali
mutation = wild type, aur phir "FLT3(ITD)" jaisi entry har doosri entry
se jud jayegi - chup-chaap, bina error. Isliye jo parse na ho uska apna
alag nishaan banta hai ("RAW:<text>"), jo sirf bilkul wahi text se
milta hai.
"""
import re

AA = "ACDEFGHIKLMNPQRSTVWY"

# Deletion pehle - warna "E746A750del" me se "E746A" point mutation
# jaisa dikh kar galat pakda jayega.
DEL = re.compile(rf"([{AA}])(\d+)[-_ ]?([{AA}])(\d+)\s*del", re.I)
DEL1 = re.compile(rf"([{AA}])(\d+)\s*del", re.I)
POINT = re.compile(rf"([{AA}])(\d{{1,4}})([{AA}])")
# Ek chunk POORA mutation-token ki zanjeer ho, tab hi maana jayega.
# Sirf "kahin mil gaya" se nahi - warna "JH1DOMAIN" me se H1D aur
# "N771INSNPG" me se N771I nikal aata hai, aur galat residue likha jata
# hai bina kisi error ke. Dono asli bug the, test ne pakde.
CHAIN = re.compile(rf"^(?:[{AA}]\d{{1,4}}[{AA}])+$")
SPLIT = re.compile(r"[,;/+\s\-_.|]+")
ALPHA = re.compile(r"^[A-Z]+$")


def normalize(raw):
    """-> (tokens, note, parsed)

    tokens: chhante hue mutation tokens ka tuple, jaise ('L858R','T790M')
    note:   bracket ke baad ka hissa (jaise Davis ka 'p' = phosphorylated)
    parsed: False agar ek bhi mutation samajh nahi aaya
    """
    if raw is None:
        return ("RAW:",), "", False
    s = str(raw).strip()
    if not s:
        return ("RAW:",), "", False

    note = ""
    # Davis jaisa "GENE(MUT)suffix" - bracket ke andar ka hissa lo
    m = re.search(r"\(([^)]*)\)(.*)$", s)
    if m:
        inner, note = m.group(1), m.group(2).strip()
        s = inner

    # Bade akshar me laate hain: amino acid ke chihn bade akshar hain,
    # aur chhote akshar wali likhai (jaise "t790m") warna samajh na aane
    # wali maani jati.
    s = s.upper()
    toks = []

    # Deletion pehle nikaal lo - unme '_' aur '-' hota hai, jo aage ke
    # batware me tootkar bekaar ho jata.
    work = s
    for mt in DEL.finditer(work):
        toks.append(f"{mt.group(1)}{mt.group(2)}_{mt.group(3)}{mt.group(4)}del")
    work = DEL.sub(" ", work)
    for mt in DEL1.finditer(work):
        toks.append(f"{mt.group(1)}{mt.group(2)}del")
    work = DEL1.sub(" ", work)

    # Ab bache hisse ko tukdon me todo. Har tukda teen me se ek hai:
    #   (a) poori mutation-zanjeer  -> le lo
    #   (b) sirf akshar, koi ank nahi (jaise "EGFR", "WILD", "TYPE")
    #       -> sajavat hai, chhod do
    #   (c) kuch aur -> samajh nahi aaya, poora naam RAW maana jayega
    for chunk in SPLIT.split(work):
        if not chunk:
            continue
        if CHAIN.match(chunk):
            for mt in POINT.finditer(chunk):
                toks.append(f"{mt.group(1)}{mt.group(2)}{mt.group(3)}")
        elif ALPHA.match(chunk):
            continue
        else:
            return (f"RAW:{s}",), note, False

    if not toks:
        return (f"RAW:{s}",), note, False

    # Dohrai hatao aur kram tay karo, taaki "L858R,T790M" aur
    # "T790M L858R" ek hi cheez banein
    return tuple(sorted(set(toks))), note, True


def drug_key(smiles):
    """Drug ko ek pehchaan do. Namak/salt forms ko ek maana jata hai.

    InChIKey ke pehle 14 akshar skeleton hain (stereo chhod kar). Davis
    aur ChEMBL ek hi dawa ko alag stereo ke saath likh sakte hain,
    isliye skeleton pe milana zyada sach dikhata hai.
    """
    from rdkit import Chem
    from rdkit.Chem import inchi

    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        k = inchi.MolToInchiKey(mol)
    except Exception:                                     # noqa: BLE001
        return None
    return k[:14] if k else None
