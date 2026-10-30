"""Har kinase ke liye: residue number -> KLIFS pocket position.

Kyun chahiye: Phase 3 kadam 1 me model ne gene yaad kar liya (random
split pe Spearman 0.752) par andekhe gene pe kuch nahi kiya (0.067).
Wajah ye ki raw position cross-gene tulnay nahi hai - ABL1 ka 315 aur
EGFR ka 315 ka koi rishta nahi. KLIFS ki 85-residue numbering wahi
rishta deti hai.

KLIFS se do cheezein milti hain (probe se naapa gaya, andaza nahi):
  /kinase_ID                  -> 85 akshar ka pocket SEQUENCE
  /interactions_match_residues -> har index (1-85) ka Xray_position
                                  aur KLIFS_position ka naam

SABSE BADA KHATRA, aur wo aaj hi dikh chuka hai: Xray_position crystal
structure ki numbering hai, aur humare mutation labels UniProt numbering
me hain. Ye hamesha nahi milti - ABL1 ki do isoforms me 19 residue ka
farak hai, jo Phase 1 me humne khud pakda tha. Na mile to har mutation
GALAT pocket position pe baith jayega aur koi error nahi aayega.

Isliye mapping khud ko jaanchti hai: har mutation pe dekha jata hai ki
us position pa wild-type amino acid wahi hai jo naam me likha hai
("T315I" me T). Na mile to pehle ek per-gene offset dhundha jata hai
(wahi tareeka jo Phase 1 me kaam kar gaya tha), aur phir bhi na mile to
wo gene istemaal NAHI hota - chup-chaap galat mapping se behtar hai.
"""
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://klifs.net/api_v2"
UA = "ai-drug-discovery/phase3 (research)"
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "work_klifs"

OFFSET_SPAN = 60        # Phase 1 me ABL1 ka asli offset -19 tha
MIN_MATCH = 0.8         # itne mutations ka wild-type AA milna hi chahiye


def get(path, params=None, tries=4):
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    last = None
    for a in range(1, tries + 1):
        try:
            r = urllib.request.Request(url, headers={
                "User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(r, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            last = f"HTTP {e.code}"
        except Exception as e:                            # noqa: BLE001
            last = repr(e)
        time.sleep(min(2 ** a, 20))
    raise RuntimeError(f"{tries} koshish ke baad bhi nahi mila: {url} ({last})")


def pick_structure(rows):
    """Sabse bharosemand structure: adhoore residue na hon, quality
    zyada, resolution kam."""
    def num(r, field, default):
        """`x or default` yahan NAHI chal sakta: Python me 0 falsy hai,
        aur missing_residues=0 sabse ACHHI haalat hai. Pehle wahi likha
        tha, to bina kisi missing residue wale structure ko 99 mil jata
        tha - yaani sabse kharab structure chuna jata. Test ne pakda."""
        v = r.get(field)
        if v is None or v == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def key(r):
        return (num(r, "missing_residues", 99.0),
                num(r, "missing_atoms", 999.0),
                -num(r, "quality_score", 0.0),
                num(r, "resolution", 99.0))
    return sorted(rows, key=key)[0]


def consensus_pocket(structs, top_n=25):
    """Kai structures ke pocket se har position ka sabse aam residue.

    Kyun zaroori: KLIFS ke bahut se structures KHUD mutant protein ke
    hote hain (jaise ABL1 T315I ka structure). Ek hi structure ka pocket
    wild-type maan lena galat hai - us position pe mutant residue baitha
    hoga aur jaanch bekaar me fail hogi. FLT3 (67%) aur MET (62%) ke
    saath shayad yahi hua.

    Sab structures ek hi KLIFS numbering pe aligned hain (85 positions,
    har jagah ek hi matlab), isliye position-wise consensus theek hai.
    """
    import collections
    pockets = [str(s.get("pocket") or "") for s in structs[:top_n]]
    pockets = [p for p in pockets if len(p) == 85]
    if not pockets:
        return ""
    out = []
    for i in range(85):
        c = collections.Counter(p[i] for p in pockets if p[i] not in "-_ ")
        out.append(c.most_common(1)[0][0] if c else "?")
    return "".join(out)


def fetch_gene(gene):
    """-> dict ya None. Cache disk pe."""
    WORK.mkdir(exist_ok=True)
    cache = WORK / f"{re.sub(r'[^A-Za-z0-9_-]', '_', gene)}.json"
    if cache.exists():
        old = json.loads(cache.read_text())
        # Purani cache nayi jaanch ko chup-chaap bekaar kar deti hai:
        # file maujood hai, isliye hum seedha lautte the aur consensus
        # kabhi banta hi nahi. Phase 2 me bilkul yahi bug nikla tha
        # (stale cache ne ek fix ko palat diya tha). Isliye ab dekha
        # jata hai ki cache me NAYE field hain ya nahi.
        if old.get("error") or "consensus_pocket" in old:
            return old
        print(f"    {gene}: purani cache (consensus nahi) - dobara")

    ks = get("/kinase_ID", {"kinase_name": gene, "species": "HUMAN"})
    if not ks:
        out = {"gene": gene, "error": "KLIFS me ye kinase nahi mila"}
        cache.write_text(json.dumps(out))
        return out
    k = ks[0]
    structs = get("/structures_list", {"kinase_ID": k["kinase_ID"]}) or []
    if not structs:
        out = {"gene": gene, "error": "koi structure nahi"}
        cache.write_text(json.dumps(out))
        return out
    st = pick_structure(structs)
    res = get("/interactions_match_residues",
              {"structure_ID": st["structure_ID"]}) or []
    out = {
        "gene": gene, "kinase_ID": k["kinase_ID"], "uniprot": k.get("uniprot"),
        "pocket": k.get("pocket"), "structure_ID": st["structure_ID"],
        "pdb": st.get("pdb"), "resolution": st.get("resolution"),
        "quality_score": st.get("quality_score"),
        "missing_residues": st.get("missing_residues"),
        "structure_pocket": st.get("pocket"),
        # Jaanch ke liye kai structures ka consensus - ek akela
        # structure khud mutant ka ho sakta hai
        "consensus_pocket": consensus_pocket(structs),
        "n_structures": len(structs),
        "residues": [{"index": int(r["index"]),
                      "xray": str(r.get("Xray_position") or ""),
                      "klifs": str(r.get("KLIFS_position") or "")}
                     for r in res],
    }
    cache.write_text(json.dumps(out))
    return out


# -------------------------------------------------------------- mapping
def region_of(klifs_label):
    """'g.l.4' -> 'g.l', 'I.1' -> 'I'. Pocket ka structural hissa."""
    s = str(klifs_label)
    m = re.match(r"^(.*?)\.?\d+$", s)
    return (m.group(1) if m else s) or "?"


def build_map(info):
    """-> {residue_number: (index, klifs_label, pocket_aa)}

    pocket_aa structure ke apne pocket sequence se aata hai, kinase ke
    aam pocket se nahi - kyunki numbering bhi usi structure se hai.
    """
    # Pehli pasand consensus: ek akela structure khud mutant protein ka
    # ho sakta hai, aur tab us position pe mutant residue dikhta hai
    # aur wild-type ki jaanch bekaar me fail hoti hai.
    pocket = (info.get("consensus_pocket") or info.get("structure_pocket")
              or info.get("pocket") or "")
    out = {}
    for r in info.get("residues", []):
        x = r["xray"]
        if not re.fullmatch(r"-?\d+", x):
            continue           # kuch structures me '_' ya khali hota hai
        i = r["index"]
        aa = pocket[i - 1] if 0 < i <= len(pocket) else "?"
        out[int(x)] = (i, r["klifs"], aa)
    return out


def best_offset(resmap, muts, span=OFFSET_SPAN):
    """Mutation labels aur structure ki numbering ke beech ek offset.

    muts: [(wild_type_aa, position), ...]
    Wahi tareeka jo Phase 1 me chala tha: har gene ke liye EK offset,
    aur wahi jo sabse zyada mutations ka wild-type AA sahi karta hai.
    Aas-paas dhundhna nahi - ek galat residue likhna kisi se behtar nahi.
    """
    if not muts:
        return 0, 0, 0
    best, best_n = 0, -1
    for off in sorted(range(-span, span + 1), key=lambda o: (abs(o), o)):
        n = 0
        for aa, pos in muts:
            hit = resmap.get(pos + off)
            if hit and hit[2] == aa:
                n += 1
        if n > best_n:
            best, best_n = off, n
    return best, best_n, len(muts)


MUT_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d{1,4})([ACDEFGHIKLMNPQRSTVWY])$")


def parse_mut(mutation):
    """'T790M' -> ('T', 790, 'M'). Multi-mutant ka pehla token."""
    tok = str(mutation).split("+")[0].strip().upper()
    m = MUT_RE.match(tok)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def verify(gene, resmap, mutations, min_match=MIN_MATCH):
    """-> dict. Gene istemaal layak hai ya nahi, aur kyun."""
    muts = [p for p in (parse_mut(m) for m in mutations) if p]
    pairs = [(aa, pos) for aa, pos, _ in muts]
    if not pairs:
        return {"gene": gene, "usable": False,
                "reason": "ek bhi point mutation parse nahi hua",
                "n_mutations": 0}
    off, n_ok, n = best_offset(resmap, pairs)
    in_pocket = sum(1 for aa, pos in pairs if (pos + off) in resmap)
    # Sirf un mutations pe match dekho jo pocket me aate hain - jo
    # pocket se bahar hain unka match hona mumkin hi nahi
    rate = (n_ok / in_pocket) if in_pocket else 0.0
    ok = in_pocket > 0 and rate >= min_match
    return {"gene": gene, "usable": bool(ok), "offset": int(off),
            "n_mutations": int(n), "n_in_pocket": int(in_pocket),
            "n_wt_matched": int(n_ok), "match_rate": float(rate),
            "reason": ("theek" if ok else
                       ("ek bhi mutation pocket me nahi" if not in_pocket
                        else f"wild-type AA sirf {rate:.0%} pe mila, "
                             f"{min_match:.0%} chahiye"))}
