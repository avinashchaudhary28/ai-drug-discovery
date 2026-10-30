"""Har gene ke liye: pocket ki har position pe ligand kitni baar lagta hai.

Probe se pakka hua (andaza nahi):
  7 interaction types hain, aur IFP ki lambai 595 = 85 x 7.
  Position i ke bits: IFP[(i-1)*7 : i*7]

Soch: pocket position akeli fail ho chuki - wo structural SLOT batati
hai, chemistry nahi. Par agar ek gene ki saari ligand-wali structures ka
IFP jod diya jaye, to har position ka ek "contact frequency" banta hai:
yahan ligand kitni baar lagta hai. Mutation agar aisi jagah hai jahan
dawa sach me lagti hai, to usse farak padna chahiye - aur jahan nahi
lagti, wahan nahi.

Leakage ka sawal: IFP crystal structures se aata hai, humare affinity
labels se nahi. Isliye label ka leak mumkin hi nahi.
"""
import json
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request

N_POS = 85
N_TYPES = 7
IFP_LEN = N_POS * N_TYPES          # 595, probe se tasdeek-shuda
TYPE_NAMES = ["apolar", "arom_f2f", "arom_e2f", "hbd_prot", "hba_prot",
              "cat_prot", "ani_prot"]

BASE = "https://klifs.net/api_v2"
UA = "ai-drug-discovery/phase3 (research)"
MAX_STRUCTS = 40                   # har gene se itni structures kaafi


def _get(path, params=None, tries=4):
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
        time.sleep(min(2 ** a, 15))
    raise RuntimeError(f"{tries} koshish ke baad nahi mila: {url} ({last})")


def parse_ifp(s):
    """-> 85 x 7 ka list-of-lists, ya None agar roop galat ho.

    Lambai ki jaanch zaroori hai: agar KLIFS kabhi dhaancha badal de to
    bits galat position pe padh liye jayenge aur koi error nahi aayega -
    bas feature chup-chaap jhootha ho jayega.
    """
    s = str(s or "").strip()
    if len(s) != IFP_LEN or not re.fullmatch(r"[01]+", s):
        return None
    return [[int(s[(i * N_TYPES) + t]) for t in range(N_TYPES)]
            for i in range(N_POS)]


def aggregate(ifps):
    """Kai IFP se har position ka contact frequency.

    -> {"n": kitni structures, "any": [85], "by_type": [85][7]}
    'any' = kitne hisse structures me is position pe KOI bhi contact tha.
    """
    good = [p for p in (parse_ifp(x) for x in ifps) if p is not None]
    n = len(good)
    if n == 0:
        return {"n": 0, "any": [0.0] * N_POS,
                "by_type": [[0.0] * N_TYPES for _ in range(N_POS)]}
    any_ = []
    by_type = []
    for i in range(N_POS):
        cols = [g[i] for g in good]
        any_.append(sum(1 for c in cols if any(c)) / n)
        by_type.append([sum(c[t] for c in cols) / n for t in range(N_TYPES)])
    return {"n": n, "any": any_, "by_type": by_type}


def fetch_gene_ifp(gene, kinase_id, work_dir, max_structs=MAX_STRUCTS):
    """-> aggregate() ka dict. Disk pe cache."""
    work = pathlib.Path(work_dir)
    work.mkdir(exist_ok=True)
    cache = work / f"ifp_{re.sub(r'[^A-Za-z0-9_-]', '_', gene)}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        # Purani cache naye fields ko chup-chaap palat deti hai - Phase 2
        # aur Phase 3 dono me ye bug nikal chuka hai
        if "by_type" in d:
            return d

    structs = _get("/structures_list", {"kinase_ID": kinase_id}) or []
    with_lig = [s for s in structs
                if str(s.get("ligand") or "").strip() not in ("", "0")]
    sids = [s["structure_ID"] for s in with_lig[:max_structs]]
    ifps = []
    for sid in sids:
        rows = _get("/interactions_get_IFP", {"structure_ID": sid}) or []
        for r in rows:
            ifps.append(r.get("IFP"))
    out = aggregate(ifps)
    out["gene"] = gene
    out["n_structures_with_ligand"] = len(with_lig)
    out["n_fetched"] = len(sids)
    out["n_bad_ifp"] = len(ifps) - out["n"]
    cache.write_text(json.dumps(out))
    return out


IFP_NAMES = (["ifp_contact_any"] + [f"ifp_{t}" for t in TYPE_NAMES]
             + ["ifp_n_structures"])


def ifp_features(agg, klifs_index):
    """Ek mutation ke liye: us position pe ligand kitni baar lagta hai.

    klifs_index 1..85, ya 0 agar mutation pocket me hi nahi hai.
    """
    import numpy as np
    z = np.zeros(len(IFP_NAMES), dtype=np.float32)
    if not agg or agg.get("n", 0) == 0 or not (1 <= int(klifs_index) <= N_POS):
        return z
    i = int(klifs_index) - 1
    vals = [agg["any"][i]] + list(agg["by_type"][i])
    # Aakhri feature: ye gene kitni structures pe tika hai - kam
    # structures wale gene ka aankda kamzor hota hai, aur model ko
    # ye jaanne ka haq hai
    vals.append(min(agg["n"], 100) / 100.0)
    return np.array(vals, dtype=np.float32)
