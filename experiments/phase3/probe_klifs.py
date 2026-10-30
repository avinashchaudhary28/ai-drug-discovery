"""KLIFS ka probe: pocket ki numbering kaise milti hai?

Phase 3 kadam 1 ne dikhaya ki model gene yaad kar leta hai par andekhe
gene pe kuch nahi kar pata (Spearman 0.752 -> 0.067). Wajah saaf hai:
raw position se cross-gene tulna hoti hi nahi - ABL1 ka residue 315 aur
EGFR ka residue 315 ka koi rishta nahi.

KLIFS ki 85-residue pocket numbering yahi deti hai: har kinase me
"pocket position 45" ek hi structural jagah hoti hai. Agar har mutation
ko uski ALIGNED pocket position mil jaye, to model ko pehli baar aisa
coordinate milega jo genes ke beech tulnay hai.

Chahiye kya:
  1. gene/UniProt -> us kinase ka pocket
  2. pocket ki 85 positions -> sequence ke kaun se residue number
  3. phir: mutation position P pocket me hai? hai to kaun si position?

API ka dhaancha mujhe NAHI pata. Aaj ChEMBL pe meri dono maanyataayein
galat nikli thin, isliye yahan andaza bilkul nahi - pehle spec maangte
hain, usme se paths padhte hain, phir unhi ko aazmate hain.

Ye script kuch likhta nahi, sirf padhta aur chhapta hai.
"""
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

UA = "ai-drug-discovery/phase3 probe (research)"
SPECS = [
    "https://klifs.net/swagger_v2/swagger.json",
    "https://klifs.net/api/swagger.json",
    "https://klifs.net/swagger/swagger.json",
    "https://klifs.net/api_v2/swagger.json",
]
BASES = ["https://klifs.net/api_v2", "https://klifs.net/api"]


def get(url, limit=400_000):
    try:
        r = urllib.request.Request(url, headers={
            "User-Agent": UA, "Accept": "application/json"})
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.read(limit).decode("utf-8", "replace"), resp.status
    except urllib.error.HTTPError as e:
        return "", e.code
    except Exception as e:                                # noqa: BLE001
        return f"__ERR__ {e!r}", None


def show(label, txt, n=700):
    print(f"\n--- {label} ---")
    if txt.startswith("__ERR__"):
        print(f"  {txt}")
        return None
    try:
        d = json.loads(txt)
    except Exception:                                     # noqa: BLE001
        print(f"  (JSON nahi) {txt[:300]}")
        return None
    s = json.dumps(d)[:n]
    print(f"  {s}")
    if isinstance(d, list) and d and isinstance(d[0], dict):
        print(f"  list of {len(d)}, pehli row ke fields: {sorted(d[0])}")
    elif isinstance(d, dict):
        print(f"  dict ke fields: {sorted(d)}")
    return d


print("=" * 70)
print("1) API ka spec milta hai kya (paths andaze se nahi, spec se)")
print("=" * 70)
paths = []
for u in SPECS:
    txt, st = get(u)
    if txt.startswith("__ERR__") or st != 200:
        print(f"  ✗ {u} -> {st or txt}")
        continue
    try:
        spec = json.loads(txt)
    except Exception:                                     # noqa: BLE001
        print(f"  ✗ {u} -> JSON nahi")
        continue
    paths = sorted(spec.get("paths", {}))
    base = spec.get("basePath") or ""
    host = spec.get("host") or ""
    print(f"  ✓ {u}")
    print(f"      host={host!r} basePath={base!r}, {len(paths)} paths")
    for p in paths:
        methods = sorted(spec["paths"][p])
        print(f"        {p}   [{', '.join(methods)}]")
    if host:
        BASES.insert(0, f"https://{host}{base}")
    break

if not paths:
    print("\n  !! Spec nahi mila. Ab bhi andaza nahi lagayenge - "
          "KLIFS ka home page dekh kar link dhundhte hain.")
    txt, st = get("https://klifs.net/")
    if not txt.startswith("__ERR__"):
        hrefs = sorted(set(re.findall(r'href=["\']([^"\']+)["\']', txt)))
        hrefs = [h for h in hrefs
                 if not re.search(r"\.(css|js|png|jpg|gif|ico|svg)$", h, re.I)]
        print(f"  home page pe {len(hrefs)} link:")
        for h in hrefs[:40]:
            mark = " <<<" if re.search(r"api|swagger|doc|download", h, re.I) \
                else ""
            print(f"      {h}{mark}")

print(f"\n{'=' * 70}\n2) kinase dhundho (ABL1, EGFR) - jo paths mile unhi se"
      f"\n{'=' * 70}")
kin_paths = [p for p in paths if re.search(r"kinase", p, re.I)]
print(f"  kinase wale paths: {kin_paths or 'koi nahi'}")

for base in BASES[:2]:
    for p in (kin_paths[:6] or ["/kinase_ID"]):
        for q in [{"kinase_name": "ABL1", "species": "HUMAN"},
                  {"kinase_name": "ABL1"}]:
            url = f"{base}{p}?" + urllib.parse.urlencode(q)
            txt, st = get(url, 8000)
            ok = st == 200 and not txt.startswith("__ERR__")
            print(f"  {'✓' if ok else '✗'} {url}  -> {st}")
            if ok:
                show(p, txt, 500)
                break

print(f"\n{'=' * 70}\n3) pocket / numbering wale paths\n{'=' * 70}")
pk = [p for p in paths
      if re.search(r"pocket|numbering|residue|interact|structure", p, re.I)]
print(f"  {pk or 'koi nahi'}")
print("\n  (inhe agle kadam me aazmayenge, jab kinase ID mil jayegi - "
       "ID ke bina inhe aazmana andaza hoga)")
