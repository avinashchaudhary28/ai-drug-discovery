"""KLIFS probe, doosra daur: pocket position -> residue number milta hai?

Pehle probe se mila: /kinase_ID seedha 85-residue pocket ka SEQUENCE deta
hai. Par asli zaroorat wo nahi hai. Chahiye ye:

    pocket position 45  ->  ABL1 ka residue 315

Uske bina ye bataya hi nahi ja sakta ki ABL1 ka T315 pocket me hai ya
nahi, aur poora Phase 3 ka tarka wahin ruk jata hai.

Pehle probe me 23 paths mile. Unme do ummeed ke layak hain:
  /interactions_match_residues
  /structure_modified_residues
Dono structure ke hisaab se kaam karte dikhte hain, isliye pehle ABL1 ki
structures nikalni padengi.

Ye andaza nahi hai - paths spec se aaye hain. Par kaun sa kaam karega, wo
aazma kar hi pata chalega.
"""
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://klifs.net/api_v2"
UA = "ai-drug-discovery/phase3 probe (research)"


def get(path, params=None, limit=200_000):
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    try:
        r = urllib.request.Request(url, headers={
            "User-Agent": UA, "Accept": "application/json"})
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.read(limit).decode("utf-8", "replace"), resp.status, url
    except urllib.error.HTTPError as e:
        return "", e.code, url
    except Exception as e:                                # noqa: BLE001
        return f"__ERR__ {e!r}", None, url


def jload(txt):
    try:
        return json.loads(txt)
    except Exception:                                     # noqa: BLE001
        return None


print("=" * 70)
print("1) ABL1 ki kinase_ID aur pocket")
print("=" * 70)
txt, st, url = get("/kinase_ID", {"kinase_name": "ABL1", "species": "HUMAN"})
d = jload(txt)
if not d:
    print(f"  !! {url} -> {st}; aage nahi badh sakte")
    sys.exit(1)
k = d[0]
kid = k["kinase_ID"]
pocket = k["pocket"]
print(f"  kinase_ID {kid}, uniprot {k['uniprot']}")
print(f"  pocket ki lambai: {len(pocket)}  (85 honi chahiye)")
print(f"  pocket: {pocket}")
if len(pocket) != 85:
    print(f"  !! 85 nahi hai - KLIFS ka dhaancha badal gaya ho sakta hai")

print(f"\n{'=' * 70}\n2) is kinase ki structures\n{'=' * 70}")
sid = None
for p in ("/structures_list", "/structure_list"):
    txt, st, url = get(p, {"kinase_ID": kid})
    d = jload(txt)
    print(f"  {'✓' if d else '✗'} {p} -> {st}, "
          f"{len(d) if isinstance(d, list) else '-'} rows")
    if isinstance(d, list) and d:
        print(f"      fields: {sorted(d[0])}")
        print(f"      pehli row: {json.dumps(d[0])[:600]}")
        sid = d[0].get("structure_ID")
        break

if sid is None:
    print("\n  !! structure_ID nahi mila - aage ke endpoint iske bina "
          "nahi aazma sakte")
    sys.exit(1)

print(f"\n{'=' * 70}\n3) pocket position -> residue number (asli sawal)"
      f"\n{'=' * 70}")
print(f"  structure_ID {sid} pe aazma rahe hain\n")
for p in ("/interactions_match_residues", "/structure_modified_residues"):
    for params in ({"structure_ID": sid}, {"structure_ID": [sid]}):
        txt, st, url = get(p, params, 60_000)
        d = jload(txt)
        ok = st == 200 and d is not None
        print(f"  {'✓' if ok else '✗'} {p} {params} -> {st}")
        if not ok:
            if txt.startswith("__ERR__"):
                print(f"      {txt[:200]}")
            continue
        if isinstance(d, list) and d:
            print(f"      list of {len(d)}")
            print(f"      fields: {sorted(d[0]) if isinstance(d[0], dict) else type(d[0])}")
            for row in d[:6]:
                print(f"      {json.dumps(row)[:220]}")
        else:
            print(f"      {json.dumps(d)[:400]}")
        break

print(f"\n{'=' * 70}\n4) ek structure ka pocket bhi dekh lo\n{'=' * 70}")
for p in ("/structure_get_pocket",):
    txt, st, url = get(p, {"structure_ID": sid}, 20_000)
    print(f"  {p} -> {st}")
    if st == 200:
        # Ye PDB text ho sakta hai, JSON nahi
        head = txt[:900]
        print(f"      pehli lines:\n{head}")

print(f"\n{'=' * 70}\nFAISLA\n{'=' * 70}")
print("  Dekhna ye hai ki upar kisi jawab me KLIFS position (1-85) aur")
print("  residue number DONO ek saath aaye hain ya nahi. Agar nahi, to")
print("  pocket sequence ko UniProt sequence se align karna padega -")
print("  wo ho sakta hai (pocket kuch contiguous tukdon ka jod hai), par")
print("  wo alag kaam hai aur uski apni jaanch chahiye.")
