"""Probe: KLIFS ka interaction fingerprint kya deta hai?

Soch ye hai: "mutation kis slot me hai" se behtar sawal hai "mutation
wahan hai jahan dawa SACH ME lagti hai". Pocket position akeli fail ho
chuki hai (Spearman -0.004, t=-0.16) - wo structural slot batati hai,
chemistry nahi.

KLIFS har structure+ligand ka interaction fingerprint rakhta hai: kaun
si pocket position ligand ko chhuti hai. Agar har gene ke kai structures
ka IFP jod kar "is position pe ligand kitni baar lagta hai" nikal lein,
to ek aisa feature banta hai jo:
  - mechanistic hai (contact hota hai ya nahi)
  - cross-gene tulnay hai (KLIFS coordinate me)

Par IFP ka dhaancha mujhe nahi pata - kitne bits, kis kram me, kis
position ka kaun sa. Isliye probe pehle, bina kisi andaze ke.
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
            return resp.read(limit).decode("utf-8", "replace"), resp.status
    except urllib.error.HTTPError as e:
        return "", e.code
    except Exception as e:                                # noqa: BLE001
        return f"__ERR__ {e!r}", None


def jload(t):
    try:
        return json.loads(t)
    except Exception:                                     # noqa: BLE001
        return None


print("=" * 70)
print("1) interaction ke types - IFP ke bits ka matlab")
print("=" * 70)
txt, st = get("/interactions_get_types")
d = jload(txt)
print(f"  status {st}")
if d:
    print(f"  {json.dumps(d)[:900]}")
    if isinstance(d, list) and d and isinstance(d[0], dict):
        print(f"  fields: {sorted(d[0])}, kul {len(d)}")
else:
    print(f"  {txt[:300]}")

print(f"\n{'=' * 70}\n2) ABL1 ki structures, phir unka IFP\n{'=' * 70}")
txt, st = get("/kinase_ID", {"kinase_name": "ABL1", "species": "HUMAN"})
k = (jload(txt) or [{}])[0]
kid = k.get("kinase_ID")
print(f"  ABL1 kinase_ID: {kid}")

txt, st = get("/structures_list", {"kinase_ID": kid})
structs = jload(txt) or []
print(f"  structures: {len(structs)}")
# Sirf wo jinme ligand hai - bina ligand ke IFP ka matlab nahi
with_lig = [s for s in structs if (s.get("ligand") or "").strip()
            not in ("", "0")]
print(f"  jinme ligand hai: {len(with_lig)}")
if not with_lig:
    print("  !! ligand wala structure nahi mila - IFP ka faida nahi")
    sys.exit(1)

sids = [s["structure_ID"] for s in with_lig[:5]]
print(f"  pehle 5: {sids}")

print(f"\n{'=' * 70}\n3) IFP ka asli roop\n{'=' * 70}")
for params in ({"structure_ID": sids[0]}, {"structure_ID": sids[:3]}):
    txt, st = get("/interactions_get_IFP", params, 40_000)
    d = jload(txt)
    print(f"\n  /interactions_get_IFP {params} -> {st}")
    if d is None:
        print(f"      {txt[:300]}")
        continue
    if isinstance(d, list) and d:
        print(f"      list of {len(d)}")
        for row in d[:3]:
            if isinstance(row, dict):
                print(f"      fields: {sorted(row)}")
                for kk, vv in row.items():
                    sv = str(vv)
                    print(f"        {kk}: {sv[:200]}"
                          + (f"   (lambai {len(sv)})" if len(sv) > 50 else ""))
            else:
                print(f"      {str(row)[:200]}")
    else:
        print(f"      {json.dumps(d)[:400]}")

print(f"\n{'=' * 70}\nFAISLA\n{'=' * 70}")
print("  Dekhna ye hai:")
print("   - IFP ek lambi string hai kya, aur uski lambai kya hai?")
print("   - 85 positions x kitne interaction types? (85*7 = 595 hota hai")
print("     agar 7 types hon)")
print("   - agar lambai 85 ka gunank nahi hai to mera samajhna galat hai")
print("     aur usse pehle theek karna padega.")
