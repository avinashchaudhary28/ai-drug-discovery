"""Ek baar ka probe: ChEMBL target ka asli structure kya hai.

classify_targets.py ka probe ruk gaya tha kyunki
'protein_classifications' field target_component me nahi mila. Andaza
lagane ke bajaye ek run me saari sambhavnaayein dekh lete hain - aur
sach me ek kinase (EGFR) aur ek non-kinase (GNRHR) dono pe, kyunki
pehchaan ka tareeka dono pe sahi chalna chahiye.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chembl_api import get                               # noqa: E402

PAIRS = [("EGFR (kinase)", "CHEMBL203"),
         ("GNRHR (kinase NAHI)", "CHEMBL1856"),
         ("ABL1 (kinase)", "CHEMBL1862")]


def show(label, obj, keys=None):
    print(f"\n--- {label} ---")
    if isinstance(obj, dict):
        print("keys:", sorted(obj))
        for k in keys or []:
            if k in obj:
                print(f"  {k} = {json.dumps(obj[k])[:400]}")
    else:
        print(json.dumps(obj)[:600])


for name, tid in PAIRS:
    print(f"\n{'=' * 66}\n{name}  {tid}\n{'=' * 66}")
    try:
        d = get(f"/target/{tid}.json")
    except Exception as e:                               # noqa: BLE001
        print("  /target fail:", e)
        continue
    show("/target", d, ["target_type", "pref_name"])
    comps = d.get("target_components") or []
    print(f"\n  components: {len(comps)}")
    if not comps:
        continue
    c0 = comps[0]
    show("target_components[0]", c0, ["accession", "component_id",
                                      "component_type"])
    xr = c0.get("target_component_xrefs") or []
    print(f"\n  xrefs: {len(xr)}")
    srcs = {}
    for x in xr:
        srcs.setdefault(x.get("xref_src_db"), []).append(x.get("xref_id"))
    for s, ids in sorted(srcs.items(), key=lambda kv: kv[0] or ""):
        print(f"    {s}: {ids[:6]}{' ...' if len(ids) > 6 else ''}")

    cid = c0.get("component_id")
    if cid:
        try:
            tc = get(f"/target_component/{cid}.json")
            show(f"/target_component/{cid}", tc)
            pcs = tc.get("protein_classifications")
            if pcs:
                print("  protein_classifications:",
                      json.dumps(pcs)[:300])
                pid = pcs[0].get("protein_classification_id")
                if pid:
                    try:
                        pc = get(f"/protein_classification/{pid}.json")
                        show(f"/protein_classification/{pid}", pc)
                    except Exception as e:               # noqa: BLE001
                        print("  /protein_classification fail:", e)
        except Exception as e:                           # noqa: BLE001
            print(f"  /target_component/{cid} fail:", e)

print(f"\n{'=' * 66}\nendpoint maujood hain ya nahi\n{'=' * 66}")
for ep in ["/protein_classification.json", "/protein_class.json",
           "/target_component.json", "/mechanism.json"]:
    try:
        d = get(ep, {"limit": 1})
        key = [k for k in d if k != "page_meta"]
        rows = d.get(key[0]) if key else None
        print(f"  ✓ {ep}  -> {key}, "
              f"total {d['page_meta'].get('total_count')}")
        if rows:
            print(f"      row keys: {sorted(rows[0])}")
            print(f"      row: {json.dumps(rows[0])[:300]}")
    except Exception as e:                               # noqa: BLE001
        print(f"  ✗ {ep}  {e}")
