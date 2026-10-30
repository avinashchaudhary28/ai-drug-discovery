"""Platinum ka probe: file milti hai, aur usme kya hai?

Platinum kyun: yahi wo dataset hai jispe published state of the art ke
numbers bane hain (796 jode, Pearson r = 0.20 unke 10-fold CV pe). Usse
jodne ka do-tarfa faida hai:
  - humara number unke number ke SAATH wali axis pe baithta hai
  - aur agar hum unke test set ko chhoote bina evaluate karein, to tulna
    imaandaar rehti hai

File chhoti hai, isliye BindingDB wali dikkat (host 594 MB ko bahut
dheema kar deta hai) yahan nahi aani chahiye - par ye bhi naap kar
dekhenge, maan kar nahi.

Ye script kuch likhta nahi, sirf padhta aur chhapta hai.
"""
import json
import os
import re
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_bindingdb import get_text, head               # noqa: E402

PAGES = [
    "https://biosig.lab.uq.edu.au/platinum/",
    "https://biosig.lab.uq.edu.au/platinum/download",
    "http://biosig.unimelb.edu.au/platinum/",
    "http://biosig.unimelb.edu.au/platinum/download",
]


def main():
    print("=" * 70)
    print("1) Platinum ka page khulta hai kya")
    print("=" * 70)
    links = []
    for page in PAGES:
        txt, status = get_text(page)
        if txt.startswith("__ERR__"):
            print(f"  ✗ {page} -> {txt}")
            continue
        found = sorted(set(re.findall(
            r'href=["\']?([^"\'>\s]+\.(?:csv|tsv|zip|txt|xlsx?|sdf))',
            txt, re.I)))
        print(f"  ✓ {page} -> HTTP {status}, {len(found)} data file link")
        for f in found[:30]:
            print(f"      {f}")
        links += [f if f.startswith("http") else urllib.parse.urljoin(page, f)
                  for f in found]
        # kuch pages pe link JS me hota hai - 'platinum' wale shabd dhundo
        for m in sorted(set(re.findall(
                r'[\w/.-]*platinum[\w/.-]*\.(?:csv|tsv|zip|txt)', txt, re.I))):
            u = m if m.startswith("http") else urllib.parse.urljoin(page, m)
            if u not in links:
                print(f"      (text me mila) {m}")
                links.append(u)

    links = list(dict.fromkeys(links))
    if not links:
        print("\n!! Koi data file ka link nahi mila. URL ka andaza nahi "
              "lagayenge - page ka HTML dekh kar aage badhna hoga.")
        for page in PAGES:
            txt, status = get_text(page, limit=4000)
            if not txt.startswith("__ERR__") and txt:
                print(f"\n  --- {page} ka shuruaati HTML ---")
                print(txt[:2500])
                break
        return 1

    print(f"\n{'=' * 70}\n2) kaun si file khulti hai aur kitni badi hai\n"
          f"{'=' * 70}")
    good = []
    for u in links[:20]:
        i = head(u)
        size = i.get("total")
        ok = i.get("status") in (200, 206) and not i.get("error")
        print(f"  {'✓' if ok else '✗'} {u}\n      {json.dumps(i)}")
        if ok:
            good.append((u, int(size) if size and size.isdigit() else None))

    if not good:
        print("\n!! Ek bhi file nahi khuli.")
        return 1

    print(f"\n{'=' * 70}\n3) sabse kaam ki lagne wali file ka shuruaati "
          f"hissa\n{'=' * 70}")
    # Mutation wala flat file dhundo - naam me 'platinum' aur chhoti ho
    good.sort(key=lambda t: (t[1] is None, t[1] or 0))
    for u, size in good[:3]:
        print(f"\n  --- {u}  ({size:,} byte)" if size else f"\n  --- {u}")
        txt, status = get_text(u, limit=60_000)
        if txt.startswith("__ERR__"):
            print(f"      {txt}")
            continue
        lines = [l for l in txt.split("\n") if l.strip()][:6]
        for l in lines:
            print(f"      {l[:300]}")
        # mutation jaisa column hai kya
        if lines:
            sep = "\t" if lines[0].count("\t") > lines[0].count(",") else ","
            hdr = [c.strip().strip('"') for c in lines[0].split(sep)]
            print(f"\n      columns ({len(hdr)}, separator {sep!r}):")
            for i, c in enumerate(hdr):
                mark = " <<<" if re.search(
                    r"mutat|variant|wild|affinity|kd|ki|ic50|pdb|uniprot|"
                    r"ligand|smiles", c, re.I) else ""
                print(f"        [{i:>2}] {c}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
