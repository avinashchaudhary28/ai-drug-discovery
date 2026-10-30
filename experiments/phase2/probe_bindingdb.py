"""Ek baar ka probe: BindingDB me mutation ka column hai ya nahi?

Kyun probe pehle: BindingDB ki poori file 1-2 GB ki hai. Use utaar kar
baad me pata chalna ki mutation ki jaankari hi nahi hai - wo ghanta
zaaya. Aur aaj ChEMBL ke baare me meri dono maanyataayein galat nikli
thin, isliye yahan andaza bilkul nahi.

Tareeka: zip ki sirf SHURUAAT utaarte hain (HTTP Range se kuch MB), zip
ka local header padhte hain, aur deflate stream ko wahin se kholte hain.
Isse pehli kuch hazaar lines mil jati hain - header row aur kuch namune -
bina poori file ke.

Ye script kuch likhta nahi, sirf padhta aur chhapta hai.
"""
import io
import json
import re
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
import zlib

UA = "ai-drug-discovery/phase2 probe (research)"
DL_PAGE = "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp"
ALT_PAGES = ["https://www.bindingdb.org/bind/downloads/",
             "https://www.bindingdb.org/rwd/bind/downloads/"]
PROBE_BYTES = 6 * 1024 * 1024        # itni shuruaat kaafi hai


def req(url, headers=None, timeout=120):
    h = {"User-Agent": UA}
    h.update(headers or {})
    return urllib.request.Request(url, headers=h)


def get_text(url, limit=400_000):
    try:
        with urllib.request.urlopen(req(url), timeout=60) as r:
            return r.read(limit).decode("utf-8", "replace"), r.status
    except urllib.error.HTTPError as e:
        return "", e.code
    except Exception as e:                                # noqa: BLE001
        return f"__ERR__ {e!r}", None


def head(url):
    try:
        with urllib.request.urlopen(
                req(url, {"Range": "bytes=0-0"}), timeout=60) as r:
            cr = r.headers.get("Content-Range") or ""
            total = cr.split("/")[-1] if "/" in cr else r.headers.get(
                "Content-Length")
            return {"status": r.status, "total": total,
                    "accept_ranges": bool(cr),
                    "type": r.headers.get("Content-Type")}
    except urllib.error.HTTPError as e:
        return {"status": e.code}
    except Exception as e:                                # noqa: BLE001
        return {"error": repr(e)}


def get_range(url, nbytes):
    with urllib.request.urlopen(
            req(url, {"Range": f"bytes=0-{nbytes - 1}"}), timeout=300) as r:
        return r.read(), r.status


def unzip_head(blob, want_bytes=4_000_000):
    """Zip ki shuruaat se pehli member ko jitna ho sake kholo."""
    if blob[:4] != b"PK\x03\x04":
        raise RuntimeError(f"zip ka header nahi mila: {blob[:8]!r}")
    # local file header: 30 byte fix, phir naam aur extra
    (_, _, method, _, _, _, _, _, nlen, elen) = struct.unpack(
        "<IHHHHIIIHH", blob[:30])
    name = blob[30:30 + nlen].decode("utf-8", "replace")
    start = 30 + nlen + elen
    print(f"  zip member: {name}  (method {method})")
    data = blob[start:]
    if method == 0:
        return name, data[:want_bytes]
    if method != 8:
        raise RuntimeError(f"compression method {method} ka intezaam nahi hai")
    d = zlib.decompressobj(-15)
    out = io.BytesIO()
    # Adhoora stream hai, isliye 'incomplete' error aana normal hai
    try:
        out.write(d.decompress(data, want_bytes))
    except zlib.error as e:
        print(f"  (deflate adhoore stream pe ruka - normal hai: {e})")
    return name, out.getvalue()


print("=" * 70)
print("1) download page khulta hai kya, aur kaun si file maujood hai")
print("=" * 70)
links = []
for page in [DL_PAGE] + ALT_PAGES:
    txt, status = get_text(page)
    if txt.startswith("__ERR__"):
        print(f"  {page} -> {txt}")
        continue
    found = sorted(set(re.findall(
        r'href=["\']?([^"\'>\s]*BindingDB[^"\'>\s]*\.(?:zip|tsv|gz))', txt,
        re.I)))
    print(f"  {page} -> HTTP {status}, {len(found)} BindingDB link")
    for f in found[:25]:
        print(f"      {f}")
    links += [f if f.startswith("http")
              else urllib.parse.urljoin(page, f) for f in found]

# TSV wali file chunno
tsv = [u for u in links if re.search(r"_All.*tsv.*\.zip$", u, re.I)]
if not tsv:
    tsv = [u for u in links if re.search(r"tsv.*\.zip$", u, re.I)]
print(f"\n  TSV jaisi file: {tsv[:5] or 'koi nahi mili'}")

if not tsv:
    print("\n!! Download page se TSV ka link nahi mila. Aage badhne ka "
          "koi faida nahi - andaza lagakar URL banana wahi galti hogi "
          "jo aaj do baar ho chuki hai.")
    sys.exit(1)

url = tsv[0]
print(f"\n{'=' * 70}\n2) file ka size aur Range chalta hai kya\n{'=' * 70}")
print(f"  {url}")
info = head(url)
print(f"  {json.dumps(info)}")
if not info.get("accept_ranges"):
    print("  !! Range nahi chalta - poori file utaarni padegi. Aaj nahi.")
    sys.exit(1)

print(f"\n{'=' * 70}\n3) shuruaat kholo aur columns dekho\n{'=' * 70}")
blob, st = get_range(url, PROBE_BYTES)
print(f"  {len(blob):,} byte utre (HTTP {st})")
name, raw = unzip_head(blob)
text = raw.decode("utf-8", "replace")
lines = text.split("\n")
print(f"  {len(lines):,} lines khuli")

hdr = lines[0].split("\t")
print(f"\n  columns: {len(hdr)}")
hits = [(i, c) for i, c in enumerate(hdr)
        if re.search(r"mutat|variant|wild|chain sequence|uniprot", c, re.I)]
print(f"\n  kaam ke lagne wale column ({len(hits)}):")
for i, c in hits:
    print(f"    [{i:>3}] {c}")

print(f"\n  pehle 20 columns:")
for i, c in enumerate(hdr[:20]):
    print(f"    [{i:>3}] {c}")

# Un columns me kuch bhara hua hai ya khali?
rows = [l.split("\t") for l in lines[1:] if l.strip()]
rows = [r for r in rows if len(r) >= len(hdr) - 5]
print(f"\n  {len(rows):,} poori rows namune me")
for i, c in hits:
    vals = [r[i] for r in rows if i < len(r) and r[i].strip()]
    uniq = sorted(set(vals))[:6]
    print(f"\n    [{i}] {c}")
    print(f"        bhari hui: {len(vals):,}/{len(rows):,}")
    if uniq:
        for v in uniq:
            print(f"        namuna: {v[:110]}")

print(f"\n{'=' * 70}\n4) faisla\n{'=' * 70}")
mut_cols = [(i, c) for i, c in hits if re.search(r"mutat|variant", c, re.I)]
if mut_cols:
    filled = 0
    for i, c in mut_cols:
        filled += sum(1 for r in rows if i < len(r) and r[i].strip())
    print(f"  mutation ka column MILA: {[c for _, c in mut_cols]}")
    print(f"  namune me bhari hui values: {filled:,}")
    print("  >>> aage badhna theek hai")
else:
    print("  mutation ka alag column NAHI mila.")
    print("  Target name me mutation likha hota hai kya - wo dekhte hain:")
    ti = [i for i, c in enumerate(hdr)
          if re.search(r"target name", c, re.I)]
    for i in ti[:2]:
        vals = [r[i] for r in rows if i < len(r) and r[i].strip()]
        braces = [v for v in vals if re.search(r"[\[(][A-Z]\d+[A-Z][\])]", v)]
        print(f"    [{i}] {hdr[i]}: {len(braces):,}/{len(vals):,} me "
              f"mutation jaisa pattern")
        for v in braces[:5]:
            print(f"        {v[:110]}")
    print("  >>> agar yahan bhi kuch nahi, to BindingDB is kaam ke liye "
          "bekaar hai aur Platinum/kinase panels pe jaana chahiye")
