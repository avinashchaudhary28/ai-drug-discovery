"""Ek baar ka probe: BindingDB me mutation ka column hai ya nahi?

Kyun probe pehle: BindingDB ki poori file 594 MB ki hai. Use utaar kar
baad me pata chalna ki mutation ki jaankari hi nahi hai - wo waqt zaaya.
Aur aaj ChEMBL ke baare me meri dono maanyataayein galat nikli thin,
isliye yahan andaza bilkul nahi.

Tareeka: zip ki sirf SHURUAAT utaarte hain (HTTP Range se kuch MB), zip
ka local header padhte hain, aur deflate stream ko wahin se kholte hain.
Isse pehli kuch hazaar lines mil jati hain - header row aur kuch namune -
bina poori file ke.

Ye script kuch likhta nahi, sirf padhta aur chhapta hai. Saara chalne
wala hissa main() me hai, taaki selftest_zip.py ise import kar sake bina
network chhue - pehle ye module level pe tha aur import karte hi chal
padta tha.
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
HOST = "https://www.bindingdb.org"
PROBE_BYTES = 6 * 1024 * 1024        # itni shuruaat kaafi hai


def req(url, headers=None):
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
    """Zip ki shuruaat se pehli member ko jitna ho sake kholo.

    Zip ka local file header 30 byte ka hai:
      signature 4, version 2, flags 2, method 2, time 2, date 2,
      crc 4, csize 4, usize 4, namelen 2, extralen 2  = 30
    Pehle maine 28 byte ka format likha tha (ek field chhoot gaya) aur
    method galat jagah se padh raha tha. selftest_zip.py isi ko pakadta
    hai, bina network ke.
    """
    if blob[:4] != b"PK\x03\x04":
        raise RuntimeError(f"zip ka header nahi mila: {blob[:8]!r}")
    (_sig, _ver, _flags, method, _mt, _md,
     _crc, _csz, _usz, nlen, elen) = struct.unpack("<IHHHHHIIIHH", blob[:30])
    name = blob[30:30 + nlen].decode("utf-8", "replace")
    start = 30 + nlen + elen
    data = blob[start:]
    if method == 0:
        # Bina compression wale member pe uski LAMBAI maanni padti hai,
        # warna uske baad ka central directory (binary) bhi aa jata hai
        # aur text decode toot jata hai. selftest_zip ne yahi pakda.
        end = min(want_bytes, _usz) if _usz else want_bytes
        return name, data[:end]
    if method != 8:
        raise RuntimeError(f"compression method {method} ka intezaam nahi hai")
    d = zlib.decompressobj(-15)
    out = io.BytesIO()
    # Adhoora stream hai, isliye beech me rukna normal hai
    try:
        out.write(d.decompress(data, want_bytes))
    except zlib.error:
        pass
    return name, out.getvalue()


def find_links():
    links = []
    for page in [DL_PAGE] + ALT_PAGES:
        txt, status = get_text(page)
        if txt.startswith("__ERR__"):
            print(f"  {page} -> {txt}")
            continue
        found = sorted(set(re.findall(
            r'href=["\']?([^"\'>\s]*BindingDB[^"\'>\s]*\.(?:zip|tsv|gz))',
            txt, re.I)))
        print(f"  {page} -> HTTP {status}, {len(found)} BindingDB link")
        for f in found[:25]:
            print(f"      {f}")
        links += [f if f.startswith("http") else urllib.parse.urljoin(page, f)
                  for f in found]
    return links


def candidates(tsv):
    """Page pe mila link, aur uske andar likha SEEDHA path.

    Seedha path link ki query me maujood hai, isliye ye andaza nahi -
    probe se mila data hai.
    """
    out = []
    for u in tsv[:3]:
        out.append(u)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(u).query)
        for f in q.get("download_file", []):
            if f.startswith("/"):
                out.append(HOST + f)
    return list(dict.fromkeys(out))


def main():
    print("=" * 70)
    print("1) download page khulta hai kya, aur kaun si file maujood hai")
    print("=" * 70)
    links = find_links()
    tsv = [u for u in links if re.search(r"_All.*tsv.*\.zip$", u, re.I)]
    if not tsv:
        tsv = [u for u in links if re.search(r"tsv.*\.zip$", u, re.I)]
    print(f"\n  TSV jaisi file: {tsv[:5] or 'koi nahi mili'}")
    if not tsv:
        print("\n!! TSV ka link nahi mila. Aage badhne ka faida nahi - "
              "URL ka andaza lagana wahi galti hogi jo aaj do baar hui.")
        return 1

    print(f"\n{'=' * 70}\n2) file ka size aur Range chalta hai kya\n{'=' * 70}")
    url, info = None, None
    for c in candidates(tsv):
        i = head(c)
        ok = i.get("accept_ranges") and "html" not in (i.get("type") or "")
        print(f"  {'✓' if ok else '✗'} {c}\n      {json.dumps(i)}")
        if ok and url is None:
            url, info = c, i
    if url is None:
        print("\n  !! Kisi raaste pe Range nahi chalta ya zip nahi milti. "
              "Poori file utaarna ek alag faisla hai, isliye yahin ruk "
              "rahe hain.")
        return 1
    print(f"\n  chuna: {url}")

    print(f"\n{'=' * 70}\n3) shuruaat kholo aur columns dekho\n{'=' * 70}")
    blob, st = get_range(url, PROBE_BYTES)
    print(f"  {len(blob):,} byte utre (HTTP {st})")
    name, raw = unzip_head(blob)
    print(f"  zip member: {name}")
    text = raw.decode("utf-8", "replace")
    lines = text.split("\n")
    print(f"  {len(lines):,} lines khuli ({len(raw):,} byte)")
    if len(lines) < 2:
        print("  !! header ke baad kuch nahi khula - aage badhna bekaar")
        return 1

    hdr = lines[0].split("\t")
    print(f"\n  columns: {len(hdr)}")
    hits = [(i, c) for i, c in enumerate(hdr)
            if re.search(r"mutat|variant|wild", c, re.I)]
    print(f"\n  mutation/variant jaise column ({len(hits)}):")
    for i, c in hits:
        print(f"    [{i:>3}] {c}")

    print(f"\n  pehle 25 columns:")
    for i, c in enumerate(hdr[:25]):
        print(f"    [{i:>3}] {c}")

    rows = [l.split("\t") for l in lines[1:] if l.strip()]
    rows = [r for r in rows if len(r) >= len(hdr) - 5]
    print(f"\n  {len(rows):,} poori rows namune me")
    for i, c in hits:
        vals = [r[i] for r in rows if i < len(r) and r[i].strip()]
        print(f"\n    [{i}] {c}  ->  bhari hui {len(vals):,}/{len(rows):,}")
        for v in sorted(set(vals))[:6]:
            print(f"        namuna: {v[:110]}")

    print(f"\n{'=' * 70}\n4) faisla\n{'=' * 70}")
    filled = sum(len([r for r in rows if i < len(r) and r[i].strip()])
                 for i, _ in hits)
    if hits and filled:
        print(f"  mutation ka column MILA: {[c for _, c in hits]}")
        print(f"  namune me bhari hui values: {filled:,}")
        print("  >>> aage badhna theek hai")
        return 0

    print("  mutation ka alag column nahi mila ya khali hai.")
    print("  Target name me mutation likha hota hai kya:")
    ti = [i for i, c in enumerate(hdr) if re.search(r"target name", c, re.I)]
    any_hit = False
    for i in ti[:3]:
        vals = [r[i] for r in rows if i < len(r) and r[i].strip()]
        pat = [v for v in vals
               if re.search(r"[\[(]\s*[ACDEFGHIKLMNPQRSTVWY]\d+"
                            r"[ACDEFGHIKLMNPQRSTVWY]\s*[\])]", v)]
        print(f"    [{i}] {hdr[i]}: {len(pat):,}/{len(vals):,} me "
              f"mutation jaisa pattern")
        for v in pat[:5]:
            print(f"        {v[:110]}")
        any_hit = any_hit or bool(pat)
    if any_hit:
        print("  >>> target name se nikalna padega - ho sakta hai, par "
              "ChEMBL ke saaf field se kamzor hai")
        return 0
    print("  >>> BindingDB is kaam ke liye bekaar hai. Platinum aur "
          "published kinase panels pe jaana chahiye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
