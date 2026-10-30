"""BindingDB ka faisla: Target Name me mutation likha hota hai ya nahi?

Probe ne ek baat PAKKI kar di: 640 columns me mutation/variant ka koi
column nahi hai. Header poori file ka hota hai, isliye wo poora jawab
hai.

Par ek baat pakki nahi hui: Target Name me mutation likha hota hai kya.
Namune me sirf 1,568 rows thin - poori file ka 0.05%, aur wo bhi
shuruaat se, yaani sabse purani entries. Us namune pe "nahi hai" kehna
galat hoga.

Isliye poori file (594 MB) stream hoti hai aur chalte-chalte khulti hai.
Disk pe kuch nahi rakha jata, memory me ek chunk se zyada nahi. Sirf
gintee hoti hai - dataset abhi nahi banta. Faisla pehle, mehnat baad me.

Nateeja: results/bindingdb_scan.json
"""
import io
import json
import os
import pathlib
import re
import sys
import urllib.request
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_bindingdb import HOST, UA, req                 # noqa: E402

RES = pathlib.Path(__file__).resolve().parent / "results"
URL = HOST + "/rwd/bind/downloads/BindingDB_All_202610_tsv.zip"
CHUNK = 4 * 1024 * 1024

AA = "ACDEFGHIKLMNPQRSTVWY"
# Mutation jaisa pattern: kisi bracket ya alag-thalag likha "T790M"
MUT = re.compile(rf"[\[(]\s*[{AA}]\d{{1,4}}[{AA}]\s*[\])]"
                 rf"|(?<![A-Za-z0-9])[{AA}]\d{{2,4}}[{AA}](?![A-Za-z0-9])")


def inflate_stream(chunks, stats=None):
    """Zip ki stream ko chalte-chalte kholo -> uncompressed bytes ke tukde.

    Pehla tukda local header rakhta hai; wahan se aage raw deflate hai.

    stats me hisaab likha jata hai (bytes_in, bytes_out, eof). Pehle ye
    nahi tha, aur stream beech me khatam ho gayi to ye chup-chaap ruk
    gaya - 73,036 rows padh kar poori file ka faisla de diya. Ab main()
    is hisaab ko milata hai aur na mile to rukta hai.
    """
    st = stats if stats is not None else {}
    st.setdefault("bytes_in", 0)
    st.setdefault("bytes_out", 0)
    st["eof"] = False
    d = None
    head = b""
    for raw in chunks:
        st["bytes_in"] += len(raw)
        if d is None:
            head += raw
            if len(head) < 64:
                continue
            if head[:4] != b"PK\x03\x04":
                raise RuntimeError(f"zip header nahi mila: {head[:8]!r}")
            nlen = int.from_bytes(head[26:28], "little")
            elen = int.from_bytes(head[28:30], "little")
            method = int.from_bytes(head[8:10], "little")
            if method != 8:
                raise RuntimeError(f"method {method} ka intezaam nahi")
            start = 30 + nlen + elen
            if len(head) < start:
                continue
            d = zlib.decompressobj(-15)
            raw = head[start:]
            head = b""
        try:
            out = d.decompress(raw)
        except zlib.error as e:
            raise RuntimeError(f"deflate toota: {e}") from e
        if out:
            st["bytes_out"] += len(out)
            yield out
        if d.eof:
            st["eof"] = True
            return


def lines_from(chunks):
    """Bytes ke tukdon se poori lines nikalo (aadhi line agli baar)."""
    buf = b""
    for part in chunks:
        buf += part
        while True:
            i = buf.find(b"\n")
            if i < 0:
                break
            yield buf[:i]
            buf = buf[i + 1:]
    if buf:
        yield buf


def scan(chunks, progress_every=100_000, stats=None):
    """-> gintee ka dict. Memory me namune ke alawa kuch nahi rakha jata."""
    hdr = None
    ti = None
    n = 0
    hit_target = 0
    hit_anywhere = 0
    examples = {}
    for raw in lines_from(inflate_stream(chunks, stats)):
        line = raw.decode("utf-8", "replace").rstrip("\r")
        if hdr is None:
            hdr = line.split("\t")
            cand = [i for i, c in enumerate(hdr)
                    if c.strip().lower() == "target name"]
            if not cand:
                cand = [i for i, c in enumerate(hdr)
                        if re.search(r"target name", c, re.I)]
            if not cand:
                raise RuntimeError(
                    f"'Target Name' column nahi mila. columns: {len(hdr)}")
            ti = cand[0]
            print(f"  columns {len(hdr)}, 'Target Name' index {ti} "
                  f"({hdr[ti]!r})", flush=True)
            continue
        n += 1
        if MUT.search(line):
            hit_anywhere += 1
        cols = line.split("\t")
        if ti < len(cols):
            name = cols[ti]
            if MUT.search(name):
                hit_target += 1
                if len(examples) < 60:
                    examples[name.strip()[:160]] = examples.get(
                        name.strip()[:160], 0) + 1
        if progress_every and n % progress_every == 0:
            cin = (stats or {}).get("bytes_in", 0)
            print(f"    {n:,} rows | target-name hits {hit_target:,} | "
                  f"line hits {hit_anywhere:,} | {cin / 1e6:,.0f} MB padhi",
                  flush=True)
    return {"rows": n, "columns": len(hdr or []),
            "target_name_index": ti,
            "hits_in_target_name": hit_target,
            "hits_anywhere_in_row": hit_anywhere,
            "examples": examples}


RANGE_SPAN = 32 * 1024 * 1024


def fetch_range(url, start, end):
    """Ek Range maango -> bytes."""
    with urllib.request.urlopen(
            req(url, {"Range": f"bytes={start}-{end}"}), timeout=600) as r:
        return r.read()


def total_size(url, fetch=None):
    fetch = fetch or fetch_range
    with urllib.request.urlopen(
            req(url, {"Range": "bytes=0-0"}), timeout=120) as r:
        cr = r.headers.get("Content-Range") or ""
        if "/" in cr:
            return int(cr.split("/")[-1])
    raise RuntimeError("file ka size pata nahi chala (Content-Range nahi aaya)")


def ranged_chunks(url, stats, total, span=RANGE_SPAN, fetch=None,
                  tries=5):
    """File ko tukdon me maango, ek lambi stream ki jagah.

    Ek hi lambe connection pe server ne 594 MB me se sirf 10.5 MB bhej
    kar connection kaat diya tha - aur generator chup-chaap ruk gaya.
    Range chalta hai (206 aata hai), isliye tukdon me maangna zyada
    bharosemand hai: har tukde ki apni koshish, aur aakhir me gintee
    milayi jati hai.
    """
    fetch = fetch or fetch_range
    stats["content_length"] = total
    got = 0
    start = 0
    while start < total:
        end = min(start + span, total) - 1
        last = None
        for attempt in range(1, tries + 1):
            try:
                blob = fetch(url, start, end)
                if not blob:
                    raise RuntimeError("khali jawab")
                break
            except Exception as e:                        # noqa: BLE001
                last = e
                blob = None
                print(f"    tukda {start:,}-{end:,} koshish {attempt} fail: "
                      f"{e!r}", flush=True)
        if blob is None:
            raise RuntimeError(
                f"{tries} koshish ke baad bhi tukda {start:,}-{end:,} nahi "
                f"aaya: {last!r}")
        got += len(blob)
        start += len(blob)
        stats["http_bytes"] = got
        yield blob
    stats["http_bytes"] = got


def http_chunks(url, stats):
    with urllib.request.urlopen(req(url), timeout=900) as r:
        total = int(r.headers.get("Content-Length") or 0)
        stats["content_length"] = total
        print(f"  HTTP {r.status}, {total:,} byte", flush=True)
        got = 0
        while True:
            b = r.read(CHUNK)
            if not b:
                break
            got += len(b)
            yield b
        stats["http_bytes"] = got


def main():
    RES.mkdir(exist_ok=True)
    print(f"stream: {URL}")
    stats = {}
    total = total_size(URL)
    print(f"  file ka size: {total:,} byte, "
          f"{RANGE_SPAN // (1024 * 1024)} MB ke tukdon me maangenge")
    res = scan(ranged_chunks(URL, stats, total), stats=stats)
    res["stream"] = dict(stats)

    # POORI file padhi gayi ya nahi - ye jaanch pehle nahi thi, aur usi
    # ki kami se pichla run 73,036 rows pe ruk kar poore faisle ka
    # elaan kar gaya. Adhoori file pe faisla dena sabse bura nateeja hai.
    cl = stats.get("content_length") or 0
    got = stats.get("bytes_in") or 0
    print(f"\n  hisaab: {got:,} / {cl:,} byte padhe, "
          f"deflate eof {stats.get('eof')}, "
          f"{stats.get('bytes_out', 0) / 1e9:.2f} GB khuli")
    problems = []
    # Mukhya niyam: deflate stream khud khatam hua ya nahi. Yahi pakka
    # pata hai. (byte ki gintee ispe kaam nahi karti - stream khatam
    # hote hi hum ruk jate hain, isliye zip ka aakhiri hissa nahi padha
    # jata aur gintee thodi kam rehti hai.)
    if not stats.get("eof"):
        problems.append(
            f"deflate stream poora khatam nahi hua - {got:,} byte padhe "
            f"the, {cl:,} declare the ({got / cl:.1%} agar cl sahi hai)")
    if res["rows"] < 1_000_000:
        problems.append(f"sirf {res['rows']:,} rows - BindingDB me lagbhag "
                        f"28 lakh hain, to ye file adhoori hai")
    if problems:
        res["verdict"] = "ADHOORA - faisla nahi"
        res["problems"] = problems
        (RES / "bindingdb_scan.json").write_text(json.dumps(res, indent=2))
        print("\n!! FILE ADHOORI PADHI GAYI - faisla nahi de rahe:")
        for p_ in problems:
            print(f"     - {p_}")
        print(f"\n  (jo gina gaya: {res['rows']:,} rows, "
              f"{res['hits_in_target_name']:,} target-name hits - ye "
              f"aankde poore nahi hain)")
        print(f"  file: {RES}/bindingdb_scan.json")
        return 1

    print(f"\n{'=' * 68}\nNATEEJA\n{'=' * 68}\n")
    print(f"  rows                        : {res['rows']:,}")
    print(f"  Target Name me mutation     : {res['hits_in_target_name']:,}")
    print(f"  kisi bhi column me mutation : {res['hits_anywhere_in_row']:,}")
    if res["examples"]:
        print(f"\n  namune ({len(res['examples'])} alag naam):")
        for k, v in sorted(res["examples"].items(), key=lambda kv: -kv[1])[:25]:
            print(f"    {v:>6,}  {k}")

    hits = res["hits_in_target_name"]
    frac = hits / max(res["rows"], 1)
    print(f"\n{'=' * 68}\nFAISLA\n{'=' * 68}")
    if hits >= 5000:
        print(f"  Target Name me {hits:,} rows ({frac:.2%}) pe mutation likha "
              f"hai.\n  >>> BindingDB kaam ka hai, par naam se parse karna "
              f"padega - ChEMBL ke saaf field se kamzor, isliye teesra "
              f"source banega, pehla nahi.")
        res["verdict"] = "usable_via_target_name"
    elif hits > 0:
        print(f"  Target Name me sirf {hits:,} rows ({frac:.3%}) pe mutation "
              f"jaisa kuch hai.\n  >>> Itna kam hai ki alag source banane "
              f"layak nahi. Platinum aur published kinase panels pe jaana "
              f"chahiye.")
        res["verdict"] = "too_sparse"
    else:
        print(f"  Target Name me ek bhi mutation nahi mila, poori file me."
              f"\n  >>> BindingDB is kaam ke liye bekaar hai. Platinum aur "
              f"published kinase panels pe jaana chahiye.")
        res["verdict"] = "unusable"
    (RES / "bindingdb_scan.json").write_text(json.dumps(res, indent=2))
    print(f"\n  file: {RES}/bindingdb_scan.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
