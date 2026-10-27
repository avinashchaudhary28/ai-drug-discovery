"""ChEMBL REST ke liye chhota, bharosemand client.

Yahan do hi kaam hain:
  - get(): retry ke saath ek request
  - fetch_all(): paginate, aur agar ChEMBL deep paging pe mana kare to
    query ko tukdon me tod kar poora data lo

Deep paging ka masla asli hai: ChEMBL bada offset reject kar deta hai.
Agar hum chup-chaap ruk jayein to aadha data milega aur koi error nahi
aayega - yaani galat jawab, bina shikayat. Isliye fetch_all() har tukde
ka count pehle poochta hai, aur jo tukda bahut bada ho use aage todta hai.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://www.ebi.ac.uk/chembl/api/data"
UA = "ai-drug-discovery/phase2 (research; contact via github)"

# ChEMBL ka offset ka hard limit. Isse neeche hi rehna hai.
MAX_OFFSET = 20000
PAGE = 1000


def get(path, params=None, tries=5):
    """Ek GET. path ya to '/activity.json' hai ya poora URL."""
    if path.startswith("http"):
        url = path
    else:
        url = BASE + path
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
    last = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            # 400 ka matlab query hi galat hai - retry bekaar hai
            if e.code in (400, 404):
                raise RuntimeError(f"{last} for {url}") from e
        except Exception as e:                       # noqa: BLE001
            last = repr(e)
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"{tries} koshish ke baad bhi nahi mila: {url} ({last})")


def count(path, params):
    """Kitne record hain - sirf page_meta.total_count ke liye ek request."""
    p = dict(params)
    p["limit"] = 1
    p["offset"] = 0
    d = get(path, p)
    return int(d["page_meta"]["total_count"])


def _page_through(path, params, cap, label, on_batch):
    """cap se chhote result set ko seedha paginate karo."""
    got = 0
    offset = 0
    while offset < cap:
        p = dict(params)
        p["limit"] = PAGE
        p["offset"] = offset
        d = get(path, p)
        key = [k for k in d if k != "page_meta"][0]
        rows = d[key]
        if not rows:
            break
        on_batch(rows)
        got += len(rows)
        offset += PAGE
        if not d["page_meta"].get("next"):
            break
    if got:
        print(f"    {label}: {got} rows", flush=True)
    return got


# Kitni rows chhooti to shikayat karni hai. 0 rakha hai: ek bhi row
# chupchaap chhootne nahi deni.
SHORTFALL = []


def fetch_all(path, params, splitters, on_batch, label=""):
    """Poora result set lao, chahe wo MAX_OFFSET se bada ho.

    splitters: [[{param: val}, ...], ...] - har andar ki list ko result set
    ka POORA batwara hona chahiye (koi row kisi tukde me na chhoote). Pehle
    splitter se todte hain, fir agle se, aur aise aage.

    Har tod ke baad tukdon ka jod parent ke count se milaya jata hai. Agar
    na mile to wo gintee SHORTFALL me darj hoti hai - isliye adhoora data
    chup-chaap nahi nikal sakta. Yahi sabse khatarnak surat thi: kam rows
    aati hain, koi error nahi aata, aur nateeja halka dikhta hai.
    """
    n = count(path, params)
    tag = label or json.dumps(
        {k: v for k, v in params.items() if k not in ("limit", "offset")})

    if n <= MAX_OFFSET or not splitters:
        if n > MAX_OFFSET:
            miss = n - MAX_OFFSET
            print(f"!! {tag}: {n} rows hain par sirf {MAX_OFFSET} tak "
                  f"pahunch sakte hain - {miss} rows CHHOOT RAHI HAIN "
                  f"(todne ke liye kuch bacha nahi).", flush=True)
            SHORTFALL.append({"slice": tag, "expected": n,
                              "got": MAX_OFFSET, "missing": miss})
        return _page_through(path, params, min(n, MAX_OFFSET), f"{tag} (n={n})",
                             on_batch)

    chunks, rest = splitters[0], splitters[1:]
    print(f"  {tag}: {n} rows - {len(chunks)} tukdon me tod rahe hain",
          flush=True)
    total = 0
    for extra in chunks:
        p = dict(params)
        p.update(extra)
        bit = ",".join(f"{k}={v}" for k, v in extra.items())
        total += fetch_all(path, p, rest, on_batch, label=f"{tag} | {bit}")

    if total != n:
        print(f"!! {tag}: tukdon ka jod {total} hai par parent count {n} - "
              f"{n - total} rows ka hisaab nahi mila. Batwara poora nahi hai.",
              flush=True)
        SHORTFALL.append({"slice": tag, "expected": n, "got": total,
                          "missing": n - total})
    return total
