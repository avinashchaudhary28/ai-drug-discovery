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


def fetch_all(path, params, split_on, on_batch, label=""):
    """Poora result set lao, chahe wo MAX_OFFSET se bada ho.

    split_on: [(param_name, [value, ...]), ...] - agar result bahut bada ho
    to pehle param se tod denge, fir dusre se, aur aise aage.
    """
    n = count(path, params)
    tag = label or json.dumps(
        {k: v for k, v in params.items() if k not in ("limit", "offset")})
    if n <= MAX_OFFSET or not split_on:
        if n > MAX_OFFSET:
            # Yahan aana matlab todne ke liye kuch bacha nahi - chup na rahein.
            print(f"!! {tag}: {n} rows hain par sirf {MAX_OFFSET} tak "
                  f"pahunch sakte hain. Ye data ADHURA hai.", flush=True)
        return _page_through(path, params, min(n, MAX_OFFSET), f"{tag} (n={n})",
                             on_batch)

    name, values = split_on[0]
    rest = split_on[1:]
    print(f"  {tag}: {n} rows - '{name}' se {len(values)} tukdon me tod rahe hain",
          flush=True)
    total = 0
    for v in values:
        p = dict(params)
        p[name] = v
        total += fetch_all(path, p, rest, on_batch, label=f"{tag} | {name}={v}")
    return total
