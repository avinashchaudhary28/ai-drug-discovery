"""Bina network ke test: poori file scan karne wala code.

Ye 594 MB wali file pe chalega, isliye uska tareeka pehle chhote zip pe
jaanch lena chahiye. Khaas taur pe teen cheezein jo galat hon to chup-chaap
galat gintee degi:

  1. aadhi line jo ek chunk ke aakhir me toot jaye - wo agle chunk se
     judni chahiye, warna rows gum ho jati hain
  2. mutation pehchanne wala pattern - "T790M" pakadna chahiye par
     "ABC123" jaisa kuch nahi
  3. 'Target Name' column ka index - galat index chup-chaap zero hits
     dega aur hum keh denge "BindingDB bekaar hai"
"""
import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_bindingdb import MUT, inflate_stream, lines_from, scan  # noqa: E402

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


def zip_bytes(text, inner="BindingDB_All.tsv"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(inner, text)
    return buf.getvalue()


def chunks_of(blob, size):
    for i in range(0, len(blob), size):
        yield blob[i:i + size]


print("mutation pehchanne wala pattern")
for s in ["Epidermal growth factor receptor [L858R]",
          "Tyrosine-protein kinase ABL1 (T315I)",
          "EGFR T790M kinase domain",
          "KIT D816V"]:
    check(f"pakadta hai: {s[:38]}", bool(MUT.search(s)), True)
for s in ["Epidermal growth factor receptor",
          "Carbonic anhydrase 2",
          "Cytochrome P450 3A4",
          "ABC123 transporter",
          "Protein kinase C alpha"]:
    check(f"nahi pakadta: {s[:38]}", bool(MUT.search(s)), False)

print("\nlines_from: aadhi line chunk ke beech me toot jaye")
data = b"\n".join(f"line{i}".encode() for i in range(200))
got = list(lines_from(chunks_of(data, 7)))      # 7 byte ke tukde = har line tooti
check("saari 200 lines mili", len(got), 200)
check("pehli line sahi", got[0], b"line0")
check("aakhiri line sahi", got[-1], b"line199")

print("\ninflate_stream: zip chalte-chalte khulti hai")
TEXT = "a\tb\nc\td\n" * 500
blob = zip_bytes(TEXT)
out = b"".join(inflate_stream(chunks_of(blob, 64)))
check("poora text khula", out.decode(), TEXT)

print("\nscan: poora raasta ek saath")
HDR = "\t".join(["id", "Ligand SMILES", "Target Name", "Ki (nM)"])
rows = []
for i in range(300):
    # har teesri row me mutation
    nm = "Kinase ABL1 (T315I)" if i % 3 == 0 else "Kinase ABL1"
    rows.append("\t".join([str(i), "CCO", nm, "10"]))
text = HDR + "\n" + "\n".join(rows) + "\n"
res = scan(chunks_of(zip_bytes(text), 100), progress_every=0)
check("saari rows gini", res["rows"], 300)
check("Target Name ka index mila", res["target_name_index"], 2)
check("mutation wali rows gini", res["hits_in_target_name"], 100)
check("namune jama hue", len(res["examples"]) >= 1, True)

print("\nscan: column ka naam badal jaye to RUKNA chahiye")
bad = "\t".join(["id", "Ligand SMILES", "Protein", "Ki (nM)"]) + "\nx\ty\tz\tw\n"
try:
    scan(chunks_of(zip_bytes(bad), 100), progress_every=0)
    check("Target Name na ho to rukta hai", "chala gaya", "RuntimeError")
except RuntimeError as e:
    check("Target Name na ho to rukta hai",
          "Target Name" in str(e), True)

print("\nscan: ek bhi mutation na ho to zero")
plain = HDR + "\n" + "\n".join(
    "\t".join([str(i), "CCO", "Kinase ABL1", "10"]) for i in range(50)) + "\n"
res = scan(chunks_of(zip_bytes(plain), 100), progress_every=0)
check("50 rows", res["rows"], 50)
check("zero hits", res["hits_in_target_name"], 0)

print("\nADHOORI stream pe hisaab galat nikalna chahiye")
# Yahi wo halat thi jisme pichla run chup-chaap faisla de gaya
full = zip_bytes(text)
half = full[: len(full) // 3]
st = {}
res = scan(chunks_of(half, 100), progress_every=0, stats=st)
check("adhoori stream pe eof FALSE", st.get("eof"), False)
check("adhoori stream pe kuch rows to aati hain", res["rows"] > 0, True)
check("par poori nahi", res["rows"] < 300, True)

st = {}
res = scan(chunks_of(full, 100), progress_every=0, stats=st)
check("poori stream pe eof TRUE", st.get("eof"), True)
check("poori stream pe saari rows", res["rows"], 300)
# inflate_stream deflate khatam hote hi ruk jata hai, isliye zip ka
# aakhiri central directory (kuch sau byte) nahi padha jata - ye theek
# hai. Isliye bytes_in poori file se thoda kam hoga.
check("bytes_in file se zyada nahi", st.get("bytes_in") <= len(full), True)
check("bytes_in lagbhag poori file", st.get("bytes_in") > len(full) - 500,
      True)
check("bytes_out gina gaya", st.get("bytes_out"), len(text.encode()))

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas (adhoori stream ki jaanch ke saath)")

print("\nranged_chunks: tukdon me maangna, aur fail hone pe dobara")
from scan_bindingdb import ranged_chunks                  # noqa: E402

blob = zip_bytes(text)
TOTAL = len(blob)


def ok_fetch(url, start, end):
    return blob[start:end + 1]


st = {}
out = b"".join(ranged_chunks("u", st, TOTAL, span=100, fetch=ok_fetch))
check("tukdon se poori file bani", out, blob)
check("gintee poori", st.get("http_bytes"), TOTAL)

# Ek tukda do baar fail ho, phir chale
state = {"fails": 0}


def flaky(url, start, end):
    if start == 100 and state["fails"] < 2:
        state["fails"] += 1
        raise OSError("connection reset")
    return blob[start:end + 1]


st = {}
out = b"".join(ranged_chunks("u", st, TOTAL, span=100, fetch=flaky))
check("fail hone pe dobara koshish", out, blob)
check("do baar fail hua tha", state["fails"], 2)


def always_fail(url, start, end):
    raise OSError("connection reset")


try:
    list(ranged_chunks("u", {}, TOTAL, span=100, fetch=always_fail, tries=2))
    check("hamesha fail ho to RUKNA chahiye", "chala gaya", "RuntimeError")
except RuntimeError as e:
    check("hamesha fail ho to RUKNA chahiye", "nahi aaya" in str(e), True)

# Poora raasta: tukdon se scan
st = {}
res = scan(ranged_chunks("u", st, TOTAL, span=100, fetch=ok_fetch),
           progress_every=0, stats=st)
check("tukdon se scan: saari rows", res["rows"], 300)
check("tukdon se scan: eof TRUE", st.get("eof"), True)

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas")
