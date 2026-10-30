"""Bina network ke test: zip ki shuruaat kholne wala code.

Pichla version Actions pe gira tha - struct ka format 28 byte ka tha
jabki zip ka local header 30 byte ka hai, aur method galat jagah se
padha ja raha tha. Ye galti yahan pakdi jani chahiye thi, 600 MB wali
file pe jaate waqt nahi.

Yahan asli zip banaya jata hai (zipfile se), aur phir usi tareeke se
khola jata hai jo BindingDB pe lagta hai - adhoore blob se bhi.
"""
import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_bindingdb import unzip_head                    # noqa: E402

FAIL = []


def check(name, got, want):
    if got == want:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}\n      mila : {got!r}\n      chahiye: {want!r}")
        FAIL.append(name)


def make_zip(inner_name, text, method=zipfile.ZIP_DEFLATED):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", method) as z:
        z.writestr(inner_name, text)
    return buf.getvalue()


HDR = "\t".join(["Col A", "Col B", "Mutation"])
BODY = "\n".join("\t".join([f"a{i}", f"b{i}", "T790M" if i % 2 else ""])
                 for i in range(2000))
TEXT = HDR + "\n" + BODY

print("deflate wala zip")
blob = make_zip("BindingDB_All.tsv", TEXT)
name, raw = unzip_head(blob)
check("member ka naam mila", name, "BindingDB_All.tsv")
out = raw.decode("utf-8", "replace")
check("header row sahi khula", out.split("\n")[0], HDR)
check("Mutation column naam me hai", "Mutation" in out.split("\n")[0], True)
check("data rows bhi aayin", out.split("\n")[1].startswith("a0\t"), True)
check("poora text khula", out.startswith(TEXT[:500]), True)

print("\nbina compression (stored) wala zip")
blob = make_zip("x.tsv", TEXT, zipfile.ZIP_STORED)
name, raw = unzip_head(blob)
check("stored bhi khulta hai", raw.decode().split("\n")[0], HDR)

print("\nADHOORA blob (jaisa HTTP Range se aata hai)")
blob = make_zip("BindingDB_All.tsv", TEXT)
# Aadhi file hi do - asli halat yahi hai
half = blob[: len(blob) // 2]
name, raw = unzip_head(half)
out = raw.decode("utf-8", "replace")
check("adhoore blob se bhi header mila", out.split("\n")[0], HDR)
check("adhoore blob se kuch data bhi mila", len(out.split("\n")) > 10, True)

print("\njo zip na ho")
try:
    unzip_head(b"<html>not a zip</html>")
    check("zip na hone pe rukta hai", "chala gaya", "RuntimeError")
except RuntimeError as e:
    check("zip na hone pe rukta hai", "header nahi mila" in str(e), True)

print()
if FAIL:
    print(f"✗ {len(FAIL)} test fail: {FAIL}")
    sys.exit(1)
print("✓ saare test paas")
