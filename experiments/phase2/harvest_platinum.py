"""Phase 2: Platinum se ddG nikalna.

Platinum kyun khaas hai: yahi wo dataset hai jispe published state of the
art ke numbers bane hain (796 jode, unka 10-fold CV pe Pearson r = 0.20).
Isse humara number unke number ke SAATH wali axis pe baithta hai.

Ek faisla pehle se, kyunki baad me lalach aayega:
**Platinum pe TRAIN nahi karna hai.** Ye unka test set hai. Uspe train
karke "humara number behtar hai" kehna jhooth hoga. Ye sirf evaluation ke
liye hai, unke hi protocol pe. Isliye nateeje me 'hold_out_only' nishaan
likha jata hai.

ddG ka hisaab (probe ke asli data se jaancha gaya, andaza nahi):
    delta_k      = k_wt - k_mt
    fold_change  = +/- (k_mt / k_wt)
    => ddG = pK_mt - pK_wt = log10(k_wt / k_mt)
Jaanch: ek row me ph=5, k_wt=1, k_mt=3.2 -> delta_k -2.2 ✓,
fold_change -3.2 ✓. Convention pakki.

ddG ek ANUPAAT hai, isliye unit se farak nahi padta - bas dono number ek
hi unit me hone chahiye, aur Platinum me unit ka ek hi column hai.

Nateeja: results/platinum_ddg.csv, results/platinum_report.json
"""
import json
import os
import pathlib
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutnorm import normalize                             # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
WORK = HERE / "work_plat"

# Ye URL probe se mila (home page -> /platinum/data -> file ka link).
# Andaza nahi lagaya gaya.
URL = ("https://biosig.lab.uq.edu.au/platinum/static/platinum_flat_file.csv")
UA = "ai-drug-discovery/phase2 (research)"


def download():
    WORK.mkdir(exist_ok=True)
    path = WORK / "platinum_flat_file.csv"
    if path.exists() and path.stat().st_size > 100_000:
        print(f"  cache se: {path.stat().st_size:,} byte")
        return path
    r = urllib.request.Request(URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(r, timeout=300) as resp:
        blob = resp.read()
    path.write_bytes(blob)
    print(f"  utra: {len(blob):,} byte")
    if len(blob) < 100_000:
        raise RuntimeError(f"file bahut chhoti hai ({len(blob)} byte) - "
                           f"shayad error page aaya hai")
    return path


def num(s):
    """Platinum me 'NA', 'NO', 'NR' jaise text bhare hote hain."""
    return pd.to_numeric(s, errors="coerce")


def compute(df):
    """-> (saaf dataframe, kaaran-wise hatai gayi gintee)"""
    drop = {}
    n0 = len(df)

    kw = num(df["affin.k_wt"])
    km = num(df["affin.k_mt"])
    ok = kw.notna() & km.notna() & (kw > 0) & (km > 0)
    drop["k_wt ya k_mt nahi mila / <=0"] = int((~ok).sum())
    d = df[ok].copy()
    kw, km = kw[ok], km[ok]

    # Agar saari rows yahin gir gayin to aage ka code uljha hua KeyError
    # deta tha. Asli file ka format badal jaye to yahi hota, aur wajah
    # samajh na aati. Isliye saaf jawab:
    if d.empty:
        print("  !! k_wt/k_mt ke baad ek bhi row nahi bachi - "
              "file ka format dekhna padega")
        for c, dt in (("ddG", "float64"), ("k_wt", "float64"),
                      ("k_mt", "float64"), ("mutation_norm", "object"),
                      ("mut_parsed", "bool"), ("is_kinase", "bool")):
            d[c] = pd.Series(dtype=dt)
        return d, drop

    # ddG = log10(k_wt / k_mt). Anupaat hai, isliye unit se farak nahi.
    d["ddG"] = np.log10(kw / km)
    d["k_wt"] = kw
    d["k_mt"] = km

    norm = d["mutation"].map(normalize)
    d["mutation_norm"] = ["+".join(t) for t, _, _ in norm]
    d["mut_parsed"] = [p for _, _, p in norm]
    bad = d[~d.mut_parsed]
    drop["mutation samajh nahi aaya"] = int(len(bad))
    if len(bad):
        print("    samajh na aaye mutation naam:")
        for k, v in bad["mutation"].value_counts().head(8).items():
            print(f"      {v:>5,}  {k}")
    d = d[d.mut_parsed].copy()

    sm = d["lig.canonical_smiles"].astype(str).str.strip()
    # Lambai ki seema 1 hai, 3 nahi: "CCO" (ethanol) sahi SMILES hai aur
    # 3 akshar ki hai. Pehle seema 3 thi aur aisi chhoti par sahi SMILES
    # chup-chaap gir jati - test ne pakda. "NA"/"NO" wagairah neeche ki
    # list se hatte hain, lambai se nahi.
    has = sm.str.len().gt(1) & ~sm.str.upper().isin(
        ["NA", "NAN", "NONE", "NO", "NR", "-", "N/A"])
    drop["SMILES nahi mila"] = int((~has).sum())
    d = d[has].copy()

    d["is_kinase"] = (d["prot.protein_class"].astype(str)
                      .str.contains("kinase", case=False, na=False))
    print(f"  {n0:,} rows -> {len(d):,} istemaal layak")
    for k, v in drop.items():
        if v:
            print(f"      hataya {v:>6,}  {k}")
    return d, drop


def sign_check(d):
    """Mera ddG aur Platinum ka fold_change ek disha me hain ya nahi?

    Sign ulta ho jana sabse khatarnak galti hai - resistance ko sudhaar
    dikha deti hai - aur wo kabhi error nahi deti. Isliye Platinum ke
    apne column se milaya jata hai.
    """
    fc = num(d["affin.fold_change"])
    both = d.assign(fc=fc)
    both = both[both.fc.notna() & (both.fc.abs() > 1.001)
                & (both.ddG.abs() > 0.01)]
    if both.empty:
        return {"compared": 0, "note": "tulna ke layak row nahi mili"}
    agree = (np.sign(both.ddG) == np.sign(both.fc)).sum()
    out = {"compared": int(len(both)), "agree": int(agree),
           "disagree": int(len(both) - agree),
           "agree_frac": float(agree / len(both))}
    print(f"\n  sign ki jaanch (Platinum ke fold_change se):")
    print(f"    tulna ki gayi : {out['compared']:,}")
    print(f"    ek disha me   : {out['agree']:,} ({out['agree_frac']:.1%})")
    print(f"    ulti disha    : {out['disagree']:,}")
    if out["agree_frac"] < 0.95:
        print(f"    !! 95% se kam - matlab mera ddG ka hisaab galat hai, "
              f"ya fold_change ka matlab wo nahi jo main samajh raha hoon. "
              f"Aage badhna theek nahi.")
        print(both[both.ddG * both.fc < 0]
              [["mutation", "k_wt", "k_mt", "ddG", "fc", "affin.unit"]]
              .head(10).to_string(index=False))
    return out


def main():
    print("[1/3] file utaro")
    path = download()
    df = pd.read_csv(path, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    print("\n[2/3] ddG ka hisaab")
    d, drop = compute(df)
    sc = sign_check(d)
    if sc.get("agree_frac", 1.0) < 0.95:
        RES.mkdir(exist_ok=True)
        (RES / "platinum_report.json").write_text(json.dumps(
            {"sign_check": sc, "verdict": "SIGN GALAT - aage nahi badhe"},
            indent=2))
        return 1

    print("\n[3/3] nateeja")
    RES.mkdir(exist_ok=True)
    cols = ["mutation", "mutation_norm", "mut.uniprot", "prot.molecule_name",
            "prot.protein_class", "prot.organism", "is_kinase",
            "lig.canonical_smiles", "affin.lig_name", "affin.unit",
            "affin.exptal_method", "k_wt", "k_mt", "ddG",
            "mut.is_single_point", "mut.in_binding_site",
            "mut.distance_to_lig", "affin.pdb_id", "mut.wt_pdb",
            "mut.mt_pdb", "mut.pmid", "mut.doi"]
    out = d[[c for c in cols if c in d.columns]].copy()
    out["source"] = "platinum"
    # Nishaan: ye sirf evaluation ke liye hai, training ke liye nahi
    out["hold_out_only"] = True
    out.sort_values("ddG").to_csv(RES / "platinum_ddg.csv", index=False)

    k = out[out.is_kinase]
    print(f"\n{'=' * 68}\nNATEEJA\n{'=' * 68}\n")
    print(f"  ddG jode              : {len(out):,}")
    print(f"  alag mutations        : {out.mutation_norm.nunique()}")
    print(f"  alag proteins (uniprot): {out['mut.uniprot'].nunique()}")
    print(f"  alag ligands          : {out['lig.canonical_smiles'].nunique()}")
    print(f"  single point mutation : "
          f"{(out['mut.is_single_point'].astype(str).str.upper() == 'YES').sum():,}")
    print(f"  binding site me       : "
          f"{(out['mut.in_binding_site'].astype(str).str.upper() == 'YES').sum():,}")
    print(f"\n  tulna: published SOTA ne Platinum ke 796 jode istemaal kiye")
    print(f"\n  KINASE: {len(k):,} jode, {k['mut.uniprot'].nunique()} proteins")
    print(f"  kinase nahi: {len(out) - len(k):,}")

    print(f"\n  ddG ka faila: min {out.ddG.min():.2f}  max {out.ddG.max():.2f}  "
          f"median {out.ddG.median():.2f}  std {out.ddG.std():.2f}")
    for thr in (0.5, 1.0, 2.0):
        n = int((out.ddG.abs() >= thr).sum())
        print(f"    |ddG| >= {thr}: {n:,} ({n / len(out):.0%})")

    print(f"\n  unit:")
    print(out["affin.unit"].value_counts().head(8).to_string())
    print(f"\n  top protein class:")
    print(out["prot.protein_class"].value_counts().head(12).to_string())
    print(f"\n  top proteins:")
    print(out["prot.molecule_name"].value_counts().head(12).to_string())

    rep = {
        "rows_raw": int(len(df)),
        "pairs": int(len(out)),
        "mutations": int(out.mutation_norm.nunique()),
        "proteins": int(out["mut.uniprot"].nunique()),
        "ligands": int(out["lig.canonical_smiles"].nunique()),
        "kinase_pairs": int(len(k)),
        "kinase_proteins": int(k["mut.uniprot"].nunique()),
        "ddG_abs_ge_1": int((out.ddG.abs() >= 1).sum()),
        "dropped": drop,
        "sign_check": sc,
        "by_unit": {str(a): int(b) for a, b in
                    out["affin.unit"].value_counts().items()},
        "by_protein_class": {str(a): int(b) for a, b in
                             out["prot.protein_class"].value_counts()
                             .head(30).items()},
        "hold_out_only": True,
        "note": ("Platinum published SOTA ka test set hai. Ispe train nahi "
                 "karna - sirf evaluation, unke hi protocol pe."),
    }
    (RES / "platinum_report.json").write_text(json.dumps(rep, indent=2))
    md = [f"**Platinum: {len(out):,} ddG jode** "
          f"({out['mut.uniprot'].nunique()} proteins, "
          f"{out.mutation_norm.nunique()} mutations)", "",
          f"- kinase: {len(k):,} jode",
          f"- |ddG| >= 1.0: {int((out.ddG.abs() >= 1).sum()):,}",
          f"- sign ki jaanch: {sc.get('agree_frac', 0):.1%} ek disha me", "",
          "**Sirf evaluation ke liye** - ye published SOTA ka test set hai."]
    (RES / "RESULT.md").write_text("\n".join(md))
    print(f"\n  files: {RES}/platinum_ddg.csv, platinum_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
