"""
Davis ke protein naam se mutation parse karke asli mutant sequence banata hai.

Kyun: Davis dataset me 442 naam hain par sirf 379 unique sequences. Saare 45
mutation-labelled proteins (ABL1(E255K), EGFR(L858R), ...) ko unke wild-type ki
HUBAHU sequence di gayi hai. Yaani model ko ek hi input de kar alag-alag jawab
maanga jaa raha hai - ye training data ka contradiction hai, model ki kami nahi.

Mutation ka naam hi bata raha hai ki kya badalna chahiye, to hum wahi kar lete hain.

Naam ke patterns jo Davis me milte hain:
  ABL1(E255K)              point mutation
  ABL1(F317I)p             point + phosphorylated suffix
  KIT(V559D-T670I)         do point mutations
  EGFR(E746A750del)        746-750 residues ka deletion
  EGFR(L747S752del,P753S)  deletion + point
  EGFR(L858R,T790M)        do point (comma se)
  EGFR(L858RT790M)         do point (bina separator)
  FLT3(ITD)                internal tandem duplication - sequence pata nahi, skip
  JAK2(JH1domain-catalytic) domain ka naam, mutation nahi - skip

Position numbering: UniProt canonical sequence ke hisaab se 1-based. Davis ki
sequence usi canonical ki hoti hai, par kabhi offset ho sakta hai - isliye hum
pehle check karte hain ki us position pe wahi amino acid hai jo naam keh raha
hai. Match na ho to +/- 30 ki window me dhundhte hain. Phir bhi na mile to us
mutation ko SKIP karte hain, galat jagah badalne se behtar hai.
"""
import json
import re

AA = set("ACDEFGHIKLMNPQRSTVWY")
POINT = re.compile(r"^([A-Z])(\d+)([A-Z])$")
DEL = re.compile(r"^([A-Z])(\d+)([A-Z]?)(\d+)del$")
MULTI = re.compile(r"(?=([A-Z]\d+[A-Z]))")


def split_edits(inside):
    """Bracket ke andar ka text -> edit strings ki list. Samajh na aaye to []."""
    inside = inside.strip()
    if not inside:
        return []
    # phosphorylation ka suffix matlab nahi rakhta sequence ke liye
    if inside.lower() in ("itd", "jh1domain-catalytic", "jh2domain-pseudokinase"):
        return []
    parts = re.split(r"[,\-]", inside)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if DEL.match(p) or POINT.match(p):
            out.append(p)
            continue
        # "L858RT790M" jaise bina separator wale -> saare X###Y nikaalo
        found = [m.group(1) for m in MULTI.finditer(p)]
        if found and "".join(found) == p:
            out.extend(found)
        elif found:
            out.extend(found)          # aadha samjha, jitna mila utna lo
        else:
            out.append("?" + p)       # samajh nahi aaya, caller skip karega
    return out


def find_pos(seq, aa, pos, offset=0):
    """
    1-based `pos` pe (gene ke offset ke saath) `aa` hona chahiye.
    Returns (0-based index, offset) ya (None, None).

    Yahan koi "aas-paas dhundho" nahi hai. Offset gene ke saare mutations se
    pehle tay hota hai (gene_offset dekho), aur bas wahi try karte hain. Agar
    us jagah wo amino acid nahi hai to mutation SKIP hota hai - kyunki paas ke
    kisi same-letter residue ko badal dena chup-chaap galat data banata hai,
    aur wo khaali chhod dene se zyada khatarnaak hai.
    """
    i = pos - 1 + offset
    if 0 <= i < len(seq) and seq[i] == aa:
        return i, offset
    return None, None


def apply_mutations(name, seq, offset=0):
    """
    Returns (new_seq, status, detail).
    status: "wt" (mutation nahi), "ok", "partial", "skip"
    """
    m = re.match(r"^([^(]+)\((.*)\)(p?)$", name)
    if not m:
        return seq, "wt", "koi bracket nahi"
    inside = m.group(2)
    edits = split_edits(inside)
    if not edits:
        return seq, "skip", f"'{inside}' se sequence nahi banaya ja sakta"

    s = list(seq)
    done, failed, shifted = [], [], []
    # deletions ko aakhir me karte hain, warna baad ke index khisak jaate hain
    for e in sorted(edits, key=lambda x: x.endswith("del")):
        if e.startswith("?"):
            failed.append(e[1:])
            continue
        pm = POINT.match(e)
        if pm:
            frm, pos, to = pm.group(1), int(pm.group(2)), pm.group(3)
            if to not in AA:
                failed.append(e)
                continue
            i, off = find_pos("".join(s), frm, pos, offset)
            if i is None:
                failed.append(e)
                continue
            s[i] = to
            done.append(e)
            if off:
                shifted.append(f"{e}@{off:+d}")
            continue
        dm = DEL.match(e)
        if dm:
            frm, a, _, b = dm.group(1), int(dm.group(2)), dm.group(3), int(dm.group(4))
            i, off = find_pos("".join(s), frm, a, offset)
            if i is None or b < a:
                failed.append(e)
                continue
            n = b - a + 1
            if i + n > len(s):
                failed.append(e)
                continue
            del s[i:i + n]
            done.append(e)
            if off:
                shifted.append(f"{e}@{off:+d}")
            continue
        failed.append(e)

    new = "".join(s)
    note = "+".join(done)
    if offset:
        note += f" (gene offset {offset:+d})"
    # "ok-shifted" ab nahi hai: offset gene ke saare mutations se tay hota hai,
    # per-mutation local search nahi. Offset report me alag column me jata hai.
    if done and not failed:
        return new, "ok", note
    if done and failed:
        return new, "partial", f"lagaye: {note} · chhoote: {'+'.join(failed)}"
    return seq, "skip", f"kuch nahi laga: {'+'.join(failed)}"


def gene_of(name):
    return name.split("(")[0].strip()


def gene_offset(seq, edits, span=60):
    """
    Ek gene ke SAARE mutations dekh kar EK offset chunta hai.

    Kyun: pehle har mutation apni alag window me position dhundhta tha. Nateeja -
    ABL1 ke 10 mutations ko offset +3, +3, +2, +4, -1 mile. Ek gene me numbering
    ek hi tarah khiskti hai, isliye alag-alag offset ka matlab hai ki kuch
    mutations ittefaq se paas ke usi amino acid pe lag gaye - yaani GALAT jagah.
    (Sabooot: ABL1(E255K) aur Q252H bilkul match nahi hue, matlab asli offset
    +-10 se bahar hai - ABL1 ke do isoform 19 residue se farak karte hain.)

    Isliye: jo offset is gene ke sabse zyada mutations ko sahi baithata hai wahi
    sab pe lagta hai. Barabari ho to 0 ke sabse nazdeek wala.
    Returns (offset, kitne match hue, kul kitne dekhe).
    """
    pts = []
    for e in edits:
        m = POINT.match(e) or DEL.match(e)
        if m:
            pts.append((m.group(1), int(m.group(2))))
    if not pts:
        return 0, 0, 0
    best, best_n = 0, -1
    for off in sorted(range(-span, span + 1), key=lambda o: (abs(o), o)):
        n = 0
        for aa, pos in pts:
            i = pos - 1 + off
            if 0 <= i < len(seq) and seq[i] == aa:
                n += 1
        if n > best_n:
            best, best_n = off, n
    return best, best_n, len(pts)


def build(proteins, span=60):
    """
    proteins: {name: seq} -> (new_proteins, report rows)

    Do chakkar: pehle har gene ka ek offset tay karo (uske saare naam ke saare
    mutations ek saath dekh kar), phir wahi offset lagao.
    """
    # chakkar 1: gene -> uske saare edits aur ek representative sequence
    per_gene = {}
    for name, seq in proteins.items():
        g = gene_of(name)
        m = re.match(r"^([^(]+)\((.*)\)(p?)$", name)
        ed = split_edits(m.group(2)) if m else []
        ed = [e for e in ed if not e.startswith("?")]
        d = per_gene.setdefault(g, {"edits": [], "seq": str(seq).upper().strip()})
        d["edits"].extend(ed)

    offsets = {}
    for g, d in per_gene.items():
        off, n, tot = gene_offset(d["seq"], d["edits"], span)
        offsets[g] = {"offset": off, "matched": n, "total": tot}

    # chakkar 2: wahi offset laga kar sequence banao
    out, rows = {}, []
    for name, seq in proteins.items():
        seq = str(seq).upper().strip()
        g = gene_of(name)
        o = offsets.get(g, {"offset": 0, "matched": 0, "total": 0})
        new, status, detail = apply_mutations(name, seq, offset=o["offset"])
        out[name] = new
        rows.append({"name": name, "gene": g, "status": status, "detail": detail,
                     "gene_offset": o["offset"],
                     "gene_mut_matched": f"{o['matched']}/{o['total']}",
                     "changed": int(new != seq), "len_before": len(seq),
                     "len_after": len(new)})
    return out, rows


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "proteins.txt"
    pro = json.load(open(src))
    new, rows = build(pro)
    uniq_before = len(set(str(v).upper().strip() for v in pro.values()))
    uniq_after = len(set(new.values()))
    print(f"{len(pro)} naam · unique sequences: {uniq_before} -> {uniq_after}")
    from collections import Counter
    print("status:", dict(Counter(r["status"] for r in rows)))
    for r in rows:
        if r["status"] in ("partial", "skip"):
            print(f"  {r['status']:8s} {r['name']:34s} {r['detail']}")
    json.dump(new, open("proteins_mutated.json", "w"))
    json.dump(rows, open("mutation_report.json", "w"), indent=1)
