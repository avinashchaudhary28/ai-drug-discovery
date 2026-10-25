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


WINDOW = 10   # isse zyada khisak gaya to bharosa nahi, skip karna behtar hai


def find_pos(seq, aa, pos, window=WINDOW):
    """
    1-based `pos` pe `aa` hona chahiye. Returns (0-based index, offset) ya (None, None).

    Exact match pehle. Na mile to +/- window me sabse nazdeek wahi amino acid
    dhundte hain (isoform numbering thoda khisak sakti hai). Window se bahar
    kuch nahi chhedte - galat jagah badalne se behtar hai mutation skip karna.
    Jo offset se mila uski report alag aati hai, taaki bharosa naapa ja sake.
    """
    i = pos - 1
    if 0 <= i < len(seq) and seq[i] == aa:
        return i, 0
    for off in range(1, window + 1):
        for j in (i - off, i + off):
            if 0 <= j < len(seq) and seq[j] == aa:
                return j, j - i
    return None, None


def apply_mutations(name, seq):
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
            i, off = find_pos("".join(s), frm, pos)
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
            i, off = find_pos("".join(s), frm, a)
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
    if shifted:
        note += f" (khiska: {', '.join(shifted)})"
    if done and not failed:
        return new, ("ok-shifted" if shifted else "ok"), note
    if done and failed:
        return new, "partial", f"lagaye: {note} · chhoote: {'+'.join(failed)}"
    return seq, "skip", f"kuch nahi laga: {'+'.join(failed)}"


def build(proteins):
    """proteins: {name: seq} -> (new_proteins, report rows)"""
    out, rows = {}, []
    for name, seq in proteins.items():
        seq = str(seq).upper().strip()
        new, status, detail = apply_mutations(name, seq)
        out[name] = new
        rows.append({"name": name, "status": status, "detail": detail,
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
