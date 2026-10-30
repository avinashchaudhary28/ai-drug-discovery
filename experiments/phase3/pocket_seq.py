"""Mutated pocket sequence banana - 85 residues, mutation lagi hui.

Kadam 1-3 me teen feature sets fail hue. Unme ek baat saanjhi thi: wo
dawa aur mutation ko ALAG-ALAG describe karte the, aur pocket ka sirf
ek number (index, ya contact frequency) dete the. Model ko kabhi ye nahi
pata chala ki pocket me AAS-PAAS kya hai.

Ab wo diya ja raha hai: poora 85-residue pocket sequence, mutation lagi
hui, aur ESM-2 se uska embedding. Saath me wild-type pocket ka embedding
bhi, aur dono ka ANTAR - kyunki asli sawal "is mutation se kya badla"
hai, "ye pocket kaisa hai" nahi.

SABSE BADA KHATRA: mutation galat index pe lag jaye. Tab sequence galat
hoga, embedding galat hoga, model kuch na kuch seekh hi lega, aur error
kabhi nahi aayega. Isliye har mutation pe ye jaancha jata hai ki us
index pe wild-type residue WAHI hai jo mutation ke naam me likha hai -
aur na ho to wo jodi chhod di jati hai, badli nahi jati.
"""
import re

AA = "ACDEFGHIKLMNPQRSTVWY"
MUT_RE = re.compile(rf"^([{AA}])(\d{{1,4}})([{AA}])$")
N_POS = 85


def parse(mutation):
    """'T790M' -> ('T', 790, 'M'). Multi-mutant ke saare tokens."""
    out = []
    for tok in str(mutation).split("+"):
        m = MUT_RE.match(tok.strip().upper())
        if m:
            out.append((m.group(1), int(m.group(2)), m.group(3)))
    return out


def wt_pocket(info):
    """Gene ka wild-type pocket (consensus, 85 akshar) ya None."""
    p = info.get("consensus_pocket") or info.get("pocket") or ""
    return p if len(p) == N_POS else None


def mutate_pocket(pocket, index, wt_aa, mt_aa):
    """-> (naya pocket, kya hua)

    index 1..85. Agar us jagah wild-type residue match na kare to
    sequence BADLA NAHI jata - galat residue likhne se behtar hai kuch
    na likhna. Yahi Phase 1 ki sabse badi seekh thi.
    """
    if pocket is None or len(pocket) != N_POS:
        return None, "pocket theek nahi"
    if not (1 <= index <= N_POS):
        return None, "index pocket se bahar"
    i = index - 1
    if pocket[i] != wt_aa:
        return None, f"us jagah {pocket[i]} hai, {wt_aa} nahi"
    return pocket[:i] + mt_aa + pocket[i + 1:], "ok"


def build_pair(gene, mutation, maps_entry, info):
    """-> (wt_seq, mut_seq, kya hua)

    maps_entry: klifs_map.json ka {offset, map}
    info: fetch_gene() ka dict (jisme consensus_pocket hai)
    """
    wt = wt_pocket(info)
    if wt is None:
        return None, None, "wild-type pocket nahi mila"
    toks = parse(mutation)
    if not toks:
        return None, None, "mutation parse nahi hua"

    seq = wt
    off = maps_entry.get("offset", 0)
    applied = 0
    for wt_aa, pos, mt_aa in toks:
        hit = maps_entry["map"].get(str(pos + off))
        if hit is None:
            continue                    # ye mutation pocket me nahi hai
        seq, why = mutate_pocket(seq, int(hit[0]), wt_aa, mt_aa)
        if seq is None:
            return None, None, f"{wt_aa}{pos}{mt_aa}: {why}"
        applied += 1
    if applied == 0:
        return None, None, "koi mutation pocket me nahi"
    return wt, seq, "ok"
