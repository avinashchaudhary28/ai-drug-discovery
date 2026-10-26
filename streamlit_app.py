import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Drug Discovery", page_icon="💊", layout="centered")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@st.cache_resource(show_spinner="Models load ho rahe hain (pehli baar ~1 minute)...")
def load_core():
    import core
    return core


def _gene_of(name):
    """Davis ka protein naam -> gene symbol (mutation/phospho suffix hata ke)."""
    return (name.split("(")[0].replace("-nonphosphorylated", "")
                .replace("-phosphorylated", "").strip())


@st.cache_data(show_spinner=False)
def load_repurposing():
    """Precomputed repurposing table. Nahi mili to (None, None)."""
    import json
    p = os.path.join(DATA, "repurposing_top.csv")
    pc = os.path.join(DATA, "repurposing_top_corrected.csv")
    m = os.path.join(DATA, "repurposing_meta.json")
    if not os.path.exists(p):
        return None, None, None
    df = pd.read_csv(p)
    # Sudhri list default hai. Naapa gaya (134 checkable targets pe):
    #   bar      raw   MW akela   MW + dawa ka mean ghata
    #   top-5     23      45              43
    #   top-10    24      46              52
    #   top-20    24      49              56
    #   rank-1    13      21              37   (out of 442)
    # top-5 pe MW akela 2 target aage hai, par baaki sab bar pe combined jeetta
    # hai aur aankh se bhi saaf behtar hai (ABL1 pe Dasatinib #3 se #1).
    dc = pd.read_csv(pc) if os.path.exists(pc) else None
    meta = json.load(open(m)) if os.path.exists(m) else {}
    return df, dc, meta


core = load_core()

EXAMPLES = {
    # cancer / kinase inhibitors
    "Imatinib — cancer (ABL1/KIT)": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "Gefitinib — cancer (EGFR)": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Erlotinib — cancer (EGFR)": "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
    "Sorafenib — cancer (multi-kinase)": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    # aam dawaein
    "Paracetamol — dard / bukhar": "CC(=O)Nc1ccc(O)cc1",
    "Aspirin — dard / khoon patla": "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen — dard / sujan": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Metformin — diabetes": "CN(C)C(=N)NC(=N)N",
    "Amoxicillin — antibiotic": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
    # dimaag tak pahunchne wali (BBB dekhne ke liye)
    "Caffeine — dimaag": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Diazepam — dimaag": "CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O",
    "Nicotine — dimaag": "CN1CCCC1c1cccnc1",
}

st.title("💊 AI Drug Discovery")

tab_mol, tab_rep = st.tabs(["🔬 Molecule analyze", "🩺 Repurposing — model ki jaanch"])

# ----------------------------------------------------------------- molecule
with tab_mol:
    st.markdown("Molecule ka **SMILES** daalo → ~670 proteins pe binding, "
                "**22 ADMET properties**, 12 toxicity tests aur solubility.")

    ex = st.selectbox("Example chuno (ya neeche apna SMILES daalo)", list(EXAMPLES))
    smiles = st.text_area("SMILES", value=EXAMPLES[ex], height=80)
    st.caption("💡 Koi bhi molecule daal sakte ho — ye sirf examples hain. "
               "Kisi dawa ka SMILES lene ke liye PubChem pe uska naam search karo aur "
               "**Canonical SMILES** copy kar lo.")

    choices = core.CHOICES
    prot = st.selectbox("Target protein", choices,
                        index=choices.index(core.default_choice()))

    if st.button("Analyze 🔬", type="primary", use_container_width=True):
        with st.spinner("Analyze ho raha hai..."):
            summary, img, top10, tox, admet = core.analyze(smiles, prot)
        st.markdown(summary, unsafe_allow_html=True)
        if top10 is not None:
            if img is not None:
                st.image(img, caption="Molecule", use_container_width=True)

            if admet is not None:
                st.subheader("ADMET report card (22 properties)")
                st.caption("A = absorption · D = distribution · "
                           "M = metabolism (liver) · E = excretion · T = toxicity")
                for group in admet["Group"].unique():
                    with st.expander(group, expanded=group.startswith("T")):
                        st.dataframe(
                            admet[admet.Group == group][["Property", "Value", "Note"]],
                            hide_index=True, use_container_width=True)

            st.subheader("Ye molecule sabse zyada kin proteins se chipkega (Top 10)")
            st.dataframe(top10, hide_index=True, use_container_width=True)
            with st.expander("Toxicity risk — Tox21 (12 tests)"):
                st.dataframe(tox, hide_index=True, use_container_width=True)

# -------------------------------------------------------------- repurposing
with tab_rep:
    rep_plain, rep_dl, meta = load_repurposing()
    if rep_plain is None:
        st.info("Repurposing table abhi is app me nahi hai. Ye alag se banti hai "
                "(`experiments/repurposing/build_table.py`) aur `data/` me aati hai.")
    else:
        use_dl = False
        if rep_dl is not None:
            use_dl = st.toggle(
                "Sudhri ranking (promiscuity correction + dawa-jaisa size)",
                value=True,
                help="Do naapi hui sudhaar ek saath: (1) sirf 250–700 Da wale "
                     "molecules — model ka jhukav bhaari molecules ki taraf hai, "
                     "uske top-50 ka median 973 Da tha; (2) har dawa ka apna "
                     "average ghata diya — kuch dawaein har target pe upar aati "
                     "thi. Nateeja: rank-1 pe sahi dawa 13/442 se 37/442, aur "
                     "top-10 recall 24/134 se 52/134. Model badla nahi hai.")
        rep = rep_dl if (use_dl and rep_dl is not None) else rep_plain
        if use_dl and rep_dl is not None:
            st.caption("Ranking **adj** se hui hai (pKd minus us dawa ka average). "
                       "`pKd` column asli value hai — padhne ke liye.")

        @st.cache_data(show_spinner=False)
        def rep_quality(df, resolved_tuple):
            """Model kitna bharose layak hai, ye ginte hain - dikhane se pehle.
            resolved tuple me leta hai: cache_data ko hashable argument chahiye,
            aur return me function nahi de sakte (cache serialize nahi kar paata)."""
            resolved = set(resolved_tuple)
            d = df.copy()
            d["gene"] = d.target.map(_gene_of)
            sub = d[d.gene.isin(resolved)]
            per = sub.groupby("target").known_for_target.sum()
            hogging = d.drug.value_counts()
            return {
                "checkable": int(per.size),
                "with_any_known": int((per > 0).sum()),
                "zero_known": int((per == 0).sum()),
                "per_target": per,
                "top_hoggers": hogging.head(8),
                "n_targets": int(d.target.nunique()),
            }

        resolved = set(meta.get("resolved_genes") or [])
        q = rep_quality(rep, tuple(sorted(resolved)))
        gene_of = _gene_of

        st.markdown("Ulta sawaal: **ek target chuno → maujooda dawaiyon me se kaun si "
                    "us par kaam kar sakti hai.** Par pehle ye dekhna zaroori hai ki "
                    "model is kaam ke layak hai ya nahi — aur abhi **nahi hai.**")

        st.error(
            f"""**Ye list candidate dhoondhne ke layak abhi NAHI hai. Kyun, ye naap ke bataya ja raha hai:**

- Jin **{q['checkable']}** targets ki jaanch ho sakti hai (known-drug list ChEMBL se mila), unme se **{q['zero_known']}** pe top-50 me **ek bhi** jaani-pehchani dawa nahi aayi — sirf **{q['with_any_known']}** pe aayi.
- Model ka **bada jhukav bhaari molecules ki taraf** hai: bina filter uske top-50 ka median **973 Da**, jabki library ka 368 Da aur asli kinase inhibitor **300–600 Da** ka hota hai. 47% top picks 1000 Da se bade — peptides, jo kinase pocket me baith hi nahi sakte.
- Upar ka filter wahi theek karta hai, aur **naapa gaya faida** deta hai — top-5 pe recall **23/146 → 45/146**. Par phir bhi **88/146 targets pe zero** rehta hai, to ranking ki kami size se alag bhi hai.
- Aur ek sabooot: bina filter wahi kuch molecules **{q['n_targets']} me se 250–290 targets** ke top-50 me baithe hain — matlab model target dekh kar rank nahi kar raha.

Isliye ye tab **model ki jaanch** ke liye hai, dawa dhoondhne ke liye nahi.""")

        with st.expander("Wahi molecules har target pe — khud dekh lo", expanded=False):
            hog = q["top_hoggers"].rename("kitne targets ke top-50 me").to_frame()
            hog["kul targets"] = q["n_targets"]
            st.dataframe(hog, use_container_width=True)
            st.caption("Agar model target ke hisaab se rank karta, to koi bhi molecule "
                       "saare targets ke top-50 me nahi hota.")

        st.divider()
        st.subheader("Kisi ek target ki list dekhni ho")

        targets = sorted(rep.target.unique())
        pref = [t for t in ("EGFR", "ALK", "ABL1", "KIT", "BRAF") if t in targets]
        tgt = st.selectbox("Target protein", targets,
                           index=targets.index(pref[0]) if pref else 0)

        full = rep[rep.target == tgt].sort_values("rank")
        checkable = gene_of(tgt) in resolved
        n_known = int(full.known_for_target.sum()) if checkable else None

        c1, c2, c3 = st.columns(3)
        c1.metric("Dawaiyan", f"{len(full)}")
        c2.metric("Sabse zyada pKd", f"{full.pKd.max():.2f}")
        if checkable:
            c3.metric("Top-50 me jaani-pehchani", f"{n_known}")
        else:
            c3.metric("Jaanch ho sakti hai?", "Nahi")

        if not checkable:
            st.warning(
                f"**{gene_of(tgt)} ki jaanch nahi ho sakti** — ChEMBL se iska known-drug "
                "list nahi mila, to hum bata nahi sakte ki neeche koi dawa pehle se "
                f"jaani-pehchani hai ya nahi. ({meta.get('known_genes_resolved', '?')} "
                "targets ka list mila hai.) Iska matlab **ye nahi** ki ye dawaiyan nayi hain.")
        elif n_known == 0:
            st.warning(
                f"**Is target pe model fail hua hai.** {gene_of(tgt)} ki jaani-pehchani "
                "dawaiyan ChEMBL me maujood hain, par model ne unme se ek bhi ko top-50 "
                "me nahi rakha. Jab model jaani-pehchani dawa hi nahi dhoondh pata, to "
                "uski 'nayi' dawaiyon par bharosa karne ki koi wajah nahi hai.")
        else:
            st.success(
                f"**Is target pe model thoda kaam kar raha hai** — {n_known} jaani-pehchani "
                "dawaiyan top-50 me aayi. Ye model ke sahi hone ka suboot hai, **nayi khoj "
                "nahi**. Jo baaki hain unme se kai chhupe hue usi tarah ke inhibitor ya "
                "galat andaza ho sakte hain.")

        cols = ["rank", "drug", "pKd"]
        for extra in ("adj", "mw"):          # sudhri list me hi hote hain
            if extra in full.columns:
                cols.append(extra)
        cols.append("max_phase")
        if checkable:
            cols.append("known_for_target")
        st.dataframe(
            full[cols].rename(columns={"rank": "#", "drug": "Dawa", "max_phase": "Phase",
                                       "mw": "MW", "adj": "adj (ranking)",
                                       "known_for_target": "Pehle se jaani?"}),
            hide_index=True, use_container_width=True)

        with st.expander("SMILES dekhne ke liye"):
            st.dataframe(full[["drug", "smiles"]].rename(
                columns={"drug": "Dawa", "smiles": "SMILES"}),
                hide_index=True, use_container_width=True)

        if meta:
            st.caption(f"{meta.get('n_drugs', '?'):,} dawaiyan × {meta.get('n_targets', '?')} "
                       f"targets · model: binding v2 · banaya {meta.get('built_utc', '?')[:10]}"
                       " · Ye computer ka andaza hai, lab ka nateeja nahi.")

st.caption("Models: Binding v2 (ESM-2 + ChemBERTa, Davis + KIBA) · "
           "ADMET v2 (TDC, 22 properties) · Tox21 · ESOL. "
           "Sirf research ke liye — medical salah nahi.")
