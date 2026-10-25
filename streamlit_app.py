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
    m = os.path.join(DATA, "repurposing_meta.json")
    if not os.path.exists(p):
        return None, None
    df = pd.read_csv(p)
    meta = json.load(open(m)) if os.path.exists(m) else {}
    return df, meta


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
    rep, meta = load_repurposing()
    if rep is None:
        st.info("Repurposing table abhi is app me nahi hai. Ye alag se banti hai "
                "(`experiments/repurposing/build_table.py`) aur `data/` me aati hai.")
    else:

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
            f"""**Ye list candidate dhoondhne ke layak NAHI hai. Kyun, ye naap ke bataya ja raha hai:**

    - Jin **{q['checkable']}** targets ki jaanch ho sakti hai (jinki known-drug list ChEMBL se mili), unme se
      **{q['zero_known']}** pe top-50 me **ek bhi** jaani-pehchani dawa nahi aayi — sirf **{q['with_any_known']}** pe aayi.
    - Jo chale wo lagbhag sab **EGFR** ke roop hain — aur Davis me EGFR ke saare roop **ek hi sequence** share karte hain, to wo asal me ek hi protein hai.
    - Sabse bada sabooot: **wahi kuch molecules {q['n_targets']} me se 250–290 targets ke top-50 me baithe hain.** Matlab model target dekh kar rank nahi kar raha — wo bas "bade aur ajeeb" molecules ko upar rakh deta hai, chahe target koi bhi ho.

    Isliye ye tab ab **model ki jaanch** ke liye hai, dawa dhoondhne ke liye nahi.""")

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

        cols = ["rank", "drug", "pKd", "max_phase"] + (
            ["known_for_target"] if checkable else [])
        st.dataframe(
            full[cols].rename(columns={"rank": "#", "drug": "Dawa", "max_phase": "Phase",
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
