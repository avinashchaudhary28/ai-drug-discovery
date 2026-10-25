import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Drug Discovery", page_icon="💊", layout="centered")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@st.cache_resource(show_spinner="Models load ho rahe hain (pehli baar ~1 minute)...")
def load_core():
    import core
    return core


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

tab_mol, tab_rep = st.tabs(["🔬 Molecule analyze", "♻️ Repurposing"])

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
    st.markdown("Ulta sawaal: **ek target chuno → maujooda dawaiyon me se kaun si "
                "us par kaam kar sakti hai.** Ye wo dawaiyan hain jo insaan pe "
                "pehle se test ho chuki hain (ChEMBL phase 1–4).")

    rep, meta = load_repurposing()
    if rep is None:
        st.info("Repurposing table abhi is app me nahi hai. Ye alag se banti hai "
                "(`experiments/repurposing/build_table.py`) aur `data/` me aati hai.")
    else:
        if meta:
            st.caption(f"{meta.get('n_drugs', '?'):,} dawaiyan × "
                       f"{meta.get('n_targets', '?')} targets · "
                       f"model: binding v2 · banaya gaya {meta.get('built_utc', '?')[:10]}")

        # 'Pehle se jaani?' column sirf un targets pe dikhata hai jinka
        # known-drug list ChEMBL se mila. Baaki pe chhupa dete hain - warna
        # har dawa 'nayi' dikhegi, jo sabse bada bharam hai.
        resolved = set(meta.get("resolved_genes") or [])

        def gene_of(name):
            return (name.split("(")[0].replace("-nonphosphorylated", "")
                        .replace("-phosphorylated", "").strip())

        targets = sorted(rep.target.unique())
        pref = [t for t in ("EGFR", "ALK", "ABL1", "KIT", "BRAF") if t in targets]
        tgt = st.selectbox("Target protein", targets,
                           index=targets.index(pref[0]) if pref else 0)

        known_ok = (not resolved) or (gene_of(tgt) in resolved)
        if resolved and not known_ok:
            st.warning(
                f"**Is target ({gene_of(tgt)}) ke liye hum nahi bata sakte ki kaun si "
                "dawa pehle se jaani-pehchani hai** — ChEMBL se iska known-drug list "
                "nahi mila. Isliye wo column chhupa diya hai. Iska matlab **ye nahi** "
                "ki neeche ki dawaiyan nayi hain — bas hum farak nahi kar sakte. "
                f"({meta.get('known_genes_resolved', '?')} targets ka list mila hai; "
                "EGFR, KIT, ABL1, BRAF, ALK jaise bade targets me se hai.)")

        hide_known = False
        if known_ok:
            hide_known = st.checkbox(
                "Sirf wo dawaiyan dikhao jo is target ke liye pehle se jaani nahi hain",
                value=False,
                help="Jaani-pehchani dawaiyan top pe aana model ke sahi hone ka "
                     "suboot hai. Unhe hata ke dekho to jo bachta hai wo 'naya' "
                     "candidate hai — par wo galat bhi ho sakta hai.")

        full = rep[rep.target == tgt].sort_values("rank")
        sub = full[~full.known_for_target] if hide_known else full

        c1, c2 = st.columns(2)
        if known_ok:
            c1.metric("Top list me jaani-pehchani",
                      f"{int(full.known_for_target.sum())} / {len(full)}")
        else:
            c1.metric("Top list me dawaiyan", f"{len(full)}")
        c2.metric("Sabse zyada pKd", f"{sub.pKd.max():.2f}" if len(sub) else "—")

        cols = ["rank", "drug", "pKd", "max_phase"] + (
            ["known_for_target"] if known_ok else [])
        show = sub[cols].rename(
            columns={"rank": "#", "drug": "Dawa", "max_phase": "Phase",
                     "known_for_target": "Pehle se jaani?"})
        st.dataframe(show, hide_index=True, use_container_width=True)

        with st.expander("SMILES dekhne ke liye"):
            st.dataframe(sub[["drug", "smiles"]].rename(
                columns={"drug": "Dawa", "smiles": "SMILES"}),
                hide_index=True, use_container_width=True)

        st.warning(
            "**Isko theek se padho.** Ye computer ka andaza hai, lab ka nateeja nahi. "
            "Jo dawaiyan top pe aati hain unme se bahut si us target ke liye **pehle "
            "se jaani-pehchani** hain — wo model ke sahi hone ka suboot hain, nayi "
            "khoj nahi. Aur jo 'nayi' lagti hain, unme se kai asal me chhupe hue "
            "usi tarah ke inhibitor hote hain ya galat andaza. Kisi bhi candidate ka "
            "matlab lab me test hone ke baad hi banta hai.")

st.caption("Models: Binding v2 (ESM-2 + ChemBERTa, Davis + KIBA) · "
           "ADMET v2 (TDC, 22 properties) · Tox21 · ESOL. "
           "Sirf research ke liye — medical salah nahi.")
