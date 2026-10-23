import streamlit as st

st.set_page_config(page_title="AI Drug Discovery", page_icon="💊", layout="centered")


@st.cache_resource(show_spinner="Models load ho rahe hain (pehli baar ~1 minute)...")
def load_core():
    import core
    return core


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
st.markdown("Molecule ka **SMILES** daalo → ~670 proteins pe binding, **22 ADMET properties**, 12 toxicity tests aur solubility.")

ex = st.selectbox("Example chuno (ya neeche apna SMILES daalo)", list(EXAMPLES))
smiles = st.text_area("SMILES", value=EXAMPLES[ex], height=80)
st.caption("💡 Koi bhi molecule daal sakte ho — ye sirf examples hain. "
           "Kisi dawa ka SMILES lene ke liye PubChem pe uska naam search karo aur "
           "**Canonical SMILES** copy kar lo.")
choices = core.CHOICES
prot = st.selectbox("Target protein", choices, index=choices.index(core.default_choice()))

if st.button("Analyze 🔬", type="primary", use_container_width=True):
    with st.spinner("Analyze ho raha hai..."):
        summary, img, top10, tox, admet = core.analyze(smiles, prot)
    st.markdown(summary, unsafe_allow_html=True)
    if top10 is not None:
        if img is not None:
            st.image(img, caption="Molecule", use_container_width=True)

        if admet is not None:
            st.subheader("ADMET report card (22 properties)")
            st.caption("A = absorption · D = distribution · M = metabolism (liver) · E = excretion · T = toxicity")
            for group in admet["Group"].unique():
                with st.expander(group, expanded=group.startswith("T")):
                    st.dataframe(admet[admet.Group == group][["Property", "Value", "Note"]],
                                 hide_index=True, use_container_width=True)

        st.subheader("Ye molecule sabse zyada kin proteins se chipkega (Top 10)")
        st.dataframe(top10, hide_index=True, use_container_width=True)
        with st.expander("Toxicity risk — Tox21 (12 tests)"):
            st.dataframe(tox, hide_index=True, use_container_width=True)

st.caption("Models: Binding v2 (ESM-2 + ChemBERTa, Davis + KIBA) · ADMET v2 (TDC, 22 properties) · Tox21 · ESOL. "
           "Sirf research ke liye — medical salah nahi.")
