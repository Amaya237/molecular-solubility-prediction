"""
Solubility predictor (web app)
-----------------------------
Run with:  streamlit run app.py
Reuses the model and helpers from predict_solubility.py.
"""

import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from predict_solubility import (
    SIMILARITY_WARN, fingerprint, load, nearest_similarity,
)


@st.cache_resource
def get_model():
    return load()


st.title("Water Solubility Predictor")
st.caption("Enter a molecule as a SMILES string, e.g. CCO (ethanol) or c1ccccc1 (benzene).")

smiles = st.text_input("SMILES", value="CCO").strip()
cutoff = st.number_input("Cutoff (logS at or above this = soluble)", value=-4.0, step=0.5)

if st.button("Predict") and smiles:
    fp = fingerprint(smiles)
    if fp is None:
        st.error("Invalid SMILES string.")
    else:
        with st.spinner("Loading model (first run trains it, this can take a minute)..."):
            model, X = get_model()
        x = fp.astype(np.float32)
        logs = float(model.predict(x.reshape(1, -1))[0])
        sim = nearest_similarity(x, X)

        st.image(Draw.MolToImage(Chem.MolFromSmiles(smiles), size=(300, 300)))
        if logs >= cutoff:
            st.success(f"Soluble (logS = {logs:.2f})")
        else:
            st.error(f"Not soluble (logS = {logs:.2f})")
        st.write(f"Similarity to training data: {sim:.2f}")
        if sim < SIMILARITY_WARN:
            st.warning("This molecule is unlike the training data, so treat the result with caution.")
