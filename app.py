"""
Solubility predictor (web app)
------------------------------
Run with:  streamlit run app.py
Reuses the model and helpers from predict_solubility.py.
"""

from urllib.parse import quote

import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from predict_solubility import (
    SIMILARITY_WARN, fingerprint, load, nearest_similarity,
)

# Faint molecule drawings (benzene rings, water, carbon chains) tiled as the background.
MOLECULES_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' width='360' height='360' viewBox='0 0 360 360'
     fill='none' stroke='white' stroke-opacity='0.17' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>
  <polygon points='70,46 99.4,63 99.4,97 70,114 40.6,97 40.6,63'/>
  <circle cx='70' cy='80' r='19' stroke-opacity='0.1'/>
  <polyline points='99.4,63 129,46 158,63'/>
  <circle cx='158' cy='63' r='8' fill='white' fill-opacity='0.14'/>
  <polyline points='243,96 270,72 297,96'/>
  <circle cx='270' cy='70' r='13' fill='white' fill-opacity='0.18'/>
  <circle cx='243' cy='97' r='7' fill='white' fill-opacity='0.12'/>
  <circle cx='297' cy='97' r='7' fill='white' fill-opacity='0.12'/>
  <polygon points='250,216 279.4,233 279.4,267 250,284 220.6,267 220.6,233'/>
  <circle cx='250' cy='250' r='19' stroke-opacity='0.1'/>
  <polyline points='220.6,233 191,216'/>
  <circle cx='191' cy='216' r='8' fill='white' fill-opacity='0.14'/>
  <polyline points='279.4,267 309,284 338,267'/>
  <circle cx='338' cy='267' r='7' fill='white' fill-opacity='0.12'/>
  <polyline points='20,320 50,303 80,320 110,303 140,320'/>
  <circle cx='140' cy='320' r='8' fill='white' fill-opacity='0.14'/>
  <circle cx='20' cy='320' r='5' fill='white' fill-opacity='0.12'/>
  <circle cx='190' cy='140' r='4' fill='white' fill-opacity='0.2'/>
  <circle cx='330' cy='170' r='6' fill='white' fill-opacity='0.14'/>
  <circle cx='40' cy='200' r='5' fill='white' fill-opacity='0.16'/>
</svg>
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

.stApp {
    background-image: url("__BG__"),
        radial-gradient(circle at 15% 10%, #1e90ff 0%, transparent 45%),
        linear-gradient(160deg, #031a3d 0%, #0a3d91 55%, #0b6fc4 100%);
    background-size: 360px 360px, auto, auto;
    background-attachment: fixed;
    font-family: 'Poppins', sans-serif;
}
header[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 760px;
    margin-top: 1.5rem;
    padding: 2.5rem 2.5rem 3rem;
    background: rgba(5, 25, 60, 0.55);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 24px;
}

.title { font-size: 2.9rem; font-weight: 800; line-height: 1.1; margin: 0 0 .4rem; }
.grad {
    background: linear-gradient(90deg, #7df9ff, #a78bfa, #ff7ad9, #ffd166, #7df9ff);
    background-size: 250% 100%;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
    filter: drop-shadow(0 0 16px rgba(125, 249, 255, .45));
    animation: shine 8s linear infinite;
}
@keyframes shine { to { background-position: 250% 0; } }

.why {
    margin: 1rem 0 1.6rem; padding: 1rem 1.2rem;
    background: rgba(255, 255, 255, 0.08);
    border-left: 4px solid #7df9ff; border-radius: 12px;
    color: #e6f3ff; font-size: .95rem; line-height: 1.65;
}
.why b { color: #7df9ff; }

.stApp label, .stApp p, .stApp [data-testid="stCaptionContainer"] { color: #e6f3ff; }

/* Make the entire input field white */
.stApp [data-baseweb="input"] {
    background-color: #ffffff !important;
    border-radius: 10px !important;
}

/* Force the input text to be visible */
.stApp [data-baseweb="input"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    background-color: transparent !important;
}

/* Make the cursor visible */
.stApp [data-baseweb="input"] input:focus {
    caret-color: #000000 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #7b61ff); color: white; border: none;
    border-radius: 999px; padding: .5rem 2rem; font-weight: 600;
    box-shadow: 0 4px 18px rgba(0, 198, 255, .4);
}
.stButton > button:hover { color: white; filter: brightness(1.15); }

.stApp img { border-radius: 16px; }
div[data-testid="stAlert"] { border-radius: 12px; font-size: 1.1rem; font-weight: 600; }
"""
st.set_page_config(page_title="Water Solubility Predictor", page_icon="💧")
st.markdown(f"<style>{CSS.replace('__BG__', 'data:image/svg+xml,' + quote(MOLECULES_SVG))}</style>",
            unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load()


st.markdown('<h1 class="title">💧 <span class="grad">Water Solubility Predictor</span></h1>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="why"><b>Why does solubility matter?</b> Solubility decides whether a molecule can '
    'actually do its job. A drug has to dissolve in the water-based fluids of the body to reach its '
    'target, and poor solubility is a common reason promising drug candidates fail. It also controls '
    'how pollutants and pesticides spread through water, how dyes and cleaning products are '
    'formulated, and which solvents chemists choose for reactions. Measuring it in the lab is slow '
    'and costly, so models like this one let scientists screen thousands of molecules on a computer first.</div>',
    unsafe_allow_html=True,
)
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

        mol = Chem.MolFromSmiles(smiles)

            drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
            drawer.drawOptions().setBackgroundColour((0.88, 0.93, 0.98))

            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()

st.image(drawer.GetDrawingText())
        if logs >= cutoff:
            st.success(f"Soluble (logS = {logs:.2f})")
        else:
            st.error(f"Not soluble (logS = {logs:.2f})")
        st.write(f"Similarity to training data: {sim:.2f}")
        if sim < SIMILARITY_WARN:
            st.warning("This molecule is unlike the training data, so treat the result with caution.")
