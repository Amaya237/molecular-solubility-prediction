"""
Solubility predictor (command line)
-----------------------------------
Usage:
    python predict_solubility.py CCO "c1ccccc1" "CC(=O)Oc1ccccc1C(=O)O"
    python predict_solubility.py CCO --cutoff -3.0

Trains a Random Forest on the ESOL dataset the first time it runs, saves it,
and reuses the saved model afterwards. Delete solubility_model.joblib to retrain.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

DATA_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
TARGET_COL = "measured log solubility in mols per litre"
MODEL_FILE = "solubility_model.joblib"
RANDOM_STATE = 42
SIMILARITY_WARN = 0.4  # below this, the molecule is unlike the training data

RDLogger.DisableLog("rdApp.*")  # we print our own "invalid SMILES" message

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else _gen.GetFingerprintAsNumPy(mol)


def train():
    df = pd.read_csv(DATA_URL)
    fps = [fingerprint(s) for s in df["smiles"]]
    keep = [i for i, f in enumerate(fps) if f is not None]
    X = np.array([fps[i] for i in keep], dtype=np.float32)
    y = df[TARGET_COL].values[keep]
    model = RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ).fit(X, y)
    joblib.dump({"model": model, "X": X}, MODEL_FILE)
    return model, X


def load():
    if os.path.exists(MODEL_FILE):
        saved = joblib.load(MODEL_FILE)
        return saved["model"], saved["X"]
    print("No saved model found. Training one (first run only)...")
    return train()


def nearest_similarity(x, X):
    """Tanimoto similarity to the closest training molecule (1.0 = identical)."""
    inter = X @ x
    return float((inter / (X.sum(axis=1) + x.sum() - inter)).max())


def main():
    parser = argparse.ArgumentParser(description="Predict if molecules are soluble in water.")
    parser.add_argument("smiles", nargs="+", help="one or more SMILES strings")
    parser.add_argument("--cutoff", type=float, default=-4.0,
                        help="logS at or above this is 'soluble' (default: -4.0)")
    args = parser.parse_args()

    model, X = load()

    for smi in args.smiles:
        fp = fingerprint(smi)
        if fp is None:
            print(f"{smi}: invalid SMILES")
            continue
        x = fp.astype(np.float32)
        logs = float(model.predict(x.reshape(1, -1))[0])
        label = "soluble" if logs >= args.cutoff else "not soluble"
        sim = nearest_similarity(x, X)
        print(f"{smi}: {label}  (logS = {logs:.2f}, similarity to training data = {sim:.2f})")
        if sim < SIMILARITY_WARN:
            print("  Warning: this molecule is unlike the training data, so treat the result with caution.")


if __name__ == "__main__":
    main()
