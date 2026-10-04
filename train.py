"""
Molecular Solubility Prediction (ESOL dataset)
------------------------------------------------
Compares Random Forest and XGBoost on random vs. scaffold splits
to show the effect of structural data leakage in molecular ML.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
TARGET_COL = "measured log solubility in mols per litre"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_URL)
    return df


def featurize(smiles_list, radius=2, n_bits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    feats, valid_mask = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mask.append(False)
            continue
        feats.append(np.array(gen.GetFingerprintAsNumPy(mol)))
        valid_mask.append(True)
    return np.array(feats), np.array(valid_mask)


def scaffold_split(smiles, test_frac=0.2):
    """Group molecules by Murcko scaffold; put whole scaffold groups into
    train or test so the test set contains structurally novel molecules."""
    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        scaffolds[scaffold].append(i)

    groups = sorted(scaffolds.values(), key=len, reverse=True)
    test_target = int(test_frac * len(smiles))

    train_idx, test_idx = [], []
    for g in groups:
        if len(test_idx) < test_target:
            test_idx.extend(g)
        else:
            train_idx.extend(g)
    return np.array(train_idx), np.array(test_idx)


def evaluate(model, Xtr, ytr, Xte, yte, label):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    rmse = mean_squared_error(yte, pred) ** 0.5
    r2 = r2_score(yte, pred)
    print(f"{label:30s} RMSE: {rmse:.3f}   R2: {r2:.3f}")
    return rmse, r2


def main():
    df = load_data()
    X, valid_mask = featurize(df["smiles"])
    y = df.loc[valid_mask, TARGET_COL].values
    smiles = df.loc[valid_mask, "smiles"].values
    print(f"Loaded {len(df)} molecules, {valid_mask.sum()} valid after featurization.\n")

    # Random split
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Scaffold split
    train_idx, test_idx = scaffold_split(smiles)
    Xtr_s, Xte_s = X[train_idx], X[test_idx]
    ytr_s, yte_s = y[train_idx], y[test_idx]

    results = {}
    results["RF_random"] = evaluate(
        RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        Xtr_r, ytr_r, Xte_r, yte_r, "Random Forest (random split)")
    results["RF_scaffold"] = evaluate(
        RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        Xtr_s, ytr_s, Xte_s, yte_s, "Random Forest (scaffold split)")
    results["XGB_random"] = evaluate(
        XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                      random_state=RANDOM_STATE, n_jobs=-1),
        Xtr_r, ytr_r, Xte_r, yte_r, "XGBoost (random split)")
    results["XGB_scaffold"] = evaluate(
        XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                      random_state=RANDOM_STATE, n_jobs=-1),
        Xtr_s, ytr_s, Xte_s, yte_s, "XGBoost (scaffold split)")

    print("\nDone. See README.md for discussion of the random-vs-scaffold gap.")
    return results


if __name__ == "__main__":
    main()
