# Molecular Solubility Prediction with Machine Learning

Predicting aqueous solubility (log S) for small molecules from the ESOL
dataset, using RDKit fingerprints and gradient-boosted / ensemble models —
with a focus on **honest evaluation** via scaffold splitting.

## Why this project

Random train/test splits on molecular datasets are misleading: structurally
similar or near-duplicate molecules often end up on both sides of the split,
which inflates test performance. This project measures that effect directly
by comparing a standard random split against a **scaffold split**, which
holds out structurally distinct molecules for testing.

## Dataset

[ESOL (Delaney) solubility dataset](https://pubs.acs.org/doi/10.1021/ci034243x),
1,128 molecules with measured aqueous solubility (log mol/L).

## Method

1. Parse SMILES with RDKit, generate 2048-bit Morgan fingerprints (radius 2).
2. Train/evaluate two models: Random Forest and XGBoost.
3. Evaluate each model under two splitting strategies:
   - **Random split** (80/20): standard baseline.
   - **Scaffold split** (80/20): molecules grouped by Murcko scaffold, with
     whole scaffold groups assigned to train or test, so the test set
     contains structurally novel molecules.

## Results

| Model | Split | RMSE | R² |
|---|---|---|---|
| Random Forest | Random | 1.159 | 0.716 |
| Random Forest | Scaffold | 2.162 | -0.408 |
| XGBoost | Random | **1.099** | **0.744** |
| XGBoost | Scaffold | 2.205 | -0.465 |

**Both models perform well under a random split but collapse under a
scaffold split** (R² turns negative, i.e. worse than predicting the mean).
This gap is the headline result: it shows that random-split performance on
this dataset substantially overstates how well the model generalizes to
genuinely new chemical structures, regardless of which model is used.

## Usage

```bash
pip install rdkit xgboost scikit-learn pandas numpy
python train.py
```

## Possible extensions

- Add a graph neural network (e.g. Chemprop or PyTorch Geometric) as a
  third model.
- Add uncertainty estimation (ensembling / conformal prediction).
- Build a small Streamlit/FastAPI demo: paste a SMILES, get a predicted
  solubility with a confidence interval.

## Project status

Baseline complete (Random Forest + XGBoost, random + scaffold splits).
