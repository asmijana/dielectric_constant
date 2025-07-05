# Dielectric Constant Prediction with Machine Learning

This repository contains a complete machine learning pipeline for predicting the **total** and **electronic dielectric constants** of materials using classical composition-based features.

## Summary

- Dataset: 1056 materials with total (`ε_total`) and electronic (`ε_elec`) dielectric constants.
- Features: 50 selected Magpie descriptors using `SelectKBest`.
- Targets: `ε_total` and `ε_elec` (log-transformed during training).
- Models: Multi target regression with `HistGradientBoostingRegressor` model jointly used for each target.
- Evaluation: Stratified train/test split (based on quantile bins) and 5-fold cross-validation.

## Key Results

|     Metric     |  Train  |    CV   |   Test  |
|----------------|---------|---------|---------|
| **RMSE total** |  6.06   |  18.96  |  10.15  |
| **R$^2$ total**|  0.91   |   0.16  |   0.42  |
| **RMSE elec**  |  5.26   |  12.30  |   5.06  |
| **R$^2$ elec** |  0.86   |   0.21  |   0.40  |

## 📁 Project Structure

```
dielectric-ml/
├── dielectric_model.ipynb       # Main notebook with model pipeline
├── models/                      # (Optional) Saved models
├── results/                     # (Optional) Plots and logs
├── requirements.txt             # Environment dependencies
└── README.md                    # This file
```

## Highlights

- Log-transform targets (`log1p`), then back-transform predictions.
- Stratified train/test split using `pd.qcut()` for fair distribution of target ranges.
- Joint models allow generalization of distinct behaviors of `ε_total` and `ε_elec`.
- Evaluated with `RMSE` and `R²` across train, CV, and test sets.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter and run:

```bash
jupyter notebook dielectric_model.ipynb
```

## Potential Extensions

- Try structural descriptors or graph models (e.g. MEGNet, CrabNet)
- Compare with MatBench benchmark for dielectric prediction
- Try other regressors 
- Try with more data points

## Author

Created by Asmita Jana (asmitajana[at]gmail[dot]com)
This project was built as a scientific benchmark for composition-based ML on small, long-tailed dielectric datasets.

---

Feel free to use this as a template or baseline for your own ML materials science projects!
