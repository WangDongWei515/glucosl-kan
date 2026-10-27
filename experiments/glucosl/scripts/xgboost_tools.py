"""TreeSHAP and permutation importance for the frozen XGBoost regression model (protocol G1, 6.4).

Runs without PyTorch, like comparators.py, because XGBoost cannot share a process with it on this platform.
"""
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STAGE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparators  # noqa: F401
from src.glucosl_data import MEAL, PERSON, shanghai_validation

OUTPUT = STAGE / 'results/glucosl/interpretability'
SEED = 20261021


def macro_mae(frame, prediction):
    return frame.assign(error=np.abs(frame.iauc - np.asarray(prediction))).groupby('patient_id').error.mean().mean()


def main():
    import shap
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validation = shanghai_validation()
    model = joblib.load(STAGE / 'results/glucosl/models/regression/xgboost.joblib')
    design = model.design(validation)
    explainer = shap.TreeExplainer(model.model)
    values = explainer.shap_values(design)
    names = PERSON + MEAL
    pd.DataFrame({'input': names, 'mean_abs_shap': np.abs(values).mean(axis=0), 'mean_shap': values.mean(axis=0)}).to_csv(OUTPUT / 'xgboost_shap_summary.csv', index=False)
    pd.DataFrame(values, columns=names).assign(record_id=validation.record_id.to_numpy()).to_csv(OUTPUT / 'xgboost_shap_values.csv', index=False)
    plt.figure(figsize=(6, 4))
    shap.summary_plot(values, features=validation[names], feature_names=names, show=False, plot_size=None)
    plt.title('XGBoost (regression): TreeSHAP values in Shanghai set B', fontsize=9)
    plt.tight_layout(); plt.savefig(OUTPUT / 'figure_xgboost_shap.png', dpi=300); plt.savefig(OUTPUT / 'figure_xgboost_shap.pdf'); plt.close()
    generator = np.random.default_rng(SEED)
    baseline = macro_mae(validation, model.predict(validation))
    rows = []
    for column in names:
        increases = []
        for _ in range(5):
            shuffled = validation.copy()
            shuffled[column] = generator.permutation(shuffled[column].to_numpy())
            increases.append(macro_mae(shuffled, model.predict(shuffled)) - baseline)
        rows.append({'model': 'xgboost', 'input': column, 'baseline_macro_mae': baseline, 'increase_macro_mae': float(np.mean(increases)), 'repeats': 5})
    pd.DataFrame(rows).to_csv(OUTPUT / 'permutation_importance_xgboost.csv', index=False)
    print('xgboost tools done')


if __name__ == '__main__':
    main()
