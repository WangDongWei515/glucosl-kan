"""Reproduce validation metrics in an independent output directory."""
from datetime import datetime, timezone
from pathlib import Path
import os
import hashlib
import json
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


STAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STAGE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparators  # noqa: F401  (the saved comparator classes live there; it does not import PyTorch)
from src.external_metrics import logit, patient_weights
from src.glucosl_data import cgmacros, shanghai_validation

RUN = Path(os.environ.get('GLUCOSL_RUN_DIR', str(STAGE / 'reproduction/default'))).resolve()
MODELS = RUN / 'models'
OUTPUT = RUN / 'evaluation'
DRAWS, SEED = int(os.environ.get("GLUCOSL_BOOTSTRAP_DRAWS", "2000")), 20261021
SHANGHAI_TRAINED = ['glucosl_kan', 'glucosl_mlp', 'single_kan', 'unconstrained_kan', 'constant', 'linear', 'xgboost', 'tabpfn']
TARGET = {'regression': 'iauc', 'binary': 'peak_ge_180'}


class IsolatedModel:
    # XGBoost cannot share a process with PyTorch here, so its saved models predict in a separate process.
    def __init__(self, task, name):
        self.task, self.name = task, name

    def predict(self, frame):
        with tempfile.TemporaryDirectory() as folder:
            frame_path, out_path = Path(folder) / 'frame.pkl', Path(folder) / 'prediction.npy'
            frame.to_pickle(frame_path)
            subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'comparators.py'), 'predict', self.task, self.name, str(frame_path), str(out_path)], check=True)
            return np.load(out_path)


def load(task, name):
    if name.startswith('xgboost'):
        return IsolatedModel(task, name)
    return joblib.load(MODELS / task / f'{name}.joblib')


def regression_metrics(frame, prediction):
    observed, prediction, groups = frame.iauc.to_numpy(float), np.asarray(prediction, float), frame.patient_id.to_numpy()
    weights = patient_weights(groups)
    error = np.abs(observed - prediction)
    spread = np.ptp(prediction) > 0
    slope = np.cov(prediction, observed, aweights=weights)[0, 1] / np.cov(prediction, aweights=weights) if spread else np.nan
    within = []
    for patient, rows in pd.Series(np.arange(len(frame))).groupby(groups):
        if len(rows) >= 8 and np.ptp(prediction[rows]) > 0 and np.ptp(observed[rows]) > 0:
            within.append(spearmanr(prediction[rows], observed[rows]).correlation)
    return {
        'macro_mae': pd.Series(error).groupby(groups).mean().mean(),
        'meal_mae': error.mean(),
        'spearman': spearmanr(prediction, observed).correlation if spread else np.nan,
        'within_patient_spearman': float(np.median(within)) if within else np.nan,
        'calibration_slope': float(slope),
        'calibration_intercept': float(np.average(observed, weights=weights) - slope * np.average(prediction, weights=weights)) if spread else np.nan,
        'mean_observed': float(np.average(observed, weights=weights)), 'mean_predicted': float(np.average(prediction, weights=weights)),
    }


def binary_metrics(frame, probability):
    observed, groups = frame.peak_ge_180.to_numpy(int), frame.patient_id.to_numpy()
    probability = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    weights = patient_weights(groups)
    result = {'auroc': np.nan, 'brier': float(np.average((observed - probability) ** 2, weights=weights)),
              'calibration_slope': np.nan, 'calibration_intercept': np.nan,
              'observed_rate': float(np.average(observed, weights=weights)), 'mean_predicted': float(np.average(probability, weights=weights))}
    if 0 < observed.sum() < len(observed):
        result['auroc'] = roc_auc_score(observed, probability, sample_weight=weights)
        if np.ptp(probability) > 0:
            model = LogisticRegression(C=np.inf, max_iter=1000).fit(logit(probability)[:, None], observed, sample_weight=weights)
            result['calibration_slope'], result['calibration_intercept'] = float(model.coef_[0, 0]), float(model.intercept_[0])
    return result


METRICS = {'regression': regression_metrics, 'binary': binary_metrics}
HEADLINE = {'regression': 'macro_mae', 'binary': 'auroc'}


def resample(frame, generator):
    # One patient-cluster bootstrap draw: all meals of each chosen patient, relabelled so duplicates stay distinct.
    groups = frame.patient_id.to_numpy()
    patients = np.unique(groups)
    index = {patient: np.flatnonzero(groups == patient) for patient in patients}
    chosen = generator.choice(patients, size=len(patients), replace=True)
    rows = np.concatenate([index[patient] for patient in chosen])
    labels = np.concatenate([np.full(len(index[patient]), position) for position, patient in enumerate(chosen)])
    return rows, labels


def bootstrap_chunk(frame, predictions, task, draws, seed):
    generator = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        rows, labels = resample(frame, generator)
        sample = frame.iloc[rows].assign(patient_id=labels)
        out.append({name: METRICS[task](sample, prediction[rows]) for name, prediction in predictions.items()})
    return out


def evaluate_set(label, frame, task, models, reference='glucosl_kan'):
    predictions = {name: np.asarray(model.predict(frame), float) for name, model in models.items()}
    point = {name: METRICS[task](frame, prediction) for name, prediction in predictions.items()}
    chunks = Parallel(n_jobs=int(os.environ.get("GLUCOSL_JOBS", "1")))(delayed(bootstrap_chunk)(frame, predictions, task, DRAWS // 10, SEED + chunk) for chunk in range(10))
    draws = [draw for chunk in chunks for draw in chunk]
    metric_rows, difference_rows = [], []
    for name in predictions:
        for metric, value in point[name].items():
            values = np.array([draw[name][metric] for draw in draws], float)
            metric_rows.append({'set': label, 'task': task, 'model': name, 'metric': metric, 'estimate': value,
                               'low': np.nanpercentile(values, 2.5) if len(values) else np.nan, 'high': np.nanpercentile(values, 97.5) if len(values) else np.nan, 'draws_defined': int(np.isfinite(values).sum())})
        if name != reference and reference in predictions:
            metric = HEADLINE[task]
            values = np.array([draw[reference][metric] - draw[name][metric] for draw in draws], float)
            difference_rows.append({'set': label, 'task': task, 'metric': metric, 'comparison': f'{reference} minus {name}',
                                    'estimate': point[reference][metric] - point[name][metric],
                                    'low': np.nanpercentile(values, 2.5) if len(values) else np.nan, 'high': np.nanpercentile(values, 97.5) if len(values) else np.nan,
                                    'patients': int(frame.patient_id.nunique()), 'meals': len(frame)})
    table = pd.DataFrame({'set': label, 'task': task, 'record_id': frame.record_id.to_numpy(), 'patient_id': frame.patient_id.to_numpy(), 'observed': frame[TARGET[task]].to_numpy(), **predictions})
    return pd.DataFrame(metric_rows), pd.DataFrame(difference_rows), table


def main():
    assert DRAWS == 0 or (DRAWS >= 10 and DRAWS % 10 == 0), 'bootstrap draws must be zero or a positive multiple of 10'
    receipt_path = OUTPUT / 'run_receipt.json'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shanghai_b = shanghai_validation()
    us = cgmacros()
    sets = {'shanghai_B': shanghai_b, 'cgmacros_diabetes_range': us.loc[us.diabetes_range].reset_index(drop=True), 'cgmacros_all': us}
    metrics, differences, predictions = [], [], []
    for task in ['regression', 'binary']:
        names = SHANGHAI_TRAINED if os.environ.get('GLUCOSL_MODEL_SET', 'all') == 'all' else ['glucosl_kan', 'xgboost', 'constant']
        models = {name: load(task, name) for name in names}
        for label, frame in sets.items():
            evaluated = {**models, 'glucosl_kan_us': load(task, 'glucosl_kan_us')} if label == 'shanghai_B' else models
            m, d, p = evaluate_set(label, frame, task, evaluated)
            metrics.append(m); differences.append(d); predictions.append(p)
            print(task, label, m.loc[m.metric.eq(HEADLINE[task]), ['model', 'estimate', 'low', 'high']].round(3).to_string(index=False), sep='\n')
        # Sensitivity 2: well-mapped meals in set B, no retraining.
        well_mapped = shanghai_b.loc[shanghai_b.specific_weight_share >= 0.9].reset_index(drop=True)
        m, d, p = evaluate_set('shanghai_B_well_mapped', well_mapped, task, {name: models[name] for name in ['glucosl_kan', 'xgboost', 'constant']})
        metrics.append(m); differences.append(d); predictions.append(p)
    # Sensitivity 1: staples as cooked weight (regression), models refit with the cooked-weight nutrients.
    cooked = shanghai_validation(variant='staples_cooked')
    m, d, p = evaluate_set('shanghai_B_staples_cooked', cooked, 'regression',
                           {'glucosl_kan': load('regression', 'glucosl_kan_staples_cooked'), 'xgboost': load('regression', 'xgboost_staples_cooked'), 'constant': load('regression', 'constant')})
    metrics.append(m); differences.append(d); predictions.append(p)
    pd.concat(metrics, ignore_index=True).to_csv(OUTPUT / 'metrics.csv', index=False)
    pd.concat(differences, ignore_index=True).to_csv(OUTPUT / 'differences.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUTPUT / 'predictions.csv', index=False)
    receipt = {'run_at_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'bootstrap_draws': DRAWS, 'seed': SEED, 'model_set': os.environ.get('GLUCOSL_MODEL_SET', 'all'),
               'sets': {label: {'patients': int(frame.patient_id.nunique()), 'meals': len(frame)} for label, frame in sets.items()},
               'well_mapped_meals': len(well_mapped), 'note': 'Calibration and mean values are patient-weighted; Spearman over all meals is unweighted.'}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print('evaluation complete')


if __name__ == '__main__':
    main()
