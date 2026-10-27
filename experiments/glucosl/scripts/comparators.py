"""Comparators of protocol G1, 4.5 (constant, linear, XGBoost, TabPFN), kept free of PyTorch.

XGBoost's OpenMP runtime crashes in a process in which PyTorch has already run (OMP error #179 on macOS), so this
file runs as its own process:

  comparators.py fit                           tunes and fits the comparators on set A (and the cooked-staples XGBoost)
  comparators.py predict TASK NAME FRAME OUT   loads a saved comparator and writes its predictions for a pickled frame
"""
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


STAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STAGE))
from src.external_metrics import patient_folds
from src.glucosl_data import MEAL, PERSON, shanghai_training
from src.glucosl_preprocess import Preprocessor

FOLD_SEED = 20261021
MODELS = STAGE / 'results/glucosl/models'
REPORTS = STAGE / 'results/glucosl/final_fit'
TARGET = {'regression': 'iauc', 'binary': 'peak_ge_180'}
XGBOOST_GRID = [dict(zip(['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight'], values))
                for values in product([2, 3, 4], [0.03, 0.1], [100, 300], [5, 20])]
LINEAR_GRID = {'regression': [{'alpha': alpha} for alpha in [0.0, 0.001, 0.01, 0.1]], 'binary': [{'C': c} for c in [0.01, 0.1, 1.0, 10.0]]}


class Comparator:
    # A scikit-learn or XGBoost model on the standardized 11-column design; the training statistics travel with it.
    def __init__(self, model, task):
        self.model, self.task = model, task

    def design(self, frame):
        return np.hstack(self.preprocessor.transform(frame))

    def fit(self, frame, target, groups):
        self.preprocessor = Preprocessor(PERSON, MEAL).fit(frame, groups)
        self.model.fit(self.design(frame), np.asarray(target))
        return self

    def predict(self, frame):
        x = self.design(frame)
        return self.model.predict_proba(x)[:, 1] if self.task == 'binary' else self.model.predict(x)


class Constant:
    def __init__(self, task):
        self.task = task

    def fit(self, frame, target, groups):
        self.value = float(np.median(target)) if self.task == 'regression' else float(np.mean(target))
        return self

    def predict(self, frame):
        return np.full(len(frame), self.value)


class TabPFNStore:
    # TabPFN has no fitted parameters: the "model" is the standardized training set plus the pretrained v2 weights.
    def __init__(self, task):
        self.task = task

    def fit(self, frame, target, groups):
        self.preprocessor = Preprocessor(PERSON, MEAL).fit(frame, groups)
        self.x, self.y = np.hstack(self.preprocessor.transform(frame)), np.asarray(target)
        return self

    def predict(self, frame):
        os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        x = np.hstack(self.preprocessor.transform(frame))
        if self.task == 'binary':
            model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cpu', random_state=FOLD_SEED)
            return model.fit(self.x, self.y.astype(int)).predict_proba(x)[:, 1]
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device='cpu', random_state=FOLD_SEED)
        return model.fit(self.x, self.y).predict(x, output_type='median')


def make_comparator(family, config, task):
    if family == 'xgboost':
        from xgboost import XGBClassifier, XGBRegressor
        shared = {'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': FOLD_SEED, 'n_jobs': 1, **config}
        model = XGBClassifier(objective='binary:logistic', **shared) if task == 'binary' else XGBRegressor(objective='reg:absoluteerror', **shared)
    elif family == 'linear':
        from sklearn.linear_model import LogisticRegression, QuantileRegressor
        model = LogisticRegression(max_iter=1000, **config) if task == 'binary' else QuantileRegressor(quantile=0.5, solver='highs', **config)
    return Comparator(model, task)


def patient_score(frame, prediction, task):
    # Tuning criterion: patient-macro MAE (regression) or patient-averaged log loss (binary).
    if task == 'binary':
        probability = np.clip(prediction, 1e-6, 1 - 1e-6)
        loss = -(frame.peak_ge_180 * np.log(probability) + (1 - frame.peak_ge_180) * np.log(1 - probability))
    else:
        loss = np.abs(frame.iauc - prediction)
    return frame.assign(loss=loss).groupby('patient_id').loss.mean().mean()


def cross_validated(family, config, frame, task, folds):
    prediction = np.full(len(frame), np.nan)
    for held in folds:
        testing = frame.patient_id.isin(held).to_numpy()
        model = make_comparator(family, config, task).fit(frame.loc[~testing], frame.loc[~testing, TARGET[task]], frame.loc[~testing, 'patient_id'])
        prediction[testing] = model.predict(frame.loc[testing])
    return patient_score(frame, prediction, task)


def tune(family, grid, frame, task, folds):
    scores = Parallel(n_jobs=-2)(delayed(cross_validated)(family, config, frame, task, folds) for config in grid)
    table = pd.DataFrame([{'task': task, 'family': family, 'config': json.dumps(config), 'cv_score': score} for config, score in zip(grid, scores)])
    return grid[int(table.cv_score.idxmin())], table


def save(model, task, name):
    path = MODELS / task / f'{name}.joblib'
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return str(path.relative_to(STAGE))


def fit_stage():
    training = shanghai_training()
    folds = patient_folds(training.patient_id, 5, FOLD_SEED)
    receipt = {'fitted_at_utc': datetime.now(timezone.utc).isoformat(), 'fold_seed': FOLD_SEED, 'selected': {}, 'models': {}}
    tables = []
    for task in ['regression', 'binary']:
        target = training[TARGET[task]]
        for family, grid in [('xgboost', XGBOOST_GRID), ('linear', LINEAR_GRID[task])]:
            best, table = tune(family, grid, training, task, folds)
            tables.append(table)
            receipt['selected'][f'{task}/{family}'] = best
            receipt['models'][f'{task}/{family}'] = save(make_comparator(family, best, task).fit(training, target, training.patient_id), task, family)
            print(task, family, best, flush=True)
        receipt['models'][f'{task}/constant'] = save(Constant(task).fit(training, target, training.patient_id), task, 'constant')
        receipt['models'][f'{task}/tabpfn'] = save(TabPFNStore(task).fit(training, target, training.patient_id), task, 'tabpfn')
    # Sensitivity analysis 1 (protocol 5.4): XGBoost retrained with staples as cooked weight, regression only.
    cooked = shanghai_training(variant='staples_cooked')
    best, table = tune('xgboost', XGBOOST_GRID, cooked, 'regression', folds)
    tables.append(table.assign(family='xgboost_staples_cooked'))
    receipt['selected']['regression/xgboost_staples_cooked'] = best
    receipt['models']['regression/xgboost_staples_cooked'] = save(make_comparator('xgboost', best, 'regression').fit(cooked, cooked.iauc, cooked.patient_id), 'regression', 'xgboost_staples_cooked')
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(REPORTS / 'comparators_tuning.csv', index=False)
    (REPORTS / 'comparators_receipt.json').write_text(json.dumps(receipt, indent=2))


def predict_stage(task, name, frame_path, out_path):
    model = joblib.load(MODELS / task / f'{name}.joblib')
    np.save(out_path, np.asarray(model.predict(pd.read_pickle(frame_path)), dtype=float))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['fit', 'predict'])
    parser.add_argument('arguments', nargs='*')
    arguments = parser.parse_args()
    # Imported under its module name so that saved objects reference `comparators.Comparator`, not `__main__`.
    import comparators as module
    if arguments.stage == 'fit':
        module.fit_stage()
    else:
        module.predict_stage(*arguments.arguments)


if __name__ == '__main__':
    main()
