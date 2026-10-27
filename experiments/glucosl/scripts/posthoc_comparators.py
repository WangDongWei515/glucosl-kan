"""Post hoc comparators requested by the mock review (not in protocol G1; all labelled post hoc):

  --part xgboost   XGBoost with a wider grid, with guideline monotone constraints (carbohydrate +, fiber -), and both;
  --part gam       shape-constrained generalized additive models (pyGAM): linear, log-scale (multiplicative) and logistic;
  --part evaluate  metrics and paired differences from the frozen GlucoSL-KAN on the same validation sets, 2,000 patient-cluster draws.

The xgboost and evaluate parts run in the study environment without PyTorch; the gam part runs in a scratch environment that
has pyGAM (which pins an older SciPy), with only numpy and pandas from this repository's code.
"""
from itertools import product
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STAGE))
from src.glucosl_data import MEAL, PERSON, cgmacros, shanghai_training, shanghai_validation
from src.glucosl_preprocess import Preprocessor

OUTPUT = STAGE / 'results/glucosl/posthoc'
FOLD_SEED, SEED, DRAWS = 20261021, 20261021, 2000
TARGET = {'regression': 'iauc', 'binary': 'peak_ge_180'}
COLUMNS = PERSON + MEAL
CARB, FIBER = COLUMNS.index('carbohydrate_g'), COLUMNS.index('fiber_g')
PROTOCOL_GRID = [dict(zip(['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight'], v)) for v in product([2, 3, 4], [0.03, 0.1], [100, 300], [5, 20])]
WIDE_GRID = [dict(zip(['max_depth', 'learning_rate', 'n_estimators', 'min_child_weight'], v)) for v in product([2, 3, 4, 5, 6], [0.01, 0.03, 0.1], [50, 100, 300, 600], [5, 20, 50])]
LAM_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def patient_folds(groups, folds, seed):
    # Same as src.external_metrics.patient_folds (copied so that the gam part needs no scikit-learn).
    patients = np.array(sorted(set(groups)))
    order = np.random.default_rng(seed).permutation(len(patients))
    return [patients[order[index::folds]] for index in range(folds)]


def sets():
    us = cgmacros()
    return {'shanghai_B': shanghai_validation(), 'cgmacros_diabetes_range': us.loc[us.diabetes_range].reset_index(drop=True), 'cgmacros_all': us}


def design(preprocessor, frame):
    return np.hstack(preprocessor.transform(frame))


def patient_score(frame, prediction, task):
    if task == 'binary':
        p = np.clip(prediction, 1e-6, 1 - 1e-6)
        loss = -(frame.peak_ge_180 * np.log(p) + (1 - frame.peak_ge_180) * np.log(1 - p))
    else:
        loss = np.abs(frame.iauc - prediction)
    return frame.assign(loss=loss).groupby('patient_id').loss.mean().mean()


def tune_and_predict(make_model, grid, training, task, folds, validation_sets):
    # Patient-grouped 5-fold CV in set A over `grid`; the best configuration is refit on all of A and applied to every validation set.
    scores = []
    for config in grid:
        prediction = np.full(len(training), np.nan)
        for held in folds:
            testing = training.patient_id.isin(held).to_numpy()
            fitted = make_model(config, training.loc[~testing])
            prediction[testing] = fitted(training.loc[testing])
        scores.append(patient_score(training, prediction, task))
    best = grid[int(np.argmin(scores))]
    fitted = make_model(best, training)
    return best, float(min(scores)), {label: fitted(frame) for label, frame in validation_sets.items()}


def xgboost_part():
    from joblib import Parallel, delayed
    from xgboost import XGBClassifier, XGBRegressor
    training, validation_sets = shanghai_training(), sets()
    folds = patient_folds(training.patient_id, 5, FOLD_SEED)

    def make(task, monotone):
        def factory(config, frame):
            preprocessor = Preprocessor(PERSON, MEAL).fit(frame, frame.patient_id)
            constraints = tuple(1 if i == CARB else -1 if i == FIBER else 0 for i in range(len(COLUMNS))) if monotone else None
            shared = {'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': FOLD_SEED, 'n_jobs': 1, **config}
            if constraints:
                shared['monotone_constraints'] = constraints
            model = XGBClassifier(objective='binary:logistic', **shared) if task == 'binary' else XGBRegressor(objective='reg:absoluteerror', **shared)
            model.fit(design(preprocessor, frame), frame[TARGET[task]].to_numpy())
            return lambda new: model.predict_proba(design(preprocessor, new))[:, 1] if task == 'binary' else model.predict(design(preprocessor, new))
        return factory

    variants = {'xgboost_wide': (WIDE_GRID, False), 'xgboost_monotone': (PROTOCOL_GRID, True), 'xgboost_monotone_wide': (WIDE_GRID, True)}

    def run(name, task):
        grid, monotone = variants[name]
        # Parallel over configurations: each job runs the 5 folds of one configuration.
        def score(config):
            prediction = np.full(len(training), np.nan)
            for held in folds:
                testing = training.patient_id.isin(held).to_numpy()
                prediction[testing] = make(task, monotone)(config, training.loc[~testing])(training.loc[testing])
            return patient_score(training, prediction, task)
        scores = Parallel(n_jobs=-2)(delayed(score)(config) for config in grid)
        best = grid[int(np.argmin(scores))]
        fitted = make(task, monotone)(best, training)
        return name, task, best, float(min(scores)), {label: fitted(frame) for label, frame in validation_sets.items()}

    rows, selected = [], {}
    for name in variants:
        for task in ['regression', 'binary']:
            name_, task_, best, cv, predictions = run(name, task)
            selected[f'{task}/{name}'] = {'config': best, 'cv_score': cv}
            print(name, task, best, round(cv, 3), flush=True)
            for label, prediction in predictions.items():
                frame = validation_sets[label]
                rows.append(pd.DataFrame({'set': label, 'task': task, 'model': name, 'record_id': frame.record_id, 'patient_id': frame.patient_id, 'prediction': prediction}))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(OUTPUT / 'comparator_predictions_xgboost.csv', index=False)
    (OUTPUT / 'comparator_selected_xgboost.json').write_text(json.dumps(selected, indent=2))


def gam_part():
    from pygam import LinearGAM, LogisticGAM, f, s
    training, validation_sets = shanghai_training(), sets()
    folds = patient_folds(training.patient_id, 5, FOLD_SEED)
    sex = COLUMNS.index('sex')

    def terms():
        parts = []
        for i, column in enumerate(COLUMNS):
            if i == sex:
                parts.append(f(i))
            elif i == CARB:
                parts.append(s(i, n_splines=10, constraints='monotonic_inc'))
            elif i == FIBER:
                parts.append(s(i, n_splines=10, constraints='monotonic_dec'))
            else:
                parts.append(s(i, n_splines=10))
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return total

    def make(kind):
        def factory(config, frame):
            preprocessor = Preprocessor(PERSON, MEAL).fit(frame, frame.patient_id)
            x = design(preprocessor, frame)
            if kind == 'gam_logistic':
                model = LogisticGAM(terms(), lam=config['lam']).fit(x, frame.peak_ge_180.to_numpy())
                return lambda new: model.predict_proba(design(preprocessor, new))
            if kind == 'gam_log':
                model = LinearGAM(terms(), lam=config['lam']).fit(x, np.log1p(frame.iauc.to_numpy()))
                return lambda new: np.clip(np.expm1(model.predict(design(preprocessor, new))), 0, None)
            model = LinearGAM(terms(), lam=config['lam']).fit(x, frame.iauc.to_numpy())
            return lambda new: np.clip(model.predict(design(preprocessor, new)), 0, None)
        return factory

    rows, selected = [], {}
    for kind, task in [('gam_linear', 'regression'), ('gam_log', 'regression'), ('gam_logistic', 'binary')]:
        best, cv, predictions = tune_and_predict(make(kind), [{'lam': lam} for lam in LAM_GRID], training, task, folds, validation_sets)
        selected[f'{task}/{kind}'] = {'config': best, 'cv_score': cv}
        print(kind, task, best, round(cv, 3), flush=True)
        for label, prediction in predictions.items():
            frame = validation_sets[label]
            rows.append(pd.DataFrame({'set': label, 'task': task, 'model': kind, 'record_id': frame.record_id, 'patient_id': frame.patient_id, 'prediction': prediction}))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(OUTPUT / 'comparator_predictions_gam.csv', index=False)
    (OUTPUT / 'comparator_selected_gam.json').write_text(json.dumps(selected, indent=2))


def evaluate_part():
    from sklearn.metrics import roc_auc_score
    frozen = pd.read_csv(STAGE / 'results/glucosl/evaluation/predictions.csv')
    new = pd.concat([pd.read_csv(OUTPUT / 'comparator_predictions_xgboost.csv'), pd.read_csv(OUTPUT / 'comparator_predictions_gam.csv')], ignore_index=True)
    rows = []
    for (set_, task), part in new.groupby(['set', 'task']):
        base = frozen[(frozen.set == set_) & (frozen.task == task)].reset_index(drop=True)
        wide = part.pivot_table(index=part.groupby(['model']).cumcount(), columns='model', values='prediction')
        assert len(wide) == len(base)
        table = base[['patient_id', 'observed', 'glucosl_kan', 'xgboost']].join(wide)
        models = ['glucosl_kan', 'xgboost'] + list(wide.columns)
        groups = table.patient_id.to_numpy()
        patients = np.unique(groups)
        index = {p: np.flatnonzero(groups == p) for p in patients}

        def metric(rows_, labels, model):
            observed, prediction = table.observed.to_numpy()[rows_], table[model].to_numpy()[rows_]
            if task == 'regression':
                return pd.Series(np.abs(observed - prediction)).groupby(labels).mean().mean()
            weights = 1 / pd.Series(labels).map(pd.Series(labels).value_counts()).to_numpy()
            return roc_auc_score(observed, prediction, sample_weight=weights) if 0 < observed.sum() < len(observed) else np.nan

        generator = np.random.default_rng(SEED)
        draws = {model: [] for model in models}
        for _ in range(DRAWS):
            chosen = generator.choice(patients, size=len(patients), replace=True)
            rows_ = np.concatenate([index[p] for p in chosen]); labels = np.concatenate([np.full(len(index[p]), i) for i, p in enumerate(chosen)])
            for model in models:
                draws[model].append(metric(rows_, labels, model))
        all_rows, all_labels = np.arange(len(table)), groups
        for model in models:
            estimate = metric(all_rows, all_labels, model)
            values = np.array(draws[model], float)
            difference = np.array(draws['glucosl_kan'], float) - values
            rows.append({'set': set_, 'task': task, 'model': model, 'metric': 'macro_mae' if task == 'regression' else 'auroc', 'estimate': estimate,
                         'low': np.nanpercentile(values, 2.5), 'high': np.nanpercentile(values, 97.5),
                         'difference_from_glucosl_kan': metric(all_rows, all_labels, 'glucosl_kan') - estimate,
                         'difference_low': np.nanpercentile(difference, 2.5), 'difference_high': np.nanpercentile(difference, 97.5), 'patients': len(patients), 'meals': len(table)})
        print(set_, task, 'done', flush=True)
    pd.DataFrame(rows).round(4).to_csv(OUTPUT / 'posthoc_comparators.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['xgboost', 'gam', 'evaluate'], required=True)
    {'xgboost': xgboost_part, 'gam': gam_part, 'evaluate': evaluate_part}[parser.parse_args().part]()
