import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def patient_weights(groups):
    groups = pd.Series(np.asarray(groups))
    return (1 / groups.map(groups.value_counts())).to_numpy(dtype=float)


def logit(probability):
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def continuous_metrics(observed, prediction, groups, reference=None):
    observed, prediction = np.asarray(observed, dtype=float), np.asarray(prediction, dtype=float)
    weights = patient_weights(groups)
    frame = pd.DataFrame({'patient': np.asarray(groups), 'absolute': np.abs(observed - prediction), 'squared': (observed - prediction) ** 2})
    mean_observed = np.average(observed, weights=weights)
    slope = np.cov(prediction, observed, aweights=weights)[0, 1] / np.cov(prediction, aweights=weights) if np.ptp(prediction) > 0 else np.nan
    result = {
        'macro_mae': frame.groupby('patient').absolute.mean().mean(),
        'rmse': np.sqrt(np.average(frame.squared, weights=weights)),
        'r2': 1 - np.average(frame.squared, weights=weights) / np.average((observed - mean_observed) ** 2, weights=weights),
        'calibration_in_the_large': mean_observed - np.average(prediction, weights=weights),
        'calibration_slope': float(slope),
    }
    if reference is not None:
        result['reference_macro_mae'] = pd.DataFrame({'patient': np.asarray(groups), 'absolute': np.abs(observed - reference)}).groupby('patient').absolute.mean().mean()
    return result


def binary_metrics(observed, probability, groups):
    observed, probability = np.asarray(observed, dtype=int), np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    weights = patient_weights(groups)
    result = {'observed_rate': np.average(observed, weights=weights), 'mean_predicted': np.average(probability, weights=weights),
              'brier': np.average((observed - probability) ** 2, weights=weights), 'auroc': np.nan, 'calibration_slope': np.nan}
    if 0 < observed.sum() < len(observed):
        result['auroc'] = roc_auc_score(observed, probability, sample_weight=weights)
        model = LogisticRegression(C=np.inf, max_iter=1000).fit(logit(probability)[:, None], observed, sample_weight=weights)
        result['calibration_slope'] = float(model.coef_[0, 0])
    return result


def patient_folds(groups, folds, seed):
    patients = np.array(sorted(set(groups)))
    order = np.random.default_rng(seed).permutation(len(patients))
    return [patients[order[index::folds]] for index in range(folds)]


def recalibrate_continuous(observed, prediction, groups, folds=5, seed=20261020):
    # Cross-fitted updates: each patient's predictions are recalibrated with parameters estimated from other patients only.
    observed, prediction, groups = np.asarray(observed, dtype=float), np.asarray(prediction, dtype=float), np.asarray(groups)
    intercept_only, linear = np.full(len(observed), np.nan), np.full(len(observed), np.nan)
    for held_out in patient_folds(groups, folds, seed):
        testing = np.isin(groups, held_out)
        weights = patient_weights(groups[~testing])
        shift = np.average(observed[~testing] - prediction[~testing], weights=weights)
        slope = np.cov(prediction[~testing], observed[~testing], aweights=weights)[0, 1] / np.cov(prediction[~testing], aweights=weights)
        intercept = np.average(observed[~testing], weights=weights) - slope * np.average(prediction[~testing], weights=weights)
        intercept_only[testing] = np.maximum(0, prediction[testing] + shift)
        linear[testing] = np.maximum(0, intercept + slope * prediction[testing])
    return intercept_only, linear


def recalibrate_binary(observed, probability, groups, folds=5, seed=20261020):
    observed, groups = np.asarray(observed, dtype=int), np.asarray(groups)
    offset = logit(np.asarray(probability, dtype=float))
    updated = np.full(len(observed), np.nan)
    for held_out in patient_folds(groups, folds, seed):
        testing = np.isin(groups, held_out)
        weights = patient_weights(groups[~testing])
        shift = 0.0
        for _ in range(50):
            fitted = 1 / (1 + np.exp(-(offset[~testing] + shift)))
            gradient = np.sum(weights * (observed[~testing] - fitted))
            shift += gradient / max(np.sum(weights * fitted * (1 - fitted)), 1e-9)
            if abs(gradient) < 1e-10:
                break
        updated[testing] = 1 / (1 + np.exp(-(offset[testing] + shift)))
    return updated


def cluster_bootstrap(frame, statistic, draws=2000, seed=20261020):
    # Resample patients with replacement; each draw keeps all meals of a selected patient, relabelled so duplicates stay distinct.
    patients = frame.patient.unique()
    by_patient = {patient: group for patient, group in frame.groupby('patient')}
    generator = np.random.default_rng(seed)
    results = []
    for _ in range(draws):
        chosen = generator.choice(patients, size=len(patients), replace=True)
        sample = pd.concat([by_patient[patient].assign(patient=f'{patient}#{index}') for index, patient in enumerate(chosen)], ignore_index=True)
        results.append(statistic(sample))
    return pd.DataFrame(results)
