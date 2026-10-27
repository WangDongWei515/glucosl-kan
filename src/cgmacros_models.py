from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


STAGE = Path(__file__).resolve().parents[1]
CONFIG = json.loads((STAGE / 'experiments/cgmacros/configs/protocol_v1.json').read_text())
CANDIDATES = ['ridge_1', 'ridge_10', 'ridge_100', 'xgb_original', 'xgb_regularized', 'histgb_7', 'histgb_15']


def features_for(recipe):
    definition = CONFIG['recipes'][recipe]
    features = CONFIG['common_features'].copy()
    if definition['nutrients']:
        features.extend(CONFIG['nutrient_features'])
    if definition['labs'] in ['routine', 'extended']:
        features.extend(CONFIG['routine_features'])
    if definition['labs'] == 'extended':
        features.extend(CONFIG['extended_features'])
    assert not any('subject' in field or 'baseline' in field or 'iauc' in field for field in features)
    return features


def subject_weights(groups):
    groups = pd.Series(np.asarray(groups))
    weights = 1 / groups.map(groups.value_counts()).to_numpy(dtype=float)
    return weights / weights.mean()


def macro_mae(observed, prediction, groups):
    return float(pd.DataFrame({'subject': np.asarray(groups), 'error': np.abs(np.asarray(observed) - np.asarray(prediction))}).groupby('subject').error.mean().mean())


def make_model(candidate):
    steps = [('impute', SimpleImputer(strategy='median'))]
    if candidate.startswith('ridge_'):
        steps.extend([('scale', StandardScaler()), ('model', Ridge(alpha=float(candidate.split('_')[1])))])
    elif candidate == 'xgb_original':
        steps.append(('model', XGBRegressor(n_estimators=80, max_depth=1, learning_rate=.2, reg_alpha=1, reg_lambda=0, objective='reg:squarederror', n_jobs=1, random_state=20261020, verbosity=0)))
    elif candidate == 'xgb_regularized':
        steps.append(('model', XGBRegressor(n_estimators=200, max_depth=2, learning_rate=.05, reg_alpha=1, reg_lambda=10, objective='reg:squarederror', n_jobs=1, random_state=20261020, verbosity=0)))
    elif candidate.startswith('histgb_'):
        steps.append(('model', HistGradientBoostingRegressor(max_iter=150, learning_rate=.05, l2_regularization=10, max_leaf_nodes=int(candidate.split('_')[1]), early_stopping=False, random_state=20261020)))
    else:
        raise ValueError(candidate)
    return Pipeline(steps)


def fit_model(candidate, frame, target, features):
    model = make_model(candidate)
    model.fit(frame[features], np.asarray(target), model__sample_weight=subject_weights(frame.subject))
    return model


def predict(model, frame, features):
    return np.maximum(0, model.predict(frame[features]))


def participant_splits(frame, folds, seed):
    participants = frame[['subject', 'hba1c_range']].drop_duplicates().sort_values('subject').reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for training_indices, testing_indices in splitter.split(participants.subject, participants.hba1c_range):
        training_subjects = participants.iloc[training_indices].subject.to_numpy()
        testing_subjects = participants.iloc[testing_indices].subject.to_numpy()
        assert not set(training_subjects).intersection(testing_subjects)
        yield training_subjects, testing_subjects


def select_candidate(frame, target_name, features, seed):
    target = frame[target_name]
    inner_splits = list(participant_splits(frame, CONFIG['inner_folds'], seed))
    scores = []
    for candidate in CANDIDATES:
        predictions = np.full(len(frame), np.nan)
        for training_subjects, testing_subjects in inner_splits:
            training = frame.subject.isin(training_subjects).to_numpy()
            testing = frame.subject.isin(testing_subjects).to_numpy()
            model = fit_model(candidate, frame.loc[training], target.loc[training], features)
            predictions[testing] = predict(model, frame.loc[testing], features)
        assert np.isfinite(predictions).all()
        scores.append({'candidate': candidate, 'inner_macro_mae': macro_mae(target, predictions, frame.subject)})
    selected = min(scores, key=lambda score: score['inner_macro_mae'])['candidate']
    return selected, scores


def mask_to_recipe(frame, recipe):
    masked = frame.copy()
    keep = set(features_for(recipe))
    for column in features_for('nutrition_extended'):
        if column not in keep:
            masked[column] = np.nan
    return masked
