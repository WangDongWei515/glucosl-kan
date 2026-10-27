"""Protocol G1, Section 9, steps 3-4: fit every final model on ShanghaiT2DM set A, and the US-trained GlucoSL-KAN on CGMacros.

Set B is never loaded here. The KAN models use the frozen recipe without tuning; the MLP ablation is tuned by
5-fold patient-grouped cross-validation in set A with one fold seed. The comparators (constant, linear, XGBoost,
TabPFN) are fitted by `comparators.py`, which runs as a separate process because XGBoost cannot share a process
with PyTorch on this platform.
"""
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import joblib
import pandas as pd
from joblib import Parallel, delayed


STAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STAGE))
from src.external_metrics import patient_folds
from src.glucosl_data import MEAL, PERSON, cgmacros, shanghai_training
from src.models.glucosl import GlucoSLEnsemble

RECIPE = json.loads((STAGE / 'experiments/glucosl/configs/final_recipe.json').read_text())
FOLD_SEED = 20261021
MODELS = STAGE / 'results/glucosl/models'
REPORTS = STAGE / 'results/glucosl/final_fit'
TARGET = {'regression': 'iauc', 'binary': 'peak_ge_180'}
MLP_GRID = [{'hidden': hidden, 'weight_decay': decay} for hidden, decay in product(RECIPE['ablation_mlp']['grid_search']['hidden'], RECIPE['ablation_mlp']['grid_search']['weight_decay'])]


def glucosl(spec, task, seeds=None, **overrides):
    settings = {key: value for key, value in spec.items() if key not in ('seeds', 'bootstrap', 'grid_search', 'tuning')}
    settings.update(overrides)
    settings['hidden'] = tuple(settings.get('hidden', ()))
    return GlucoSLEnsemble(PERSON, MEAL, seeds or spec['seeds'], bootstrap=spec.get('bootstrap', False), task=task, **settings)


def patient_score(frame, prediction, task):
    import numpy as np
    if task == 'binary':
        probability = np.clip(prediction, 1e-6, 1 - 1e-6)
        loss = -(frame.peak_ge_180 * np.log(probability) + (1 - frame.peak_ge_180) * np.log(1 - probability))
    else:
        loss = np.abs(frame.iauc - prediction)
    return frame.assign(loss=loss).groupby('patient_id').loss.mean().mean()


def cross_validated_mlp(config, frame, task, folds):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    mlp = RECIPE['ablation_mlp']
    prediction = np.full(len(frame), np.nan)
    for held in folds:
        testing = frame.patient_id.isin(held).to_numpy()
        model = glucosl(mlp, task, seeds=[mlp['seeds'][0]], **config).fit(frame.loc[~testing], frame.loc[~testing, TARGET[task]], frame.loc[~testing, 'patient_id'])
        prediction[testing] = model.predict(frame.loc[testing])
    return patient_score(frame, prediction, task)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(model, task, name):
    path = MODELS / task / f'{name}.joblib'
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return str(path.relative_to(STAGE))


def main():
    import torch
    # Single-threaded, deterministic CPU training so that a re-run reproduces the model files byte for byte.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    training = shanghai_training()
    folds = patient_folds(training.patient_id, 5, FOLD_SEED)
    receipt = {'fitted_at_utc': datetime.now(timezone.utc).isoformat(), 'fold_seed': FOLD_SEED,
               'protocol_sha256': sha256(STAGE / 'docs/protocol_glucosl_v1.md'),
               'recipe_sha256': sha256(STAGE / 'experiments/glucosl/configs/final_recipe.json'),
               'training_patients': int(training.patient_id.nunique()), 'training_meals': len(training), 'selected': {}, 'models': {}}
    tables = []
    for task in ['regression', 'binary']:
        target = training[TARGET[task]]
        for name, spec in [('glucosl_kan', RECIPE['glucosl_kan']), ('single_kan', RECIPE['ablation_single_kan']), ('unconstrained_kan', RECIPE['ablation_unconstrained_kan'])]:
            receipt['models'][f'{task}/{name}'] = save(glucosl(spec, task).fit(training, target, training.patient_id), task, name)
            print(task, name, 'fitted', flush=True)
        scores = Parallel(n_jobs=-2)(delayed(cross_validated_mlp)(config, training, task, folds) for config in MLP_GRID)
        table = pd.DataFrame([{'task': task, 'family': 'glucosl_mlp', 'config': json.dumps(config), 'cv_score': score} for config, score in zip(MLP_GRID, scores)])
        tables.append(table)
        best = MLP_GRID[int(table.cv_score.idxmin())]
        receipt['selected'][f'{task}/glucosl_mlp'] = best
        receipt['models'][f'{task}/glucosl_mlp'] = save(glucosl(RECIPE['ablation_mlp'], task, **best).fit(training, target, training.patient_id), task, 'glucosl_mlp')
        print(task, 'glucosl_mlp', best, flush=True)
    # US-trained GlucoSL-KAN (protocol 4.6): frozen recipe, CGMacros statistics.
    us = cgmacros()
    for task in ['regression', 'binary']:
        receipt['models'][f'{task}/glucosl_kan_us'] = save(glucosl(RECIPE['glucosl_kan'], task).fit(us, us[TARGET[task]], us.patient_id), task, 'glucosl_kan_us')
    # Sensitivity analysis 1 (protocol 5.4): staples as cooked weight, regression only.
    cooked = shanghai_training(variant='staples_cooked')
    receipt['models']['regression/glucosl_kan_staples_cooked'] = save(glucosl(RECIPE['glucosl_kan'], 'regression').fit(cooked, cooked.iauc, cooked.patient_id), 'regression', 'glucosl_kan_staples_cooked')
    print('KAN and MLP models fitted; fitting comparators in a separate process', flush=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'comparators.py'), 'fit'], check=True)
    comparators = json.loads((REPORTS / 'comparators_receipt.json').read_text())
    receipt['selected'].update(comparators['selected'])
    receipt['models'].update(comparators['models'])
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.concat(tables + [pd.read_csv(REPORTS / 'comparators_tuning.csv')], ignore_index=True).to_csv(REPORTS / 'tuning.csv', index=False)
    receipt['model_sha256'] = {key: sha256(STAGE / path) for key, path in receipt['models'].items()}
    (REPORTS / 'fit_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt['selected'], indent=2))


if __name__ == '__main__':
    main()
