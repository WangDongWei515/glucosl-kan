import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.glucosl_preprocess import Preprocessor  # noqa: F401  (re-exported for the development scripts)
from .kan import KAN


def mlp(widths):
    layers = []
    for inputs, outputs in zip(widths[:-2], widths[1:-1]):
        layers += [nn.Linear(inputs, outputs), nn.SiLU()]
    return nn.Sequential(*layers, nn.Linear(widths[-2], widths[-1]))


def make_part(kind, inputs, hidden, grid, outputs=1, monotone=None):
    widths = [inputs, *hidden, outputs]
    return KAN(widths, grid, monotone) if kind == 'kan' else mlp(widths)


# Meal columns in src.glucosl_data.MEAL order: carbohydrate, protein, fat, fiber, hour_sin, hour_cos.
CARBOHYDRATE, PROTEIN, FAT, FIBER, TIME = [0], [1], [2], [3], [4, 5]


class TwoPart(nn.Module):
    # Prediction = sum over channels of sensitivity(person) x load(meal); softplus keeps every part positive.
    # timed: meal time moves from the load to the sensitivity (structure B).
    # channels=2: separate carbohydrate and fat-protein sensitivities and loads (structure C).
    # meal_monotone: {meal column index: +1 or -1}, applied to the load parts (e.g. carbohydrate non-decreasing).
    def __init__(self, kind, person_inputs, meal_inputs, hidden, grid, timed=False, channels=1, meal_monotone=None):
        super().__init__()
        self.timed = timed
        time = [] if timed else TIME
        self.columns = ([CARBOHYDRATE + PROTEIN + FAT + FIBER + time] if channels == 1
                        else [CARBOHYDRATE + FIBER + time, PROTEIN + FAT])
        self.person = make_part(kind, person_inputs + (len(TIME) if timed else 0), hidden, grid, outputs=channels)
        signs = lambda columns: [meal_monotone.get(column, 0) for column in columns] if meal_monotone else None
        self.loads = nn.ModuleList(make_part(kind, len(columns), hidden, grid, monotone=signs(columns)) for columns in self.columns)

    def parts(self, person, meal):
        if self.timed:
            person = torch.cat([person, meal[:, TIME]], dim=1)
        sensitivity = F.softplus(self.person(person))
        load = torch.cat([F.softplus(part(meal[:, columns])) for part, columns in zip(self.loads, self.columns)], dim=1)
        return sensitivity, load

    def forward(self, person, meal):
        sensitivity, load = self.parts(person, meal)
        return (sensitivity * load).sum(dim=1)


class Single(nn.Module):
    # Ablation 2: one network over person and meal inputs (additive inside softplus for a single KAN layer).
    # meal_monotone uses meal column indices, as in TwoPart; the person columns are unconstrained.
    def __init__(self, kind, person_inputs, meal_inputs, hidden, grid, meal_monotone=None):
        super().__init__()
        signs = [0] * person_inputs + [meal_monotone.get(column, 0) for column in range(meal_inputs)] if meal_monotone else None
        self.net = make_part(kind, person_inputs + meal_inputs, hidden, grid, monotone=signs)

    def forward(self, person, meal):
        return F.softplus(self.net(torch.cat([person, meal], dim=1))).squeeze(-1)


def spline_penalty(module):
    terms = [part.spline_penalty() for part in module.modules() if isinstance(part, KAN)]
    return torch.stack(terms).mean() if terms else torch.zeros(())


def smoothness_penalty(module):
    terms = [part.smoothness_penalty() for part in module.modules() if isinstance(part, KAN)]
    return torch.stack(terms).mean() if terms else torch.zeros(())


class GlucoSLModel:
    # task='regression': prediction = sensitivity x load (iAUC). task='binary': logit = log(sensitivity x load) + bias,
    # so sensitivity and load act multiplicatively on the odds (protocol G1, 4.2).
    # weighting='patient' scores each patient equally in the loss and in early stopping, matching patient-macro MAE.
    def __init__(self, person, meal, structure='two_part', kind='kan', hidden=(), grid=5, penalty=0.0, weight_decay=0.0,
                 learning_rate=0.01, epochs=3000, patience=150, seed=0, timed=False, channels=1, weighting='meal',
                 meal_monotone=None, smoothness=0.0, task='regression'):
        self.preprocessor = Preprocessor(person, meal)
        self.task = task
        self.structure, self.kind, self.hidden, self.grid = structure, kind, tuple(hidden), grid
        self.penalty, self.weight_decay, self.learning_rate = penalty, weight_decay, learning_rate
        self.epochs, self.patience, self.seed = epochs, patience, seed
        self.timed, self.channels, self.weighting = timed, channels, weighting
        # meal_monotone uses meal column names, e.g. {'carbohydrate_g': 1, 'fiber_g': -1}.
        self.meal_monotone, self.smoothness = meal_monotone, smoothness

    def _tensors(self, frame):
        return [torch.from_numpy(values) for values in self.preprocessor.transform(frame, clip=True)]

    def fit(self, frame, target, groups):
        groups, target = np.asarray(groups), np.asarray(target, dtype=np.float32)
        self.preprocessor.fit(frame, groups)
        self.scale = float(target.std()) if self.task == 'regression' else 1.0
        # Early stopping on 10% of the training patients, held out inside this fit.
        patients = np.unique(groups)
        generator = np.random.default_rng(self.seed)
        stopping = generator.choice(patients, size=max(1, round(0.1 * len(patients))), replace=False)
        held = np.isin(groups, stopping)
        person, meal = self._tensors(frame)
        outcome = torch.from_numpy(target / self.scale)
        counts = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy(dtype=np.float32)
        weights = torch.from_numpy(1 / counts if self.weighting == 'patient' else np.ones_like(counts))
        torch.manual_seed(self.seed)
        monotone = {self.preprocessor.meal.index(name): sign for name, sign in self.meal_monotone.items()} if self.meal_monotone else None
        if self.structure == 'two_part':
            network = TwoPart(self.kind, person.shape[1], meal.shape[1], self.hidden, self.grid, self.timed, self.channels, monotone)
        else:
            network = Single(self.kind, person.shape[1], meal.shape[1], self.hidden, self.grid, monotone)
        if self.task == 'binary':
            # Bias starts at the training log-odds; the network's positive output enters through its logarithm.
            rate = float(np.clip(target[~held].mean(), 1e-3, 1 - 1e-3))
            self.bias = nn.Parameter(torch.tensor(float(np.log(rate / (1 - rate)))))
            parameters = list(network.parameters()) + [self.bias]
        else:
            parameters = network.parameters()
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        fitting, stopping_rows = torch.from_numpy(~held), torch.from_numpy(held)

        def error(rows):
            output = network(person[rows], meal[rows])
            if self.task == 'binary':
                losses = F.binary_cross_entropy_with_logits(torch.log(output + 1e-6) + self.bias, outcome[rows], reduction='none')
            else:
                losses = (output - outcome[rows]).abs()
            return (weights[rows] * losses).sum() / weights[rows].sum()

        best, best_state, waited = np.inf, None, 0
        for epoch in range(self.epochs):
            network.train()
            optimizer.zero_grad()
            loss = error(fitting) + self.penalty * spline_penalty(network) + self.smoothness * smoothness_penalty(network)
            loss.backward()
            optimizer.step()
            network.eval()
            with torch.no_grad():
                validation = float(error(stopping_rows))
            if validation < best - 1e-6:
                best, best_state, waited = validation, copy.deepcopy(network.state_dict()), 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        network.load_state_dict(best_state)
        self.network, self.epochs_run = network.eval(), epoch + 1
        return self

    def predict(self, frame):
        # Regression: predicted iAUC. Binary: predicted probability.
        with torch.no_grad():
            output = self.network(*self._tensors(frame))
            if self.task == 'binary':
                return torch.sigmoid(torch.log(output + 1e-6) + self.bias).numpy()
            return output.numpy() * self.scale

    def parts(self, frame):
        # Personal sensitivities and meal loads (one column per channel) on the internal scale;
        # the sum over channels of their products, times `scale`, is the prediction.
        with torch.no_grad():
            sensitivity, load = self.network.parts(*self._tensors(frame))
        return sensitivity.numpy(), load.numpy()


GlucoSLRegressor = GlucoSLModel  # name used by the development scripts


class GlucoSLEnsemble:
    # Members differ in seed (initialization and early-stopping patients); with bootstrap=True each member is also
    # trained on its own bootstrap sample of patients. The prediction (iAUC or probability) is the members' mean.
    def __init__(self, person, meal, seeds, bootstrap=False, **settings):
        self.members = [GlucoSLModel(person, meal, seed=seed, **settings) for seed in seeds]
        self.seeds, self.bootstrap = seeds, bootstrap

    def fit(self, frame, target, groups):
        frame = frame.assign(_target=np.asarray(target, dtype=float), _group=np.asarray(groups))
        for seed, member in zip(self.seeds, self.members):
            sample = frame
            if self.bootstrap:
                patients = frame._group.unique()
                chosen = np.random.default_rng(seed).choice(patients, size=len(patients), replace=True)
                # Duplicated patients are relabelled so that each copy is its own group.
                sample = pd.concat([frame.loc[frame._group.eq(patient)].assign(_group=f'{patient}#{index}') for index, patient in enumerate(chosen)], ignore_index=True)
            member.fit(sample, sample._target, sample._group)
        return self

    def predict(self, frame):
        return np.mean([member.predict(frame) for member in self.members], axis=0)
