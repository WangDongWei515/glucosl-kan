"""Protocol G1, Section 6: descriptive interpretability outputs from the frozen regression models.

Reads the frozen models and the evaluation predictions; writes figures and tables to results/glucosl/interpretability.
Definitions (ensemble of M members, member m has internal sensitivity s_m, load l_m and outcome scale c_m):
  relative sensitivity of a person  = mean_m s_m(person) / mean(s_m over set-A patients)
  load of a meal (mg/dL*h)          = mean_m l_m(meal) * mean(s_m over set-A patients) * c_m
so that relative sensitivity x load is the predicted iAUC of that meal for a patient of average sensitivity.
"""
from pathlib import Path
import json
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
from src.glucosl_data import MEAL, PERSON, cgmacros, shanghai_training, shanghai_validation

MODELS = STAGE / 'results/glucosl/models/regression'
OUTPUT = STAGE / 'results/glucosl/interpretability'
LABELS = {'age': 'Age (years)', 'sex': 'Sex', 'bmi': 'BMI (kg/m²)', 'hba1c_pct': 'HbA1c (%)', 'fasting_glucose_mgdl': 'Fasting glucose (mg/dL)',
          'carbohydrate_g': 'Carbohydrate (g)', 'protein_g': 'Protein (g)', 'fat_g': 'Fat (g)', 'fiber_g': 'Fiber (g)', 'hour': 'Clock time (h)'}
SEED = 20261021
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'figure.dpi': 150})
BLUE, TEAL, RED, GREY, DARK = '#1f4e79', '#2a9d8f', '#c0504d', '#8a8a8a', '#333333'  # Shanghai A, Shanghai B, US, structure, text


def letter(axis, text):
    axis.text(-0.08, 1.06, text, transform=axis.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right')


def parts(model, frame):
    # Per-member internal sensitivity and load (first channel) for every row of `frame`.
    sensitivity, load = [], []
    for member in model.members:
        s, l = member.parts(frame)
        sensitivity.append(s[:, 0]); load.append(l[:, 0])
    return np.array(sensitivity), np.array(load)


class Scales:
    # Set-A reference quantities for the ensemble: per-member mean sensitivity over patients, and outcome scales.
    def __init__(self, model, reference):
        patients = reference.groupby('patient_id').first().reset_index()
        s, _ = parts(model, patients)
        self.mean_sensitivity = s.mean(axis=1)
        self.scale = np.array([member.scale for member in model.members])

    def relative_sensitivity(self, model, frame):
        s, _ = parts(model, frame)
        return (s / self.mean_sensitivity[:, None]).mean(axis=0)

    def load(self, model, frame):
        _, l = parts(model, frame)
        return (l * (self.mean_sensitivity * self.scale)[:, None]).mean(axis=0)

    def relative_sensitivity_members(self, model, frame):
        s, _ = parts(model, frame)
        return s / self.mean_sensitivity[:, None]

    def load_members(self, model, frame):
        _, l = parts(model, frame)
        return l * (self.mean_sensitivity * self.scale)[:, None]


def reference_profile(training):
    person = training.groupby('patient_id').first()[PERSON].median()
    meal = training[MEAL].median()
    noon = 12.5
    meal['hour_sin'], meal['hour_cos'] = np.sin(2 * np.pi * noon / 24), np.cos(2 * np.pi * noon / 24)
    return person, meal


def grid_frame(person, meal, column, values):
    frame = pd.DataFrame([pd.concat([person, meal])] * len(values))
    if column == 'hour':
        frame['hour_sin'], frame['hour_cos'] = np.sin(2 * np.pi * values / 24), np.cos(2 * np.pi * values / 24)
    else:
        frame[column] = values
    return frame


def partial_dependence(model, scales, training, label):
    person, meal = reference_profile(training)
    patients = training.groupby('patient_id').first()
    rows = []
    for column in PERSON:
        values = np.array([0, 1]) if column == 'sex' else np.linspace(*np.percentile(patients[column].dropna(), [5, 95]), 41)
        members = scales.relative_sensitivity_members(model, grid_frame(person, meal, column, values))
        base = scales.relative_sensitivity_members(model, grid_frame(person, meal, column, [person[column]]))
        ratio = members / base  # relative to the reference profile, per member
        for value, mean, low, high in zip(values, ratio.mean(axis=0), ratio.min(axis=0), ratio.max(axis=0)):
            rows.append({'model': label, 'part': 'sensitivity', 'input': column, 'value': value, 'estimate': mean, 'low': low, 'high': high})
    for column in ['carbohydrate_g', 'protein_g', 'fat_g', 'fiber_g', 'hour']:
        values = np.linspace(6, 21, 46) if column == 'hour' else np.linspace(*np.percentile(training[column], [5, 95]), 41)
        members = scales.load_members(model, grid_frame(person, meal, column, values))
        for value, mean, low, high in zip(values, members.mean(axis=0), members.min(axis=0), members.max(axis=0)):
            rows.append({'model': label, 'part': 'load', 'input': column, 'value': value, 'estimate': mean, 'low': low, 'high': high})
    return pd.DataFrame(rows)


def plot_partial_dependence(table, path, models, training, clip_hba1c=None):
    # Rows share their y-axis so that slopes are comparable; rugs show the set-A distribution of each input.
    inputs = [('sensitivity', c) for c in PERSON] + [('load', c) for c in ['carbohydrate_g', 'protein_g', 'fat_g', 'fiber_g', 'hour']]
    figure, axes = plt.subplots(2, 5, figsize=(11, 4.8), sharey='row')
    colors = {'Shanghai-trained': BLUE, 'US-trained': RED}
    patients = training.groupby('patient_id').first()
    hours = training.time.dt.hour + training.time.dt.minute / 60
    for index, (axis, (part, column)) in enumerate(zip(axes.flat, inputs)):
        for label in models:
            rows = table[(table.model == label) & (table.part == part) & (table.input == column)]
            if column == 'sex':
                axis.errorbar(rows.value + (0.08 if label == 'US-trained' else -0.08), rows.estimate, yerr=[rows.estimate - rows.low, rows.high - rows.estimate], fmt='o', color=colors[label], label=label, capsize=2, markersize=4, linewidth=1)
                axis.set_xticks([0, 1]); axis.set_xticklabels(['female', 'male']); axis.set_xlim(-0.6, 1.6)
            else:
                axis.plot(rows.value, rows.estimate, color=colors[label], linewidth=1.4, label=label)
                axis.fill_between(rows.value, rows.low, rows.high, color=colors[label], alpha=0.12, linewidth=0)
        if column != 'sex':
            values = hours if column == 'hour' else (patients[column] if part == 'sensitivity' else training[column])
            values = values.dropna()
            curve = table[(table.part == part) & (table.input == column)].value
            margin = 0.04 * (curve.max() - curve.min())
            axis.set_xlim(curve.min() - margin, curve.max() + margin)
            values = values[(values >= curve.min() - margin) & (values <= curve.max() + margin)]
            axis.plot(values, np.full(len(values), 15 if part == 'load' else 0.5), '|', color=GREY, markersize=4, alpha=0.35, markeredgewidth=0.5, clip_on=False)
        if column == 'hba1c_pct' and clip_hba1c is not None:
            axis.axvspan(clip_hba1c, axis.get_xlim()[1], color=GREY, alpha=0.12, linewidth=0)
            axis.text(axis.get_xlim()[1] - 0.1, 0.53, 'US inputs clipped', fontsize=6, color=GREY, va='bottom', ha='right')
        axis.set_xlabel(LABELS[column])
        if part == 'sensitivity':
            axis.axhline(1, color=GREY, linewidth=0.6, linestyle=':')
            axis.set_ylim(0.5, 1.5)
        else:
            axis.set_ylim(15, 100)
        letter(axis, 'abcdefghij'[index])
    axes[0, 0].set_ylabel('Relative sensitivity'); axes[1, 0].set_ylabel('Meal load (mg/dL·h)')
    axes[0, 0].legend(frameon=False, loc='lower left')
    figure.tight_layout(w_pad=1.0)
    figure.savefig(path.with_suffix('.pdf')); figure.savefig(path.with_suffix('.png'), dpi=300)
    plt.close(figure)


def plot_heatmap(model, scales, training, validation, us, path, gradient=None):
    from scipy.stats import gaussian_kde
    labels = ['Shanghai A (development)', 'Shanghai B (temporal validation)', 'CGMacros, HbA1c ≥6.5%']
    frames = dict(zip(labels, [training, validation, us]))
    colors = dict(zip(labels, [BLUE, TEAL, RED]))
    sens = {label: scales.relative_sensitivity(model, frame.groupby('patient_id').first().reset_index()) for label, frame in frames.items()}
    load = {label: scales.load(model, frame) for label, frame in frames.items()}
    s_range, l_range = np.percentile(sens[labels[0]], [5, 95]), np.percentile(load[labels[0]], [5, 95])
    s_grid, l_grid = np.linspace(*s_range, 60), np.linspace(*l_range, 60)
    values = np.outer(s_grid, l_grid)
    figure = plt.figure(figsize=(6.0, 5.4))
    grid = figure.add_gridspec(2, 2, width_ratios=[4, 1.0], height_ratios=[1.0, 4], hspace=0.06, wspace=0.06)
    main = figure.add_subplot(grid[1, 0]); top = figure.add_subplot(grid[0, 0], sharex=main); right = figure.add_subplot(grid[1, 1], sharey=main)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('paper_blues', gradient or ['#f4f6fa', '#c2cfdf', '#8ea9c6', '#4f7aa3', BLUE, '#122c47'])  # light grey to navy, the paper's blue (option C of the colour comparison)
    image = main.pcolormesh(l_grid, s_grid, values, cmap=cmap, shading='auto')
    low = main.contour(l_grid, s_grid, values, levels=[20, 40], colors=DARK, linewidths=0.6)
    high = main.contour(l_grid, s_grid, values, levels=[60, 80, 100, 120], colors='white', linewidths=0.6)
    main.clabel(low, fmt='%d', fontsize=6); main.clabel(high, fmt='%d', fontsize=6)
    main.set_xlabel('Meal load (mg/dL·h; predicted iAUC for a patient of average sensitivity)')
    main.set_ylabel('Relative personal sensitivity')
    corner = figure.add_subplot(grid[0, 1]); corner.axis('off')
    for label in labels:
        xs = np.linspace(*l_range, 200); top.plot(xs, gaussian_kde(load[label])(xs), color=colors[label], linewidth=1.2, label=label)
        ys = np.linspace(*s_range, 200); right.plot(gaussian_kde(sens[label])(ys), ys, color=colors[label], linewidth=1.2)
    handles, names = top.get_legend_handles_labels()
    corner.legend(handles, names, frameon=False, loc='center left', bbox_to_anchor=(-0.15, 0.5), fontsize=7)
    top.set_yticks([]); right.set_xticks([]); top.tick_params(labelbottom=False); right.tick_params(labelleft=False)
    for axis in (top, right):
        for side in ['top', 'right', 'left' if axis is top else 'bottom']:
            axis.spines[side].set_visible(False)
    top.set_ylabel('meals', color=GREY, fontsize=7); right.set_xlabel('patients', color=GREY, fontsize=7)
    colorbar = figure.colorbar(image, ax=[main, right], location='bottom', pad=0.14, aspect=40, shrink=0.75)
    colorbar.set_label('Predicted 2-h iAUC (mg/dL·h)')
    figure.savefig(path.with_suffix('.pdf'), bbox_inches='tight'); figure.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(figure)
    return pd.DataFrame({'quantity': ['relative_sensitivity'] * 3 + ['load'] * 3, 'set': ['Shanghai A', 'Shanghai B', 'CGMacros, HbA1c ≥6.5%'] * 2,
                         'p5': [np.percentile(sens[l], 5) for l in labels] + [np.percentile(load[l], 5) for l in labels],
                         'median': [np.median(sens[l]) for l in labels] + [np.median(load[l]) for l in labels],
                         'p95': [np.percentile(sens[l], 95) for l in labels] + [np.percentile(load[l], 95) for l in labels]})


def permutation_importance(models, frame, repeats=5):
    generator = np.random.default_rng(SEED)
    rows = []
    for name, model in models.items():
        baseline = macro_mae(frame, model.predict(frame))
        for column in PERSON + MEAL:
            increases = []
            for _ in range(repeats):
                shuffled = frame.copy()
                shuffled[column] = generator.permutation(shuffled[column].to_numpy())
                increases.append(macro_mae(shuffled, model.predict(shuffled)) - baseline)
            rows.append({'model': name, 'input': column, 'baseline_macro_mae': baseline, 'increase_macro_mae': float(np.mean(increases)), 'repeats': repeats})
        print(name, 'permutation importance done', flush=True)
    return pd.DataFrame(rows)


def macro_mae(frame, prediction):
    return frame.assign(error=np.abs(frame.iauc - np.asarray(prediction))).groupby('patient_id').error.mean().mean()


def main(figures_only=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    training, validation, us = shanghai_training(), shanghai_validation(), cgmacros()
    us_dm = us.loc[us.diabetes_range].reset_index(drop=True)
    local, us_model = joblib.load(MODELS / 'glucosl_kan.joblib'), joblib.load(MODELS / 'glucosl_kan_us.joblib')
    scales = Scales(local, training)
    us_scales = Scales(us_model, training)  # the US-trained model's sensitivities are also normalized on Shanghai A patients
    us_patients = us.groupby('patient_id').first()
    clip_hba1c = float(us_patients.hba1c_pct.mean() + 3 * us_patients.hba1c_pct.std())  # beyond this the US-trained model's HbA1c input is clipped
    if figures_only:
        table = pd.read_csv(OUTPUT / 'partial_dependence.csv')
        plot_heatmap(local, scales, training, validation, us_dm, OUTPUT / 'figure_heatmap')
        plot_partial_dependence(table, OUTPUT / 'figure_learned_functions', ['Shanghai-trained', 'US-trained'], training, clip_hba1c)
        plot_partial_dependence(table[table.model.eq('Shanghai-trained')], OUTPUT / 'figure_learned_functions_shanghai', ['Shanghai-trained'], training)
        print('figures redrawn'); return
    # 1. Heatmap with marginal distributions.
    ranges = plot_heatmap(local, scales, training, validation, us_dm, OUTPUT / 'figure_heatmap')
    ranges.to_csv(OUTPUT / 'sensitivity_load_ranges.csv', index=False)
    # Per-patient sensitivity and per-meal load for the supplement.
    rows = []
    for label, frame in [('shanghai_A', training), ('shanghai_B', validation), ('cgmacros_all', us)]:
        patients = frame.groupby('patient_id').first().reset_index()
        rows.append(pd.DataFrame({'set': label, 'patient_id': patients.patient_id, 'relative_sensitivity': scales.relative_sensitivity(local, patients), 'hba1c_pct': patients.hba1c_pct, 'fasting_glucose_mgdl': patients.fasting_glucose_mgdl, 'bmi': patients.bmi, 'age': patients.age, 'sex': patients.sex}))
    pd.concat(rows, ignore_index=True).to_csv(OUTPUT / 'patient_sensitivity.csv', index=False)
    # 2 and 3. Partial dependence of each part on each input, Shanghai-trained and US-trained, overlaid.
    table = pd.concat([partial_dependence(local, scales, training, 'Shanghai-trained'), partial_dependence(us_model, us_scales, training, 'US-trained')], ignore_index=True)
    table.to_csv(OUTPUT / 'partial_dependence.csv', index=False)
    plot_partial_dependence(table, OUTPUT / 'figure_learned_functions', ['Shanghai-trained', 'US-trained'], training, clip_hba1c)
    plot_partial_dependence(table[table.model.eq('Shanghai-trained')], OUTPUT / 'figure_learned_functions_shanghai', ['Shanghai-trained'], training)
    # 4. Permutation importance in set B for the PyTorch models, linear and TabPFN; XGBoost is added by xgboost_tools.py.
    models = {name: joblib.load(MODELS / f'{name}.joblib') for name in ['glucosl_kan', 'unconstrained_kan', 'single_kan', 'glucosl_mlp', 'linear', 'tabpfn']}
    permutation_importance(models, validation).to_csv(OUTPUT / 'permutation_importance_torch_process.csv', index=False)
    json.dump({'reference_profile': {k: float(v) for k, v in pd.concat(reference_profile(training)).items()}}, open(OUTPUT / 'reference_profile.json', 'w'), indent=2)
    print('interpretability outputs written')


if __name__ == '__main__':
    main(figures_only='--figures-only' in sys.argv)
