"""Tables (LaTeX) and figures for the manuscript, from the frozen evaluation, post hoc and interpretability outputs."""
from pathlib import Path
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[3]
RESULTS = STAGE / 'results/glucosl'
PAPER = STAGE / 'paper'
TABLES, FIGURES = PAPER / 'tables', PAPER / 'figures'
ESCAPE = {'>=': r'$\geq$', '%': r'\%', '≥': r'$\geq$', '≤': r'$\leq$', 'α': r'$\alpha$', 'ρ': r'$\rho$', '→': r'$\rightarrow$', '–': '--', '−': '$-$', '×': r'$\times$', '_': r'\_', '&': r'\&', '²': '$^2$', '·': r'$\cdot$', '#': r'\#'}
NAMES = {'glucosl_kan': 'GlucoSL-KAN', 'xgboost': 'XGBoost', 'tabpfn': 'TabPFN', 'linear': 'Linear (median / logistic)', 'constant': 'Constant',
         'glucosl_mlp': 'GlucoSL-MLP (ablation 1)', 'single_kan': 'Single KAN (ablation 2)', 'unconstrained_kan': 'Unconstrained GlucoSL-KAN (ablation 3)',
         'glucosl_kan_us': 'GlucoSL-KAN, US-trained'}
SETS = {'shanghai_B': 'ShanghaiT2DM set B', 'cgmacros_diabetes_range': 'CGMacros, HbA1c ≥6.5%', 'cgmacros_all': 'CGMacros, all participants',
        'shanghai_B_well_mapped': 'Set B, well-mapped meals', 'shanghai_B_staples_cooked': 'Set B, staples as cooked weight'}
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6})
BLUE, TEAL, RED, GREY, DARK, ORANGE = '#1f4e79', '#2a9d8f', '#c0504d', '#8a8a8a', '#333333', '#e07b39'  # Shanghai A, Shanghai B, US, structure, text, meal part


def letter(axis, text):
    axis.text(-0.08, 1.04, text, transform=axis.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='right')


def tex(text):
    text = str(text)
    for old, new in ESCAPE.items():
        text = text.replace(old, new)
    return text


def number(value, digits=1):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'NA'
    text = f'{value:.{digits}f}'
    return '0.' + '0' * digits if text.strip('-') == '0.' + '0' * digits else text.replace('-', '−')


def interval(estimate, low, high, digits=1):
    joiner = ' to ' if (low < 0 or high < 0) else '–'
    return f'{number(estimate, digits)} ({number(low, digits)}{joiner}{number(high, digits)})'


def latex_table(path, header, rows, widths, first_left=True):
    columns = ''.join(('>{\\raggedright\\arraybackslash}' if index == 0 and first_left else '>{\\raggedright\\arraybackslash}') + f'p{{{width}cm}}' for index, width in enumerate(widths))
    lines = [f'\\begin{{tabular}}{{{columns}}}', '\\toprule', ' & '.join(f'\\textbf{{{tex(h)}}}' for h in header) + ' \\\\', '\\midrule']
    for row in rows:
        if row == 'midrule':
            lines.append('\\midrule')
        else:
            lines.append(' & '.join(tex(cell) for cell in row) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    path.write_text('\n'.join(lines) + '\n')


def metric(table, set_, task, model, name):
    row = table[(table.set == set_) & (table.task == task) & (table.model == model) & (table.metric == name)]
    return row.iloc[0] if len(row) else None


def difference(table, set_, task, model):
    row = table[(table.set == set_) & (table.task == task) & (table.comparison == f'glucosl_kan minus {model}')]
    return row.iloc[0] if len(row) else None


def table_characteristics(characteristics):
    rows_wanted = [('patients', 'Patients'), ('recordings', 'Recordings'), ('meals', 'Meals'), ('meals per patient', 'Meals per patient'),
                   ('first recording', 'First recording (date)'), ('age', 'Age (years)'), ('female %', 'Female (%)'), ('BMI', 'BMI (kg/m²)'),
                   ('HbA1c %', 'HbA1c (%)'), ('HbA1c missing', 'HbA1c missing (patients)'), ('fasting glucose', 'Fasting glucose (mg/dL)'), ('diabetes years', 'Diabetes duration (years)'),
                   ('insulin use % (patients)', 'Insulin use (% of patients)'), ('metformin %', 'Metformin (%)'), ('alpha_glucosidase_inhibitor %', 'α-glucosidase inhibitor (%)'),
                   ('sulfonylurea %', 'Sulfonylurea (%)'), ('dpp4_inhibitor %', 'DPP-4 inhibitor (%)'), ('sglt2_inhibitor %', 'SGLT2 inhibitor (%)'),
                   ('carbohydrate g', 'Carbohydrate (g per meal)'), ('protein g', 'Protein (g)'), ('fat g', 'Fat (g)'), ('fiber g', 'Fiber (g)'),
                   ('specific mapping share', 'Share of meal weight mapped to specific entries'), ('onset glucose', 'Glucose at meal onset (mg/dL)'),
                   ('iAUC median', '2-h iAUC (mg/dL·h)'), ('iAUC mean', '2-h iAUC, mean'), ('peak >=180 %', 'Peak ≥180 mg/dL (% of meals)')]
    rows = []
    for key, label in rows_wanted:
        values = [characteristics.loc[key, column] if key in characteristics.index else 'NA' for column in characteristics.columns]
        rows.append([label] + ['NA' if (isinstance(v, float) and np.isnan(v)) or v == 'nan' else str(v).replace('alpha', 'α') for v in values])
    latex_table(TABLES / 'table1.tex', ['Characteristic', 'ShanghaiT2DM set A (development)', 'ShanghaiT2DM set B (temporal validation)', 'CGMacros, HbA1c ≥6.5%'], rows, [4.0, 2.8, 2.8, 2.6])


def table_regression_b(metrics, differences):
    rows = []
    for group, models in [('Main comparison', ['glucosl_kan', 'xgboost', 'tabpfn', 'linear', 'constant']), ('Ablations', ['glucosl_mlp', 'single_kan', 'unconstrained_kan']), ('Training data', ['glucosl_kan_us'])]:
        rows.append([f'\\textit{{{group}}}'] + [''] * 5)
        for model in models:
            m = lambda name: metric(metrics, 'shanghai_B', 'regression', model, name)
            d = difference(differences, 'shanghai_B', 'regression', model)
            rows.append([NAMES[model], interval(m('macro_mae').estimate, m('macro_mae').low, m('macro_mae').high),
                         '—' if d is None else interval(d.estimate, d.low, d.high, 2), number(m('meal_mae').estimate),
                         number(m('spearman').estimate, 2) + ' / ' + number(m('within_patient_spearman').estimate, 2),
                         number(m('calibration_slope').estimate, 2) + ' / ' + number(m('calibration_intercept').estimate)])
    latex_table(TABLES / 'table2.tex', ['Model', 'Patient-macro MAE (95% CI)', 'Difference from GlucoSL-KAN (95% CI)', 'Meal-level MAE', 'Spearman, all meals / within patients', 'Calibration slope / intercept'],
                rows, [3.0, 2.1, 2.2, 1.2, 1.7, 1.6])


def table_binary_b(metrics, differences):
    rows = []
    for group, models in [('Main comparison', ['glucosl_kan', 'xgboost', 'tabpfn', 'linear', 'constant']), ('Ablations', ['glucosl_mlp', 'single_kan', 'unconstrained_kan']), ('Training data', ['glucosl_kan_us'])]:
        rows.append([f'\\textit{{{group}}}'] + [''] * 4)
        for model in models:
            m = lambda name: metric(metrics, 'shanghai_B', 'binary', model, name)
            d = difference(differences, 'shanghai_B', 'binary', model)
            rows.append([NAMES[model], interval(m('auroc').estimate, m('auroc').low, m('auroc').high, 3), '—' if d is None else interval(d.estimate, d.low, d.high, 3),
                         number(m('brier').estimate, 3), number(m('calibration_slope').estimate, 2) + ' / ' + number(m('calibration_intercept').estimate, 2)])
    latex_table(TABLES / 'table3.tex', ['Model', 'AUROC (95% CI)', 'Difference from GlucoSL-KAN (95% CI)', 'Brier score', 'Calibration slope / intercept'], rows, [3.6, 2.3, 2.5, 1.4, 2.2])


def table_transport(metrics, differences):
    rows = []
    for set_ in ['cgmacros_diabetes_range', 'cgmacros_all', 'shanghai_B_staples_cooked', 'shanghai_B_well_mapped']:
        rows.append([f'\\textit{{{SETS[set_]}}}'] + [''] * 4)
        models = ['glucosl_kan', 'xgboost', 'tabpfn', 'linear', 'constant', 'glucosl_mlp', 'single_kan', 'unconstrained_kan'] if set_.startswith('cgmacros') else ['glucosl_kan', 'xgboost', 'constant']
        for model in models:
            m = metric(metrics, set_, 'regression', model, 'macro_mae')
            if m is None:
                continue
            d = difference(differences, set_, 'regression', model)
            a = metric(metrics, set_, 'binary', model, 'auroc')
            ad = difference(differences, set_, 'binary', model)
            rows.append([NAMES[model], interval(m.estimate, m.low, m.high), '—' if d is None else interval(d.estimate, d.low, d.high, 2),
                         'NA' if a is None else interval(a.estimate, a.low, a.high, 3), '—' if ad is None else interval(ad.estimate, ad.low, ad.high, 3)])
    latex_table(TABLES / 'table4.tex', ['Set and model', 'Patient-macro MAE (95% CI)', 'MAE difference from GlucoSL-KAN (95% CI)', 'AUROC (95% CI)', 'AUROC difference from GlucoSL-KAN (95% CI)'],
                rows, [3.4, 2.2, 2.3, 1.9, 2.2])


def supplementary_tables(metrics, differences, log, tuning, receipt):
    # S1 development log.
    rows = []
    def splits(value, round_):
        text = str(value)
        if text in ('nan', ''):
            return 'round-1 split'
        if text == 'three splits':
            return 'three splits'
        if text.startswith('['):
            return 'three new splits'
        return 'round-1 split' if int(round_) < 3 or text == '20261021' else 'new split'
    for _, row in log.iterrows():
        change = str(row.change).split(': ', 1)[-1] if ': ' in str(row.change) and not str(row.change).startswith('Initial') else str(row.change)
        change = change.replace('Round-1 best configuration, one seed (reproduces round 1)', 'round-1 best, one seed (reproduces round 1)')
        if pd.notna(row.get('glucosl_cv_macro_mae')):
            rows.append([int(row['round']), change, number(row.glucosl_cv_macro_mae, 2), number(row.xgboost_cv_macro_mae, 2),
                         interval(row.difference, row.difference_low, row.difference_high, 2) if pd.notna(row.difference) else '—', splits(row.fold_seed, row['round'])])
        elif pd.notna(row.get('glucosl_binary_auroc')):
            rows.append([int(row['round']), change, f'AUROC {number(row.glucosl_binary_auroc, 3)}', f'AUROC {number(row.xgboost_binary_auroc, 3)}', f'log loss {number(row.glucosl_binary_log_loss, 3)} vs {number(row.xgboost_binary_log_loss, 3)}', splits(row.fold_seed, row['round'])])
    latex_table(TABLES / 'tableS_development.tex', ['Round', 'Change', 'GlucoSL-KAN CV MAE', 'XGBoost CV MAE', 'Difference (95% bootstrap)', 'Splits'], rows, [0.8, 5.0, 1.3, 1.3, 2.2, 1.2])
    # S2 selected configurations.
    rows = [[key.replace('/', ' / '), json.dumps(value)] for key, value in receipt['selected'].items()]
    latex_table(TABLES / 'tableS_selected.tex', ['Task / model', 'Selected configuration (5-fold patient-grouped CV in set A)'], rows, [3.6, 8.3])
    # S3 all regression metrics; S4 all binary metrics.
    for task, name, columns in [('regression', 'tableS_regression_all', ['macro_mae', 'meal_mae', 'spearman', 'within_patient_spearman', 'calibration_slope', 'calibration_intercept', 'mean_predicted']),
                                ('binary', 'tableS_binary_all', ['auroc', 'brier', 'calibration_slope', 'calibration_intercept', 'mean_predicted', 'observed_rate'])]:
        rows = []
        for set_ in SETS:
            sub = metrics[(metrics.set == set_) & (metrics.task == task)]
            if not len(sub):
                continue
            rows.append([f'\\textit{{{SETS[set_]}}}'] + [''] * len(columns))
            for model in [m for m in NAMES if m in set(sub.model)]:
                cells = []
                for column in columns:
                    r = metric(metrics, set_, task, model, column)
                    digits = 3 if column in ('auroc', 'brier') else 2 if column in ('spearman', 'within_patient_spearman', 'calibration_slope', 'mean_predicted', 'observed_rate') and task == 'binary' or column in ('spearman', 'within_patient_spearman', 'calibration_slope') else 1
                    cells.append('NA' if r is None or np.isnan(r.estimate) else interval(r.estimate, r.low, r.high, digits) if column in ('macro_mae', 'auroc') else number(r.estimate, digits))
                rows.append([NAMES[model]] + cells)
        short = {'macro_mae': 'Patient-macro MAE (95% CI)', 'meal_mae': 'Meal MAE', 'spearman': 'ρ, all meals', 'within_patient_spearman': 'ρ, within patients', 'calibration_slope': 'Cal. slope',
                 'calibration_intercept': 'Cal. intercept', 'mean_predicted': 'Mean predicted', 'auroc': 'AUROC (95% CI)', 'brier': 'Brier', 'observed_rate': 'Observed rate'}
        header = ['Set and model'] + [short[c] for c in columns]
        widths = [2.5, 2.5] + [round(6.9 / (len(columns) - 1), 2)] * (len(columns) - 1)
        latex_table(TABLES / f'{name}.tex', header, rows, widths)
    # S5 all paired differences.
    rows = [[SETS[r.set], r.task, r.metric.replace('_', ' '), NAMES[r.comparison.replace('glucosl_kan minus ', '')], interval(r.estimate, r.low, r.high, 3 if r.metric == 'auroc' else 2)] for _, r in differences.iterrows()]
    latex_table(TABLES / 'tableS_differences.tex', ['Set', 'Task', 'Metric', 'Comparison', 'Estimate (95% CI)'], rows, [3.0, 1.3, 1.3, 3.6, 2.8])


def posthoc_tables():
    posthoc = RESULTS / 'posthoc'
    decomposition = pd.read_csv(posthoc / 'ranking_decomposition.csv')
    rows = [[SETS[r.set], r.task, NAMES[r.model], number(r.pooled_unweighted, 3), number(r.between_patient_spearman, 3), number(r.within_patient_median, 3), int(r.patients_within)] for _, r in decomposition.iterrows()]
    latex_table(TABLES / 'tableS_ranking.tex', ['Set', 'Task', 'Model', 'Pooled (Spearman / AUROC, unweighted)', 'Between patients (Spearman of patient means)', 'Within patients (median Spearman / AUROC)', 'Patients (within)'], rows, [2.0, 1.3, 2.8, 1.6, 1.7, 1.7, 0.9])
    hba1c = pd.read_csv(posthoc / 'hba1c_associations.csv')
    rows = [[r.set, int(r.patients), number(r.spearman_hba1c_mean_iauc, 2), number(r.spearman_hba1c_event_rate, 2), number(r.spearman_fpg_mean_iauc, 2)] for _, r in hba1c.iterrows()]
    latex_table(TABLES / 'tableS_hba1c.tex', ['Set', 'Patients', 'Spearman: HbA1c vs mean iAUC', 'Spearman: HbA1c vs share of peaks ≥180', 'Spearman: fasting glucose vs mean iAUC'], rows, [3.2, 1.3, 2.4, 2.6, 2.5])
    strata = pd.read_csv(posthoc / 'insulin_strata.csv')
    rows = []
    for (task, use), part in strata.groupby(['task', 'insulin_use']):
        rows.append([task, 'insulin' if use else 'no insulin', int(part.patients.iloc[0]), int(part.meals.iloc[0])] + [number(part[part.model == m].value.iloc[0], 3 if task == 'binary' else 1) if len(part[part.model == m]) else 'NA' for m in ['glucosl_kan', 'xgboost', 'constant', 'glucosl_kan_us']])
    latex_table(TABLES / 'tableS_insulin.tex', ['Task', 'Stratum (set B)', 'Patients', 'Meals', 'GlucoSL-KAN', 'XGBoost', 'Constant', 'GlucoSL-KAN, US-trained'], rows, [1.5, 1.7, 1.1, 1.0, 1.7, 1.5, 1.4, 1.8])
    onset = pd.read_csv(posthoc / 'onset_180.csv')
    rows = [[r.set, int(r.meals), number(100 * r.onset_ge_180_share, 0), int(r.events), number(100 * r.events_with_onset_ge_180_share, 0)] for _, r in onset.iterrows()]
    latex_table(TABLES / 'tableS_onset.tex', ['Set', 'Meals', 'Meals with onset ≥180 mg/dL (%)', 'Events (peak ≥180)', 'Events with onset ≥180 (%)'], rows, [3.6, 1.4, 2.6, 2.0, 2.4])
    slot = pd.read_csv(posthoc / 'meal_slot.csv')
    rows = []
    for _, r in slot.iterrows():
        if pd.notna(r.meals):
            rows.append([r.set, r.meal_type, int(r.meals), number(r.mean_onset), number(r.mean_iauc), number(r.median_carbohydrate), '—'])
        else:
            rows.append([r.set, r.meal_type.replace('adjusted: ', ''), '', '', '', '', interval(r.coefficient, r.ci_low, r.ci_high, 2)])
    latex_table(TABLES / 'tableS_mealslot.tex', ['Set', 'Meal slot / adjusted term', 'Meals', 'Mean onset glucose (mg/dL)', 'Mean iAUC', 'Median carbohydrate (g)', 'Within-patient coefficient (95% CI)'], rows, [1.7, 3.0, 1.0, 1.8, 1.4, 1.7, 2.6])
    extra = pd.read_csv(posthoc / 'extra_differences.csv')
    rows = [[SETS[r.set], 'Unconstrained GlucoSL-KAN minus GlucoSL-MLP', interval(r.estimate, r.low, r.high, 2), int(r.patients)] for _, r in extra.iterrows()]
    latex_table(TABLES / 'tableS_extra.tex', ['Set', 'Comparison (patient-macro MAE)', 'Difference (95% CI)', 'Patients'], rows, [3.6, 4.6, 2.8, 1.2])
    comparators = posthoc / 'posthoc_comparators.csv'
    if comparators.exists():
        c = pd.read_csv(comparators)
        names = {**NAMES, 'xgboost_wide': 'XGBoost, wider grid', 'xgboost_monotone': 'XGBoost, monotone constraints', 'xgboost_monotone_wide': 'XGBoost, monotone constraints, wider grid',
                 'gam_linear': 'Constrained GAM (linear scale)', 'gam_log': 'Constrained GAM (log scale, multiplicative)', 'gam_logistic': 'Constrained logistic GAM'}
        rows = []
        for set_ in ['shanghai_B', 'cgmacros_diabetes_range', 'cgmacros_all']:
            for task in ['regression', 'binary']:
                part = c[(c.set == set_) & (c.task == task)]
                if not len(part):
                    continue
                rows.append([f'\\textit{{{SETS[set_]}: {"patient-macro MAE" if task == "regression" else "AUROC"}}}', '', ''])
                for _, r in part.iterrows():
                    digits = 3 if task == 'binary' else 1
                    rows.append([names[r.model], interval(r.estimate, r.low, r.high, digits), '—' if r.model == 'glucosl_kan' else interval(r.difference_from_glucosl_kan, r.difference_low, r.difference_high, 3 if task == 'binary' else 2)])
        latex_table(TABLES / 'tableS_posthoc_comparators.tex', ['Set, task and model', 'Estimate (95% CI)', 'GlucoSL-KAN minus model (95% CI)'], rows, [6.0, 3.0, 3.2])
    trend = pd.read_csv(posthoc / 'calendar_trend.csv')
    rows = [[r.half_year, int(r.patients), r.sets, number(r.mean_iauc), number(r.event_rate, 2), number(r.hba1c), number(100 * r.insulin, 0), number(r.carbohydrate)] for _, r in trend.iterrows()]
    latex_table(TABLES / 'tableS_calendar.tex', ['Half-year of first recording', 'Patients', 'Set', 'Mean iAUC', 'Share of peaks ≥180', 'Median HbA1c (%)', 'Insulin use (%)', 'Median carbohydrate (g)'], rows, [2.2, 1.1, 0.8, 1.3, 1.6, 1.6, 1.4, 1.7])
    interpretability = RESULTS / 'interpretability'
    importance = pd.concat([pd.read_csv(p) for p in [interpretability / 'permutation_importance_torch_process.csv', interpretability / 'permutation_importance_xgboost.csv'] if p.exists()], ignore_index=True)
    if len(importance):
        pivot = importance.pivot(index='input', columns='model', values='increase_macro_mae')
        order = [m for m in NAMES if m in pivot.columns]
        rows = [[i] + [number(pivot.loc[i, m], 2) for m in order] for i in ['age', 'sex', 'bmi', 'hba1c_pct', 'fasting_glucose_mgdl', 'carbohydrate_g', 'protein_g', 'fat_g', 'fiber_g', 'hour_sin', 'hour_cos'] if i in pivot.index]
        latex_table(TABLES / 'tableS_permutation.tex', ['Input'] + [NAMES[m] for m in order], rows, [2.4] + [round(9.1 / len(order), 2)] * len(order))
    shap_path = interpretability / 'xgboost_shap_summary.csv'
    if shap_path.exists():
        shap = pd.read_csv(shap_path).sort_values('mean_abs_shap', ascending=False)
        latex_table(TABLES / 'tableS_shap.tex', ['Input', 'Mean |SHAP| (mg/dL·h)', 'Mean SHAP'], [[r.input, number(r.mean_abs_shap, 2), number(r.mean_shap, 2)] for _, r in shap.iterrows()], [3.0, 2.6, 2.0])
    ranges = interpretability / 'sensitivity_load_ranges.csv'
    if ranges.exists():
        r = pd.read_csv(ranges)
        latex_table(TABLES / 'tableS_ranges.tex', ['Quantity', 'Set', '5th percentile', 'Median', '95th percentile'], [[x.quantity.replace('_', ' '), x.set, number(x.p5, 2), number(x['median'], 2), number(x.p95, 2)] for _, x in r.iterrows()], [2.8, 3.4, 1.8, 1.8, 1.8])


def figure_forest(differences):
    panels = [('shanghai_B', 'regression'), ('cgmacros_diabetes_range', 'regression'), ('cgmacros_all', 'regression'), ('shanghai_B', 'binary'), ('cgmacros_diabetes_range', 'binary'), ('cgmacros_all', 'binary')]
    order = ['xgboost', 'tabpfn', 'linear', 'constant', 'glucosl_mlp', 'single_kan', 'unconstrained_kan', 'glucosl_kan_us']
    titles = {'shanghai_B': 'ShanghaiT2DM set B', 'cgmacros_diabetes_range': 'CGMacros, HbA1c ≥6.5%', 'cgmacros_all': 'CGMacros, all'}
    figure, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey=True)
    for index, (axis, (set_, task)) in enumerate(zip(axes.flat, panels)):
        panel_color = BLUE if set_ == 'shanghai_B' else RED
        y = np.arange(len(order))[::-1]
        low_limit = -14 if task == 'regression' else -0.2
        for m in order:
            d = difference(differences, set_, task, m)
            if d is None:
                continue
            position = y[order.index(m)]
            excludes = d.low > 0 or d.high < 0
            if d.low < low_limit:  # off scale: draw what fits and annotate the estimate
                axis.annotate('', xy=(low_limit, position), xytext=(low_limit + 1.5, position), arrowprops=dict(arrowstyle='->', color=panel_color, linewidth=0.8))
                axis.text(low_limit + 1.7, position + 0.25, f'{d.estimate:.1f} ({d.low:.1f} to {d.high:.1f})', fontsize=6, color=DARK, va='bottom')
                continue
            axis.errorbar(d.estimate, position, xerr=[[d.estimate - d.low], [d.high - d.estimate]], fmt='o', color=panel_color, markerfacecolor=panel_color if excludes else 'white', capsize=2, markersize=4, linewidth=0.9)
        axis.axvline(0, color=GREY, linewidth=0.7, linestyle=':')
        axis.set_yticks(y); axis.set_yticklabels([NAMES[m].replace(' (ablation', '\n(ablation').replace(', US-trained', ',\nUS-trained') for m in order])
        axis.set_title(f'{titles[set_]}: {"MAE" if task == "regression" else "AUROC"}')
        axis.set_xlabel('GlucoSL-KAN minus comparator' + (' (mg/dL·h)' if task == 'regression' else ''))
        axis.set_xlim(low_limit, 6 if task == 'regression' else 0.42)
        letter(axis, 'abcdef'[index])
    figure.tight_layout(w_pad=1.2)
    figure.savefig(FIGURES / 'Figure_4.pdf'); figure.savefig(FIGURES / 'Figure_4.png', dpi=300); plt.close(figure)


def figure_calibration(predictions, draws=200):
    # Flexible calibration curves (lowess of observed on predicted, meal level) with patient-cluster bootstrap bands,
    # the diagonal of perfect calibration, and the distribution of the predictions below each panel.
    from statsmodels.nonparametric.smoothers_lowess import lowess
    figure = plt.figure(figsize=(11, 4.4))
    grid = figure.add_gridspec(2, 3, height_ratios=[4, 1], hspace=0.08, wspace=0.3)
    sets = [('shanghai_B', 'ShanghaiT2DM set B'), ('cgmacros_diabetes_range', 'CGMacros, HbA1c ≥6.5%'), ('cgmacros_all', 'CGMacros, all')]
    generator = np.random.default_rng(20261021)
    for index, (set_, label) in enumerate(sets):
        axis = figure.add_subplot(grid[0, index]); hist = figure.add_subplot(grid[1, index], sharex=axis)
        frame = predictions[(predictions.set == set_) & (predictions.task == 'regression')].reset_index(drop=True)
        models = [('glucosl_kan', BLUE), ('xgboost', GREY)] + ([('glucosl_kan_us', RED)] if set_ == 'shanghai_B' else [])
        limit = float(max(frame[[m for m, _ in models]].quantile(.99).max(), frame.observed.quantile(.9)))
        groups = frame.patient_id.to_numpy(); patients = np.unique(groups); rows_of = {p: np.flatnonzero(groups == p) for p in patients}
        for model, color in models:
            x, y = frame[model].to_numpy(float), frame.observed.to_numpy(float)
            xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 60)
            curve = lowess(y, x, frac=0.6, it=0, xvals=xs)
            band = []
            for _ in range(draws):
                chosen = np.concatenate([rows_of[p] for p in generator.choice(patients, len(patients), replace=True)])
                band.append(lowess(y[chosen], x[chosen], frac=0.6, it=0, xvals=xs))
            band = np.array(band)
            axis.fill_between(xs, np.nanpercentile(band, 2.5, axis=0), np.nanpercentile(band, 97.5, axis=0), color=color, alpha=0.15, linewidth=0)
            axis.plot(xs, curve, color=color, linewidth=1.6, label=NAMES[model])
            hist.hist(x, bins=np.linspace(0, limit, 30), histtype='stepfilled', color=color, alpha=0.25, linewidth=0)
            hist.hist(x, bins=np.linspace(0, limit, 30), histtype='step', color=color, linewidth=0.8)
        top = max(limit, 1.02 * max(line.get_ydata().max() for line in axis.get_lines()))
        axis.plot([0, top], [0, top], color=DARK, linewidth=0.7, linestyle=':')
        axis.text(limit * 0.98, limit * 0.98, 'perfect calibration', color=GREY, fontsize=6.5, ha='right', va='bottom', rotation=45, rotation_mode='anchor')
        axis.set_xlim(0, limit); axis.set_ylim(0, top); axis.tick_params(labelbottom=False)
        axis.set_ylabel('Observed 2-h iAUC (mg/dL·h)'); axis.set_title(label)
        hist.set_xlabel('Predicted 2-h iAUC (mg/dL·h)'); hist.set_yticks([]); hist.spines['left'].set_visible(False)
        letter(axis, 'abc'[index])
        if index == 0:
            axis.legend(frameon=False, loc='upper left')
    figure.savefig(FIGURES / 'Figure_5.pdf', bbox_inches='tight'); figure.savefig(FIGURES / 'Figure_5.png', dpi=300, bbox_inches='tight'); plt.close(figure)


def figure_schematic():
    figure, axis = plt.subplots(figsize=(10, 5.6)); axis.axis('off')
    def box(x, y, w, h, text, color='#f0f4fa', edge='#4a6fa5', size=8, bold=False):
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.02', facecolor=color, edgecolor=edge, linewidth=1))
        axis.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, fontweight='bold' if bold else 'normal', wrap=True)
    def arrow(x0, y0, x1, y1):
        axis.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color='#4a6fa5', linewidth=1.2))
    # Panel a: model.
    axis.text(0.01, 0.97, 'a  GlucoSL-KAN', fontsize=10, fontweight='bold', va='top')
    box(0.02, 0.66, 0.2, 0.2, 'Person\nage, sex, BMI,\nHbA1c, fasting glucose')
    box(0.02, 0.36, 0.2, 0.2, 'Meal\ncarbohydrate, protein,\nfat, fiber, clock time')
    box(0.3, 0.66, 0.22, 0.2, 'KAN$_s$: one layer of\nlearnable 1-D functions\n→ softplus', color='#fff4e0', edge='#d08c00')
    box(0.3, 0.36, 0.22, 0.2, 'KAN$_l$: one layer,\ncarbohydrate ↑, fiber ↓\n(guideline constraints)\n→ softplus', color='#fff4e0', edge='#d08c00')
    box(0.6, 0.51, 0.16, 0.2, 'sensitivity\n×\nload', color='#e8f5e9', edge='#2e7d32', bold=True)
    box(0.82, 0.66, 0.16, 0.16, '2-h iAUC\n(regression)')
    box(0.82, 0.4, 0.16, 0.16, 'P(peak ≥180 mg/dL)\n= σ(log s + log l + β₀)')
    for y in (0.76, 0.46):
        arrow(0.22, y, 0.3, y)
    arrow(0.52, 0.76, 0.6, 0.66); arrow(0.52, 0.46, 0.6, 0.56); arrow(0.76, 0.64, 0.82, 0.73); arrow(0.76, 0.58, 0.82, 0.49)
    # Panel b: study design.
    axis.text(0.01, 0.27, 'b  Study design', fontsize=10, fontweight='bold', va='top')
    box(0.02, 0.04, 0.26, 0.17, 'ShanghaiT2DM set A\n67 patients, 1,952 meals\nfirst recording ≤ 2021-06-29\n→ development (logged); protocol and\nmodels frozen before validation', color='#e3f2fd')
    box(0.36, 0.04, 0.26, 0.17, 'ShanghaiT2DM set B\n33 patients, 670 meals\nfirst recording ≥ 2021-07-03\n→ temporal validation (primary)', color='#fce4ec')
    box(0.7, 0.04, 0.28, 0.17, 'CGMacros (US)\n14 participants with HbA1c ≥6.5% (353 meals);\nall 44 (1,147 meals)\n→ cross-country validation; US-trained model', color='#e8f5e9')
    arrow(0.28, 0.125, 0.36, 0.125); arrow(0.62, 0.125, 0.7, 0.125)
    axis.set_xlim(0, 1); axis.set_ylim(0, 1)
    figure.savefig(FIGURES / 'Figure_1.pdf', bbox_inches='tight'); figure.savefig(FIGURES / 'Figure_1.png', dpi=300, bbox_inches='tight'); plt.close(figure)


def figure_schematic_v2():
    """Render the approved red/blue architecture and nested-cohort layout."""
    import runpy
    renderer = runpy.run_path(str(PAPER / 'scripts/render_figure1.py'), run_name='figure1_renderer')
    renderer['build'](FIGURES)


def figure_shap():
    """Figure S1 in the paper's own style: one strip of per-meal SHAP values per input, ordered by mean |SHAP|,
    coloured by the percentile of the input value, with the mean |SHAP| printed at the right margin."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    sys.path.insert(0, str(STAGE))
    from src.glucosl_data import shanghai_validation
    values = pd.read_csv(RESULTS / 'interpretability/xgboost_shap_values.csv')
    frame = shanghai_validation()
    inputs = ['age', 'sex', 'bmi', 'hba1c_pct', 'fasting_glucose_mgdl', 'carbohydrate_g', 'protein_g', 'fat_g', 'fiber_g', 'hour_sin', 'hour_cos']
    person = {'age', 'sex', 'bmi', 'hba1c_pct', 'fasting_glucose_mgdl'}
    labels = {'age': 'Age', 'sex': 'Sex', 'bmi': 'BMI', 'hba1c_pct': 'HbA1c', 'fasting_glucose_mgdl': 'Fasting glucose', 'carbohydrate_g': 'Carbohydrate', 'protein_g': 'Protein', 'fat_g': 'Fat', 'fiber_g': 'Fiber', 'hour_sin': 'Clock time (sin)', 'hour_cos': 'Clock time (cos)'}
    importance = values[inputs].abs().mean().sort_values(ascending=True)
    order = importance.index.tolist()
    cmap = LinearSegmentedColormap.from_list('paper', [BLUE, '#d9d9d9', RED])
    figure, axis = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(order))
    generator = np.random.default_rng(20261021)
    for position, column in zip(y, order):
        rank = frame[column].rank(pct=True).to_numpy()  # percentile colouring so that skewed inputs still span the scale
        axis.scatter(values[column], position + generator.uniform(-0.3, 0.3, len(rank)), c=rank, cmap=cmap, norm=Normalize(0, 1), s=6, alpha=0.7, linewidths=0, rasterized=True)
        axis.text(1.0, position, f'{importance[column]:.1f}', transform=axis.get_yaxis_transform(), ha='left', va='center', fontsize=7.5, color=DARK, clip_on=False)
    axis.text(1.0, len(order) - 0.35, 'mean |SHAP|', transform=axis.get_yaxis_transform(), ha='left', va='bottom', fontsize=7, color=GREY, style='italic', clip_on=False)
    axis.axvline(0, color=DARK, linewidth=0.7, linestyle=':')
    axis.set_yticks(y); axis.set_yticklabels([labels[c] for c in order])
    for tick, column in zip(axis.get_yticklabels(), order):
        tick.set_color(BLUE if column in person else ORANGE)
    axis.set_ylim(-0.6, len(order) - 0.4)
    axis.set_xlabel('SHAP value: contribution to the predicted 2-h iAUC (mg/dL·h)')
    axis.spines['left'].set_visible(False); axis.tick_params(axis='y', length=0)
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=axis, pad=0.1, aspect=30, shrink=0.8)
    colorbar.set_ticks([0, 1]); colorbar.set_ticklabels(['low', 'high']); colorbar.set_label('Input value (percentile)')
    figure.tight_layout()
    figure.savefig(FIGURES / 'Figure_S1.pdf', dpi=300); figure.savefig(FIGURES / 'Figure_S1.png', dpi=300); plt.close(figure)


def figure_sensitivity_scatter():
    path = RESULTS / 'interpretability/patient_sensitivity.csv'
    if not path.exists():
        return
    table = pd.read_csv(path)
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    colors = {'shanghai_A': BLUE, 'shanghai_B': TEAL, 'cgmacros_all': RED}
    labels = {'shanghai_A': 'ShanghaiT2DM A', 'shanghai_B': 'ShanghaiT2DM B', 'cgmacros_all': 'CGMacros (all)'}
    for index, (axis, column, label) in enumerate([(axes[0], 'hba1c_pct', 'HbA1c (%)'), (axes[1], 'fasting_glucose_mgdl', 'Fasting glucose (mg/dL)')]):
        for set_, part in table.groupby('set'):
            axis.scatter(part[column], part.relative_sensitivity, s=12, alpha=0.7, color=colors[set_], label=labels[set_], linewidths=0)
        axis.set_xlabel(label); axis.set_ylabel('Relative personal sensitivity')
        letter(axis, 'ab'[index])
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout(); figure.savefig(FIGURES / 'Figure_S2.pdf'); figure.savefig(FIGURES / 'Figure_S2.png', dpi=300); plt.close(figure)


def main():
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS / 'evaluation/metrics.csv'); differences = pd.read_csv(RESULTS / 'evaluation/differences.csv')
    predictions = pd.read_csv(RESULTS / 'evaluation/predictions.csv')
    characteristics = pd.read_csv(RESULTS / 'posthoc/set_characteristics.csv', index_col=0)
    log = pd.read_csv(RESULTS / 'development_log.csv'); tuning = pd.read_csv(RESULTS / 'final_fit/tuning.csv'); receipt = json.loads((RESULTS / 'final_fit/fit_receipt.json').read_text())
    table_characteristics(characteristics); table_regression_b(metrics, differences); table_binary_b(metrics, differences); table_transport(metrics, differences)
    supplementary_tables(metrics, differences, log, tuning, receipt); posthoc_tables()
    figure_schematic_v2(); figure_forest(differences); figure_calibration(predictions); figure_sensitivity_scatter(); figure_shap()
    for source, target in [('figure_learned_functions', 'Figure_2'), ('figure_heatmap', 'Figure_3')]:
        for suffix in ['.pdf', '.png']:
            if (RESULTS / f'interpretability/{source}{suffix}').exists():
                shutil.copy(RESULTS / f'interpretability/{source}{suffix}', FIGURES / f'{target}{suffix}')
    print(sorted(p.name for p in TABLES.iterdir()), sorted(p.name for p in FIGURES.iterdir()))


if __name__ == '__main__':
    main()
