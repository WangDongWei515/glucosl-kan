"""Post hoc analyses after the one-time evaluation (protocol G1, step 7). Everything here is descriptive and labelled post hoc.

1. Characteristics of set A (training), set B (temporal validation) and the CGMacros diabetes-range participants.
2. Decomposition of ranking in set B into between-patient and within-patient components, regression and binary.
3. HbA1c associations with patient-level responses in each set.
4. Performance in set B by insulin use (medication was not a model input).
5. Calendar trend of responses across recording start dates, and the recruiting-period note.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


STAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STAGE))
from src.glucosl_data import cgmacros
from src.shanghai_t2dm import drug_classes

OUTPUT = STAGE / 'results/glucosl/posthoc'
EVALUATION = STAGE / 'results/glucosl/evaluation'
CLASSES = ['metformin', 'alpha_glucosidase_inhibitor', 'sulfonylurea', 'glinide', 'dpp4_inhibitor', 'sglt2_inhibitor', 'thiazolidinedione', 'glp1_receptor_agonist']


def iqr(series):
    values = series.dropna()
    return f'{values.median():.1f} ({values.quantile(.25):.1f}–{values.quantile(.75):.1f})' if len(values) else 'NA'


def load_shanghai():
    cohort = pd.read_csv(STAGE / 'data/processed/shanghai_t2dm/external_cohort.csv', parse_dates=['time'])
    cohort = cohort.loc[cohort.cohort_all_records].copy()
    split = pd.read_csv(STAGE / 'data/processed/glucosl/split.csv', parse_dates=['first_recording'])
    cohort = cohort.merge(split[['patient_id', 'set', 'first_recording']], on='patient_id')
    records = pd.read_csv(STAGE / 'data/interim/shanghai_t2dm/records.csv', parse_dates=['start', 'end'])
    for name in CLASSES:
        records[name] = records.apply(lambda row: name in drug_classes(row.agents) | drug_classes(row.summary_classes) | set(str(row.noninsulin_classes).split(';')), axis=1)
    cohort = cohort.merge(records[['record_id', 'diabetes_years', 'start', 'span_days'] + CLASSES], on='record_id', how='left')
    cohort['iauc'] = cohort.iauc_15min
    return cohort


def characteristics(cohort, us):
    rows = []
    frames = {'Shanghai A (training)': cohort[cohort.set.eq('A')], 'Shanghai B (temporal validation)': cohort[cohort.set.eq('B')]}
    for label, frame in frames.items():
        patients = frame.sort_values('time').groupby('patient_id').first()
        per_patient_insulin = frame.groupby('patient_id').any_insulin.max()
        rows.append({
            'set': label, 'patients': len(patients), 'recordings': frame.record_id.nunique(), 'meals': len(frame),
            'meals per patient': iqr(frame.groupby('patient_id').size()),
            'first recording': f'{frame.first_recording.min().date()} to {frame.first_recording.max().date()}',
            'age': iqr(patients.age), 'female %': f'{100 * (patients.sex == 0).mean():.0f}', 'BMI': iqr(patients.bmi),
            'HbA1c %': iqr(patients.hba1c_pct), 'HbA1c missing': int(patients.hba1c_pct.isna().sum()),
            'fasting glucose': iqr(patients.fasting_glucose_mgdl), 'diabetes years': iqr(patients.diabetes_years),
            'insulin use % (patients)': f'{100 * per_patient_insulin.mean():.0f}',
            **{f'{name} %': f'{100 * frame.groupby("patient_id")[name].max().mean():.0f}' for name in CLASSES},
            'carbohydrate g': iqr(frame.primary_carbohydrate_g), 'protein g': iqr(frame.primary_protein_g), 'fat g': iqr(frame.primary_fat_g), 'fiber g': iqr(frame.primary_fiber_g),
            'energy kcal': iqr(frame.primary_energy_kcal), 'specific mapping share': iqr(frame.specific_weight_share),
            'onset glucose': iqr(frame.onset_mgdl), 'iAUC median': iqr(frame.iauc), 'iAUC mean': f'{frame.iauc.mean():.1f}', 'peak >=180 %': f'{100 * frame.peak_ge_180.mean():.0f}',
            'breakfast/lunch/dinner %': '/'.join(f'{100 * (frame.meal_type == m).mean():.0f}' for m in ['breakfast', 'lunch', 'dinner']),
        })
    dm = us.loc[us.diabetes_range]
    patients = dm.groupby('patient_id').first()
    rows.append({'set': 'CGMacros, HbA1c >=6.5%', 'patients': len(patients), 'recordings': len(patients), 'meals': len(dm), 'meals per patient': iqr(dm.groupby('patient_id').size()),
                 'age': iqr(patients.age), 'female %': f'{100 * (patients.sex == 0).mean():.0f}', 'BMI': iqr(patients.bmi), 'HbA1c %': iqr(patients.hba1c_pct), 'HbA1c missing': 0,
                 'fasting glucose': iqr(patients.fasting_glucose_mgdl), 'carbohydrate g': iqr(dm.carbohydrate_g), 'protein g': iqr(dm.protein_g), 'fat g': iqr(dm.fat_g), 'fiber g': iqr(dm.fiber_g),
                 'iAUC median': iqr(dm.iauc), 'iAUC mean': f'{dm.iauc.mean():.1f}', 'peak >=180 %': f'{100 * dm.peak_ge_180.mean():.0f}'})
    return pd.DataFrame(rows).set_index('set').T


def ranking_decomposition(predictions, cohort_b):
    rows = []
    for set_, task, models in [('shanghai_B', 'regression', ['glucosl_kan', 'xgboost', 'unconstrained_kan', 'glucosl_kan_us']), ('shanghai_B', 'binary', ['glucosl_kan', 'xgboost', 'unconstrained_kan', 'glucosl_kan_us']),
                               ('cgmacros_diabetes_range', 'regression', ['glucosl_kan', 'xgboost', 'unconstrained_kan']), ('cgmacros_diabetes_range', 'binary', ['glucosl_kan', 'xgboost', 'unconstrained_kan'])]:
        frame = predictions[(predictions.set == set_) & (predictions.task == task)]
        for model in models:
            grouped = frame.groupby('patient_id')
            between = spearmanr(grouped[model].mean(), grouped['observed'].mean()).correlation
            within = []
            for _, rows_ in grouped:
                if len(rows_) < 8 or rows_.observed.nunique() < 2 or rows_[model].nunique() < 2:
                    continue
                within.append(roc_auc_score(rows_.observed, rows_[model]) if task == 'binary' else spearmanr(rows_[model], rows_.observed).correlation)
            pooled = roc_auc_score(frame.observed, frame[model]) if task == 'binary' else spearmanr(frame[model], frame.observed).correlation
            rows.append({'set': set_, 'task': task, 'model': model, 'pooled_unweighted': pooled, 'between_patient_spearman': between,
                         'within_patient_median': float(np.median(within)) if within else np.nan, 'patients_within': len(within)})
    return pd.DataFrame(rows)


def hba1c_associations(cohort, us, interpretability):
    rows = []
    for label, frame in [('Shanghai A', cohort[cohort.set.eq('A')]), ('Shanghai B', cohort[cohort.set.eq('B')]), ('CGMacros, HbA1c >=6.5%', us[us.diabetes_range]), ('CGMacros, all', us)]:
        patients = frame.groupby('patient_id').agg(hba1c=('hba1c_pct', 'first'), fpg=('fasting_glucose_mgdl', 'first'), mean_iauc=('iauc', 'mean'), event_rate=('peak_ge_180', 'mean')).dropna(subset=['hba1c'])
        rows.append({'set': label, 'patients': len(patients), 'spearman_hba1c_mean_iauc': spearmanr(patients.hba1c, patients.mean_iauc).correlation,
                     'spearman_hba1c_event_rate': spearmanr(patients.hba1c, patients.event_rate).correlation,
                     'spearman_fpg_mean_iauc': spearmanr(patients.fpg.fillna(patients.fpg.median()), patients.mean_iauc).correlation})
    table = pd.DataFrame(rows)
    if interpretability.exists():
        pd_table = pd.read_csv(interpretability)
        for model in ['Shanghai-trained', 'US-trained']:
            curve = pd_table[(pd_table.model == model) & (pd_table.part == 'sensitivity') & (pd_table.input == 'hba1c_pct')]
            table[f'{model}: relative sensitivity at HbA1c p95 vs p5'] = round(curve.estimate.iloc[-1] / curve.estimate.iloc[0], 3)
    return table


def insulin_strata(predictions, cohort_b):
    insulin = cohort_b.groupby('record_id').any_insulin.first()
    rows = []
    for task, metric in [('regression', 'macro_mae'), ('binary', 'auroc')]:
        frame = predictions[(predictions.set == 'shanghai_B') & (predictions.task == task)].copy()
        frame['insulin'] = frame.record_id.map(insulin).astype(bool)
        for stratum, part in frame.groupby('insulin'):
            for model in ['glucosl_kan', 'xgboost', 'constant', 'glucosl_kan_us']:
                if task == 'regression':
                    value = part.assign(error=(part[model] - part.observed).abs()).groupby('patient_id').error.mean().mean()
                else:
                    weights = 1 / part.patient_id.map(part.patient_id.value_counts())
                    value = roc_auc_score(part.observed, part[model], sample_weight=weights) if part.observed.nunique() == 2 and part[model].nunique() > 1 else np.nan
                rows.append({'task': task, 'metric': metric, 'insulin_use': stratum, 'patients': part.patient_id.nunique(), 'meals': len(part), 'model': model, 'value': value})
    return pd.DataFrame(rows)


def calendar_trend(cohort):
    patients = cohort.groupby('patient_id').agg(first_recording=('first_recording', 'first'), set=('set', 'first'), mean_iauc=('iauc', 'mean'), event_rate=('peak_ge_180', 'mean'),
                                                hba1c=('hba1c_pct', 'first'), insulin=('any_insulin', 'max'), carbohydrate=('primary_carbohydrate_g', 'median'))
    trend = {'spearman_date_mean_iauc': spearmanr(patients.first_recording.astype('int64'), patients.mean_iauc).correlation,
             'spearman_date_event_rate': spearmanr(patients.first_recording.astype('int64'), patients.event_rate).correlation}
    patients['half_year'] = patients.first_recording.dt.year.astype(str) + 'H' + ((patients.first_recording.dt.month > 6) + 1).astype(str)
    by_half = patients.groupby('half_year').agg(patients=('set', 'size'), sets=('set', lambda s: '/'.join(sorted(set(s)))), mean_iauc=('mean_iauc', 'mean'), event_rate=('event_rate', 'mean'), hba1c=('hba1c', 'median'), insulin=('insulin', 'mean'), carbohydrate=('carbohydrate', 'median')).round(2)
    return trend, by_half


def onset_table(predictions, cohort, us):
    # Meals whose onset glucose is already >= 180 mg/dL have the binary outcome by definition.
    rows = []
    frames = {'Shanghai A': cohort[cohort.set.eq('A')], 'Shanghai B': cohort[cohort.set.eq('B')]}
    for label, frame in frames.items():
        rows.append({'set': label, 'meals': len(frame), 'onset_ge_180_share': frame.onset_mgdl.ge(180).mean(), 'events': int(frame.peak_ge_180.sum()),
                     'events_with_onset_ge_180_share': frame.loc[frame.peak_ge_180.eq(1), 'onset_mgdl'].ge(180).mean()})
    cg = pd.read_csv(STAGE / 'data/processed/cgmacros/primary_cohort.csv')[['meal_id', 'libre_baseline']].rename(columns={'meal_id': 'record_id', 'libre_baseline': 'onset_mgdl'})
    for label, frame in [('CGMacros, HbA1c >=6.5%', us[us.diabetes_range]), ('CGMacros, all', us)]:
        frame = frame.merge(cg, on='record_id')
        rows.append({'set': label, 'meals': len(frame), 'onset_ge_180_share': frame.onset_mgdl.ge(180).mean(), 'events': int(frame.peak_ge_180.sum()),
                     'events_with_onset_ge_180_share': frame.loc[frame.peak_ge_180.eq(1), 'onset_mgdl'].ge(180).mean()})
    table = pd.DataFrame(rows)
    # AUROC in set B restricted to meals with onset < 180 (post hoc), patient-weighted.
    frame = predictions[(predictions.set == 'shanghai_B') & (predictions.task == 'binary')].copy()
    frame['order'] = frame.groupby('record_id').cumcount()
    frame = frame.merge(cohort_b_onset(cohort), on=['record_id', 'order'], how='left', validate='one_to_one')
    assert frame.onset_mgdl.notna().all()
    restricted = frame[frame.onset_mgdl < 180]
    weights = 1 / restricted.patient_id.map(restricted.patient_id.value_counts())
    for model in ['glucosl_kan', 'xgboost', 'glucosl_kan_us']:
        table[f'auroc_B_onset_lt_180_{model}'] = roc_auc_score(restricted.observed, restricted[model], sample_weight=weights)
    table['B_meals_onset_lt_180'] = len(restricted)
    table['B_event_rate_onset_lt_180'] = restricted.observed.mean()
    return table


def cohort_b_onset(cohort):
    # predictions.csv identifies a meal by record_id only; the evaluation kept the cohort file's row order, so onset is
    # matched by meal order within the record (the cohort file is sorted by record and time).
    b = cohort[cohort.set.eq('B')]
    assert b.groupby('record_id').time.apply(lambda t: t.is_monotonic_increasing).all()
    return b[['record_id', 'onset_mgdl']].assign(order=b.groupby('record_id').cumcount())


def meal_slot_table(cohort):
    import statsmodels.formula.api as smf
    rows = []
    for label, frame in [('Shanghai A', cohort[cohort.set.eq('A')]), ('Shanghai B', cohort[cohort.set.eq('B')])]:
        for meal_type in ['breakfast', 'lunch', 'dinner']:
            part = frame[frame.meal_type.eq(meal_type)]
            rows.append({'set': label, 'meal_type': meal_type, 'meals': len(part), 'mean_onset': part.onset_mgdl.mean(), 'mean_iauc': part.iauc.mean(), 'median_carbohydrate': part.primary_carbohydrate_g.median()})
        model = smf.ols('iauc ~ C(meal_type, Treatment("dinner")) + onset_mgdl + primary_carbohydrate_g + C(patient_id)', data=frame).fit(cov_type='cluster', cov_kwds={'groups': frame.patient_id})
        for term, name in [('C(meal_type, Treatment("dinner"))[T.breakfast]', 'breakfast vs dinner'), ('C(meal_type, Treatment("dinner"))[T.lunch]', 'lunch vs dinner'), ('onset_mgdl', 'onset per mg/dL'), ('primary_carbohydrate_g', 'carbohydrate per g')]:
            rows.append({'set': label, 'meal_type': f'adjusted: {name}', 'coefficient': model.params[term], 'ci_low': model.conf_int().loc[term, 0], 'ci_high': model.conf_int().loc[term, 1]})
    return pd.DataFrame(rows)


def extra_differences(predictions, draws=2000, seed=20261021):
    # Paired patient-macro MAE difference, unconstrained GlucoSL-KAN minus GlucoSL-MLP (post hoc; the evaluation only computed differences from GlucoSL-KAN).
    generator = np.random.default_rng(seed)
    rows = []
    for set_ in ['shanghai_B', 'cgmacros_diabetes_range', 'cgmacros_all']:
        frame = predictions[(predictions.set == set_) & (predictions.task == 'regression')]
        per_patient = frame.assign(a=(frame.unconstrained_kan - frame.observed).abs(), b=(frame.glucosl_mlp - frame.observed).abs()).groupby('patient_id')[['a', 'b']].mean()
        difference = (per_patient.a - per_patient.b).to_numpy()
        boot = [generator.choice(difference, len(difference)).mean() for _ in range(draws)]
        rows.append({'set': set_, 'comparison': 'unconstrained_kan minus glucosl_mlp', 'estimate': difference.mean(), 'low': np.percentile(boot, 2.5), 'high': np.percentile(boot, 97.5), 'patients': len(difference)})
    return pd.DataFrame(rows)


def model_checks(cohort_b):
    import joblib
    import torch
    from src.glucosl_data import MEAL, PERSON
    model = joblib.load(STAGE / 'results/glucosl/models/regression/glucosl_kan.joblib')
    parameters = sum(p.numel() for p in model.members[0].network.parameters()) + 1
    frame = cohort_b.rename(columns={'primary_carbohydrate_g': 'carbohydrate_g', 'primary_protein_g': 'protein_g', 'primary_fat_g': 'fat_g', 'primary_fiber_g': 'fiber_g'})[PERSON + MEAL].reset_index(drop=True)
    checks = {}
    for column, sign in [('carbohydrate_g', 1), ('fiber_g', -1)]:
        values = np.linspace(frame[column].min(), frame[column].max(), 60)
        predictions = np.stack([model.predict(frame.assign(**{column: value})) for value in values])  # (60, meals)
        steps = np.diff(predictions, axis=0) * sign
        checks[column] = {'direction': 'non-decreasing' if sign == 1 else 'non-increasing', 'meals': len(frame), 'grid_points': len(values), 'violations': int((steps < -1e-6).sum()), 'min_step': float(steps.min())}
    return {'parameters_per_member': int(parameters), 'members': len(model.members), 'monotonicity_set_B': checks}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cohort, us = load_shanghai(), cgmacros()
    predictions = pd.read_csv(EVALUATION / 'predictions.csv')
    cohort_b = cohort[cohort.set.eq('B')]
    characteristics(cohort, us).to_csv(OUTPUT / 'set_characteristics.csv')
    ranking_decomposition(predictions, cohort_b).round(3).to_csv(OUTPUT / 'ranking_decomposition.csv', index=False)
    hba1c_associations(cohort, us, STAGE / 'results/glucosl/interpretability/partial_dependence.csv').round(3).to_csv(OUTPUT / 'hba1c_associations.csv', index=False)
    insulin_strata(predictions, cohort_b).round(3).to_csv(OUTPUT / 'insulin_strata.csv', index=False)
    onset_table(predictions, cohort, us).round(3).to_csv(OUTPUT / 'onset_180.csv', index=False)
    meal_slot_table(cohort).round(2).to_csv(OUTPUT / 'meal_slot.csv', index=False)
    extra_differences(predictions).round(3).to_csv(OUTPUT / 'extra_differences.csv', index=False)
    (OUTPUT / 'model_checks.json').write_text(json.dumps(model_checks(cohort_b), indent=2))
    trend, by_half = calendar_trend(cohort)
    by_half.to_csv(OUTPUT / 'calendar_trend.csv')
    note = {**{k: round(v, 3) for k, v in trend.items()},
            'recruiting_periods_in_data_descriptor': 'Shanghai East Hospital, September 2019 to March 2021; Shanghai Fourth People\'s Hospital, June 2021 to November 2021 (Zhao et al. 2023). Hospital is not recorded per patient.',
            'set_A_first_recordings': f'{cohort[cohort.set.eq("A")].first_recording.min().date()} to {cohort[cohort.set.eq("A")].first_recording.max().date()}',
            'set_B_first_recordings': f'{cohort_b.first_recording.min().date()} to {cohort_b.first_recording.max().date()}',
            'note': 'The split date (2021-06-29) falls at the start of the second recruiting period, so set B is likely to come mostly from the second hospital; this cannot be verified from the released data.'}
    (OUTPUT / 'calendar_note.json').write_text(json.dumps(note, indent=2))
    print(characteristics(cohort, us).to_string()); print(ranking_decomposition(predictions, cohort_b).round(3).to_string(index=False))
    print(hba1c_associations(cohort, us, STAGE / 'results/glucosl/interpretability/partial_dependence.csv').round(3).to_string(index=False))
    print(insulin_strata(predictions, cohort_b).round(3).to_string(index=False)); print(by_half.to_string()); print(json.dumps(note, indent=2))


if __name__ == '__main__':
    main()
