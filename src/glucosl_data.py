from pathlib import Path

import pandas as pd


STAGE = Path(__file__).resolve().parents[1]
PERSON = ['age', 'sex', 'bmi', 'hba1c_pct', 'fasting_glucose_mgdl']
MEAL = ['carbohydrate_g', 'protein_g', 'fat_g', 'fiber_g', 'hour_sin', 'hour_cos']
NUTRIENTS = ['carbohydrate_g', 'protein_g', 'fat_g', 'fiber_g']


COLUMNS = ['patient_id', 'record_id', 'time'] + PERSON + MEAL + ['iauc', 'peak_ge_180']


def _shanghai(which, variant):
    cohort = pd.read_csv(STAGE / 'data/processed/shanghai_t2dm/external_cohort.csv', parse_dates=['time'])
    cohort = cohort.loc[cohort.cohort_all_records]
    split = pd.read_csv(STAGE / 'data/processed/glucosl/split.csv')
    patients = split.loc[split.set.eq(which), 'patient_id']
    cohort = cohort.loc[cohort.patient_id.isin(patients)].copy()
    cohort[NUTRIENTS] = cohort[[f'{variant}_{name}' for name in NUTRIENTS]].to_numpy()
    cohort = cohort.rename(columns={'iauc_15min': 'iauc'})
    cohort['peak_ge_180'] = cohort.peak_ge_180.astype(int)
    return cohort[COLUMNS + ['specific_weight_share']].reset_index(drop=True)


def shanghai_training(variant='primary'):
    # Development data: set A only. Set B rows are dropped here, before any modelling code sees them.
    cohort = _shanghai('A', variant)
    assert cohort.patient_id.nunique() == 67
    return cohort


def shanghai_validation(variant='primary'):
    # Temporal validation set B. Only the evaluation script may call this (protocol G1, Section 9).
    cohort = _shanghai('B', variant)
    assert cohort.patient_id.nunique() == 33
    return cohort


def cgmacros():
    # All 44 CGMacros participants with outcomes on the 15-min grid; `diabetes_range` marks HbA1c >= 6.5%.
    cohort = pd.read_csv(STAGE / 'data/processed/cgmacros/primary_cohort.csv', parse_dates=['meal_time'])
    outcomes = pd.read_csv(STAGE / 'data/processed/cgmacros/harmonized_outcomes_15min.csv')
    cohort = cohort.merge(outcomes, on='meal_id', validate='one_to_one')
    cohort = cohort.rename(columns={'subject': 'patient_id', 'meal_id': 'record_id', 'meal_time': 'time', 'iauc_15min': 'iauc'})
    cohort['diabetes_range'] = cohort.hba1c_range.eq('diabetes_range')
    cohort['peak_ge_180'] = cohort.peak_ge_180.astype(int)
    assert cohort.patient_id.nunique() == 44 and len(cohort) == 1147
    return cohort[COLUMNS + ['diabetes_range']].reset_index(drop=True)
