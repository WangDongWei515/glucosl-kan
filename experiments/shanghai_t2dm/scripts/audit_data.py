from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STAGE / 'src'))
from shanghai_t2dm import (CONVERSIONS, INSULIN_COLUMNS, RECORD_COLUMNS, SOURCE, SUMMARY_FIELDS, drug_classes, parse_intake,
                           read_record, read_summary, record_paths, unmatched_agent_terms)

INTERIM = STAGE / 'data/interim/shanghai_t2dm'
METADATA = STAGE / 'data/metadata'
OUTPUT = STAGE / 'results/shanghai_t2dm/reports'
WINDOW_OFFSETS = pd.to_timedelta(np.arange(0, 121, 15), unit='min')


def describe(series):
    values = pd.to_numeric(series, errors='coerce')
    statistics = {'mean': values.mean(), 'sd': values.std(), 'min': values.min(), 'median': values.median(), 'max': values.max()}
    return {'n': int(values.notna().sum()), 'missing': int(values.isna().sum()), **{key: None if pd.isna(value) else round(float(value), 4) for key, value in statistics.items()}}


def plain(value):
    return value.item() if hasattr(value, 'item') else str(value)


def main():
    for directory in [INTERIM, METADATA, OUTPUT]:
        directory.mkdir(parents=True, exist_ok=True)
    summary = read_summary()
    paths = record_paths()
    assert sorted(path.stem for path in paths) == sorted(summary.record_id)
    records, cgm, intake, items, medications, profiles = [], [], [], [], [], []
    for path in paths:
        record_id = path.stem
        frame = read_record(path)
        steps = frame.time.diff().dt.total_seconds().div(60)
        monitored = frame.loc[frame.cgm_mgdl.notna(), ['time', 'cgm_mgdl']]
        glucose = monitored.set_index('time').cgm_mgdl
        event_classes = set().union(*[drug_classes(text) for text in frame.oral_agents.dropna()]) if frame.oral_agents.notna().any() else set()
        profiles.append({
            'record_id': record_id, 'rows': len(frame), 'cgm_readings': len(monitored), 'start': frame.time.min(), 'end': frame.time.max(),
            'span_days': (frame.time.max() - frame.time.min()).total_seconds() / 86400,
            'duplicate_timestamps': int(frame.time.duplicated().sum()), 'backward_steps': int((steps < 0).sum()),
            'step_minutes': Counter(steps.dropna().round().astype(int).tolist()),
            'event_insulin': bool(frame[INSULIN_COLUMNS].notna().any().any()) or 'insulin' in event_classes,
            'event_classes': sorted(event_classes - {'insulin'}),
            'unmatched_event_terms': sorted({term for text in frame.oral_agents.dropna() for term in unmatched_agent_terms(text)}),
        })
        cgm.append(monitored.assign(record_id=record_id))
        meals = frame.loc[frame.diet_en.notna() | frame.diet_zh.notna()]
        for time, english, chinese in meals[['time', 'diet_en', 'diet_zh']].itertuples(index=False):
            parsed = parse_intake(english, chinese)
            window = glucose.reindex(time + WINDOW_OFFSETS)
            intake.append({'record_id': record_id, 'time': time, 'diet_en': english, 'diet_zh': chinese, 'lines': len(parsed),
                           'not_recorded': bool(parsed) and all(item['not_recorded'] for item in parsed),
                           'all_lines_parsed': bool(parsed) and all(item['parsed'] for item in parsed),
                           'on_cgm_row': bool(np.isfinite(glucose.get(time, np.nan))), 'complete_120min_window': bool(window.notna().all())})
            items.extend({'record_id': record_id, 'time': time, **item} for item in parsed)
        for column in INSULIN_COLUMNS + ['oral_agents']:
            for time, value in frame.loc[frame[column].notna(), ['time', column]].itertuples(index=False):
                medications.append({'record_id': record_id, 'time': time, 'field': column, 'value': value})
    profiles = pd.DataFrame(profiles)
    table = summary.merge(profiles, on='record_id', validate='one_to_one')
    table['summary_classes'] = table.agents.map(lambda text: sorted(drug_classes(text) - {'insulin'}))
    table['summary_insulin'] = table.agents.map(lambda text: 'insulin' in drug_classes(text))
    table['any_insulin'] = table.summary_insulin | table.event_insulin
    table['noninsulin_classes'] = [sorted(set(left) | set(right)) for left, right in zip(table.summary_classes, table.event_classes)]
    table['metformin_only_or_none'] = ~table.any_insulin & table.noninsulin_classes.map(lambda classes: set(classes) <= {'metformin'})
    intake = pd.DataFrame(intake)
    items = pd.DataFrame(items)
    for name, frame in [('records', table.drop(columns=['step_minutes'])), ('cgm', pd.concat(cgm)[['record_id', 'time', 'cgm_mgdl']]),
                        ('intake_events', intake), ('intake_items', items), ('medication_events', pd.DataFrame(medications))]:
        frame.to_csv(INTERIM / f'{name}.csv', index=False)
    recorded = items.loc[~items.not_recorded]
    frequency = recorded.assign(key=recorded.item_en.fillna(recorded.text_en).str.lower().str.strip()).merge(table[['record_id', 'patient_id']], on='record_id')
    frequency = frequency.groupby('key').agg(
        lines=('line', 'size'), records=('record_id', 'nunique'), patients=('patient_id', 'nunique'), parsed_lines=('parsed', 'sum'),
        example_en=('text_en', 'first'), item_zh=('item_zh', lambda values: values.dropna().mode().iloc[0] if values.notna().any() else ''),
        units=('unit', lambda values: ','.join(sorted(values.dropna().unique()))), median_amount=('amount', 'median'),
    ).sort_values('lines', ascending=False).reset_index()
    frequency['cumulative_line_share'] = frequency.lines.cumsum() / frequency.lines.sum()
    frequency.to_csv(INTERIM / 'food_item_frequency.csv', index=False)
    dictionary = [{'source': 'Shanghai_T2DM_Summary.xlsx', 'source_field': source, 'field': field, 'note': ''} for source, field in SUMMARY_FIELDS.items()]
    dictionary += [{'source': 'Shanghai_T2DM_Summary.xlsx', 'source_field': source, 'field': field, 'note': note} for field, source, note in [
        ('sex', 'Gender (Female=1, Male=2)', 'Recoded to CGMacros convention: female 0, male 1'),
        ('hba1c_pct', 'HbA1c (mmol/mol)', 'NGSP % = 0.09148 x IFCC mmol/mol + 2.152'),
        ('insulin', 'Fasting Insulin (pmol/L)', 'uIU/mL = pmol/L / 6.0'),
        ('cholesterol', 'Total Cholesterol (mmol/L)', 'mg/dL = mmol/L x 38.67'), ('hdl', 'High-Density Lipoprotein Cholesterol (mmol/L)', 'mg/dL = mmol/L x 38.67'),
        ('ldl', 'Low-Density Lipoprotein Cholesterol (mmol/L)', 'mg/dL = mmol/L x 38.67'), ('triglycerides', 'Triglyceride (mmol/L)', 'mg/dL = mmol/L x 88.57'),
        ('homa_ir', 'Fasting Insulin; Fasting Plasma Glucose', 'insulin uIU/mL x fasting glucose mg/dL / 405, as in CGMacros')]]
    dictionary += [{'source': 'Shanghai_T2DM/<record>.xls[x]', 'source_field': f'column {position + 1}', 'field': field,
                    'note': 'Mapped by position; header spellings vary across files'} for position, field in enumerate(RECORD_COLUMNS)]
    assert set(CONVERSIONS) <= {row['field'] for row in dictionary}
    pd.DataFrame(dictionary).to_csv(METADATA / 'shanghai_t2dm_field_dictionary.csv', index=False)
    steps = sum(profiles.step_minutes, Counter())
    glucose_values = pd.concat(cgm).cgm_mgdl
    patients = table.drop_duplicates('patient_id')
    subsets = {'all_records': table.index == table.index, 'no_insulin': ~table.any_insulin, 'metformin_only_or_none': table.metformin_only_or_none}
    eligible = intake.loc[~intake.not_recorded & intake.complete_120min_window]
    report = {
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': {'dataset': 'ShanghaiT2DM', 'figshare_version': 3, 'doi': '10.6084/m9.figshare.20444397.v3', 'license': 'CC BY 4.0',
                   'record_files': len(paths), 'file_sha256': {str(path.relative_to(SOURCE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [SOURCE / 'Shanghai_T2DM_Summary.xlsx', *paths]}},
        'records': len(table), 'patients': int(table.patient_id.nunique()),
        'records_per_patient': {int(key): int(value) for key, value in table.patient_id.value_counts().value_counts().sort_index().items()},
        'sex_patients': patients.sex.map({0: 'female', 1: 'male'}).value_counts().to_dict(),
        'monitoring': {'span_days': describe(table.span_days), 'cgm_readings': int(len(glucose_values)), 'step_minutes': {str(key): value for key, value in sorted(steps.items())},
                       'duplicate_timestamps': int(table.duplicate_timestamps.sum()), 'backward_steps': int(table.backward_steps.sum()),
                       'cgm_mgdl': describe(glucose_values), 'cgm_below_40': int((glucose_values < 40).sum()), 'cgm_above_400': int((glucose_values > 400).sum())},
        'intake': {'events': len(intake), 'events_on_cgm_rows': int(intake.on_cgm_row.sum()), 'events_not_recorded': int(intake.not_recorded.sum()),
                   'events_all_lines_parsed': int(intake.all_lines_parsed.sum()), 'events_with_complete_120min_cgm': int(intake.complete_120min_window.sum()),
                   'lines': len(items), 'not_recorded_lines': int(items.not_recorded.sum()), 'parsed_lines': int(items.parsed.sum()),
                   'unparsed_recorded_lines': recorded.loc[~recorded.parsed, 'text_en'].value_counts().to_dict(),
                   'events_with_unequal_en_zh_lines': int((~items.groupby(['record_id', 'time']).zh_line_count_matches.first()).sum()),
                   'distinct_items': len(frequency), 'top_200_line_share': float(frequency.cumulative_line_share.iloc[min(199, len(frequency) - 1)])},
        'medication_subsets': {name: {'records': int(mask.sum()), 'patients': int(table.loc[mask, 'patient_id'].nunique()),
                                      'recorded_intake_events_with_complete_120min_cgm': int(eligible.record_id.isin(table.loc[mask, 'record_id']).sum())}
                               for name, mask in subsets.items()},
        'insulin_source_agreement': pd.crosstab(table.summary_insulin, table.event_insulin).to_dict(),
        'noninsulin_class_records': Counter(name for classes in table.noninsulin_classes for name in classes),
        'unmatched_agent_terms': sorted({term for text in table.agents for term in unmatched_agent_terms(text)} | {term for terms in table.unmatched_event_terms for term in terms}),
        'baseline': {field: describe(table[field]) for field in ['age', 'bmi', 'diabetes_years', 'hba1c_pct', 'fasting_glucose_mgdl', 'insulin', 'homa_ir', 'cholesterol', 'hdl', 'ldl', 'triglycerides', 'fasting_cpeptide_nmoll', 'glycated_albumin_pct']},
        'note': 'Counts only. Postprandial outcomes were not computed in this audit; cohort rules and the external validation protocol are fixed separately.',
    }
    (OUTPUT / 'data_audit_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=plain))
    print(json.dumps({key: report[key] for key in ['records', 'patients', 'records_per_patient', 'intake', 'medication_subsets']}, indent=2, ensure_ascii=False, default=plain))


if __name__ == '__main__':
    main()
