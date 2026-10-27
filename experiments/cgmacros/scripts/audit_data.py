from pathlib import Path
import csv
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[3]
SOURCE = STAGE / 'data/raw/cgmacros/1.0.0/extracted/CGMacros'
OUTPUT = STAGE / 'results/cgmacros/reports'
METADATA = STAGE / 'data/metadata'


def cleaned(path):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    return frame


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    bio = cleaned(SOURCE / 'bio.csv')
    bio['subject'] = pd.to_numeric(bio['subject'], errors='coerce')
    bio_ids = set(bio.subject.dropna().astype(int))
    profiles = []
    meals = []
    dictionaries = []
    file_profiles = {}
    for filename in ['bio.csv', 'gut_health_test.csv', 'microbes.csv']:
        frame = cleaned(SOURCE / filename)
        identifiers = pd.to_numeric(frame.subject, errors='coerce')
        file_profiles[filename] = {
            'rows': len(frame), 'columns': len(frame.columns),
            'subject_ids': sorted(identifiers.dropna().astype(int).unique().tolist()),
            'extra_subject_ids': sorted(set(identifiers.dropna().astype(int)) - bio_ids),
            'missing_subject_ids': sorted(bio_ids - set(identifiers.dropna().astype(int))),
            'invalid_identifier_rows': int(identifiers.isna().sum()),
            'duplicate_subject_rows': int(identifiers.dropna().duplicated().sum()),
        }
        for column in frame:
            numeric = pd.to_numeric(frame[column], errors='coerce')
            dictionaries.append({'source': filename, 'field': column, 'dtype': str(frame[column].dtype), 'nonmissing': int(frame[column].notna().sum()), 'missing': int(frame[column].isna().sum()), 'unique': int(frame[column].nunique()), 'numeric_min': numeric.min(), 'numeric_max': numeric.max()})
    for path in sorted(SOURCE.glob('CGMacros-*/CGMacros-*.csv')):
        subject = int(re.search(r'CGMacros-(\d+)\.csv', path.name).group(1))
        frame = cleaned(path)
        timestamp = pd.to_datetime(frame.Timestamp, errors='coerce', format='mixed')
        if timestamp.isna().any():
            raise ValueError(f'Unparseable timestamps: {path}')
        frame['time'] = timestamp
        differences = timestamp.diff().dt.total_seconds().div(60)
        meal_frame = frame.loc[frame['Meal Type'].notna()].copy()
        profile = {
            'subject': subject, 'rows': len(frame), 'start': str(timestamp.min()), 'end': str(timestamp.max()),
            'span_days': (timestamp.max() - timestamp.min()).total_seconds() / 86400,
            'duplicate_timestamps': int(timestamp.duplicated().sum()), 'backward_steps': int((differences < 0).sum()),
            'nonminute_steps': int((differences.dropna() != 1).sum()), 'meal_records': len(meal_frame),
        }
        for sensor in ['Libre GL', 'Dexcom GL']:
            values = pd.to_numeric(frame[sensor], errors='coerce')
            profile[sensor + '_missing'] = float(values.isna().mean())
            profile[sensor + '_min'] = values.min()
            profile[sensor + '_max'] = values.max()
        profiles.append(profile)
        for position, (_, meal) in enumerate(meal_frame.iterrows()):
            anchor = meal.time
            record = {'subject': subject, 'meal_time': str(anchor), 'meal_type': meal['Meal Type']}
            record['next_meal_minutes'] = (meal_frame.iloc[position + 1].time - anchor).total_seconds() / 60 if position + 1 < len(meal_frame) else np.nan
            for column in ['Calories', 'Carbs', 'Protein', 'Fat', 'Fiber', 'Amount Consumed']:
                record[column] = pd.to_numeric(pd.Series([meal.get(column)]), errors='coerce').iloc[0]
            window = frame.loc[frame.time.between(anchor, anchor + pd.Timedelta(minutes=120))]
            record['window_rows'] = len(window)
            for sensor in ['Libre GL', 'Dexcom GL']:
                values = pd.to_numeric(window[sensor], errors='coerce')
                record[sensor + '_coverage'] = float(values.notna().sum() / 121)
                record[sensor + '_baseline'] = float(meal[sensor]) if pd.notna(meal[sensor]) else np.nan
            image_path = meal.get('Image path')
            record['image_available'] = bool(isinstance(image_path, str) and (path.parent / image_path.replace('\\', '/')).is_file())
            meals.append(record)
    profile_table = pd.DataFrame(profiles)
    meal_table = pd.DataFrame(meals)
    profile_table.to_csv(METADATA / 'cgmacros_participant_profile.csv', index=False)
    meal_table.to_csv(STAGE / 'data/interim/cgmacros_meal_audit.csv', index=False)
    pd.DataFrame(dictionaries).to_csv(METADATA / 'cgmacros_source_dictionary.csv', index=False)
    glycemic = pd.cut(bio['A1c PDL (Lab)'], [-np.inf, 5.7 - 1e-10, 6.5 - 1e-10, np.inf], labels=['normoglycemia', 'prediabetes_range', 'diabetes_range'])
    summary = {
        'participants': len(profile_table), 'minute_rows': int(profile_table.rows.sum()),
        'meal_records': len(meal_table), 'meal_types': meal_table.meal_type.value_counts().to_dict(),
        'bio_hba1c_groups': glycemic.value_counts().to_dict(), 'supplemental_tables': file_profiles,
        'duplicate_timestamps': int(profile_table.duplicate_timestamps.sum()),
        'backward_steps': int(profile_table.backward_steps.sum()),
        'span_days': profile_table.span_days.describe().to_dict(),
        'meal_amounts': meal_table['Amount Consumed'].value_counts(dropna=False).to_dict(),
        'missing_meal_features': meal_table[['Calories', 'Carbs', 'Protein', 'Fat', 'Fiber']].isna().sum().to_dict(),
        'meal_coverage': {},
    }
    for sensor in ['Libre GL', 'Dexcom GL']:
        complete = meal_table[sensor + '_coverage'].eq(1) & meal_table[sensor + '_baseline'].notna() & meal_table.window_rows.eq(121)
        isolated = meal_table.next_meal_minutes.ge(120) | meal_table.next_meal_minutes.isna()
        summary['meal_coverage'][sensor] = {'complete_120min': int(complete.sum()), 'complete_and_isolated': int((complete & isolated).sum()), 'by_meal_type': meal_table.loc[complete & isolated].meal_type.value_counts().to_dict(), 'participants_complete_isolated': int(meal_table.loc[complete & isolated].subject.nunique())}
    (OUTPUT / 'data_audit_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
