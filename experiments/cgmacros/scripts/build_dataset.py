from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[3]
SOURCE = STAGE / 'data/raw/cgmacros/1.0.0/extracted/CGMacros'
CONFIG = json.loads((STAGE / 'experiments/cgmacros/configs/protocol_v1.json').read_text())


def positive_area(values, minutes):
    delta = np.asarray(values, dtype=float) - float(values[0])
    interval = np.diff(np.asarray(minutes, dtype=float))
    left, right = delta[:-1], delta[1:]
    area = np.zeros_like(interval)
    both = (left >= 0) & (right >= 0)
    area[both] = (left[both] + right[both]) * interval[both] / 2
    falling = (left > 0) & (right < 0)
    area[falling] = left[falling] ** 2 * interval[falling] / (2 * (left[falling] - right[falling]))
    rising = (left < 0) & (right > 0)
    area[rising] = right[rising] ** 2 * interval[rising] / (2 * (right[rising] - left[rising]))
    return float(area.sum() / 60)


def main():
    assert positive_area([100, 110, 100], [0, 60, 120]) == 10
    assert positive_area([100, 90, 90], [0, 60, 120]) == 0
    assert abs(positive_area([100, 110, 90], [0, 60, 120]) - 7.5) < 1e-10
    bio = pd.read_csv(SOURCE / 'bio.csv')
    bio.columns = bio.columns.str.strip()
    bio = bio.set_index('subject')
    assert bio.index.is_unique
    records = []
    curves = []
    source_hashes = {}
    for path in sorted(SOURCE.glob('CGMacros-*/CGMacros-*.csv')):
        subject = int(re.search(r'CGMacros-(\d+)\.csv', path.name).group(1))
        source_hashes[str(path.relative_to(SOURCE))] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert subject in bio.index
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        frame['time'] = pd.to_datetime(frame.Timestamp, format='mixed', errors='raise')
        frame = frame.set_index('time').sort_index()
        assert frame.index.is_unique
        meals = frame.loc[frame['Meal Type'].notna()]
        phenotype = bio.loc[subject]
        for position, (anchor, meal) in enumerate(meals.iterrows()):
            meal_type = str(meal['Meal Type']).strip().lower()
            if meal_type in ['snacks', 'snack 1']:
                meal_type = 'snack'
            next_time = meals.index[position + 1] if position + 1 < len(meals) else None
            next_minutes = (next_time - anchor).total_seconds() / 60 if next_time is not None else np.inf
            record = {
                'meal_id': f'{subject:03d}_{anchor.strftime("%Y%m%dT%H%M")}',
                'subject': subject, 'meal_time': str(anchor), 'meal_type': meal_type,
                'age': float(phenotype.Age), 'sex': {'F': 0, 'M': 1}.get(str(phenotype.Gender).strip()),
                'bmi': float(phenotype.BMI), 'hour_sin': float(np.sin(2 * np.pi * (anchor.hour + anchor.minute / 60) / 24)),
                'hour_cos': float(np.cos(2 * np.pi * (anchor.hour + anchor.minute / 60) / 24)),
                'is_lunch': int(meal_type == 'lunch'), 'is_dinner': int(meal_type == 'dinner'),
                'next_meal_minutes': next_minutes,
            }
            for column, output in [('Calories', 'energy_kcal'), ('Carbs', 'carbohydrate_g'), ('Protein', 'protein_g'), ('Fat', 'fat_g'), ('Fiber', 'fiber_g')]:
                record[output] = float(meal[column]) if pd.notna(meal[column]) else np.nan
            for column, output in [('A1c PDL (Lab)', 'hba1c_pct'), ('Fasting GLU - PDL (Lab)', 'fasting_glucose_mgdl'), ('Insulin', 'insulin'), ('Triglycerides', 'triglycerides'), ('Cholesterol', 'cholesterol'), ('HDL', 'hdl'), ('Non HDL', 'non_hdl'), ('LDL (Cal)', 'ldl'), ('VLDL (Cal)', 'vldl'), ('Cho/HDL Ratio', 'chol_hdl_ratio')]:
                record[output] = float(phenotype[column])
            record['homa_ir'] = record['insulin'] * record['fasting_glucose_mgdl'] / 405
            record['hba1c_range'] = 'normoglycemia' if record['hba1c_pct'] < 5.7 else ('prediabetes_range' if record['hba1c_pct'] < 6.5 else 'diabetes_range')
            record['main_meal'] = meal_type in CONFIG['primary_meal_types']
            record['isolated_120min'] = next_minutes >= 120
            record['valid_demographics'] = all(pd.notna(record[name]) and np.isfinite(record[name]) for name in ['age', 'sex', 'bmi'])
            record['valid_nutrients'] = record['energy_kcal'] > 0 and all(pd.isna(record[name]) or record[name] >= 0 for name in CONFIG['nutrient_features'])
            if not all(pd.notna(record[name]) for name in ['energy_kcal', 'carbohydrate_g', 'protein_g', 'fat_g']):
                record['valid_nutrients'] = False
            timestamps = pd.date_range(anchor, periods=121, freq='min')
            window = frame.reindex(timestamps)
            sensor_curves = []
            for source_sensor, sensor in [('Libre GL', 'libre'), ('Dexcom GL', 'dexcom')]:
                values = pd.to_numeric(window[source_sensor], errors='coerce').to_numpy(dtype=float)
                complete = bool(np.isfinite(values).all())
                record[sensor + '_complete'] = complete
                record[sensor + '_baseline'] = values[0]
                record[sensor + '_iauc'] = positive_area(values, np.arange(121)) if complete else np.nan
                record[sensor + '_signed_iauc'] = float(np.trapezoid(values - values[0]) / 60) if complete else np.nan
                record[sensor + '_auc'] = float(np.trapezoid(values) / 60) if complete else np.nan
                record[sensor + '_rise120'] = float(values[-1] - values[0]) if complete else np.nan
                sensor_curves.append(values)
            record['primary_eligible'] = all(record[name] for name in ['main_meal', 'isolated_120min', 'valid_demographics', 'valid_nutrients', 'libre_complete', 'dexcom_complete'])
            records.append(record)
            curves.append(sensor_curves)
    table = pd.DataFrame(records)
    assert table.meal_id.is_unique
    criteria = ['main_meal', 'isolated_120min', 'valid_demographics', 'valid_nutrients', 'libre_complete', 'dexcom_complete']
    flow = [{'criterion': 'all_recorded_intakes', 'meals': len(table), 'participants': int(table.subject.nunique())}]
    keep = np.ones(len(table), dtype=bool)
    for criterion in criteria:
        previous = keep.sum()
        keep &= table[criterion].to_numpy(dtype=bool)
        flow.append({'criterion': criterion, 'meals': int(keep.sum()), 'participants': int(table.loc[keep].subject.nunique()), 'excluded_at_step': int(previous - keep.sum())})
    assert np.array_equal(keep, table.primary_eligible)
    processed = STAGE / 'data/processed/cgmacros'
    processed.mkdir(parents=True, exist_ok=True)
    table.to_csv(processed / 'all_meal_records.csv', index=False)
    table.loc[keep].to_csv(processed / 'primary_cohort.csv', index=False)
    np.savez_compressed(processed / 'aligned_glucose_curves.npz', meal_id=table.meal_id.to_numpy(dtype=str), glucose=np.asarray(curves), minutes=np.arange(121))
    output = STAGE / 'results/cgmacros/reports'
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flow).to_csv(output / 'cohort_flow.csv', index=False)
    cohort = table.loc[keep]
    summary = {
        'built_at_utc': datetime.now(timezone.utc).isoformat(), 'flow': flow,
        'primary_meals': len(cohort), 'primary_participants': int(cohort.subject.nunique()),
        'primary_meal_types': cohort.meal_type.value_counts().to_dict(),
        'primary_participant_ranges': cohort.drop_duplicates('subject').hba1c_range.value_counts().to_dict(),
        'zero_iauc': {sensor: int(cohort[sensor + '_iauc'].eq(0).sum()) for sensor in CONFIG['sensors']},
        'outcome_summary': cohort[[sensor + '_iauc' for sensor in CONFIG['sensors']]].describe().to_dict(),
        'config_sha256': hashlib.sha256((STAGE / 'experiments/cgmacros/configs/protocol_v1.json').read_bytes()).hexdigest(),
        'source_csv_sha256': source_hashes,
    }
    (output / 'cohort_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({name: value for name, value in summary.items() if name != 'source_csv_sha256'}, indent=2))


if __name__ == '__main__':
    main()
