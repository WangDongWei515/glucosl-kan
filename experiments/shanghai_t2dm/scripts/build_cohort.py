from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STAGE / 'experiments/cgmacros/scripts'))
from build_dataset import positive_area

INTERIM = STAGE / 'data/interim/shanghai_t2dm'
OFFSETS = np.arange(0, 121, 15)
WINDOWS = {'breakfast': (5, 10), 'lunch': (10, 15), 'dinner': (16, 21)}
FLOOR_MGDL = 40


def meal_type(time):
    return next((name for name, (start, end) in WINDOWS.items() if start <= time.hour < end), 'snack')


def main():
    records = pd.read_csv(INTERIM / 'records.csv')
    events = pd.read_csv(INTERIM / 'intake_events.csv', parse_dates=['time'])
    items = pd.read_csv(INTERIM / 'intake_items.csv', parse_dates=['time'])
    glucose = pd.read_csv(INTERIM / 'cgm.csv', parse_dates=['time']).set_index(['record_id', 'time']).cgm_mgdl
    weights = items.groupby(['record_id', 'time']).agg(recorded_weight=('amount', 'sum'), lines_with_weight=('parsed', 'sum'))
    events = events.merge(weights, on=['record_id', 'time'], how='left').sort_values(['record_id', 'time']).reset_index(drop=True)
    events['window'] = events.time.map(meal_type)
    events['date'] = events.time.dt.date
    # Within one window of one day, the heaviest recorded intake is the meal; other events in that window count as snacks.
    ranked = events.loc[events.window != 'snack'].sort_values(['recorded_weight', 'time'], ascending=[False, True])
    chosen = ranked.drop_duplicates(['record_id', 'date', 'window']).index
    events['meal_type'] = np.where(events.index.isin(chosen), events.window, 'snack')
    following = events.groupby('record_id').time.shift(-1)
    events['next_intake_minutes'] = (following - events.time).dt.total_seconds().div(60).fillna(np.inf)
    rows = []
    for event in events.itertuples(index=False):
        values = glucose.reindex([(event.record_id, event.time + pd.Timedelta(minutes=int(offset))) for offset in OFFSETS]).to_numpy(dtype=float)
        complete = bool(np.isfinite(values).all())
        rows.append({'onset_mgdl': values[0], 'complete_window': complete,
                     'iauc_15min': positive_area(values, OFFSETS) if complete else np.nan,
                     'peak_mgdl': float(values.max()) if complete else np.nan})
    events = pd.concat([events, pd.DataFrame(rows)], axis=1)
    events['peak_ge_180'] = np.where(events.complete_window, events.peak_mgdl >= 180, np.nan)
    events = events.merge(records[['record_id', 'patient_id', 'age', 'sex', 'bmi', 'hba1c_pct', 'fasting_glucose_mgdl', 'any_insulin', 'metformin_only_or_none']], on='record_id', validate='many_to_one')
    hours = events.time.dt.hour + events.time.dt.minute / 60
    events['hour_sin'], events['hour_cos'] = np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)
    events['is_lunch'], events['is_dinner'] = (events.meal_type == 'lunch').astype(int), (events.meal_type == 'dinner').astype(int)
    criteria = {
        'main_meal': events.meal_type != 'snack',
        'isolated_120min': events.next_intake_minutes > 120,
        'complete_window': events.complete_window,
        'onset_above_floor': events.onset_mgdl > FLOOR_MGDL,
        'diet_recorded': ~events.not_recorded,
        'all_lines_weighted': events.all_lines_parsed,
        'valid_demographics': events[['age', 'sex', 'bmi']].notna().all(axis=1),
    }
    for name, mask in criteria.items():
        events[name] = mask.to_numpy(dtype=bool)
    populations = {'no_insulin': ~events.any_insulin, 'all_records': events.index >= 0, 'metformin_only_or_none': events.metformin_only_or_none}
    flow = []
    for population, base in populations.items():
        keep = np.array(base, dtype=bool)
        flow.append({'population': population, 'criterion': 'all_intake_events', 'meals': int(keep.sum()), 'patients': int(events.loc[keep, 'patient_id'].nunique())})
        for name in criteria:
            keep &= events[name].to_numpy()
            flow.append({'population': population, 'criterion': name, 'meals': int(keep.sum()), 'patients': int(events.loc[keep, 'patient_id'].nunique())})
    processed = STAGE / 'data/processed/shanghai_t2dm'
    processed.mkdir(parents=True, exist_ok=True)
    events.drop(columns=['date']).to_csv(processed / 'meal_candidates.csv', index=False)
    output = STAGE / 'results/shanghai_t2dm/reports'
    pd.DataFrame(flow).to_csv(output / 'cohort_flow_pre_nutrients.csv', index=False)
    summary = {
        'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'meal_type_counts': events.meal_type.value_counts().to_dict(),
        'note': 'Flow only. Outcome values are stored for the frozen analysis but deliberately not summarized here; nutrient completeness is applied after the food mapping is fixed.',
    }
    (output / 'cohort_build_pre_nutrients.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(pd.DataFrame(flow).to_string(index=False))


if __name__ == '__main__':
    main()
