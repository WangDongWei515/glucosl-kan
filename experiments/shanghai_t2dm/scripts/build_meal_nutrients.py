from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[3]
INTERIM = STAGE / 'data/interim/shanghai_t2dm'
METADATA = STAGE / 'data/metadata'
PROCESSED = STAGE / 'data/processed/shanghai_t2dm'
NUTRIENTS = ['energy_kcal', 'carbohydrate_g', 'protein_g', 'fat_g', 'fiber_g']
VARIANTS = ['primary', 'no_oil', 'group_only', 'usda', 'staples_cooked']


def main():
    # Line-level nutrients, including amount overrides and weight-state rules, come from the food-mapping build.
    lines = pd.read_csv(INTERIM / 'intake_line_nutrients.csv', parse_dates=['time'])
    usable = lines.usable.astype(bool)
    nutrient_columns = [f'{variant}_{name}' for variant in VARIANTS for name in NUTRIENTS]
    lines[nutrient_columns] = np.where(usable.to_numpy()[:, None], lines[nutrient_columns].to_numpy(dtype=float), np.nan)
    lines['specific_weight'] = np.where(usable & lines.mapping_level.eq('specific'), lines.amount_g_used, 0.0)
    lines['usable_weight'] = np.where(usable, lines.amount_g_used, 0.0)
    lines['usable'] = usable
    meals = lines.groupby(['record_id', 'time']).agg(
        lines=('line', 'size'), unusable_lines=('usable', lambda values: int((~values).sum())),
        recorded_weight_g=('usable_weight', 'sum'), specific_weight_g=('specific_weight', 'sum'),
        **{column: (column, lambda values: values.sum(min_count=len(values))) for column in nutrient_columns},
    ).reset_index()
    meals['nutrient_complete'] = meals.unusable_lines.eq(0) & meals.primary_energy_kcal.notna()
    meals['specific_weight_share'] = meals.specific_weight_g / meals.recorded_weight_g.replace(0, np.nan)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    meals.to_csv(PROCESSED / 'meal_nutrients.csv', index=False)
    candidates = pd.read_csv(PROCESSED / 'meal_candidates.csv', parse_dates=['time'])
    candidates = candidates.merge(meals[['record_id', 'time', 'nutrient_complete', 'specific_weight_share'] + nutrient_columns], on=['record_id', 'time'], how='left', validate='one_to_one')
    candidates['nutrient_complete'] = candidates.nutrient_complete.fillna(False).astype(bool)
    # nutrient_complete uses adjudicated amounts, so it replaces the pre-override all_lines_weighted check.
    criteria = ['main_meal', 'isolated_120min', 'complete_window', 'onset_above_floor', 'diet_recorded', 'valid_demographics', 'nutrient_complete']
    populations = {'no_insulin': ~candidates.any_insulin, 'all_records': candidates.index >= 0, 'metformin_only_or_none': candidates.metformin_only_or_none}
    flow = []
    for population, base in populations.items():
        keep = np.array(base, dtype=bool)
        flow.append({'population': population, 'criterion': 'all_intake_events', 'meals': int(keep.sum()), 'patients': int(candidates.loc[keep, 'patient_id'].nunique())})
        for name in criteria:
            keep &= candidates[name].to_numpy(dtype=bool)
            flow.append({'population': population, 'criterion': name, 'meals': int(keep.sum()), 'patients': int(candidates.loc[keep, 'patient_id'].nunique())})
        candidates[f'cohort_{population}'] = keep
    candidates.to_csv(PROCESSED / 'external_cohort.csv', index=False)
    output = STAGE / 'results/shanghai_t2dm/reports'
    pd.DataFrame(flow).to_csv(output / 'cohort_flow.csv', index=False)
    primary = candidates.loc[candidates.cohort_no_insulin]
    receipt = {
        'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'mapping_sha256': hashlib.sha256((METADATA / 'shanghai_t2dm_food_mapping.csv').read_bytes()).hexdigest(),
        'line_nutrients_sha256': hashlib.sha256((INTERIM / 'intake_line_nutrients.csv').read_bytes()).hexdigest(),
        'protocol_sha256': hashlib.sha256((STAGE / 'docs/protocol_glucosl_v1.md').read_bytes()).hexdigest(),
        'primary_meals': len(primary), 'primary_patients': int(primary.patient_id.nunique()),
        'primary_specific_weight_share_median': float(primary.specific_weight_share.median()),
        'primary_meals_specific_share_ge_90pct': int((primary.specific_weight_share >= .9).sum()),
        'primary_nutrient_medians': primary[[f'primary_{name}' for name in NUTRIENTS]].median().round(2).to_dict(),
        'note': 'Predictor summaries only; glucose outcomes are not summarized before the protocol is frozen.',
    }
    (output / 'meal_nutrients_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False))
    print(pd.DataFrame(flow).to_string(index=False))
    print(json.dumps({key: value for key, value in receipt.items() if not key.endswith('sha256')}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
