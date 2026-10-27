from pathlib import Path
import sys

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STAGE / 'experiments/cgmacros/scripts'))
from build_dataset import positive_area

OFFSETS = np.arange(0, 121, 15)


def main():
    # Recompute the CGMacros Libre outcomes on the same 0, 15, ..., 120-minute grid used for ShanghaiT2DM.
    cohort = pd.read_csv(STAGE / 'data/processed/cgmacros/primary_cohort.csv')
    curves = np.load(STAGE / 'data/processed/cgmacros/aligned_glucose_curves.npz')
    position = {meal: index for index, meal in enumerate(curves['meal_id'])}
    libre = curves['glucose'][[position[meal] for meal in cohort.meal_id], 0][:, OFFSETS]
    assert np.isfinite(libre).all()
    outcomes = pd.DataFrame({
        'meal_id': cohort.meal_id,
        'iauc_15min': [positive_area(values, OFFSETS) for values in libre],
        'peak_ge_180': (libre.max(axis=1) >= 180).astype(int),
    })
    outcomes.to_csv(STAGE / 'data/processed/cgmacros/harmonized_outcomes_15min.csv', index=False)
    agreement = np.corrcoef(outcomes.iauc_15min, cohort.libre_iauc)[0, 1]
    print(f'{len(outcomes)} meals; correlation with the original 1-minute Libre iAUC: {agreement:.4f}')


if __name__ == '__main__':
    main()
