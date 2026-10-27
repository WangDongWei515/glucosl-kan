# -*- coding: utf-8 -*-
"""Numerical invariants for the data and patient-level evaluation procedures."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/cgmacros/scripts'))
from build_dataset import positive_area
sys.path.insert(0, str(ROOT / 'experiments/glucosl/scripts'))
from evaluate_glucosl import regression_metrics, resample
from src.glucosl_preprocess import Preprocessor


class ReproductionChecks(unittest.TestCase):
    def test_incremental_area_units(self):
        self.assertAlmostEqual(positive_area([100, 140, 100], [0, 30, 60]), 20.0)

    def test_baseline_crossing_excludes_negative_area(self):
        self.assertAlmostEqual(positive_area([100, 140, 60], [0, 30, 60]), 15.0)

    def test_below_baseline_is_zero(self):
        self.assertEqual(positive_area([100, 90, 80], [0, 60, 120]), 0.0)

    def test_patient_averaging_does_not_favor_many_meals(self):
        frame = pd.DataFrame({'patient_id': ['A', 'A', 'B'], 'iauc': [10.0, 10.0, 100.0]})
        result = regression_metrics(frame, np.zeros(3))
        self.assertAlmostEqual(result['macro_mae'], 55.0)
        self.assertAlmostEqual(result['meal_mae'], 40.0)

    def test_bootstrap_relabels_repeated_patient_clusters(self):
        class RepeatedPatient:
            @staticmethod
            def choice(*args, **kwargs):
                return np.array(['A', 'A'])
        frame = pd.DataFrame({'patient_id': ['A', 'A', 'B']})
        rows, groups = resample(frame, RepeatedPatient())
        np.testing.assert_array_equal(rows, [0, 1, 0, 1])
        np.testing.assert_array_equal(groups, [0, 0, 1, 1])

    def test_preprocessor_does_not_refit_on_validation(self):
        training = pd.DataFrame({'age': [40.0, 60.0], 'carbs': [20.0, 40.0]})
        preprocessor = Preprocessor(['age'], ['carbs']).fit(training, ['A', 'B'])
        before = preprocessor.person_mean.copy()
        preprocessor.transform(pd.DataFrame({'age': [1000.0], 'carbs': [1000.0]}), clip=True)
        pd.testing.assert_series_equal(preprocessor.person_mean, before)
        self.assertEqual(float(preprocessor.person_mean['age']), 50.0)


if __name__ == '__main__':
    unittest.main()
