#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare public data, fit fixed study models, and reproduce validation metrics."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'xgboost', 'joblib', 'openpyxl', 'xlrd', 'statsmodels']
PREPARE = [
    'experiments/cgmacros/scripts/build_dataset.py',
    'experiments/external_validation/scripts/harmonize_cgmacros.py',
    'experiments/shanghai_t2dm/scripts/audit_data.py',
    'experiments/shanghai_t2dm/scripts/build_cohort.py',
    'experiments/shanghai_t2dm/scripts/query_cncdc.py',
    'experiments/shanghai_t2dm/scripts/build_food_mapping.py',
    'experiments/shanghai_t2dm/scripts/build_meal_nutrients.py',
    'experiments/glucosl/scripts/make_split.py',
]


def run(script, env, arguments=()):
    print(f'\nRunning {script}', flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *arguments], cwd=ROOT, env=env, check=True)


def check_dependencies(args):
    if sys.version_info < (3, 10):
        raise RuntimeError('Python 3.10 or newer is required; Python 3.12 matches the study environment.')
    packages = REQUIRED + (['tabpfn'] if args.models == 'all' else [])
    versions, missing = {}, []
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise RuntimeError('Missing packages: ' + ', '.join(missing) + '. Install requirements.txt and, for all models, requirements-tabpfn.txt.')
    print(json.dumps({'python': sys.version.split()[0], 'packages': versions}, indent=2), flush=True)


def validate_prepared():
    sys.path.insert(0, str(ROOT))
    from src.glucosl_data import shanghai_training, shanghai_validation, cgmacros
    a, b, us = shanghai_training(), shanghai_validation(), cgmacros()
    dm = us.loc[us.diabetes_range]
    counts = {'Shanghai A': (a.patient_id.nunique(), len(a)), 'Shanghai B': (b.patient_id.nunique(), len(b)),
              'CGMacros': (us.patient_id.nunique(), len(us)), 'CGMacros HbA1c >=6.5%': (dm.patient_id.nunique(), len(dm)),
              'well-mapped B meals': int((b.specific_weight_share >= 0.9).sum())}
    expected = {'Shanghai A': (67, 1952), 'Shanghai B': (33, 670), 'CGMacros': (44, 1147),
                'CGMacros HbA1c >=6.5%': (14, 353), 'well-mapped B meals': 347}
    assert counts == expected, f'Prepared cohorts differ from the study: {counts}'
    assert not set(a.patient_id) & set(b.patient_id), 'Patients overlap between training and validation.'
    for frame in [a, b, us]:
        assert frame.iauc.notna().all()
        assert set(frame.peak_ge_180.unique()) <= {0, 1}
    print('Prepared cohort checks passed:', counts, flush=True)
    return counts


def prepare(args, env):
    raw = ROOT / 'data/raw'
    if args.raw_dir:
        source = Path(args.raw_dir).expanduser().resolve()
        if not source.is_dir():
            raise RuntimeError(f'Raw source directory not found: {source}')
        raw.parent.mkdir(parents=True, exist_ok=True)
        if not raw.exists():
            raw.symlink_to(source, target_is_directory=True)
        elif raw.resolve() != source:
            raise RuntimeError('data/raw already points elsewhere; use a separate checkout for different source data.')
    if args.cn_cache_dir:
        source = Path(args.cn_cache_dir).expanduser().resolve()
        if not source.is_dir():
            raise RuntimeError(f'Food-composition cache not found: {source}')
        destination = ROOT / 'data/interim/shanghai_t2dm/cncdc_query_cache'
        if source != destination.resolve():
            shutil.copytree(source, destination, dirs_exist_ok=True)
    if args.download:
        if args.offline or args.raw_dir:
            raise RuntimeError('--download cannot be combined with --offline or --raw-dir.')
        for script in ['experiments/cgmacros/scripts/download_data.py',
                       'experiments/shanghai_t2dm/scripts/download_data.py',
                       'experiments/shanghai_t2dm/scripts/download_usda.py']:
            run(script, env)
    needed = ['cgmacros/1.0.0/extracted/CGMacros', 'shanghai_t2dm/v3/extracted',
              'usda_fdc/sr_legacy_2018-04/extracted/FoodData_Central_sr_legacy_food_csv_2018-04']
    missing = [name for name in needed if not (raw / name).exists()]
    if missing:
        raise RuntimeError('Missing source data: ' + ', '.join(missing) + '. Run prepare --download or supply --raw-dir.')
    for script in PREPARE:
        run(script, env)
    validate_prepared()


def verify(run_dir):
    import pandas as pd
    reference = pd.read_csv(ROOT / 'results/glucosl/evaluation/metrics.csv')
    reproduced = pd.read_csv(run_dir / 'evaluation/metrics.csv')
    keys = ['set', 'task', 'model', 'metric']
    merged = reproduced.merge(reference[keys + ['estimate']], on=keys, suffixes=('_reproduced', '_reference'), validate='one_to_one')
    assert len(merged) == len(reproduced), 'Some reproduced metrics have no study reference.'
    differences = (merged.estimate_reproduced - merged.estimate_reference).abs()
    finite = merged.estimate_reproduced.notna() & merged.estimate_reference.notna()
    unexpected_na = merged.estimate_reproduced.isna() ^ merged.estimate_reference.isna()
    max_difference = float(differences.loc[finite].max())
    report = {'compared_metrics': len(merged), 'maximum_absolute_point_difference': max_difference,
              'point_tolerance': 1e-6, 'unexpected_missing_metrics': int(unexpected_na.sum()),
              'point_estimates_match': bool(max_difference <= 1e-6 and not unexpected_na.any()),
              'confidence_intervals_compared': False}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report['point_estimates_match']:
        raise RuntimeError('Reproduced point estimates differ from study references; inspect verification.json and input versions.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['check', 'prepare', 'check-data', 'fit', 'evaluate', 'posthoc', 'verify', 'all'])
    parser.add_argument('--models', choices=['core', 'all'], default='all',
                        help='core: GlucoSL-KAN, XGBoost, constant and US/cooked-weight variants; all: add ablations, linear models and TabPFN v2')
    parser.add_argument('--run-dir', default='reproduction/default', help='Independent output directory; supplied study results are never overwritten.')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--bootstrap-draws', type=int, default=2000,
                        help='Use 2000 for study uncertainty estimates; zero checks point estimates only.')
    parser.add_argument('--download', action='store_true', help='Download the pinned official datasets before preparation.')
    parser.add_argument('--raw-dir', help='Use an existing raw-source directory with the documented layout.')
    parser.add_argument('--cn-cache-dir', help='Copy an existing local food-composition lookup cache.')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive.')
    if args.bootstrap_draws != 0 and (args.bootstrap_draws < 10 or args.bootstrap_draws % 10):
        parser.error('--bootstrap-draws must be zero or a positive multiple of 10.')
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir = run_dir.resolve()
    forbidden = (ROOT / 'results/glucosl').resolve()
    if run_dir == forbidden or forbidden in run_dir.parents:
        parser.error('--run-dir must not overwrite the supplied study reference results.')
    env = os.environ.copy()
    env.update(GLUCOSL_RUN_DIR=str(run_dir), GLUCOSL_MODEL_SET=args.models, GLUCOSL_JOBS=str(args.jobs),
               GLUCOSL_BOOTSTRAP_DRAWS=str(args.bootstrap_draws))
    if args.offline:
        env['GLUCOSL_OFFLINE'] = '1'
    if args.step == 'check':
        check_dependencies(args)
        return
    if args.step == 'verify':
        verify(run_dir)
        return
    check_dependencies(args)
    if args.step in ['prepare', 'all']:
        prepare(args, env)
    if args.step == 'check-data':
        validate_prepared()
    if args.step in ['fit', 'all']:
        validate_prepared()
        run('experiments/glucosl/scripts/fit_final_models.py', env)
    if args.step in ['evaluate', 'all']:
        validate_prepared()
        run('experiments/glucosl/scripts/evaluate_glucosl.py', env)
    if args.step == 'posthoc':
        if args.models != 'all':
            parser.error('Descriptive post hoc analysis requires --models all.')
        run('experiments/glucosl/scripts/posthoc_glucosl.py', env)
    if args.step == 'all':
        verify(run_dir)


if __name__ == '__main__':
    main()
