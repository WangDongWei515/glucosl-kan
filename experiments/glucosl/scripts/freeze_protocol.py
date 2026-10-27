"""Protocol G1, Section 9, steps 2 and 5: record and verify SHA-256 hashes.

  --stage protocol   records the protocol, recipe, data, code and development-log hashes (step 2);
  --stage models     verifies those hashes still match, then adds every fitted model file (step 5);
  --verify           checks every recorded hash and exits non-zero on any mismatch (used by the evaluation script).
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys


STAGE = Path(__file__).resolve().parents[3]
RECORD = STAGE / 'docs/protocol_glucosl_v1_freeze.json'
PROTOCOL_FILES = [
    'docs/protocol_glucosl_v1.md',
    'experiments/glucosl/configs/final_recipe.json',
    'data/processed/glucosl/split.csv',
    'data/processed/shanghai_t2dm/external_cohort.csv',
    'data/processed/cgmacros/primary_cohort.csv',
    'data/processed/cgmacros/harmonized_outcomes_15min.csv',
    'data/metadata/shanghai_t2dm_food_mapping.csv',
    'data/interim/shanghai_t2dm/intake_line_nutrients.csv',
    'src/models/kan.py',
    'src/models/glucosl.py',
    'src/glucosl_preprocess.py',
    'src/glucosl_data.py',
    'src/external_metrics.py',
    'experiments/glucosl/scripts/fit_final_models.py',
    'experiments/glucosl/scripts/comparators.py',
    'results/glucosl/development_log.csv',
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(paths):
    return {str(path.relative_to(STAGE)): sha256(path) for path in paths}


def tabpfn_weights():
    # The pretrained v2 checkpoints that TabPFN loaded; their location is reported by the library's own resolver.
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import resolve_model_path
    found = []
    for task in ['regressor', 'classifier']:
        paths, _, _, _ = resolve_model_path(None, task, ModelVersion.V2.value)
        found += [Path(path) for path in paths if Path(path).exists()]
    assert len(found) == 2, f'expected the v2 regressor and classifier checkpoints, found {found}'
    return found


def verify(record):
    mismatched = []
    for section in ['protocol', 'models']:
        for name, expected in record.get(section, {}).items():
            path = STAGE / name if not name.startswith('/') else Path(name)
            if not path.exists() or sha256(path) != expected:
                mismatched.append(name)
    return mismatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['protocol', 'models'])
    parser.add_argument('--verify', action='store_true')
    arguments = parser.parse_args()
    if arguments.verify:
        mismatched = verify(json.loads(RECORD.read_text()))
        print('all hashes match' if not mismatched else 'MISMATCH: ' + ', '.join(mismatched))
        sys.exit(1 if mismatched else 0)
    if arguments.stage == 'protocol':
        assert not RECORD.exists(), f'{RECORD} already exists; the protocol stage is recorded once'
        record = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol': hashes(STAGE / name for name in PROTOCOL_FILES)}
    else:
        record = json.loads(RECORD.read_text())
        mismatched = verify(record)
        assert not mismatched, 'protocol-stage files changed after freezing: ' + ', '.join(mismatched)
        assert 'models' not in record, 'the model stage is recorded once'
        models = sorted((STAGE / 'results/glucosl/models').glob('*/*.joblib'))
        record['models'] = hashes(models)
        record['models'].update({str(path): sha256(path) for path in tabpfn_weights()})
        record['models_frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    RECORD.write_text(json.dumps(record, indent=2))
    print(f'{arguments.stage} stage recorded: {len(record[arguments.stage])} files')


if __name__ == '__main__':
    main()
