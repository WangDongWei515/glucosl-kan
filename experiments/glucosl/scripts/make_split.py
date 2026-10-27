from pathlib import Path

import pandas as pd


STAGE = Path(__file__).resolve().parents[3]
CUTOFF = pd.Timestamp('2021-06-29')


def main():
    # Protocol G1, 2.1: patients ordered by the start date of their first recording (from the recording identifier); dates only, no outcomes.
    cohort = pd.read_csv(STAGE / 'data/processed/shanghai_t2dm/external_cohort.csv', usecols=['record_id', 'patient_id', 'cohort_all_records'])
    cohort = cohort.loc[cohort.cohort_all_records]
    cohort['recording_start'] = pd.to_datetime(cohort.record_id.str.split('_').str[2], format='%Y%m%d')
    patients = cohort.groupby('patient_id').agg(first_recording=('recording_start', 'min'), meals=('record_id', 'size')).reset_index()
    patients = patients.sort_values(['first_recording', 'patient_id']).reset_index(drop=True)
    patients['set'] = patients.first_recording.le(CUTOFF).map({True: 'A', False: 'B'})
    assert patients.set.value_counts().to_dict() == {'A': 67, 'B': 33}
    output = STAGE / 'data/processed/glucosl'
    output.mkdir(parents=True, exist_ok=True)
    patients.assign(first_recording=patients.first_recording.dt.date).to_csv(output / 'split.csv', index=False)
    print(patients.groupby('set').agg(patients=('patient_id', 'size'), meals=('meals', 'sum')).to_string())


if __name__ == '__main__':
    main()
