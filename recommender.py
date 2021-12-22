import argparse
import json
import numpy as np
import pandas as pd

def load_data(path):
    with open(args.dataset_path, 'r') as f:
        data = json.load(f)
    conditions = pd.DataFrame.from_dict(data['Conditions']).set_index('id')
    therapies = pd.DataFrame.from_dict(data['Therapies']).set_index('id')
    # Split patient data into three dataframes:
    patients, p_conditions, p_trials = [], [], []
    for p in data['Patients']:
        p_conditions.extend([{'patient': p['id'], **c} for c in p.pop('conditions')])
        p_trials.extend([{'patient': p['id'], **t} for t in p.pop('trials')])
        patients.append(p)
    patients = pd.DataFrame.from_dict(patients).set_index('id')
    p_conditions = pd.DataFrame.from_dict(p_conditions).set_index(['patient', 'id'])
    p_trials = pd.DataFrame.from_dict(p_trials).set_index(['patient', 'id'])
    return conditions, therapies, patients, p_conditions, p_trials


if __name__ == '__main__':
    # Script arguments
    parser = argparse.ArgumentParser(description='Therapy recommender system.', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))
    #parser.add_argument('dataset_path', type=str, help='path to the dataset file.', default='./data/generated/dataset.json')
    #parser.add_argument('patient_id', type=str, help='id of the patient for which to compute the recommendation.')
    #parser.add_argument('condition_id', type=str, help='id of the condition of the patient for which to compute the recommendation.')
    
    #TODO: remove
    parser.set_defaults(dataset_path='./data/generated/dataset.json')
    parser.set_defaults(patient_id='100')
    parser.set_defaults(condition_id='pc621')
    #TODO: remove

    args = parser.parse_args()

    # Load dataset
    conditions, therapies, patients, p_conditions, p_trials = load_data(args.dataset_path)