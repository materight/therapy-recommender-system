import json
import numpy as np
import pandas as pd


class Dataset():
    """Helper class for loading the dataset and convert it to multiple pandas dataframes."""
    
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r') as f:
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
        p_conditions = pd.DataFrame.from_dict(p_conditions).astype({'kind': 'category'}).set_index(['patient', 'id'])
        p_trials = pd.DataFrame.from_dict(p_trials).astype({'condition': 'category', 'therapy': 'category', 'successful': int}).set_index(['patient', 'id'])
        
        # Post-processing
        p_conditions.replace({'cured': 'Null'}, value=np.nan, inplace=True)
        p_conditions['diagnosed'], p_conditions['cured'] = pd.to_datetime(p_conditions['diagnosed']), pd.to_datetime(p_conditions['cured'])
        p_trials.replace({'end': 'Null'}, value=np.nan, inplace=True)
        p_trials['start'], p_trials['end'] = pd.to_datetime(p_trials['start']), pd.to_datetime(p_trials['end'])
        
        # Store dataframes
        self.patients = patients
        self.conditions = conditions
        self.therapies = therapies
        self.p_conditions = p_conditions
        self.p_trials = p_trials
