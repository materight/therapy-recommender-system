import json
import numpy as np
import pandas as pd


class Dataset():
    """Helper class for loading the dataset and convert it to multiple pandas dataframes."""
    
    def __init__(self, dataset_path: str, val_ratio: float = 0.1):
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
        
        # Create dataframes
        patients = pd.DataFrame.from_dict(patients) \
                     .set_index('id')
        p_conditions = pd.DataFrame.from_dict(p_conditions) \
                         .astype({'patient': 'category', 'kind': 'category'}) \
                         .set_index(['patient', 'id'])
        p_trials = pd.DataFrame.from_dict(p_trials) \
                     .astype({'patient': 'category', 'condition': 'category', 'therapy': 'category', 'successful': int}) \
                     .set_index(['patient', 'id'])
        
        # Post-processing
        p_conditions.replace({'cured': 'Null'}, value=np.nan, inplace=True)
        p_conditions['diagnosed'], p_conditions['cured'] = pd.to_datetime(p_conditions['diagnosed']), pd.to_datetime(p_conditions['cured'])
        p_trials.replace({'end': 'Null'}, value=np.nan, inplace=True)
        p_trials['start'], p_trials['end'] = pd.to_datetime(p_trials['start']), pd.to_datetime(p_trials['end'])
        
        # Generate validation split, using last trials from a subset of conditions and patients
        if val_ratio > 0:
            # TODO: make split computation reproducible
            val_patients_idx = patients.sample(frac=val_ratio).index
            val_patients_idx = val_patients_idx.intersection(p_conditions.index.get_level_values('patient'))
            val_conditions_idx = p_conditions.loc[val_patients_idx].sample(frac=val_ratio).index
            val_trials_idx = p_trials[p_trials.condition.isin(val_conditions_idx.get_level_values('id'))] \
                            .reset_index() \
                            .sort_values('start') \
                            .groupby(['patient', 'condition'], observed=True) \
                            .last()['id'].values
            val_trials_mask = p_trials.index.get_level_values('id').isin(val_trials_idx)
            val_trials = p_trials[val_trials_mask]
            p_trials = p_trials[~val_trials_mask]
        else:
            val_trials = None

        # Store dataframes
        self.patients = patients
        self.conditions = conditions
        self.therapies = therapies
        self.p_conditions = p_conditions
        self.p_trials = p_trials
        self.val_trials = val_trials
