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
    # Post-processing
    p_conditions['diagnosed'], p_conditions['cured'] = pd.to_datetime(p_conditions['diagnosed']), pd.to_datetime(p_conditions['cured'])
    p_trials['start'], p_trials['end'] = pd.to_datetime(p_trials['start']), pd.to_datetime(p_trials['end'])
    p_trials['successfull'] = p_trials['successfull'].astype(int)
    return conditions, therapies, patients, p_conditions, p_trials


def svd(p_trials):
    pt_matrix = p_trials.pivot_table(index='patient', columns=['condition', 'therapy'], values='successfull', fill_value=0) # Compute matrix of patient-therapies for SVD
    u, s, vh = np.linalg.svd(pt_matrix.values, full_matrices=False)
    reconstructed = (u * s) @ vh
    return u, s, vh


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

    # Merge everything
    merged = (p_trials.reset_index()
        .merge(p_conditions, how='left', left_on=['patient', 'condition'], right_on=['patient', 'kind']).drop(['kind'], axis=1) # TODO solve issue: if a condition for a patient appears multiple times, the trial gets duplicated
        .merge(conditions.add_prefix('condition_'), how='left', left_on=['condition'], right_on=['id']).drop(['condition_name'], axis=1)
        .merge(therapies.add_prefix('therapy_'), how='left', left_on=['therapy'], right_on=['id']).drop(['therapy_name'], axis=1)
        .merge(patients.add_prefix('patient_'), how='left', left_on=['patient'], right_on=['id']).drop(['patient_name'], axis=1)
    ).set_index(['patient', 'id'])

    # Mine frequent itemsets
    from mlxtend.frequent_patterns import apriori
    from scipy.sparse import csr_matrix
    trials_itemset = merged.pivot_table(index=['patient', 'condition'], columns='therapy', values='successfull')
    apriori_input = pd.DataFrame.sparse.from_spmatrix(csr_matrix(trials_itemset.fillna(0).astype(bool).values), index=trials_itemset.index, columns=trials_itemset.columns)
    frequent_therapies = apriori(apriori_input, min_support=0.1, use_colnames=True)