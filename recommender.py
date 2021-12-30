import argparse
import json
import numpy as np
import pandas as pd
import editdistance


def load_data(path):
    with open(path, 'r') as f:
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
    p_trials = pd.DataFrame.from_dict(p_trials).astype({'condition': 'category', 'therapy': 'category', 'successfull': int}).set_index(['patient', 'id'])
    # Post-processing
    p_conditions['diagnosed'], p_conditions['cured'] = pd.to_datetime(p_conditions['diagnosed']), pd.to_datetime(p_conditions['cured'])
    p_trials['start'], p_trials['end'] = pd.to_datetime(p_trials['start']), pd.to_datetime(p_trials['end'])
    return conditions, therapies, patients, p_conditions, p_trials


def svd(p_trials):
    pt_matrix = p_trials.pivot_table(index='patient', columns=['condition', 'therapy'], values='successfull', fill_value=0) # Compute matrix of patient-therapies for SVD
    u, s, vh = np.linalg.svd(pt_matrix.values, full_matrices=False)
    reconstructed = (u * s) @ vh
    return u, s, vh


def patients_features():
    pass

def p_conditions_features(p_conditions, conditions):
    features = pd.get_dummies(p_conditions['kind'])
    features = features.groupby('patient').sum()
    features = features.where(features <= 1, 1) # Set back to 1 values greater than 1
    features = features[conditions.index]
    return features


def trials_distance(patient_id, condition_id, p_trials, p_conditions):
    condition_kind = p_conditions.loc[(patient_id, condition_id), 'kind']
    merged_trials = p_trials.merge(p_conditions, how='left', left_on=['patient', 'condition'], right_on=['patient', 'id'])
    current_trials = merged_trials.loc[patient_id].pipe(lambda df: df[df.condition == condition_id]).sort_values('start')['therapy'].to_list()
    other_trials = merged_trials[(merged_trials.index != patient_id) & (merged_trials.condition != condition_id) & (merged_trials.kind == condition_kind)]
    other_trials = other_trials.groupby('patient').apply(lambda p: p.sort_values('start')['therapy'].to_list())
    other_trials = other_trials.rename('therapies').to_frame()
    other_trials['distance'] = np.nan
    for tls in other_trials.itertuples():
        other_trials.loc[tls.Index, 'distance'] = editdistance.eval(current_trials, tls.therapies) # TODO: use edit distance ignoring suffixes
    other_trials = other_trials.sort_values('distance')
    return other_trials


if __name__ == '__main__':
    # Script arguments
    parser = argparse.ArgumentParser(description='Therapy recommender system.', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))
    #parser.add_argument('dataset_path', type=str, help='path to the dataset file.', default='./data/generated/dataset.json')
    #parser.add_argument('patient_id', type=int, help='id of the patient for which to compute the recommendation.')
    #parser.add_argument('condition_id', type=str, help='id of the condition of the patient for which to compute the recommendation.')
    
    #TODO: remove
    parser.set_defaults(dataset_path='./data/generated/dataset.json')
    parser.set_defaults(patient_id=1000)
    parser.set_defaults(condition_id='pc5952')
    #TODO: remove

    args = parser.parse_args()

    # Load dataset
    conditions, therapies, patients, p_conditions, p_trials = load_data(args.dataset_path)

    # Merge everything
    merged = (p_trials.reset_index()
        .merge(p_conditions, how='left', left_on=['patient', 'condition'], right_on=['patient', 'id'])
        .merge(conditions.add_prefix('condition_'), how='left', left_on=['kind'], right_on=['id']).drop(['condition_name'], axis=1)
        .merge(therapies.add_prefix('therapy_'), how='left', left_on=['therapy'], right_on=['id']).drop(['therapy_name'], axis=1)
        .merge(patients.add_prefix('patient_'), how='left', left_on=['patient'], right_on=['id']).drop(['patient_name'], axis=1)
    ).set_index(['patient', 'id'])

    # Mine frequent itemsets
    # from mlxtend.frequent_patterns import apriori
    # from scipy.sparse import csr_matrix
    # trials_itemset = merged.pivot_table(index=['patient', 'condition'], columns='therapy', values='successfull')
    # apriori_input = pd.DataFrame.sparse.from_spmatrix(csr_matrix(trials_itemset.fillna(0).astype(bool).values), index=trials_itemset.index, columns=trials_itemset.columns)
    # frequent_therapies = apriori(apriori_input, min_support=0.1, use_colnames=True)


    patient_id = args.patient_id
    condition_id = args.condition_id