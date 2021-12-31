import argparse
import numpy as np
import pandas as pd
import editdistance

from recommender.dataset import Dataset

def svd(p_trials):
    pt_matrix = p_trials.pivot_table(index='patient', columns=['condition', 'therapy'], values='successfull', fill_value=0) # Compute matrix of patient-therapies for SVD
    u, s, vh = np.linalg.svd(pt_matrix.values, full_matrices=False)
    reconstructed = (u * s) @ vh
    return u, s, vh


def patients_features():
    pass

def p_conditions_features(dataset):
    features = pd.get_dummies(dataset.p_conditions['kind'])
    features = features.groupby('patient').sum()
    features = features.where(features <= 1, 1) # Set back to 1 values greater than 1
    features = features[dataset.conditions.index]
    return features


def trials_similarity(patient_id, condition_id, dataset, normalized=True):
    condition_kind = dataset.p_conditions.loc[(patient_id, condition_id), 'kind']
    merged_trials = dataset.p_trials.merge(dataset.p_conditions, how='left', left_on=['patient', 'condition'], right_on=['patient', 'id'])
    target_trials = merged_trials.loc[patient_id].pipe(lambda df: df[df.condition == condition_id]).sort_values('start')['therapy'].to_list()
    candidate_trials = merged_trials[(merged_trials.index != patient_id) & (merged_trials.condition != condition_id) & (merged_trials.kind == condition_kind)]
    candidate_trials = candidate_trials.groupby(['patient', 'condition']).apply(lambda p: p.sort_values('start')['therapy'].to_list())
    candidate_trials = candidate_trials.rename('therapies').to_frame()
    candidate_trials['similarity'] = np.nan
    for tls in candidate_trials.itertuples():
        similarity = max(len(tls.therapies), len(target_trials)) - editdistance.eval(target_trials, tls.therapies)
        if normalized:
            similarity = similarity / len(target_trials) # Normalize similarity
        candidate_trials.loc[tls.Index, 'similarity'] = similarity
    # Remove trials wich does not have any therapy in common with the target trials
    candidate_trials = candidate_trials[candidate_trials.similarity > 0]
    candidate_trials = candidate_trials.sort_values('similarity', ascending=False)
    # Compute set of therapies not included in the target trials
    candidate_trials['new_therapies'] = candidate_trials['therapies'].apply(lambda l: l[max([l.index(t) for t in target_trials if t in l])+1:]).to_frame() # Remove therapies that are before the last matched therapy present in target_trials
    candidate_therapies = candidate_trials.drop('therapies', axis=1).explode('new_therapies').rename(columns={'new_therapies': 'therapy'}).dropna()
    # Add success rate to the candidate therapies
    candidate_therapies = candidate_therapies.merge(dataset.p_trials[['condition', 'therapy', 'successfull']], how='left', left_on=['patient', 'condition', 'therapy'], right_on=['patient', 'condition', 'therapy'])
    candidate_therapies = candidate_therapies[['condition', 'therapy', 'successfull', 'similarity']]
    # Compute final scaore combining similarity and success rate
    candidate_therapies['score'] = candidate_therapies.similarity * (candidate_therapies.successfull/100)
    candidate_therapies = candidate_therapies.drop_duplicates()
    candidate_therapies = candidate_therapies.sort_values('score', ascending=False)
    return target_trials, candidate_trials, candidate_therapies


def patients_similarity(patient_id, condition_id, dataset):
    condition_kind = dataset.p_conditions.loc[(patient_id, condition_id), 'kind']
    # Get patients that have the same condition of the target one
    filtered_patients_ids = dataset.p_conditions[dataset.p_conditions.kind == condition_kind].index.unique(level='patient')
    filtered_patients_ids = np.intersect1d(filtered_patients_ids, dataset.p_trials.index.levels[0]) # Remove patients that have no trials
    filtered_p_conditions = dataset.p_conditions.loc[filtered_patients_ids]
    # Convert p_conditions to a binary representation of 0 or 1 values, with one row for each patient and one column for each condition.
    patients_features = pd.get_dummies(filtered_p_conditions['kind'])
    patients_features = patients_features.groupby('patient').sum().astype(bool)
    patients_features = patients_features[dataset.conditions.index]
    # Compute sets
    target_patient = patients_features.loc[patient_id]
    other_patients = patients_features.drop(patient_id)
    # Compute Jaccard similarity
    intersection = (target_patient & other_patients).sum(axis=1)
    union = (target_patient | other_patients).sum(axis=1)
    patients_similarities = (intersection / union).rename('similarity')
    # Compute set of therapies not already tested on the target patient
    merged_trials = dataset.p_trials.merge(dataset.p_conditions, how='left', left_on=['patient', 'condition'], right_on=['patient', 'id'])
    candidate_therapies = merged_trials.loc[patients_similarities.index]
    candidate_therapies = candidate_therapies[candidate_therapies.kind == condition_kind]
    candidate_therapies = candidate_therapies[~candidate_therapies.therapy.isin(dataset.p_trials.loc[patient_id].therapy)]
    candidate_therapies = candidate_therapies[['condition', 'therapy', 'successfull']]
    # Add computed similarity to candidate therapies
    candidate_therapies = candidate_therapies.merge(patients_similarities, how='left', left_on='patient', right_on='patient')
    # Compute final score combining similarity and success rate
    candidate_therapies['score'] = candidate_therapies.similarity * (candidate_therapies.successfull/100)
    candidate_therapies = candidate_therapies.drop_duplicates()
    candidate_therapies = candidate_therapies.sort_values('score', ascending=False)
    return patients_similarities, candidate_therapies


if __name__ == '__main__':
    # Script arguments
    parser = argparse.ArgumentParser(description='Therapy recommender system.', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))
    #parser.add_argument('dataset_path', type=str, help='path to the dataset file.', default='./data/generated/dataset.json')
    #parser.add_argument('patient_id', type=int, help='id of the patient for which to compute the recommendation.')
    #parser.add_argument('condition_id', type=str, help='id of the condition of the patient for which to compute the recommendation.')
    
    #TODO: remove
    tests = pd.read_csv('data/generated/test.csv')
    patient_id, condition_id = tests.iloc[0]
    parser.set_defaults(dataset_path='./data/generated/dataset.json')
    parser.set_defaults(patient_id=patient_id)
    parser.set_defaults(condition_id=condition_id)
    #TODO: remove

    args = parser.parse_args()

    # Load dataset
    print('Loading dataset...')
    dataset = Dataset(args.dataset_path)

    # Compute trials similarities
    print('Computing trials history similarities...')
    target_trials, candidate_trials, candidate_therapies1 = trials_similarity(args.patient_id, args.condition_id, dataset)
    
    # Compute patients similarities
    print('Computing patients similarities...')
    patients_similarities, candidate_therapies2 = patients_similarity(args.patient_id, args.condition_id, dataset)
    
    print('Done')

