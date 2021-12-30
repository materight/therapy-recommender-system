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
    # Compute final scaore combining similarity and success reate
    candidate_therapies['score'] = candidate_therapies.similarity * (candidate_therapies.successfull/100)
    candidate_therapies = candidate_therapies.sort_values('score', ascending=False)
    return target_trials, candidate_trials, candidate_therapies


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

    # # Merge everything
    # merged = (dataset.p_trials.reset_index()
    #     .merge(dataset.p_conditions, how='left', left_on=['patient', 'condition'], right_on=['patient', 'id'])
    #     .merge(dataset.conditions.add_prefix('condition_'), how='left', left_on=['kind'], right_on=['id']).drop(['condition_name'], axis=1)
    #     .merge(dataset.therapies.add_prefix('therapy_'), how='left', left_on=['therapy'], right_on=['id']).drop(['therapy_name'], axis=1)
    #     .merge(dataset.patients.add_prefix('patient_'), how='left', left_on=['patient'], right_on=['id']).drop(['patient_name'], axis=1)
    # ).set_index(['patient', 'id'])
    # # Mine frequent itemsets
    # from mlxtend.frequent_patterns import apriori
    # from scipy.sparse import csr_matrix
    # trials_itemset = merged.pivot_table(index=['patient', 'condition'], columns='therapy', values='successfull')
    # apriori_input = pd.DataFrame.sparse.from_spmatrix(csr_matrix(trials_itemset.fillna(0).astype(bool).values), index=trials_itemset.index, columns=trials_itemset.columns)
    # frequent_therapies = apriori(apriori_input, min_support=0.1, use_colnames=True)

    # Compute trials distances
    print('Computing trials history similarities...')
    target_trials, candidate_trials, candidate_therapies = trials_similarity(args.patient_id, args.condition_id, dataset)
    
    
    
    print('Done')

