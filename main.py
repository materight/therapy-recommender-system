import argparse
import numpy as np
import pandas as pd

from recommender.algorithms import HybridRecommender, CollaborativeFilteringRecommender
from recommender.dataset import Dataset

def svd(p_trials):
    pt_matrix = p_trials.pivot_table(index='patient', columns=['condition', 'therapy'], values='successfull', fill_value=0) # Compute matrix of patient-therapies for SVD
    u, s, vh = np.linalg.svd(pt_matrix.values, full_matrices=False)
    reconstructed = (u * s) @ vh
    return u, s, vh

if __name__ == '__main__':
    """
    New ideas to try:
    - Content-based: define a profile for therapies, compute a profile for each condition as average of the therapies applied for it (weigthed by their success), and finally
                     compute a patient profile as average of the profiles of the conditions he has. Then, suggest a therapy based on the distance between the patient profile or the
                     condition profile.
    """

    # Script arguments
    parser = argparse.ArgumentParser(description='Therapy recommender system.', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))
    #parser.add_argument('dataset_path', type=str, help='path to the dataset file.', default='./data/generated/dataset.json')
    #parser.add_argument('patient_id', type=str, help='id of the patient for which to compute the recommendation.')
    #parser.add_argument('condition_id', type=str, help='id of the condition of the patient for which to compute the recommendation.')
    
    #TODO: remove
    dataset_path = './data/generated'
    tests = pd.read_csv(f'{dataset_path}/test.csv', sep='\t')
    patient_id, condition_id = tests.iloc[0]
    parser.set_defaults(dataset_path=f'{dataset_path}/dataset.json')
    parser.set_defaults(patient_id=str(patient_id))
    parser.set_defaults(condition_id=str(condition_id))
    #TODO: remove

    args = parser.parse_args()

    # Load dataset
    print('Loading dataset...')
    dataset = Dataset(args.dataset_path)

    print('Init recommender...')
    recommender = HybridRecommender([
        CollaborativeFilteringRecommender(similarity='levenshtein', n_neighbors=50)
    ])

    print('Fit recommender on dataset...')
    recommender.fit(dataset)
    
    print('Run prediction...')
    results = recommender.predict(args.patient_id, args.condition_id)

    print('Done')

