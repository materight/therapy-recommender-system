import argparse
import pandas as pd

from recommender.algorithms import (HybridRecommender, NearestNeighborsRecommender, CollaborativeFilteringRecommender, LatentFactorRecommender)
from recommender.dataset import Dataset

if __name__ == '__main__':
    """
    New ideas to try:
    - Content-based: define a profile for therapies, compute a profile for each condition as average of the therapies applied for it (weigthed by their success), and finally
                     compute a patient profile as average of the profiles of the conditions he has. Then, suggest a therapy based on the distance between the patient profile or the
                     condition profile.
    - Dimensionality reduction: Try SVD but also Non-negative Matrix Factorization
    - Collaborative filtering: try also item-based
    - Handle case when the patient has no trials available for a given condition
    - Hybrid: check https://www.quora.com/How-can-I-combine-the-recommendation-results-from-User-based-collaborative-and-item-based-and-return-the-best-of-these-two-using-Apache-Mahout

    Eval:
    - Do a ablation study removing different recommenders
    - RMSE
    - MAE
    - NDCG (check: https://benjlindsay.com/posts/comparing-collaborative-filtering-methods#algorithm-comparisons)
    """

    # Script arguments
    parser = argparse.ArgumentParser(description='Therapy recommender system.', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))
    #parser.add_argument('dataset_path', type=str, help='path to the dataset file.', default='./data/generated/dataset.json')
    #parser.add_argument('patient_id', type=str, help='id of the patient for which to compute the recommendation.')
    #parser.add_argument('condition_id', type=str, help='id of the condition of the patient for which to compute the recommendation.')
    
    #TODO: remove
    dataset_path = './data/generated'
    tests = pd.read_csv(f'{dataset_path}/test.csv', sep='\t')
    patient_id, condition_id = tests.iloc[2]
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
        #NearestNeighborsRecommender(method='patient-profile', similarity='hamming', n_neighbors=50), # For patients without registered conditions
        #NearestNeighborsRecommender(method='trials-sequence', similarity='levenshtein', n_neighbors=50),
        #CollaborativeFilteringRecommender(method='user-user', similarity='pearson', n_neighbors=50),
        #CollaborativeFilteringRecommender(method='item-item', similarity='pearson', n_neighbors=50),
        LatentFactorRecommender(method='svd')
    ])

    print('Fit recommender on dataset...')
    recommender.fit(dataset)
    
    print('Run prediction...')
    results = recommender.predict(args.patient_id, args.condition_id)

    print('Done')

