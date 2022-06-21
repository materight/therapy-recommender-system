import os
import argparse
import numpy as np
import pandas as pd

from recommender.algorithms import (HybridRecommender, NearestNeighborsRecommender, CollaborativeFilteringRecommender, LatentFactorRecommender)
from recommender.dataset import Dataset

if __name__ == '__main__':
    # Script arguments
    parser = argparse.ArgumentParser(description='Therapy recommender system.', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))
    parser.add_argument('--dataset_path', '-d', type=str, help='path to the dataset file.', required=True)
    parser.add_argument('--patient_id', '-p', type=str, help='id of the patient for which to compute the recommendation.', default=None)
    parser.add_argument('--condition_id', '-c', type=str, help='id of the condition of the patient for which to compute the recommendation.', default=None)
    parser.add_argument('--test_file', '-t', type=str, help='path of CSV file with (patient_id, patient_condition_id) tuples to predict for.', default=None)
    parser.add_argument('--num', '-n', type=str, help='number of recommended therapies to return (default: %(default)s).', default=5)
    parser.add_argument('--out', '-o', type=str, help='output file for the computed results', default=None)
    args = parser.parse_args()
    if args.test_file is not None:
        patients = pd.read_csv(args.test_file, sep='\t').astype(str).to_records(index=False)
    elif args.patient_id is not None and args.condition_id is not None:
        patients = [(args.patient_id, args.condition_id)]
    else:
        print('Please set either a value for --patient_id and --condition_id or for --test_file.')
        exit()

    # Load dataset
    print('Loading dataset...')
    dataset = Dataset(args.dataset_path, val_ratio=0)

    # Init hybrid recommender
    print('Init recommender...')
    recommender = HybridRecommender(method='cascade', recommenders=[
        NearestNeighborsRecommender(method='demographic', similarity='hamming', n_neighbors=50), # For patients without registered conditions
        NearestNeighborsRecommender(method='conditions-profile', similarity='jaccard', n_neighbors=50),
        NearestNeighborsRecommender(method='trials-sequence', similarity='levenshtein', n_neighbors=50),
        CollaborativeFilteringRecommender(method='user-user', similarity='pearson', n_neighbors=50),
        CollaborativeFilteringRecommender(method='item-item', similarity='pearson', n_neighbors=50),
        LatentFactorRecommender(method='svd', latent_size=100, epochs=20, lr=0.005, reg=0.02),
        LatentFactorRecommender(method='svd++', latent_size=20, epochs=20, lr=0.005, reg=0.02),
    ])

    # Fit recommender
    print('Fit recommender on dataset...')
    recommender.fit(dataset)
    
    # Predict
    predictions = []
    for patient_id, condition_id in patients:
        print(f'\n\nPredict for patient "{patient_id}" and condition "{condition_id}":')
        prediction = recommender.predict(patient_id, condition_id)
        prediction = dataset.parse_results(prediction, args.num)
        print('Recommendations:\n', prediction.to_string(max_rows=np.inf, index=False))
        prediction['patient_id'], prediction['condition_id']  = patient_id, condition_id
        predictions.append(prediction)

    if args.out is not None:
        predictions = pd.concat(predictions, axis=0)
        predictions = predictions[['patient_id', 'condition_id', 'therapy_id', 'name', 'score']]
        predictions.to_csv(args.out, sep='\t', float_format='%.2f', index=False)

    print('Done')
