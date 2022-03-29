import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from recommender.algorithms import (HybridRecommender, NearestNeighborsRecommender, CollaborativeFilteringRecommender, LatentFactorRecommender)
from recommender.dataset import Dataset

if __name__ == '__main__':
    # Script arguments
    parser = argparse.ArgumentParser(description='Run a benchmark evaluation for the therapy recommender system.', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))
    #parser.add_argument('dataset_path', type=str, help='path to the dataset file.', default='./data/generated/dataset.json')
    parser.add_argument('--val_split', type=float, help='fraction of trials samples to be used for validation.', default=0.1)
    #TODO: remove
    dataset_path = './data/generated'
    parser.set_defaults(dataset_path=f'{dataset_path}/dataset.json')
    #TODO: remove
    args = parser.parse_args()

    # Load dataset
    print('Loading dataset...')
    dataset = Dataset(args.dataset_path, val_ratio=args.val_split)

    print('Init recommender...')
    recommender = HybridRecommender([
        #NearestNeighborsRecommender(method='patient-profile', similarity='hamming', n_neighbors=50), # For patients without registered conditions
        #NearestNeighborsRecommender(method='trials-sequence', similarity='levenshtein', n_neighbors=50),
        #CollaborativeFilteringRecommender(method='user-user', similarity='pearson', n_neighbors=50),
        #CollaborativeFilteringRecommender(method='item-item', similarity='pearson', n_neighbors=50),
        LatentFactorRecommender(method='svd', latent_size=100, epochs=20, lr=0.005, reg=0.02),
        LatentFactorRecommender(method='svd++', latent_size=20, epochs=20, lr=0.005, reg=0.02),
    ])

    print('Fit recommender on dataset...')
    recommender.fit(dataset)
    
    print('Evaluate over validation split...')
    K = 10 # top-k accuracy evaluation
    n_samples = len(dataset.val_trials)
    rmse, mae = 0, 0
    top_k = dict(zip(np.arange(1,K+1), np.zeros(K)))
    for trial in tqdm(dataset.val_trials.reset_index().itertuples(), total=n_samples):
        prediction = recommender.predict(trial.patient, trial.condition, verbose=False)
        diff = prediction.loc[trial.therapy] - trial.successful
        rmse += np.power(diff, 2)
        mae += np.abs(diff)
        nlargest = prediction.nlargest(K)
        for k in top_k:
            top_k[k] += trial.therapy in nlargest[:k]
    rmse = np.sqrt(rmse / n_samples)
    mae = mae / n_samples
    top_k = {k: (v / n_samples) * 100 for k, v in top_k.items() }
    print('\nEvaluation results:')
    print(f'- RMSE: {rmse:.4f}')
    print(f'- MAE: {mae:.4f}')
    print(f'- Top-1: {top_k[1]:.2f}%')
    print(f'- Top-5: {top_k[5]:.2f}%')
    print(f'- Top-10: {top_k[10]:.2f}%')
