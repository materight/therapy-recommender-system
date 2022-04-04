import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

from recommender.algorithms import (HybridRecommender, NearestNeighborsRecommender, CollaborativeFilteringRecommender, LatentFactorRecommender)
from recommender.dataset import Dataset

"""TODO: 
    - add NDCG metric for evaluation
    - implement cross validation with multiple folds
"""

def evaluate(dataset: Dataset, recommender: HybridRecommender, active_recommenders: List[str]):
    print(", ".join(active_recommenders))
    recommender.set_active_recommenders(active_recommenders)
    n_samples = len(dataset.val_trials)
    rmse, mae = 0, 0
    for trial in tqdm(dataset.val_trials.reset_index().itertuples(), total=n_samples):
        prediction = recommender.predict(trial.patient, trial.condition, trial.therapy, verbose=False)
        prediction = prediction.loc[trial.therapy]
        diff = prediction - trial.successful if not np.isnan(prediction) else 0 # penalize missing predictions
        rmse += np.power(diff, 2)
        mae += np.abs(diff)
    rmse = np.sqrt(rmse / n_samples)
    mae = mae / n_samples
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'----------------------')
    return rmse, mae

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
    recommender = HybridRecommender(method='cascade', recommenders=[
        NearestNeighborsRecommender(method='demographic', similarity='hamming', n_neighbors=50), # For patients without registered conditions
        NearestNeighborsRecommender(method='conditions-profile', similarity='jaccard', n_neighbors=50),
        NearestNeighborsRecommender(method='trials-sequence', similarity='levenshtein', n_neighbors=50),
        CollaborativeFilteringRecommender(method='user-user', similarity='pearson', n_neighbors=50),
        CollaborativeFilteringRecommender(method='item-item', similarity='pearson', n_neighbors=50),
        LatentFactorRecommender(method='svd', latent_size=100, epochs=20, lr=0.005, reg=0.02),
        LatentFactorRecommender(method='svd++', latent_size=20, epochs=20, lr=0.005, reg=0.02),
    ])

    print('\nFit recommender on dataset...')
    recommender.fit(dataset)
    
    print('\nSingle algorithms evaluation...')
    results = []
    algorithms = [ rec.method for rec in recommender.recommenders ]
    for i, algo in enumerate(algorithms, start=1):
        print(f'\n[{i}/{len(algorithms)}] ', end='')
        rmse, mae = evaluate(dataset, recommender, [ algo ])
        results.append(('single', algo, rmse, mae))

    print('\nAblation evaluation...')
    for i, algo in enumerate(algorithms, start=1):
        print(f'\n[{i}/{len(algorithms)}] ', end='')
        rmse, mae = evaluate(dataset, recommender, [ a for a in algorithms if a != algo ])
        results.append(('ablation', algo, rmse, mae))

    print('\nSaving results...')
    df = pd.DataFrame(results, columns=['type', 'algorithm', 'rmse', 'mae'])
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    dataset_dirname = os.path.basename(os.path.dirname(os.path.abspath(args.dataset_path)))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f'{dataset_dirname}_benchmark.csv'), index=False, sep='\t', float_format='%.4f')
