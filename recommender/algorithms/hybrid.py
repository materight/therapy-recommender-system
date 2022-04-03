import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

from recommender.dataset import Dataset
from recommender.algorithms.utils import BaseRecommender

class HybridRecommender(BaseRecommender):
    def __init__(self, method: str, recommenders: List[BaseRecommender]):
        """
        Hybrid recommender, combines multiple recommender systems.

        Args:
            method (str): How the results from the different recommenders are combined. Supported values: 'avg', 'weighted', 'mixed'.
            recommenders (list): List of recommenders systems to combine.
        """
        super().__init__()
        self.method = method
        self.recommenders = recommenders


    def fit(self, dataset: Dataset):
        # Compute utility matrix (common for all recommenders)
        utility_matrix = self._get_utility_matrix(dataset.p_trials, dataset.therapies)
        # Compute global baseline estimates
        global_baseline = self._get_baseline_estimates(utility_matrix, dataset.p_conditions)
        # Fit recommenders on dataset
        pbar = tqdm(self.recommenders)
        for recommender in pbar:
            pbar.set_description(recommender.method)
            recommender.init_state(utility_matrix=utility_matrix, global_baseline=global_baseline)
            recommender.fit(dataset)

    def _combine(self, predictions: pd.DataFrame):
        """Combine multiple predictions into single result."""
        if self.method == 'avg':
            predictions = predictions.mean(axis=1, skipna=True)
        else:
            raise ValueError(f'Method {self.method} is not supported.')
        return predictions

    def predict(self, patient_id: str, condition_id: str, verbose=True):
        # TODO: rank aggregation (check: https://people.orie.cornell.edu/dpw/talks/RankAggDec2012.pdf)
        results = []
        pbar = tqdm(self.recommenders, disable=not verbose)
        for recommender in pbar:
            pbar.set_description(recommender.method)
            result = recommender.predict(patient_id, condition_id)
            results.append(result)
        # Combine predictions into single result
        predictions = pd.concat(results, axis=1)
        predictions = self._combine(predictions)
        return predictions
