import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

from recommender.dataset import Dataset
from recommender.algorithms.utils import BaseRecommender

class HybridRecommender(BaseRecommender):

    # Order of available recommender for the `cascade` hybrid method
    METHODS_ORDER = [('svd++', 'svd'), ('item-item', ), ('user-user', 'trials-sequence'), ('conditions-profile', ), ('demographic', )]

    def __init__(self, method: str, recommenders: List[BaseRecommender]):
        """
        Hybrid recommender, combines multiple recommender systems.

        Args:
            method (str): How the results from the different recommenders are combined. Supported values: 'avg', 'cascade'.
            recommenders (list): List of recommenders systems to combine.
        """
        super().__init__()
        self.method = method
        self.recommenders = recommenders
        self.active_recommenders = None


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

    def set_active_recommenders(self, active_recommenders: List[str]):
        """Set which recommenders should be used for prediction. Set to None to enable all."""
        self.active_recommenders = active_recommenders

    def _combine(self, predictions: pd.DataFrame):
        """Combine multiple predictions into single result."""
        if self.method == 'avg':
            final_predictions = predictions.mean(axis=1, skipna=True)
        elif self.method == 'cascade':
            assert predictions.columns.isin(x for g in self.METHODS_ORDER for x in g).all(), f"One recommender in {predictions.columns} is not supported by the hybrid cascade method."
            final_predictions = pd.Series(np.nan, index=predictions.index)
            for group in self.METHODS_ORDER: 
                group_mean = predictions[predictions.columns.intersection(group)].mean(axis=1, skipna=True) # Compute mean ratings for each group of methods
                final_predictions[final_predictions.isna()] = group_mean[final_predictions.isna()] # Replace missing values with mean of group of current level
        else:
            raise ValueError(f'Method {self.method} is not supported.')
        return final_predictions

    def predict(self, patient_id: str, condition_id: str, therapy_id: str = None, verbose=True):
        results = []
        pbar = tqdm(self.recommenders, disable=not verbose)
        for recommender in pbar:
            if self.active_recommenders is None or recommender.method in self.active_recommenders:
                pbar.set_description(recommender.method)
                result = recommender.predict(patient_id, condition_id, therapy_id)
                results.append(result.rename(recommender.method))
        # Combine predictions into single result
        predictions = pd.concat(results, axis=1)
        predictions = self._combine(predictions)
        return predictions
