import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

from .utils import BaseRecommender 

class HybridRecommender(BaseRecommender):
    def __init__(self, recommenders: List[BaseRecommender]):
        """
        Hybrid recommender, combines multiple recommender systems.

        Args:
            recommenders: List of recommenders systems to combine.
        """
        super().__init__()
        self.recommenders = recommenders


    def fit(self, dataset):
        # Compute utility matrix (common for all recommenders)
        utility_matrix = self._get_utility_matrix(dataset.p_trials, dataset.therapies)
        # Compute global baseline estimates
        global_baseline = self._get_baseline_estimates(utility_matrix)
        # Fit recommenders on dataset
        for recommender in tqdm(self.recommenders):
            recommender.init_state(utility_matrix=utility_matrix, global_baseline=global_baseline)
            recommender.fit(dataset)

    def predict(self, patient_id: str, condition_id: str):
        # TODO: rank aggregation (check: https://people.orie.cornell.edu/dpw/talks/RankAggDec2012.pdf)
        results = []
        for recommender in tqdm(self.recommenders):
            result = recommender.predict(patient_id, condition_id)
            results.append(result)
        # Combine predictions into single result
        predictions = pd.concat(results, axis=1)
        predictions = predictions.mean(axis=1)
        return predictions