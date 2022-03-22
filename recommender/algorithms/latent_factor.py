import numpy as np
import pandas as pd

from .utils import BaseRecommender

class LatentFactorRecommender(BaseRecommender):
    def __init__(self, algorithm: str):
        """
        Collaborative filtering recommender.

        Args:
            algorithm (str): Which algorithm to use. Supported values: 'funk_svd'.
        """
        super().__init__()
        self.algorithm = algorithm

    def fit(self, dataset):
        # Compute utility matrix
        self.utility = self._get_utility_matrix(dataset.p_trials, dataset.therapies)
        # Compute normalization factor as average rating of users
        avg_rating = np.nanmean(self.utility, axis=1, keepdims=True)
        # Use SVD decomposition to compute the latent factors
        matrix = self.utility.values - avg_rating
        matrix[np.isnan(matrix)] = 0 # Fill nan values with average rating
        # TODO: use funk_svd
        u, s, v = np.linalg.svd(matrix, full_matrices=False)
        self.predictions = (u * s) @ v + avg_rating
        self.predictions = pd.DataFrame(self.predictions, index=self.utility.index, columns=self.utility.columns)

    def predict(self, patient_id: str, condition_id: str):
        pred_ratings = self.predictions.loc[condition_id]
        return pred_ratings
