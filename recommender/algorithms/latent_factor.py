import numpy as np
import pandas as pd

from recommender.dataset import Dataset
from recommender.algorithms.utils import BaseRecommender

class LatentFactorRecommender(BaseRecommender):
    def __init__(self, method: str):
        """
        Collaborative filtering recommender.

        Args:
            method (str): Which algorithm to use. Supported values: 'funk_svd'.
        """
        super().__init__()
        self.method = method


    def init_state(self, utility_matrix: pd.DataFrame, **kwargs):
        self.utility_matrix = utility_matrix


    def fit(self, dataset: Dataset):
        # Compute normalization factor as average rating of users
        avg_rating = np.nanmean(self.utility_matrix, axis=1, keepdims=True)
        # Use SVD decomposition to compute the latent factors
        matrix = self.utility_matrix.values - avg_rating
        matrix[np.isnan(matrix)] = 0 # Fill nan values with average rating
        # TODO: use funk_svd
        u, s, v = np.linalg.svd(matrix, full_matrices=False)
        self.predictions = (u * s) @ v + avg_rating
        self.predictions = pd.DataFrame(self.predictions, index=self.utility_matrix.index, columns=self.utility_matrix.columns)


    def predict(self, patient_id: str, condition_id: str):
        pred_ratings = self.predictions.loc[condition_id]
        return pred_ratings
