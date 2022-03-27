import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit

from recommender.dataset import Dataset
from recommender.algorithms.utils import BaseRecommender


@njit
def _svd(ratings, rows, cols, matrix_shape, latent_size, epochs, lr, reg):
    """Funk SVD algorithm. Source: https://github.com/NicolasHug/Surprise/blob/46b9914995e6c8c7d227b46f2eaeef2d4600580f/surprise/prediction_algorithms/matrix_factorization.pyx#L159"""
    # Initialize matrices
    np.random.seed(0)
    global_mean = ratings.mean()
    bu = np.zeros(matrix_shape[0], np.double) # User bias
    bi = np.zeros(matrix_shape[1], np.double) # Item bias
    P = np.random.normal(0, 0.1, (matrix_shape[0], latent_size))
    Q = np.random.normal(0, 0.1, (matrix_shape[1], latent_size))
    # Run optimization
    for _ in range(epochs):
        for r, u, i in zip(ratings, rows, cols):
            # Compute current error
            dot_prod = np.dot(P[u], Q[i])
            err = r - (global_mean + bu[u] + bi[i] + dot_prod) # Compute prediction with bias terms
            # Update biases
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            # Update factors
            for f in range(latent_size):
                puf = P[u, f]
                qif = Q[i, f]
                P[u, f] += lr * (err * qif - reg * puf)
                Q[i, f] += lr * (err * puf - reg * qif)
    # Compute final predictions as dot product
    baseline_estimate = global_mean + (bu.reshape(-1,1) + bi.reshape(1,-1))
    predictions = (P @ Q.T) + baseline_estimate
    return predictions


class LatentFactorRecommender(BaseRecommender):
    def __init__(self, method: str, latent_size: int = 100, epochs: int = 20):
        """
        Collaborative filtering recommender.

        Args:
            method (str): Which algorithm to use. Supported values: 'svd'.
            latent_size (int): Number of latent factors to compute.
        """
        super().__init__()
        self.method = method
        self.latent_size = latent_size
        self.epochs = epochs
        self.rng = np.random.RandomState(0)


    def init_state(self, utility_matrix: pd.DataFrame, **kwargs):
        self.utility_matrix = utility_matrix


    def fit(self, dataset: Dataset):
        if self.method == 'svd':
            R = self.utility_matrix.astype(pd.SparseDtype(float, np.nan)).sparse.to_coo() # Convert to scipy sparse matrix for easy iteration
            predictions = _svd(R.data, R.row, R.col, R.shape, self.latent_size, self.epochs, lr=0.005, reg=0.02)
        else:
            raise ValueError(f'Unknown method: {self.method}')
        self.predictions = pd.DataFrame(predictions, index=self.utility_matrix.index, columns=self.utility_matrix.columns)


    def predict(self, patient_id: str, condition_id: str):
        # https://towardsdatascience.com/introduction-to-latent-matrix-factorization-recommender-systems-8dfc63b94875
        pred_ratings = self.predictions.loc[condition_id]
        return pred_ratings
