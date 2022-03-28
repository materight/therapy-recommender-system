import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit

from recommender.dataset import Dataset
from recommender.algorithms.utils import BaseRecommender


@njit
def _svd(ratings, indptr, indices, n_users, n_items, latent_size, epochs, lr, reg, with_implicit_ratings):
    """Funk SVD algoritm using SGD optimization. Source: https://github.com/NicolasHug/Surprise/blob/46b9914995e6c8c7d227b46f2eaeef2d4600580f/surprise/prediction_algorithms/matrix_factorization.pyx#L159"""
    # TODO: fix NaN results with svd++
    # Initialize matrices
    np.random.seed(0)
    global_mean = ratings.mean()
    bu = np.zeros(n_users, np.double) # User bias
    bi = np.zeros(n_items, np.double) # Item bias
    P = np.random.normal(0, 0.1, (n_users, latent_size))
    Q = np.random.normal(0, 0.1, (n_items, latent_size))
    Y = np.random.normal(0, 0.1, (n_items, latent_size)) # Item implicit factors
    # Run optimization
    for _ in range(epochs):
        for u in range(len(indptr)-1): # For each user
            s, e = indptr[u], indptr[u+1] # Get index pointers for current row
            if with_implicit_ratings:
                Iu = indices[s:e] # Items rated by u
                sqrt_Iu = np.sqrt(len(Iu))
            # Iterate over each rating value in each column
            for r, i in zip(ratings[s:e], indices[s:e]):
                # Compute implicit feedback of current user if needed, oterwise set to 0
                impl_fdb = Y[Iu].sum(axis=0) / sqrt_Iu if with_implicit_ratings else np.zeros(latent_size, np.double)
                # Compute current error
                dot = np.dot(P[u] + impl_fdb, Q[i])
                err = r - (global_mean + bu[u] + bi[i] + dot) # Compute prediction error
                # Update biases
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])
                # Update factors
                for f in range(latent_size):
                    puf, qif = P[u, f], Q[i, f]
                    P[u, f] += lr * (err * qif - reg * puf)
                    Q[i, f] += lr * (err * (puf + impl_fdb[f]) - reg * qif)
                    if with_implicit_ratings:
                        for j in Iu:
                            Y[j, f] += lr * (err * qif / sqrt_Iu - reg * Y[j, f])
    # Compute final implicit feedbacks of users
    F = np.zeros((n_users, latent_size), np.double)
    if with_implicit_ratings:
        for u in range(len(indptr)-1):
            Iu = indices[indptr[u]:indptr[u+1]]
            F[u] = Y[Iu].sum(axis=0) / np.sqrt(len(Iu))
    # Compute final predictions with dot product and baselines
    baseline_estimate = global_mean + (bu.reshape(-1,1) + bi.reshape(1,-1))
    predictions = baseline_estimate + ((P + F) @ Q.T)
    return predictions


class LatentFactorRecommender(BaseRecommender):
    def __init__(self, method: str, latent_size: int = 100, epochs: int = 20, lr: float = 0.005, reg: float = 0.02):
        """
        Collaborative filtering recommender.

        Args:
            method (str): Which algorithm to use. Supported values: 'svd', 'svd++'.
            latent_size (int): Number of latent factors to compute.
            epochs (int): Number of iterations to run.
            lr (float): Learning rate.
            reg (float): Regularization parameter.
        """
        super().__init__()
        self.method = method
        self.latent_size = latent_size
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.rng = np.random.RandomState(0)


    def init_state(self, utility_matrix: pd.DataFrame, **kwargs):
        self.utility_matrix = utility_matrix


    def fit(self, dataset: Dataset):
        if self.method == 'svd':
            R = self.utility_matrix.astype(pd.SparseDtype(float, np.nan)).sparse.to_coo().tocsr() # Convert to scipy sparse matrix for easy iteration
            predictions = _svd(R.data, R.indptr, R.indices, R.shape[0], R.shape[1], self.latent_size, self.epochs, lr=self.lr, reg=self.reg, with_implicit_ratings=False)
        elif self.method == 'svd++':
            R = self.utility_matrix.astype(pd.SparseDtype(float, np.nan)).sparse.to_coo().tocsr() # Convert to scipy sparse matrix for easy iteration
            predictions = _svd(R.data, R.indptr, R.indices, R.shape[0], R.shape[1], self.latent_size, self.epochs, lr=self.lr, reg=self.reg, with_implicit_ratings=True)
        else:
            raise ValueError(f'Unknown method: {self.method}')
        self.predictions = pd.DataFrame(predictions, index=self.utility_matrix.index, columns=self.utility_matrix.columns)


    def predict(self, patient_id: str, condition_id: str):
        # https://towardsdatascience.com/introduction-to-latent-matrix-factorization-recommender-systems-8dfc63b94875
        pred_ratings = self.predictions.loc[condition_id]
        return pred_ratings
