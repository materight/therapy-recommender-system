import numpy as np
import pandas as pd
from tqdm import tqdm

from .nearest_neighbors import NearestNeighborsRecommender

class CollaborativeFilteringRecommender(NearestNeighborsRecommender):
    def __init__(self, method: str, similarity: str, n_neighbors: int):
        """
        Collaborative filtering recommender.

        Args:
            method (str): Type of collaborative filtering. 
                          Supported values: 'user-user', 'item-item'.
            similarity (str): Which similarity metric to use.
                              Supported values: 'jaccard', 'pearson'.
            n_neightbors (int): Number of similar objects to consider when computing the rating average.
        """
        super().__init__(method, similarity, n_neighbors)


    def init_state(self, utility_matrix, global_baseline, **kwargs):
        super().init_state(utility_matrix, global_baseline, **kwargs)
        if self.method == 'item-item':
            self.utility_matrix = self.utility_matrix.T
            self.global_baseline = self.global_baseline.T


    def fit(self, dataset):
        self.dataset = dataset


    def _get_features(self, object_id: str):
        # Split features into target and others
        target_features = self.utility_matrix.loc[self.utility_matrix.index == object_id]
        others_features = self.utility_matrix.loc[self.utility_matrix.index != object_id]
        return target_features, others_features


    def _predict_user_user(self, condition_id: str):
        target_features, other_features = self._get_features(condition_id) # Compute features
        neighbors_similarities = self._get_neighbors(target_features, other_features)  # Get neighbors with similarity values
        pred_ratings = self._predict_ratings(condition_id, neighbors_similarities)  # Compute ratings predictions
        return pred_ratings


    def _predict_item_item(self, condition_id: str):
        # Predict ratings item by item and aggregate results for the given condition
        pred_ratings = pd.Series(index=self.utility_matrix.index)
        for therapy_id in tqdm(self.utility_matrix.index):
            target_features, other_features = self._get_features(therapy_id) # Compute features
            neighbors_similarities = self._get_neighbors(target_features, other_features) # Get neighbors with similarity values
            therapy_pred_ratings = self._predict_ratings(therapy_id, neighbors_similarities) # Compute ratings predictions for the current therapy
            pred_ratings.loc[therapy_id] = therapy_pred_ratings.loc[condition_id]
        return pred_ratings


    def predict(self, patient_id: str, condition_id: str):
        if self.method == 'user-user':
            return self._predict_user_user(condition_id)
        elif self.method == 'item-item':
            return self._predict_item_item(condition_id)
        else:
            raise ValueError(f'Method {self.method} is not supported.')
