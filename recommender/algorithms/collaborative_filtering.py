import numpy as np

from .utils import BaseRecommender

class CollaborativeFilteringRecommender(BaseRecommender):
    """
    Build a NxM utility matrix, where:
    - N is the number of conditions of each patient (i.e. the "users")
    - M is the number of available therapies (i.e. the "items")
    The similarity between conditions is computed using a string distance metric, to take into account the order of the applied therapies.
    """
    def fit(self, dataset):
        users_col, items_col = 'condition', 'therapy'
        self.utility = dataset.p_trials.pivot_table(index=users_col, columns=items_col, values='successful', fill_value=0)


    def _get_top_similar_k(self, condition_id):
        pass

    def predict(self, patient_id, condition_id):
        pass
