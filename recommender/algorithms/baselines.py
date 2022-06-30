import numpy as np
import pandas as pd

from recommender.dataset import Dataset
from recommender.algorithms.utils import BaseRecommender

class BaselineRecommender(BaseRecommender):
    def __init__(self, method: str):
        """
        Baseline recommenders.

        Args:
            method (str): Type of baseline. Supported values: 'random', 'mean'.
        """
        super().__init__()
        self.method = method


    def fit(self, dataset: Dataset):
        self.dataset = dataset
        if self.method == 'mean':
            self.global_mean_scores = self.dataset.p_trials.groupby('therapy')['successful'].mean()


    def predict(self, patient_id: str, condition_id: str, therapy_id: str = None, verbose=True):
        if self.method == 'random':
            random_scores = np.random.randint(0, 100, len(self.dataset.therapies.index))
            return pd.Series(random_scores, index=self.dataset.therapies.index)
        elif self.method == 'mean':
            target_condition_kind = self.dataset.p_conditions[self.dataset.p_conditions.index.get_level_values('id') == condition_id].kind.iloc[0]
            relevant_conditions_ids = self.dataset.p_conditions[self.dataset.p_conditions.kind == target_condition_kind].index.get_level_values('id')
            relevant_trials = self.dataset.p_trials[self.dataset.p_trials.condition.isin(relevant_conditions_ids)]
            mean_scores = relevant_trials.groupby('therapy')['successful'].mean()
            mean_scores.fillna(self.global_mean_scores)
            return mean_scores
        else:
            raise ValueError(f'Method {self.method} is not supported.')
