import numpy as np
import pandas as pd

from .utils import BaseRecommender

class CollaborativeFilteringRecommender(BaseRecommender):
    def __init__(self, similarity: str, n_neighbors: int):
        """
        Collaborative filtering recommender.

        Args:
            similarity (str): Which similarity to use. Supported values: 'levenshtein', 'cosine'.
            n_neightbors (int): Number of similar conditions to consider when computing the rating average.
        """
        super().__init__()
        self.similarity = similarity
        self.n_neighbors = n_neighbors

    def fit(self, dataset):
        self.dataset = dataset
        # Compute the utility matrix, using as value the 'successful' column. The result is a NxM utility matrix, where:
        # - N is the number of conditions of each patient (i.e. the "users")
        # - M is the number of available therapies (i.e. the "items")
        self.utility = self._get_utility_matrix(dataset.p_trials, dataset.therapies)
        # Compute global baseline estimates
        global_avg_rating = np.nanmean(self.utility)
        users_rating_deviation = global_avg_rating - self.utility.mean(axis=1, skipna=True).values
        items_rating_deviation = global_avg_rating - self.utility.mean(axis=0, skipna=True).values
        self.global_baseline = global_avg_rating + (users_rating_deviation.reshape(-1,1) + items_rating_deviation.reshape(1,-1))
        self.global_baseline = pd.DataFrame(self.global_baseline, index=self.utility.index, columns=self.utility.columns)

    def _get_top_similar_k(self, condition_id: str):
        """Compute the k most similar conditions to the given one."""
        # Get trials of condition_id
        target_features = self.utility[self.utility.index == condition_id]
        other_features = self.utility[self.utility.index != condition_id]
        if self.similarity == 'levenshtein':
            # Compute Jaccard similarity between the given condition and all the other conditions to reduce the number of conditons to consider to compute the Levenshtein distance
            similarities = self._jaccard_similarity(target_features, other_features)
            # Compute sorted sequence of trials from the most similar conditions
            relevant_conditions = similarities.nlargest(self.n_neighbors * 10).index # TODO: change this k*10 to something else, understand relation between Jaccard distance and edit distance
            target_sequence = self._get_trials_sequences(self._get_trials(self.dataset.p_trials, [condition_id]))
            other_sequences = self._get_trials_sequences(self._get_trials(self.dataset.p_trials, relevant_conditions))
            similarities = self._levenshtein_similarity(target_sequence, other_sequences)
        elif self.similarity == 'cosine':
            similarities = self._cosine_similarity(target_features, other_features)
        top_k = similarities.nlargest(self.n_neighbors)
        return top_k

    def _predict_ratings(self, condition_id: str, top_k_similarities: pd.DataFrame):
        """Predict the therapy ratings using the given k most similar conditions."""
        top_k_ratings = self.utility.loc[top_k_similarities.index] # Get the ratings of the top k similar conditions
        # Generate matrix of weigths to be applied. Weights are the similairty scores
        weights = np.tile(top_k_similarities.values.reshape(-1, 1), len(top_k_ratings.columns))
        weights = pd.DataFrame(weights, index=top_k_ratings.index, columns=top_k_ratings.columns)
        weights[top_k_ratings.isna()] = 0 # Set weights of empty ratings to 0
        # Compute weighted average of rating (with global baseline estimate)
        weighted_ratings = weights * (top_k_ratings - self.global_baseline.loc[top_k_ratings.index])  # Multiply every rating by the similarity score
        pred_ratings = weighted_ratings.sum(axis='index') / weights.sum(axis='index')
        pred_ratings = pred_ratings + self.global_baseline.loc[condition_id]
        pred_ratings = pred_ratings.fillna(0)
        pred_ratings = pred_ratings.clip(0, 100)
        return pred_ratings

    def predict(self, patient_id: str, condition_id: str):
        # Get top-k conditions
        top_k_similarities = self._get_top_similar_k(condition_id)
        # Compute ratings predictions
        pred_ratings = self._predict_ratings(condition_id, top_k_similarities)
        return pred_ratings
