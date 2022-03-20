import numpy as np
import pandas as pd

from .utils import BaseRecommender

class CollaborativeFilteringRecommender(BaseRecommender):
    def __init__(self, similarity):
        """
        Collaborative filtering recommender.

        Args:
            similarity (str, optional): Which similarity to use. Supported values: 'levenshtein', 'cosine'.
        """
        super().__init__()
        self.similarity = similarity


    def fit(self, dataset):
        self.dataset = dataset
        # Compute the utility matrix, using as value the 'successful' column. The result is a NxM utility matrix, where:
        # - N is the number of conditions of each patient (i.e. the "users")
        # - M is the number of available therapies (i.e. the "items")
        self.utility = self._get_trials_vectors(dataset.p_trials, dataset.therapies)

    def _get_top_similar_k(self, condition_id, k):
        """Compute the k most similar conditions to the given one."""
        # Get trials of condition_id
        # TODO: distance based n k-grams
        target_features = self.utility[self.utility.index == condition_id]
        other_features = self.utility[self.utility.index != condition_id]
        if self.similarity == 'levenshtein':
            # Compute Jaccard similarity between the given condition and all the other conditions to reduce the number of conditons to consider to compute the Levenshtein distance
            similarities = self._jaccard_similarity(target_features, other_features)
            # Compute sorted sequence of trials from the most similar conditions
            relevant_conditions = similarities.nlargest(k * 10).index
            target_sequence = self._get_trials_sequences(self._get_trials(self.dataset.p_trials, [condition_id]))
            other_sequences = self._get_trials_sequences(self._get_trials(self.dataset.p_trials, relevant_conditions))
            similarities = self._levenshtein_similarity(target_sequence, other_sequences)
        elif self.similarity == 'cosine':
            # TODO: finish implement cosine similarity
            # TODO: add nomrlaization for cosine, subtract aveage value
            similarities = self._cosine_similarity(target_features, other_features)
        top_k = similarities.nlargest(k)
        return top_k

    def _predict_ratings(self, condition_id, top_k):
        """Predict the therapy ratings using the given k most similar conditions."""
        top_k_ratings = self.utility.loc[top_k.index] # Get the ratings of the top k similar conditions
        # Generate matrix of weigths to be applied. Weights are the similairty scores
        weights = pd.concat([top_k] * len(self.dataset.therapies), axis=1)
        weights.columns = self.dataset.therapies.index # 
        weights[top_k_ratings == 0] = 0 # Set weights of empty ratings to 0
        # Compute weighted average of weights
        weighted_ratings = top_k_ratings * weights # Multiply every rating by the similarity score
        pred_ratings = weighted_ratings.sum(axis='index') / weights.sum(axis='index')
        return pred_ratings

    def predict(self, patient_id, condition_id, k=50):
        # Get top-k conditions
        top_k = self._get_top_similar_k(condition_id, k)
        # Compute ratings predictions
        pred_ratings = self._predict_ratings(condition_id, top_k)