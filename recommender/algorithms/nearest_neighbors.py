import numpy as np
import pandas as pd
from sklearn import neighbors

from recommender.dataset import Dataset
from recommender.algorithms.utils import BaseRecommender
class NearestNeighborsRecommender(BaseRecommender):
    def __init__(self, method: str, similarity: str, n_neighbors: int):
        """
        Nearest neighbors recommender.

        Args:
            method (str): Type of features to be used to compute distances between objects. 
                          Supported values: 'demographic', 'trials-sequence'.
            similarity (str): Which similarity metric to use.
                              Supported values: 'jaccard', 'pearson', 'levenshtein'.
            n_neightbors (int): Number of similar objects to consider when computing the rating average.
        """
        super().__init__()
        self.method = method
        self.similarity = similarity
        self.n_neighbors = n_neighbors
        self.utility_matrix = None
        self.global_baseline = None
        self.features = None


    def init_state(self, utility_matrix: pd.DataFrame, global_baseline: pd.DataFrame, **kwargs):
        self.utility_matrix = utility_matrix
        self.global_baseline = global_baseline


    def fit(self, dataset: Dataset):
        self.dataset = dataset
        # Compute features to be used to find similar objects
        if self.method == 'demographic': # Demographic approach
            self.features = self._build_patients_demographic_profiles(self.dataset.patients)
        elif self.method == 'conditions-profile':
            self.features = self._build_patients_conditions_profiles(self.dataset.p_conditions, self.dataset.conditions)
        elif self.method == 'trials-sequence':
            self.features = self._get_trials_sequences(self.dataset.p_trials)
        else: 
            raise ValueError(f'Method {self.method} is not supported.')


    def _get_features(self, patient_id: str, condition_id: str):
        """Get the features according to the given features type."""
        if self.method in ['demographic', 'conditions-profile']:
            # Remap patients profiles to conditions indexes, using only conditions of the same type as the target condition
            target_condition_kind = self.dataset.p_conditions[self.dataset.p_conditions.index.get_level_values('id') == condition_id].kind.iloc[0]
            relevant_conditions = self.dataset.p_conditions[self.dataset.p_conditions.kind == target_condition_kind].reset_index()
            final_features = relevant_conditions[['id', 'patient']].merge(self.features, left_on='patient', right_on='id', how='left').set_index('id').drop('patient', axis=1)
            # Re-map patients features to conditions indexes
            target_features = final_features[final_features.index == condition_id]
            others_features = final_features[final_features.index != condition_id]
        elif self.method == 'trials-sequence':
            # Compute Jaccard similarity between the given condition and all the others to reduce the number of conditons to consider when computing the Levenshtein distance
            utility_mask = self.utility_matrix.index == condition_id
            jaccard_similarities = self._jaccard_similarity(self.utility_matrix[utility_mask], self.utility_matrix[~utility_mask])
            # Compute sorted sequence of trials from the most similar conditions
            relevant_conditions = jaccard_similarities.nlargest(self.n_neighbors * 10).index
            target_features = self.features[self.features.index == condition_id]
            others_features = self.features[self.features.index.isin(relevant_conditions)]
        else:
            raise ValueError(f'Method {self.method} is not supported.')
        return target_features, others_features


    def _get_neighbors(self, target_features: pd.DataFrame, other_features: pd.DataFrame):
        """Compute the k most similar objects to the target."""
        if self.similarity == 'jaccard':
            similarities = self._jaccard_similarity(target_features, other_features)
        elif self.similarity == 'hamming':
            similarities = self._hamming_similarity(target_features, other_features)
        elif self.similarity == 'pearson':
            similarities = self._pearson_correlation(target_features, other_features)
        elif self.similarity == 'levenshtein' and self.method == 'trials-sequence':
            similarities = self._levenshtein_similarity(target_features, other_features)
        else:
            raise ValueError(f'Similarity {self.similarity} with {self.method} method is not supported.')
        similarities = similarities[similarities > 0] # Use only items with positive similarities
        similarities = similarities.loc[similarities.index.intersection(self.utility_matrix.index)] # Use only items that are present in the utility matrix (i.e. that have been rated)
        neighbors_similarities = similarities.nlargest(self.n_neighbors)
        return neighbors_similarities


    def _predict_ratings(self, object_id: str, neighbors_similarities: pd.DataFrame):
        """Predict the therapy ratings using the given k most similar conditions."""
        neighbors_ratings = self.utility_matrix.loc[neighbors_similarities.index] # Get the ratings of the top k similar conditions
        # Generate matrix of weigths to be applied. Weights are the similairty scores
        weights = np.tile(neighbors_similarities.values.reshape(-1, 1), len(neighbors_ratings.columns))
        weights = pd.DataFrame(weights, index=neighbors_ratings.index, columns=neighbors_ratings.columns)
        weights[neighbors_ratings.isna()] = 0 # Set weights of empty ratings to 0
        # Compute weighted average of rating (with global baseline estimate)
        weighted_ratings = weights * (neighbors_ratings - self.global_baseline.loc[neighbors_ratings.index])  # Multiply every rating by the similarity score
        pred_ratings = weighted_ratings.sum(axis='index') / weights.sum(axis='index')
        pred_ratings = pred_ratings + self.global_baseline.loc[object_id]
        pred_ratings = pred_ratings.clip(0, 100)
        return pred_ratings


    def predict(self, patient_id: str, condition_id: str):
        target_features, other_features = self._get_features(patient_id, condition_id) # Compute features
        neighbors_similarities = self._get_neighbors(target_features, other_features) # Get neighbors with similarity values
        pred_ratings = self._predict_ratings(condition_id, neighbors_similarities) # Compute ratings predictions
        return pred_ratings
