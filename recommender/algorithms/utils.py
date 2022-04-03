from abc import abstractmethod
import numpy as np
import pandas as pd
import editdistance
from typing import List

from recommender.dataset import Dataset

class BaseRecommender:
    """Base class for all recommender algorithms."""

    def init_state(self, **kwargs):
        """Initialize the recommender state with additional data if needed."""
        pass

    @abstractmethod
    def fit(self, dataset: Dataset):
        """Fit the recommender algorithm to the given data."""

    @abstractmethod
    def predict(self, patient_id: str, condition_id: str):
        """Recommend a list of k therapies with a predicted success rate."""

    @staticmethod
    def _get_trials(p_trials, condition_ids: List[str]):
        """Return the trials of the given conditions."""
        return p_trials[p_trials.condition.isin(condition_ids)]

    @staticmethod
    def _get_utility_matrix(p_trials: pd.DataFrame, therapies: pd.DataFrame):
        """Compute the utility matrix, using as value the 'successful' column. The result is a NxM utility matrix, where:
            - N is the number of conditions of each patient (i.e. the "users")
            - M is the number of available therapies (i.e. the "items")"""
        features = p_trials.pivot_table(index='condition', columns='therapy', values='successful', aggfunc='mean')
        features = features.reindex(columns=therapies.index)
        return features

    @staticmethod
    def _get_baseline_estimates(utility_matrix: pd.DataFrame, p_conditions: pd.DataFrame):
        """Compute the baseline estimates on the given utility matrix."""
        global_avg_rating = np.nanmean(utility_matrix)
        utility_matrix = utility_matrix.reindex(index=p_conditions.index.get_level_values('id')) # Re-index to include global baseline for conditions without therapies
        users_rating_deviation = utility_matrix.mean(axis=1, skipna=True).values - global_avg_rating
        items_rating_deviation = utility_matrix.mean(axis=0, skipna=True).values - global_avg_rating
        global_baseline = global_avg_rating + (users_rating_deviation.reshape(-1,1) + items_rating_deviation.reshape(1,-1))
        global_baseline = pd.DataFrame(global_baseline, index=utility_matrix.index, columns=utility_matrix.columns)
        return global_baseline

    @staticmethod
    def _get_trials_sequences(p_trials: pd.DataFrame):
        """Convert the given conditions in p_trials to a list of trials, ordered by start time."""
        sequences = p_trials.sort_values('start').groupby(['condition'], observed=True)['therapy'].apply(list)
        return sequences

    @staticmethod
    def _build_patients_demographic_profiles(patients: pd.DataFrame):
        """Build the patients profiles based on the available demographic data."""
        patients = patients.copy().drop(columns=['name', 'email', 'birthdate'], errors='ignore')
        if 'age' in patients.columns: # Convert age to category
            patients['age'] = pd.cut(patients['age'], bins=[0, 2, 18, 30, 50, 70, np.inf], labels=['<2', '2-18', '18-30', '30-50', '50-70', '>70'])
        return patients

    @staticmethod
    def _build_patients_conditions_profiles(p_conditions: pd.DataFrame, conditions: pd.DataFrame):
        """Build the patients profiles based on their medical conditions."""
        profiles = pd.crosstab(p_conditions.index.get_level_values('patient'), p_conditions['kind']).astype(bool)
        profiles = profiles.reindex(columns=conditions.index)
        profiles.index.rename('id', inplace=True)
        return profiles

    @staticmethod
    def _jaccard_similarity(target_item: pd.DataFrame, other_items: pd.DataFrame):
        """Compute the Jaccard similarity between the target item and all the other items."""
        assert target_item.shape[0] == 1, 'target_item must contain 1 element'
        other_index = other_items.index
        target_item, other_items = target_item.fillna(0).values.astype(bool), other_items.fillna(0).values.astype(bool)
        intersection = (target_item & other_items).sum(axis=1)
        union = (target_item | other_items).sum(axis=1)
        similarities = intersection / union
        similarities = pd.Series(similarities, index=other_index)
        return similarities

    @staticmethod
    def _hamming_similarity(target_item: pd.DataFrame, other_items: pd.DataFrame):
        """Compute the normalized hamming similarity between the target item and all the other items."""
        assert target_item.shape[0] == 1, 'target_item must contain 1 element'
        intersection = (target_item.iloc[0] == other_items).sum(axis=1)
        similarities = intersection / target_item.shape[1]
        return similarities

    @staticmethod
    def _pearson_correlation(target_item: pd.DataFrame, other_items: pd.DataFrame):
        """Compute the centered cosine similarity (pearson correlation) between the target item and all the other items."""
        assert target_item.shape[0] == 1, 'target_item must contain 1 element'
        other_index = other_items.index
        target_item, other_items = target_item.values, other_items.values
        # Normalize using rows means (centered cosine similarity)
        target_item = target_item - np.nanmean(target_item, axis=1, keepdims=True)
        other_items = other_items - np.nanmean(other_items, axis=1, keepdims=True)
        # Compute cosine similarity
        target_item[np.isnan(target_item)], other_items[np.isnan(other_items)] = 0, 0 # Set missing ratings to 0
        dot_prods = (other_items @ target_item.T).ravel() # Compute dot product between target_item and all other_items
        target_norm = np.linalg.norm(target_item, ord=2, axis=1)
        other_norms = np.linalg.norm(other_items, ord=2, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            similarities = dot_prods / (target_norm * other_norms)
        similarities = pd.Series(similarities, index=other_index)
        similarities = similarities.fillna(-1) # If NaN, one of the vectors is zeros-only. Set similarity to -1.
        return similarities

    @staticmethod
    def _levenshtein_similarity(target_item: pd.DataFrame, other_items: pd.DataFrame):
        """Compute the Levenshtein distance between the target item and all the other items."""
        assert target_item.shape[0] == 1, 'target_item must contain 1 element'
        target_sequence = target_item.iloc[0]
        similarities = pd.Series(index=other_items.index)
        for condition_id, sequence in other_items.iteritems():
            distance = editdistance.eval(target_sequence, sequence)
            max_len = max(len(target_sequence), len(sequence))
            similarity = max_len - distance # Convert distance to similarity
            similarities.loc[condition_id] = similarity
        return similarities
