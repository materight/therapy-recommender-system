import numpy as np
import pandas as pd
import editdistance
from typing import List

from recommender.dataset import Dataset

class BaseRecommender:
    """Base class for all recommender algorithms."""

    def fit(self, dataset: Dataset):
        """Fit the recommender algorithm to the given data."""
        raise NotImplemented

    def predict(self, patient_id: str, condition_id: str):
        """Recommend a list of k therapies with a predicted success rate."""
        raise NotImplemented
    
    @staticmethod
    def _get_trials(p_trials, condition_ids: List[str]):
        """Return the trials of the given conditions."""
        return p_trials[p_trials.condition.isin(condition_ids)]

    @staticmethod
    def _get_trials_vectors(p_trials: pd.DataFrame, therapies: pd.DataFrame):
        """Convert the given conditions in p_trials to a feature vector, where 
           columns=therapies and values=successful score."""
        features = p_trials.pivot_table(index='condition', columns='therapy', values='successful', fill_value=0).astype(int)
        features = features.reindex(columns=therapies.index, fill_value=0)
        return features

    @staticmethod
    def _get_trials_sequences(p_trials: pd.DataFrame):
        """Convert the given conditions in p_trials to a list of trials, ordered by start or end time."""
        sequences = p_trials.groupby(['condition'], observed=True)['therapy'].apply(list)
        return sequences

    @staticmethod
    def _jaccard_similarity(target_item: pd.DataFrame, other_items: pd.DataFrame):
        """Compute the Jaccard similarity between the target item and all the other items."""
        assert target_item.shape[0] == 1, 'target_item must contain 1 element'
        other_index = other_items.index
        target_item, other_items = target_item.values.astype(bool), other_items.values.astype(bool)
        intersection = (target_item & other_items).sum(axis=1)
        union = (target_item | other_items).sum(axis=1)
        similarities = intersection / union
        similarities = pd.Series(similarities, index=other_index)
        return similarities

    @staticmethod
    def _cosine_similarity(target_item: pd.DataFrame, other_items: pd.DataFrame, centered: bool = True):
        """Compute the cosine similarity between the target item and all the other items."""
        assert target_item.shape[0] == 1, 'target_item must contain 1 element'
        other_index = other_items.index
        target_item, other_items = target_item.values, other_items.values
        # Normalize using rows means (centered cosine similarity)
        if centered:
            target_item = target_item - np.nanmean(np.where(target_item == 0, np.nan, target_item), axis=1, keepdims=True)
            other_items = other_items - np.nanmean(np.where(other_items == 0, np.nan, other_items), axis=1, keepdims=True)
        # Compute cosine similarity
        dot_prods = (other_items @ target_item.T).ravel() # Compute dot product between target_item and all other_items
        target_norm = np.linalg.norm(target_item, ord=2, axis=1)
        other_norms = np.linalg.norm(other_items, ord=2, axis=1)
        similarities = dot_prods / (target_norm * other_norms)
        similarities = pd.Series(similarities, index=other_index)     
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
            similarity = (max_len - distance) / max_len # Convert distance to similarity in [0, 1]
            similarities.loc[condition_id] = similarity
        return similarities
