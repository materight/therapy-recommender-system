from typing import List

from .utils import BaseRecommender


class HybridRecommender(BaseRecommender):
    def __init__(self, recommenders: List[BaseRecommender]):
        """
        Hybrid recommender, combines multiple recommender systems.

        Args:
            recommenders: List of recommenders systems to combine.
        """
        super().__init__()
        self.recommenders = recommenders

    def fit(self, dataset):
        for recommender in self.recommenders:
            recommender.fit(dataset)

    def predict(self, patient_id: str, condition_id: str):
        # TODO: rank aggregation (check: https://people.orie.cornell.edu/dpw/talks/RankAggDec2012.pdf)
        results = []
        for recommender in self.recommenders:
            result = recommender.predict(patient_id, condition_id)
            results.append(result)
        return results[0]