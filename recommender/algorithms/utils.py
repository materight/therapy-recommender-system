

class BaseRecommender:
    """Base class for all recommender algorithms."""
    def __init__(self):
        pass

    def fit(self, dataset):
        """Fit the recommender algorithm to the given data."""
        pass

    def predict(self, patient_id, condition_id):
        """Recommend a list of therapies with their predicted usccess rates."""
        pass