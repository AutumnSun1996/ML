class BaseEstimator:
    def __init__(self):
        self.predictor = None
        self.kwargs = None

    def predict(self, x, **kwargs):
        return self.predictor.predict(x, **kwargs)
