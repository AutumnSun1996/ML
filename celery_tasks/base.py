import numpy as np
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from config import log


class BaseEstimator:
    def __init__(self):
        self.predictor = None
        self.kwargs = None

    def predict(self, x, **kwargs):
        return self.predictor.predict(x, **kwargs)


class TuningEstimator(BaseEstimator):
    def __init__(self):
        super().__init__()

    def tuning(self, x, y, params, cv, gp):
        space = []
        names = []
        for key in params:
            names.append(key)
            space.append(params[key])

        def objective(p):
            self.predictor.set_params(**dict(zip(names, p)))
            return -np.mean(cross_val_score(self.predictor, x, y, **cv))

        result = gp_minimize(objective, space, **gp)
        self.predictor.set_params(**dict(zip(names, result.x)))
        self.predictor.fit(x, y)


def tuning(predictor, x, y, params, cv, gp):
    space = []
    names = []
    for key in params:
        names.append(key)
        space.append(params[key])

    def objective(p):
        log(0x26, 'Set:', dict(zip(names, p)))
        predictor.set_params(**dict(zip(names, p)))
        return -np.mean(cross_val_score(predictor, x, y, **cv))

    result = gp_minimize(objective, space, **gp)
    predictor.set_params(**dict(zip(names, result.x)))
    predictor.fit(x, y)
    return result