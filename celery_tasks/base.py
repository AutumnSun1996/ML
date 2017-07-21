import numpy as np
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from config import log
from catboost import CatBoostRegressor


class MyCatBoostRegressor(CatBoostRegressor):
    def set_params(self, **params):
        for k in params:
            obj = params[k]
            if isinstance(obj, np.integer):
                params[k] = int(obj)
            elif isinstance(obj, np.floating):
                params[k] = float(obj)
            elif isinstance(obj, np.ndarray):
                params[k] = obj.tolist()
        super().set_params(**params)


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
