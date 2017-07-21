import numpy as np
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from sklearn.datasets import load_boston
from celery_tasks.base import MyCatBoostRegressor


def tuning(predictor, x, y, params, cv, gp):
    space = []
    names = []
    for key in params:
        names.append(key)
        space.append(params[key])

    def objective(p):
        print('Set:', dict(zip(names, p)))
        predictor.set_params(**dict(zip(names, p)))
        return -np.mean(cross_val_score(predictor, x, y, **cv))

    result = gp_minimize(objective, space, **gp)
    predictor.set_params(**dict(zip(names, result.x)))
    predictor.fit(x, y)
    return result


if __name__ == '__main__':
    b_x, b_y = load_boston(return_X_y=True)

    cbr = MyCatBoostRegressor()

    params = {
        'depth': (1, 5),
        'learning_rate': (1e-05, 1, 'log-uniform'),
    }
    cv = {
        'cv': 5,
        'n_jobs': -1,
        'scoring': 'neg_mean_absolute_error'
    }
    gp = {
        'n_calls': 10,
        'random_state': 0,
        'verbose': True
    }
    # params = cbr.get_params()
    # for k in params:
    #     print(k, ':', type(params[k]).__name__)
    # cbr.set_params(depth=5)
    # params = cbr.get_params()
    # for k in params:
    #     print(k, ':', type(params[k]).__name__)
    res_cbr = tuning(cbr, b_x, b_y, params, cv, gp)
