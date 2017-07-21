from celery_tasks.base import tuning
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
if __name__ == '__main__':
    b_x, b_y =  load_boston(return_X_y=True)

    reg = GradientBoostingRegressor(n_estimators=50, random_state=0)
    params = {
        'max_depth': (1, 5),
        'learning_rate': (1e-05, 1, 'log-uniform'),
        'max_features': (1, b_x.shape[1]),
        'min_samples_split': (2, 100),
        'min_samples_leaf': (1, 100)
    }
    cv = {
        'cv': 5,
        'n_jobs': -1,
        'scoring': 'neg_mean_absolute_error'
    }
    gp = {
        'n_calls': 50,
        'random_state': 0,
        'verbose': True
    }
    res = tuning(reg, b_x, b_y, params, cv, gp)
    print(res)


    #
    # from catboost import CatBoostRegressor
    #
    # b_x, b_y = load_boston(return_X_y=True)
    # cbr = CatBoostRegressor()
    # params = {
    #     'depth': (1, 5),
    #     'learning_rate': (1e-05, 1, 'log-uniform'),
    #     #     'max_features': (1, b_features),
    #     #     'min_samples_split': (2, 100),
    #     #     'min_samples_leaf': (1, 100)
    # }
    # cv = {
    #     'cv': 5,
    #     'n_jobs': -1,
    #     'scoring': 'neg_mean_absolute_error'
    # }
    # gp = {
    #     'n_calls': 10,
    #     'random_state': 0,
    #     'verbose': True
    # }
    # print(cbr.get_params())
    # cbr.set_params(depth=5)
    # res_cbr = tuning(cbr, b_x, b_y, params, cv, gp)
