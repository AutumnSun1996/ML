from celery_tasks.base import tuning
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
if __name__ == '__main__':

    boston = load_boston()
    X, y = boston.data, boston.target
    n_features = X.shape[1]

    reg = GradientBoostingRegressor(n_estimators=50, random_state=0)

    names = ['max_depth',
             'learning_rate',
             'max_features',
             'min_samples_split',
             'min_samples_leaf',
             ]
    p = [(1, 5),  # max_depth
         (10 ** -5, 10 ** 0, "log-uniform"),  # learning_rate
         (1, n_features),  # max_features
         (2, 100),  # min_samples_split
         (1, 100)]
    params = {
        'max_depth': (1, 5),
        'learning_rate': (1e-05, 1, 'log-uniform'),
        'max_features': (1, n_features),
        'min_samples_split': (2, 100),
        'min_samples_leaf': (1, 100)
    }
    cv = {
        'cv': 5,
        'n_jobs': -1,
        'scoring': 'neg_mean_absolute_error'
    }

    from catboost import CatBoostRegressor

    b_x, b_y = load_boston(return_X_y=True)
    cbr = CatBoostRegressor()
    params = {
        'depth': (1, 5),
        'learning_rate': (1e-05, 1, 'log-uniform'),
        #     'max_features': (1, b_features),
        #     'min_samples_split': (2, 100),
        #     'min_samples_leaf': (1, 100)
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
    print(cbr.get_params())
    cbr.set_params(depth=5)
    res_cbr = tuning(cbr, b_x, b_y, params, cv, gp)
