import time
import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_iris
from sklearn.metrics.scorer import log_loss as error_func
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import cross_val_predict, KFold
from xgboost import XGBRegressor, XGBClassifier

from sklearn.linear_model.logistic import LogisticRegression as MetaEstimator

from base import tuning
from config import get_setting
from data_loader import *

# from config import log


class Ensemble:
    def __init__(self, base_estimators=None, random_state=0):
        self.base_estimators = base_estimators
        self.estimator = MetaEstimator()
        self.random_state = random_state

    def fit(self, X, y):
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        predictions = []
        for estimator in self.base_estimators:
            log(0x25, 'cross_val_predict start', estimator.__class__.__name__)
            prediction = cross_val_predict(estimator, X, y, cv=cv, method='predict_proba')
            log(0x25, 'cross_val_predict end', estimator.__class__.__name__)
            print('prediction of', estimator.__class__.__name__)
            print(prediction)
            predictions.append(prediction.T[0])
        print('all predictions')
        print(np.array(predictions), y)
        self.estimator.fit(np.array(predictions).T, y)
        for estimator in self.base_estimators:
            log(0x25, 'fit start', estimator.__class__.__name__)
            estimator.fit(X, y)
            log(0x25, 'fit end:', estimator.__class__.__name__)

    def predict(self, X, margin):
        return np.array(self.predict_proba(X)[:, 0]) > margin

    def predict_proba(self, X):
        predictions = []
        for estimator in self.base_estimators:
            # predictions.extend(estimator.predict_proba(X).T)
            predictions.append(estimator.predict_proba(X).T[0])
        return self.estimator.predict_proba(np.array(predictions).T)


estimator_classes = [
    CatBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    LGBMClassifier,
    XGBClassifier,
]


def check_result(y_true, y_pred):
    # print(pd.DataFrame(y_pred, dtype='object').describe())
    # print(set(y_pred))
    return error_func(y_true, y_pred)


def check(estimator, data, tune=True):
    log(0x25, '~Default Setting~', estimator.__class__.__name__)
    tick = time.time()
    estimator.fit(**data['train'])
    if estimator.__class__.__name__ == 'Ensemble':
        log(0x25, 'Base Estimators:', ', '.join(['%s' % e.__class__.__name__ for e in estimator.base_estimators]))
        log(0x25, 'Ceof:', estimator.estimator.coef_, 'intercept:', estimator.estimator.intercept_)
    score = check_result(data['test']['y'], estimator.predict_proba(data['test']['X']))
    log(0x25, 'Time:', time.time() - tick)
    log(0x25, 'Score:', score)

    if not tune:
        return
    log(0x25, '~Tuned~', estimator.__class__.__name__)
    tick = time.time()
    tuning(estimator, **data['train'], **get_setting(estimator.__class__.__name__))
    # estimator.fit(**data['train'])
    score = check_result(data['test']['y'], estimator.predict_proba(data['test']['X']))
    log(0x25, 'Params:', estimator.get_params())
    log(0x25, 'Time:', time.time() - tick)
    log(0x25, 'Score:', score)


if __name__ == '__main__':
    # check(LGBMClassifier, load_Amazon())
    # data = load_adult()
    # data = load_Amazon()
    data = load_orange()
    random_state = 0
    ensemble = Ensemble([
        RandomForestClassifier(random_state=random_state),
        GradientBoostingClassifier(random_state=random_state),
        LGBMClassifier(seed=random_state),
        XGBClassifier(seed=random_state),
        CatBoostClassifier(random_seed=random_state),
        LogisticRegression(random_state=random_state),
    ])
    check(ensemble, data, tune=False)
    for estimator in ensemble.base_estimators:
        check(estimator, data, tune=False)
        # except Exception as e:
        #     log(0x22, e)
        #     pass
        log(0x25, '~Default Setting~', estimator.__class__.__name__)
        score = check_result(data['test']['y'], estimator.predict_proba(data['test']['X']))
        log(0x25, 'Score:', score)
