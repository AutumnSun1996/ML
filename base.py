# from sklearn.linear_model import ElasticNet as MetaEstimator
from sklearn.linear_model.logistic import LogisticRegression as MetaEstimator
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
import sklearn.metrics
from skopt import gp_minimize
from config import log
from sklearn.ensemble import BaseEnsemble


# from catboost import CatBoostRegressor, CatBoostClassifier
#
#
# class MyCatBoostRegressor(CatBoostRegressor):
#     def set_params(self, **params):
#         for k in params:
#             if isinstance(params[k], np.integer):
#                 params[k] = int(params[k])
#             elif isinstance(params[k], np.floating):
#                 params[k] = float(params[k])
#             elif isinstance(params[k], np.ndarray):
#                 params[k] = params[k].tolist()
#         super().set_params(**params)
#
#
# class MyCatBoostClassifier(CatBoostClassifier):
#     def set_params(self, **params):
#         for k in params:
#             if isinstance(params[k], np.integer):
#                 params[k] = int(params[k])
#             elif isinstance(params[k], np.floating):
#                 params[k] = float(params[k])
#             elif isinstance(params[k], np.ndarray):
#                 params[k] = params[k].tolist()
#         super().set_params(**params)
#

def tuning(predictor, X, y, params, cv, gp, scoring='log_loss'):
    log(0x25, 'tuning: predictor=', predictor)
    log(0x25, 'tuning: x=', type(X), X.shape)
    log(0x25, 'tuning: y=', type(y), y.shape)
    log(0x25, 'tuning: params=', params)
    log(0x25, 'tuning: cv=', cv)
    log(0x25, 'tuning: gp=', gp)
    space = []
    names = []
    for key in params:
        names.append(key)
        space.append(params[key])

    def objective(p):
        for k, v in enumerate(p):
            if isinstance(v, np.integer):
                p[k] = int(v)
            elif isinstance(v, np.floating):
                p[k] = float(v)
        print('Set:', dict(zip(names, p)))
        predictor.set_params(**dict(zip(names, p)))
        prediction = cross_val_predict(predictor, X[:], y[:], **cv)
        return sklearn.metrics.__getattribute__(scoring)(y[:], prediction)

    result = gp_minimize(objective, space, **gp)
    predictor.set_params(**dict(zip(names, result.x)))
    predictor.fit(X, y)
    return result


class Ensemble:
    def __init__(self, base_estimators=None, random_state=0):
        self.base_estimators = base_estimators
        self.estimator = MetaEstimator()
        self.random_state = random_state

    def fit(self, X, y):
        cv = 2
        # folds = KFold(n_splits=2, shuffle=True, random_state=self.random_state)
        predictions = []
        idx = 0
        # for train, test in folds.split(X):
        #     estimator = self.base_estimators[idx]
        #     idx += 1
        # estimator.fit(X[train], y[train])
        # prediction = estimator.predict_proba(X)
        for estimator in self.base_estimators:
            prediction = cross_val_predict(estimator, X, y, cv=cv, method='predict_proba')
            print('prediction of', estimator.__class__.__name__)
            print(prediction)
            # predictions.extend(prediction.T)
            predictions.append(prediction.T[0])
        print('all predictions')
        print(np.array(predictions), y)
        self.estimator.fit(np.array(predictions).T, y)
        for estimator in self.base_estimators:
            estimator.fit(X, y)

    def predict(self, X, margin):
        return np.array(self.predict_proba(X)) > margin

    def predict_proba(self, X):
        predictions = []
        for estimator in self.base_estimators:
            # predictions.extend(estimator.predict_proba(X).T)
            predictions.append(estimator.predict_proba(X).T[0])
        return self.estimator.predict_proba(np.array(predictions).T)

#
#
# class Ensemble:
#     def __init__(self, base_estimators=None, random_state=0):
#         self.base_estimators = base_estimators
#         self.estimator = LogisticRegression()
#         self.random_state = random_state
#
#     def fit(self, X, y):
#         folds = KFold(n_splits=len(self.base_estimators), shuffle=True, random_state=self.random_state)
#         predictions = []
#         idx = 0
#         for train, test in folds.split(X):
#             estimator = self.base_estimators[idx]
#             idx += 1
#             estimator.fit(X[train], y[train])
#             prediction = estimator.predict_proba(X)
#             print('prediction of', estimator.__class__.__name__)
#             print(prediction)
#             predictions.extend(prediction.T)
#         print('all predictions')
#         print(np.array(predictions), y)
#         self.estimator.fit(np.array(predictions).T, y)
#         for estimator in self.base_estimators:
#             estimator.fit(X, y)
#
#     def predict(self, X, margin):
#         return np.array(self.predict_proba(X)) > margin
#
#     def predict_proba(self, X):
#         predictions = []
#         for estimator in self.base_estimators:
#             predictions.extend(estimator.predict_proba(X).T)
#         return self.estimator.predict_proba(np.array(predictions).T)
