import time

from sklearn.metrics.scorer import log_loss as error_func
from sklearn.linear_model.logistic import LogisticRegression as MetaEstimator
from sklearn.model_selection import cross_val_predict, KFold

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBRegressor, XGBClassifier

from base import tuning
from config import get_setting
from data_loader import *


# from config import log


class Ensemble:
    def __init__(self, base_estimators=None, random_state=0, cv=3):
        self.base_estimators = base_estimators
        self.estimator = MetaEstimator()
        self.random_state = random_state
        self.fit_cv = cv

    def fit(self, X, y):
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        predictions = []
        for estimator in self.base_estimators:
            name = estimator.__class__.__name__
            log(0x25, 'cross_val_predict start', name)
            prediction = cross_val_predict(estimator, X, y, cv=cv, method='predict_proba')
            log(0x25, 'cross_val_predict end', name)
            # print('prediction of', estimator.__class__.__name__)
            # print(prediction)
            log(0x25, 'CV Score', name, check_result(y, prediction))
            predictions.append(prediction.T[0])
        # print('all predictions')
        # print(np.array(predictions), y)
        self.estimator.fit(np.array(predictions).T, y)
        for estimator in self.base_estimators:
            name = estimator.__class__.__name__
            log(0x25, 'fit start', name)
            estimator.fit(X, y)
            log(0x25, 'fit end:', name)

    def predict(self, X, margin):
        return np.array(self.predict_proba(X)[:, 0]) > margin

    def predict_proba(self, X):
        predictions = []
        for estimator in self.base_estimators:
            # predictions.extend(estimator.predict_proba(X).T)
            predictions.append(estimator.predict_proba(X).T[0])
        return self.estimator.predict_proba(np.array(predictions).T)


def check_result(y_true, y_pred):
    # print(pd.DataFrame(y_pred, dtype='object').describe())
    # print(set(y_pred))
    return error_func(y_true, y_pred)


def check(estimator, data, tune=True, fit=True):
    log(0x25, '~Default Setting~', estimator.__class__.__name__)
    tick = time.time()
    if fit:
        estimator.fit(**data['train'])
        log(0x25, 'Fit in:', time.time() - tick)
    if estimator.__class__.__name__ == 'Ensemble':
        log(0x25, 'Base Estimators:', ', '.join(['%s' % e.__class__.__name__ for e in estimator.base_estimators]))
        log(0x25, 'Ceof:', estimator.estimator.coef_, 'intercept:', estimator.estimator.intercept_)
    tick = time.time()
    prediction = estimator.predict_proba(data['test']['X'])
    log(0x25, 'Predict in:', time.time() - tick)
    score = check_result(data['test']['y'], prediction)
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
    cv = 5
    log(0x24, 'random_state:', random_state, 'cv:', cv)
    ensemble = Ensemble(
        base_estimators=[
            RandomForestClassifier(random_state=random_state),
            # GradientBoostingClassifier(random_state=random_state),
            LGBMClassifier(seed=random_state),
            XGBClassifier(seed=random_state),
            CatBoostClassifier(random_seed=random_state),
            LogisticRegression(random_state=random_state),
        ],
        random_state=random_state,
        cv=cv,
    )
    check(ensemble, data, tune=False)
    for estimator in ensemble.base_estimators:
        check(estimator, data, tune=False, fit=False)
