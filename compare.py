import datetime
import time

import pandas as pd
import numpy as np
from sklearn.metrics.scorer import mean_squared_error as error_func
from sklearn.datasets import load_iris
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier

from base import tuning
from config import get_setting

regressor_classes = [CatBoostRegressor, LGBMRegressor, XGBRegressor]
classifier_classes = [
    LGBMClassifier,
    XGBClassifier,
    CatBoostClassifier,
]

compare_data = {'Amazon': 'ACTION'}


def log(level, *messages, **kwargs):
    timestamp = datetime.datetime.now()
    print('LOG: %02X' % level, timestamp, *messages, **kwargs)
    kwargs.update({'file': log_file, 'flush': True})
    print('%02X' % level, timestamp, *messages, **kwargs)


def check_result(y_true, y_pred):
    print(pd.DataFrame([y_true, y_pred], columns=['True', 'Pred']).describe())
    return error_func(y_true, y_pred)


def process_data(all_data, target, frac=0.8):
    train = all_data.sample(frac=frac, random_state=0)
    train_y = np.array(train[target])
    train_x = np.array(train.drop(target, axis=1))
    test = all_data.drop(train.index)
    test_y = np.array(test[target])
    test_x = np.array(test.drop(target, axis=1))

    return {'train': {'X': train_x, 'y': train_y}, 'test': {'X': test_x, 'y': test_y}}
    # return train_x[:], train_y[:], test_x[:], test_y[:]


def check(estimator_class, data):
    if estimator_class.__name__ == 'CatBoostClassifier':
        estimator = estimator_class(loss_function='MultiClass', classes_count=len(set(data['train']['y'])))
    else:
        estimator = estimator_class()
    log('~Fit With Default Setting~', estimator_class.__name__)
    tick1 = time.time()
    estimator.fit(**data['train'])
    score = error_func(data['test']['y'], estimator.predict(data['test']['X']))
    tick2 = time.time()
    log('Score:', score)
    log('Time Usage:', tick2 - tick1)

    if estimator_class.__name__ == 'CatBoostClassifier':
        estimator = estimator_class(loss_function='MultiClass', classes_count=len(set(data['train']['y'])))
    else:
        estimator = estimator_class()
    log('~Tuning~', estimator_class.__name__)
    tick1 = time.time()
    tuning(estimator, **data['train'], **get_setting(estimator_class.__name__))
    score = error_func(data['test']['y'], estimator.predict(data['test']['X']))
    tick2 = time.time()
    log('Score:', score)
    log('Time Usage:', tick2 - tick1)


if __name__ == '__main__':

    log_file = open('./compare.log', 'w')
    iris_x, iris_y = load_iris(return_X_y=True)
    iris = pd.DataFrame(iris_x)
    iris['y'] = iris_y
    data = process_data(iris, 'y')
    for each in classifier_classes:
        check(each, data)

    log_file.close()
