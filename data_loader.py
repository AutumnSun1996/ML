import datetime

import numpy as np
import pandas as pd

log_name = 'log/{}.log'.format(datetime.datetime.now())
log_file = open(log_name, 'w', -1, 'utf8')
print('Log To:', log_name)


def log(level, *messages, **kwargs):
    timestamp = datetime.datetime.now()
    print('LOG: %02X' % level, timestamp, *messages, **kwargs)
    kwargs.update({'file': log_file, 'flush': True})
    print('%02X' % level, timestamp, *messages, **kwargs)


def process_data(all_data, target, frac=0.5):
    train = all_data.sample(frac=frac, random_state=0)
    train_y = np.array(train[target])
    train_x = np.array(train.drop(target, axis=1))
    test = all_data.drop(train.index)
    test_y = np.array(test[target])
    test_x = np.array(test.drop(target, axis=1))

    # return {'train': {'X': train_x, 'y': train_y}, 'test': {'X': test_x, 'y': test_y}}
    return {'train': {'X': train_x[:], 'y': train_y[:]}, 'test': {'X': test_x[:], 'y': test_y[:]}}


def load_orange(frac=0.5):
    log(0x24, 'Use Data: orange')
    data = pd.read_csv('data/orange/train.data', sep='\t')
    data = data.dropna(axis=1, how='all')
    mean_val = data.mean()
    indices = mean_val.index
    data[indices] = (data[indices] - mean_val) / data[indices].std()
    data = pd.get_dummies(data).fillna(0)
    data['Target'] = pd.read_csv('data/orange/train_appetency.labels', header=None)[0].apply(
        lambda a: 1 if a > 0 else 0)
    return process_data(data, 'Target', frac=frac)


def load_Amazon(frac=0.5):
    log(0x24, 'Use Data: Amazon')
    train = pd.read_csv('data/Amazon/train.csv', dtype='category')
    train['ACTION'] = train['ACTION'].astype('int32')
    return process_data(pd.get_dummies(train), 'ACTION', frac)


def load_adult():
    log(0x24, 'Use Data: Adult')
    import re
    items = re.findall('(?m)^\s*(.+): (.+)$', '''
    age: continuous.
    workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    fnlwgt: continuous.
    education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    education-num: continuous.
    marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    sex: Female, Male.
    capital-gain: continuous.
    capital-loss: continuous.
    hours-per-week: continuous.
    native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    ''')
    items.append(('Target', ''))
    # print(names)
    names = []
    types = {}
    for each in items:
        names.append(each[0])
        types[each[0]] = np.float32 if each[1] == 'continuous.' else 'object'
        # print(each[0], types[each[0]])
    settings = dict(names=names, dtype=types, skiprows=1, header=0, index_col=False, na_values=['?'],
                    skipinitialspace=True, )
    train = pd.read_csv('data/adult/adult.data', **settings, )
    train_dummy = pd.get_dummies(train)
    # print(train_dummy.head())
    train_dummy = train_dummy.drop('Target_<=50K', axis=1)
    columns = train_dummy.columns
    # print(train.describe(include='all'))

    test = pd.read_csv('data/adult/adult.test', **settings)
    test['Target'] = test['Target'].apply(lambda a: a.strip('.'))
    # print(test.describe(include='all'))
    test_dummy = pd.get_dummies(test)
    test_dummy = test_dummy.loc[:, columns].fillna(0)
    target = 'Target_>50K'
    # print(train_dummy[[target]].describe())
    # print(test_dummy[[target]].describe())
    # print(train_dummy[[target]].describe())
    # print(test_dummy[[target]].describe())
    # np.ravel()
    return {'train': {'X': np.array(train_dummy.drop(target, axis=1)), 'y': np.ravel(train_dummy[[target]])},
            'test': {'X': np.array(test_dummy.drop(target, axis=1)), 'y': np.ravel(test_dummy[[target]])}, }
    # return {'train': {'X': train_dummy.drop(target, axis=1), 'y': train_dummy[[target]]},
    #         'test': {'X': test_dummy.drop(target, axis=1), 'y': test_dummy[[target]]}, }
