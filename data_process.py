import numpy as np
import pandas as pd
from config import log


class Processor:
    def __init__(self, data, records=None):
        self.data = data.copy()
        self.records = [] if records is None else records

    def one_hot_encoding(self, columns=None, record=True):
        self.data = pd.get_dummies(self.data)
        if columns is None:
            columns = list(self.data.columns)
        else:
            self.data = self.data.loc[:, columns].fillna(0, columns=columns)
        if record:
            self.records.append({'name': 'one_hot_encoding', 'args': {'columns': columns}})

    def add_column(self, name, source, func, record=True):
        # noinspection PyUnresolvedReferences
        self.data[name] = self.data[source].apply(np.__getattribute__(func))
        if record:
            self.records.append({'name': 'add_column', 'args': {'name': name, 'source': source, 'func': func}})

    def drop(self, names, record=True):
        self.data.drop(names, axis=1, inplace=True)
        if record:
            self.records.append({'name': 'drop', 'args': {'names': names}})

    def normalize(self, norm=None, record=True):
        num_cols = self.data.columns[self.data.dtypes != "object"]
        if norm is None:
            norm = (list(self.data[num_cols].std()), list(self.data[num_cols].mean()))
        self.data[num_cols] = (self.data[num_cols] - norm[1]) / norm[0]
        if record:
            self.records.append({'name': 'normalize', 'args': {'norm': norm}})

    def redo(self):
        for each in self.records:
            log(0x16, 'Processor.redo:', each['name'])
            func = self.__getattribute__(each['name'])
            if func:
                func(**each['args'], record=False)
            else:
                raise NameError("Unknown Function %s" % each['name'])

    def fill_missing(self, fill_with='mean', record=True):
        columns = self.data.columns[self.data.dtypes != 'object']
        if self.records[-1]['name'] == 'normalize' and fill_with == 'mean':
            fill_with = 0

        fill = {}
        if isinstance(fill_with, str):
            for column in columns:
                fill[column] = self.data[column].__getattribute__(fill_with)()
        elif isinstance(fill_with, (int, float)):
            for column in columns:
                fill[column] = fill_with
        elif not isinstance(fill_with, dict):
            raise TypeError('fill_with Should be number or dict or method name, get {!r}'.format(fill_with))
        else:
            fill = fill_with

        for column in fill.keys():
            self.data[column].fillna(fill.get(column), inplace=True)

        if record:
            self.records.append({'name': 'fill_missing', 'args': {'fill_with': fill}})
