import numpy as np
from config import log


class Processor:
    def __init__(self, data):
        self.data = data.copy()
        self.records = []

    def one_hot(self, record=True):
        temp = self.data.loc[:, self.data.dtypes == 'object']
        items = {}
        for col in temp.columns:
            items[col] = list(temp[col].dropna().drop_duplicates())
        self.one_hot_encoding(items, record)

    def one_hot_encoding(self, items, record=True):
        for column in items:
            for choice in items[column]:
                self.data['{0}_{1}'.format(column, choice)] = self.data[column].apply(
                    lambda a: 1 if a is choice else 0)
            log(0x19, 'Processor.one_hot_encoding', 'one hot for %s' % column)
            self.data.drop(column, axis=1, inplace=True)
        if record:
            self.records.append({'name': 'one_hot_encoding', 'args': {'items': items}})

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

    def redo(self, records):
        for each in records:
            log(0x16, 'Processor.redo:', each['name'])
            func = self.__getattribute__(each['name'])
            if func:
                func(**each['args'], record=False)
            else:
                raise NameError("Unknown Function %s" % each['name'])

    def fill_missing(self, fill_with='mean', record=True):
        columns = self.data.columns[self.data.dtypes != 'object']
        if self.records[-1]['name'] == 'normalize':
            fill_with = 0
        for column in columns:
            fill = self.data[column].__getattribute__(fill_with)() if isinstance(fill_with, str) else fill_with
            self.fillna(column, fill, record)

    def fillna(self, column, fill_with, record=True):
        self.data[column].fillna(fill_with, inplace=True)
        if record:
            self.records.append({'name': 'fillna', 'args': {'column': column, 'fill_with': fill_with}})
