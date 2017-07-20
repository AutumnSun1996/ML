import numpy


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
            print('one hot for %s' % column)
            self.data.drop(column, axis=1, inplace=True)
        if record:
            self.records.append({'name': 'one_hot_encoding', 'args': {'items': items}})

    def add_column(self, name, source, func, record=True):
        self.data[name] = self.data[source].apply(numpy.__getattribute__(func))
        if record:
            self.records.append({'name': 'add_column', 'args': {'name': name, 'source': source, 'func': func}})

    def drop(self, names, record=True):
        self.data.drop(names, axis=1, inplace=True)
        if record:
            self.records.append({'name': 'drop', 'args': {'names': names}})

    def normalize(self, norm=None, record=True):
        if norm is None:
            norm = (self.data.std, self.data.mean)
        self.data = (self.data - norm[1]) / norm[0]
        if record:
            self.records.append({'name': 'normalize', 'args': {'norm': norm}})

    def redo(self, records):
        for each in records:
            print('Do:', each['name'])
            func = self.__getattribute__(each['name'])
            if func:
                func(**each['args'], record=False)
            else:
                raise NameError("Unknown Function %s" % each['name'])

    def fill_missing(self, fill_with='mean', record=True):
        # num_data = self.data.index[]
        columns = self.data.columns[self.data.dtypes != 'object']
        for column in columns:
            self.fillna(column, self.data[column].__getattribute__(fill_with)(), record)

    def fillna(self, column, fill_with, record=True):
        self.data[column].fillna(fill_with, inplace=True)
        if record:
            self.records.append({'name': 'fillna', 'args': {'column': column, 'fill_with': fill_with}})
