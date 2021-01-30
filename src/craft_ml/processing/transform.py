import numpy as np
import pandas as pd
from .model import TrainableModel
from ..data.dataset import Dataset, TableData, TableDataset
from ..mltypes import Identifier
import typing as t


def column_iterator(data: TableData) -> t.Generator[t.Tuple[Identifier, t.Union[np.ndarray, pd.Series]], None, None]:
    if isinstance(data, pd.DataFrame):
        return ((col_name, data[col_name]) for col_name in data.columns)
    else:
        return (data[:, col_idx] for col_idx in range(data.shape[1]))


def get_unique(column: t.Union[np.ndarray, pd.Series]):
    if isinstance(column, pd.Series):
        return column.unique()
    else:
        return np.unique(column)


class ToCategory(TrainableModel):
    def __init__(self, max_unique: int = 20):
        super().__init__()
        self.max_unique = max_unique
        self.map = {}

    def fit(self, dataset: TableDataset) -> None:
        for column_id, column in column_iterator(dataset.table_data):
            unique = get_unique(column)
            if len(unique) > self.max_unique:
                continue
            self.map[column_id] = unique

    def predict(self, dataset: TableDataset) -> TableDataset:
        data = dataset.table_data.copy()
        for column_id, unique_values in self.map.items():
            if column_id not in data.columns:
                continue
            for cur_idx, cur_value in enumerate(unique_values):
                data[column_id] = data[column_id].replace(cur_value, cur_idx)
        return TableDataset(data, dataset.target_columns)

    def predict_proba(self, dataset: TableDataset) -> TableDataset:
        return self.predict(dataset)


class DropStrings(TrainableModel):
    def __init__(self):
        super().__init__()
        self.drop_ids = []

    def fit(self, dataset: TableDataset) -> None:
        for column_id, column in column_iterator(dataset.objects_data):
            for value in column:
                if isinstance(value, str):
                    self.drop_ids.append(column_id)
                    break

    def predict(self, dataset: TableDataset) -> TableDataset:
        if len(self.drop_ids) < 1:
            return dataset
        data = dataset.table_data
        if isinstance(data, pd.DataFrame):
            data = data.drop(columns=self.drop_ids, inplace=False)
        else:
            data = np.delete(data, self.drop_ids, axis=1)
        return TableDataset(data, dataset.target_columns)

    def predict_proba(self, dataset: TableDataset) -> TableDataset:
        return self.predict(dataset)
