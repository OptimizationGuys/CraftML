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
        for column_id, column in column_iterator(dataset.objects_data):
            unique = get_unique(column)
            if len(unique) > self.max_unique:
                continue
            self.map[column_id] = unique

    def predict(self, dataset: TableDataset) -> TableDataset:
        if len(self.map) < 1:
            return dataset

        for column_id, column in column_iterator(dataset.objects_data):
            if column_id not in self.map:
                continue
            for cur_idx, cur_value in enumerate(self.map[column_id]):
                column[column == cur_value] = cur_idx
        return dataset

    def predict_proba(self, dataset: TableDataset) -> TableDataset:
        return self.predict(dataset)
