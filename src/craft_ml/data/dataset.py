import pandas as pd
import numpy as np
from functools import cache, lru_cache
import typing as t
from ..mltypes import Object, Label, Identifier
from ..utils.logger import Logger


class Dataset:
    """
    Dataset class provides an interface for accessing general data through __getitem__ method.
    """
    def __init__(self, ids: t.Sequence[Identifier], dataset_name: t.Optional[str] = None):
        if dataset_name is None:
            dataset_name = 'data'
        self.dataset_name = dataset_name
        self.ids = ids

    def __getitem__(self, idx: Identifier) -> t.Tuple[Object, t.Optional[Label]]:
        raise NotImplementedError

    def get_objects(self) -> t.Sequence[Object]:
        return [self[cur_id][0] for cur_id in self.ids]

    def get_labels(self) -> t.Sequence[t.Optional[Label]]:
        return [self[cur_id][1] for cur_id in self.ids]


TableData = t.Union[pd.DataFrame, np.ndarray]


class TableDataset(Dataset):

    def __init__(self,
                 table_data: TableData,
                 target_columns: t.Optional[t.Sequence[Identifier]] = None,
                 dataset_name: t.Optional[str] = None
                 ):
        super().__init__(list(range(len(table_data))), dataset_name)
        self.table_data = table_data
        self.target_columns = target_columns

    @staticmethod
    # TODO: fix caching
    def get_columns(table_data: TableData) -> t.Sequence[Identifier]:
        if isinstance(table_data, np.ndarray):
            return np.array(range(table_data.shape[1]))
        elif isinstance(table_data, pd.DataFrame):
            return np.array(list(sorted(table_data.columns)))
        else:
            raise TypeError("")  # TODO: Add an error message

    @property
    def columns(self) -> t.Sequence[Identifier]:
        return self.get_columns(self.table_data)

    @staticmethod
    # TODO: fix caching
    def get_train_columns(all_columns: t.Sequence[Identifier],
                          target_columns: t.Sequence[Identifier]
                          ) -> t.Sequence[Identifier]:
        return np.setdiff1d(all_columns, target_columns)

    @property
    def train_columns(self):
        if self.target_columns is None:
            return self.columns
        return self.get_train_columns(self.columns, self.target_columns)

    @staticmethod
    # TODO: fix caching
    def _get_column_data(table: TableData, columns: t.Sequence[Identifier]) -> TableData:
        if isinstance(table, np.ndarray):
            return table[:, columns]
        elif isinstance(table, pd.DataFrame):
            return table[columns]
        else:
            raise TypeError("")  # TODO: Add an error message

    @staticmethod
    # TODO: fix caching
    def _get_row_data(table: TableData, rows: t.Sequence[Identifier]) -> TableData:
        if isinstance(table, np.ndarray):
            return table[rows]
        elif isinstance(table, pd.DataFrame):
            return table.iloc[rows]
        else:
            raise TypeError("")  # TODO: Add an error message

    @property
    def objects_data(self) -> TableData:
        return self._get_column_data(self.table_data, self.train_columns)

    @property
    def labels_data(self) -> TableData:
        return self._get_column_data(self.table_data, self.target_columns)

    def __getitem__(self, idx: Identifier) -> t.Tuple[Object, t.Optional[Label]]:
        obj = self._get_column_data(self.objects_data, [idx])
        label = self._get_column_data(self.labels_data, [idx]) if self.labels_data is not None else None
        return obj, label

    def get_objects(self) -> TableData:
        return self._get_row_data(self.objects_data, self.ids)

    def get_labels(self) -> t.Optional[TableData]:
        if self.labels_data is None:
            return None
        return self._get_row_data(self.labels_data, self.ids)
