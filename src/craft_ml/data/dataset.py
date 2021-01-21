import pandas as pd
import numpy as np
from functools import cached_property
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
        if self.target_columns is None:
            self.objects_data = self.table_data
            self.labels_data = None
        else:
            self.labels_data = self._get_column_data(self.table_data, self.target_columns)
            train_columns = np.setdiff1d(self.columns, self.target_columns)
            self.objects_data = self._get_column_data(self.table_data, train_columns)

    @cached_property
    def columns(self) -> t.Sequence[Identifier]:
        if isinstance(self.table_data, np.ndarray):
            return np.array(range(self.table_data.shape[1]))
        elif isinstance(self.table_data, pd.DataFrame):
            return self.table_data.columns.to_numpy()
        else:
            raise TypeError("")  # TODO: Add an error message

    @staticmethod
    def _get_column_data(table: TableData, columns: t.Sequence[Identifier]) -> TableData:
        if isinstance(table, np.ndarray):
            return table[:, columns]
        elif isinstance(table, pd.DataFrame):
            return table[columns]
        else:
            raise TypeError("")  # TODO: Add an error message

    @staticmethod
    def _get_row_data(table: TableData, rows: t.Sequence[Identifier]) -> TableData:
        raise NotImplementedError

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
