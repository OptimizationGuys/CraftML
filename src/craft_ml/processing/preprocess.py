import numpy as np
import pandas as pd
from ..data.dataset import TableDataset
from ..mltypes import Identifier
import typing as t


def findna(dataset: TableDataset) -> TableDataset:
    data = dataset.table_data
    data[data == 'nan'] = None
    data[data == 'NaN'] = None
    data[data == 'NA'] = None
    data[data == 'na'] = None
    dataset.table_data = data
    return dataset


def fillna(dataset: TableDataset) -> TableDataset:
    if isinstance(dataset.table_data, pd.DataFrame):
        dataset.table_data.fillna(method='ffill', inplace=True)
    else:
        mask = np.isnan(dataset.table_data)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        dataset.table_data = dataset.table_data[np.arange(idx.shape[0])[:, None], idx]
    return dataset


def to_category(dataset: TableDataset, max_unique: int = 20) -> TableDataset:
    def column_iterator(dataset: TableDataset) -> t.Generator[t.Union[np.ndarray, pd.Series]]:
        data = dataset.table_data
        if isinstance(data, pd.DataFrame):
            return (data[col_name] for col_name in data.columns)
        else:
            return (data[:, col_idx] for col_idx in range(data.shape[1]))

    def get_unique(column: t.Union[np.ndarray, pd.Series]):
        if isinstance(column, pd.Series):
            return column.unique()
        else:
            return np.unique(column)

    for column in column_iterator(dataset):
        unique = get_unique(column)
        if len(unique) > max_unique:
            continue
        for cur_idx, cur_value in enumerate(unique):
            column[column == cur_value] = cur_idx
    return dataset


def drop_index(dataset: TableDataset, index_id: Identifier = 0) -> TableDataset:
    data = dataset.table_data
    if isinstance(data, pd.DataFrame):
        column_ids = data.columns
        if index_id not in column_ids:
            index_id = data.columns[index_id]
        data.drop(columns=index_id, inplace=True)
    else:
        assert isinstance(index_id, int)
        pos_id = index_id if index_id >= 0 else data.shape[1] + index_id
        data = np.delete(data, pos_id, axis=1)
        dataset.table_data = data
    return dataset


def drop_strings(dataset: TableDataset) -> TableDataset:
    pass
