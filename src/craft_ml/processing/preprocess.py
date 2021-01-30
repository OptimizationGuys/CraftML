import numpy as np
import pandas as pd
from ..data.dataset import TableDataset
from ..mltypes import Identifier
import typing as t


def findna(dataset: TableDataset) -> TableDataset:
    data = dataset.table_data
    data = data.replace(('nan', 'NaN', 'NA', 'na', np.nan), None, inplace=False)
    return TableDataset(data, dataset.target_columns)


def fillna(dataset: TableDataset) -> TableDataset:
    data = dataset.table_data
    if isinstance(data, pd.DataFrame):
        data = data.fillna(method='ffill', inplace=False)
        data.fillna(method='bfill', inplace=True)
    else:
        mask = np.isnan(dataset.table_data)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        data = data[np.arange(idx.shape[0])[:, None], idx]
    return TableDataset(data, dataset.target_columns)


def drop_index(dataset: TableDataset, index_id: Identifier = 0) -> TableDataset:
    data = dataset.table_data
    if isinstance(data, pd.DataFrame):
        column_ids = data.columns
        if index_id not in column_ids:
            index_id = data.columns[index_id]
        data = data.drop(columns=index_id, inplace=False)
    else:
        assert isinstance(index_id, int)
        pos_id = index_id if index_id >= 0 else data.shape[1] + index_id
        data = np.delete(data, pos_id, axis=1)
    return TableDataset(data, dataset.target_columns)


def drop_columns(dataset: TableDataset, columns: t.Sequence[Identifier]) -> TableDataset:
    data = dataset.table_data.drop(columns, axis=1)
    return TableDataset(data, dataset.target_columns)
