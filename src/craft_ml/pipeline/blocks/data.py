import numpy as np
import pandas as pd
import typing as t
from ...data.dataset import Dataset, TableDataset
from ..block import Block


class PandasLoader(Block):

    def run(self, inputs: t.Tuple[str, str, str]) -> t.Tuple[TableDataset, TableDataset]:
        train_path, test_path, target_column = inputs
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        if target_column is None:
            train_columns = train_data.columns.to_numpy()
            test_columns = test_data.columns.to_numpy()
            target_columns = np.setdiff1d(train_columns, test_columns).tolist()
        else:
            target_columns = [target_column]
        conform_columns = list(set(train_data.columns).intersection(set(test_data.columns)))
        train_data = train_data[conform_columns + target_columns]
        test_data = test_data[conform_columns]
        train_dataset = TableDataset(train_data, target_columns)
        test_dataset = TableDataset(test_data)
        return train_dataset, test_dataset


class NextSplit(Block):
    def run(self, inputs: t.Generator[t.Tuple[Dataset, Dataset], None, None]) -> t.Tuple[Dataset, Dataset]:
        return next(inputs)
