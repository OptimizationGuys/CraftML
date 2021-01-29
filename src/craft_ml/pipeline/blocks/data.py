import numpy as np
import pandas as pd
import typing as t
from ...data.dataset import TableDataset
from ..block import Block


class PandasLoader(Block):

    def run(self, inputs: t.Tuple[str, str]) -> t.Tuple[TableDataset, TableDataset]:
        train_path, test_path = inputs
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        train_columns = train_data.columns.to_numpy()
        test_columns = test_data.columns.to_numpy()
        target_columns = np.setdiff1d(train_columns, test_columns)
        train_dataset = TableDataset(train_data, target_columns)
        test_dataset = TableDataset(test_data, target_columns)
        return train_dataset, test_dataset
