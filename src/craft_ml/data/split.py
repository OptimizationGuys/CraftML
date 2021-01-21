import numpy as np
from sklearn import model_selection
import typing as t
from copy import copy
from ..mltypes import RandomState
from ..data.dataset import Dataset


class DataSplit:

    def get_splits(self, dataset: Dataset) -> t.Generator[t.Tuple[Dataset, Dataset]]:
        raise NotImplementedError


class TrainTestSplit(DataSplit):

    def __init__(self,
                 train_size: t.Union[float, int] = 0.7,
                 random_state: RandomState = None,
                 shuffle: bool = True
                 ):
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle

    def get_splits(self, dataset: Dataset) -> t.Generator[t.Tuple[Dataset, Dataset]]:
        all_rows = dataset.ids
        train_rows, test_rows = model_selection.train_test_split(all_rows,
                                                                 train_size=self.train_size,
                                                                 random_state=self.random_state,
                                                                 shuffle=self.shuffle)
        train_dataset = copy(dataset)
        train_dataset.ids = train_rows
        test_dataset = copy(dataset)
        test_dataset.ids = test_rows
        yield train_dataset, test_dataset
