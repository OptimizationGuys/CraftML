import pandas as pd
import numpy as np
from ..block import Block
from ...data.dataset import Dataset, TableDataset
from ...processing.model import TrainableModel
from ...utils.common import get_class, forward_compose
from ...mltypes import KwArgs
import typing as t


class TrainModel(Block):
    def __init__(self, use_wrapper: t.Optional[str] = None):
        self.wrapper = lambda x: x
        if use_wrapper is not None:
            self.wrapper = get_class(use_wrapper)
        self.model = None

    def run(self, inputs: t.Tuple[t.Any, Dataset]) -> TrainableModel:
        if self.model is None:
            self.model = self.wrapper(inputs[0])
        self.model.fit(inputs[1])
        return self.model


class InferenceModel(Block):
    def run(self, inputs: t.Tuple[TrainableModel, Dataset]) -> np.ndarray:
        return inputs[0].predict_proba(inputs[1])


class TablePreprocess(Block):
    def __init__(self, fn_names: t.Union[str, t.Sequence[str]]):
        if isinstance(fn_names, list):
            fns = []
            for cur_name in fn_names:
                fns.append(get_class(cur_name))
            self.fn = forward_compose(fns)
        else:
            self.fn = get_class(fn_names)

    def run(self, inputs: TableDataset) -> TableDataset:
        return self.fn(inputs)
