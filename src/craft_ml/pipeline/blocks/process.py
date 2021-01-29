import numpy as np
from ..block import Block
from ...data.dataset import Dataset
from ...processing.model import TrainableModel
from ...utils.common import initialize_class
from ...mltypes import KwArgs
import typing as t


class TrainModel(Block):
    def __init__(self, use_wrapper: t.Optional[str] = None):
        self.wrapper = lambda x: x
        if use_wrapper:
            # TODO: add import
            self.wrapper = None
        self.model = None

    def run(self, inputs: t.Tuple[t.Any, Dataset]) -> TrainableModel:
        if self.model is None:
            self.model = self.wrapper(inputs[0])
        self.model.fit(inputs[1])
        return self.model


class InferenceModel(Block):
    def run(self, inputs: t.Tuple[TrainableModel, Dataset]) -> np.ndarray:
        return inputs[0].predict_proba(inputs[1])
