import numpy as np
from ..block import Block
from .common import Wrapper
from ...data.dataset import Dataset
from ...processing.model import TrainableModel
from ...utils.common import initialize_class
from ...mltypes import KwArgs
import typing as t


class TrainModel(Block):
    def __init__(self, class_name: str, arguments: KwArgs):
        self.model: TrainableModel = initialize_class(class_name, arguments)

    def run(self, inputs: t.Any) -> TrainableModel:
        self.model.fit(inputs)
        return self.model


class InferenceModel(Block):
    def run(self, inputs: t.Tuple[TrainableModel, Dataset]) -> np.ndarray:
        return inputs[0].predict_proba(inputs[1])
