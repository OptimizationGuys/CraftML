import numpy as np
import typing as t
from ..data.dataset import Dataset
from ..mltypes import Label


Predictor = t.Callable[[Dataset], np.ndarray]


class Validator:

    def __init__(self, validation_metric: t.Callable[[np.ndarray, t.Sequence[Label]], np.ndarray]):
        self.validation_metric = validation_metric

    def validate(self, dataset: Dataset, predictor: Predictor) -> np.ndarray:
        return self.validation_metric(predictor(dataset), dataset.get_labels())
