import numpy as np
import sklearn
import typing as t
from ..data.dataset import Dataset
from ..utils.common import ensure_numpy
from ..mltypes import Label


class TrainableModel:
    """
    Base interface for trainable models
    """
    def __init__(self, model: t.Any):
        self.model = model

    def fit(self, dataset: Dataset) -> None:
        raise NotImplementedError

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        raise NotImplementedError

    def predict(self, dataset: Dataset) -> t.Sequence[Label]:
        raise NotImplementedError


class SklearnClassifier(TrainableModel):

    def __init__(self, classifier: sklearn.base.ClassifierMixin):
        super().__init__(classifier)
        assert hasattr(classifier, 'fit')
        assert hasattr(classifier, 'predict')
        assert hasattr(classifier, 'predict_proba')

    def fit(self, dataset: Dataset) -> None:
        objects = ensure_numpy(dataset.get_objects())
        labels = ensure_numpy(dataset.get_labels())
        self.model.fit(objects, labels)

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        objects = ensure_numpy(dataset.get_objects())
        return self.model.predict_proba(objects)

    def predict(self, dataset: Dataset) -> np.ndarray:
        objects = ensure_numpy(dataset.get_objects())
        return self.model.predict(objects)
