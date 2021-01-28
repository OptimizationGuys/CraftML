import numpy as np
from ..mltypes import KwArgs
import typing as t


def ensure_numpy(array_like_object) -> np.ndarray:
    if isinstance(array_like_object, np.ndarray):
        return array_like_object
    return np.asarray(array_like_object)


def initialize_class(class_name: str, arguments: KwArgs) -> object:
    raise NotImplementedError
