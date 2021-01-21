import numpy as np
import typing as t


def ensure_numpy(array_like_object) -> np.ndarray:
    if isinstance(array_like_object, np.ndarray):
        return array_like_object
    return np.asarray(array_like_object)
