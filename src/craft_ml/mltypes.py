import numpy as np
import typing as t

Identifier = t.Union[int, str]
Object = t.Any
Label = t.Union[int, np.ndarray]
RandomState = t.Union[None, int, np.random.RandomState]
