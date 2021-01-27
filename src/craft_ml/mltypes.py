import numpy as np
import typing as t

KwArgs = t.Dict[str, t.Any]
Identifier = t.Union[int, str]
Object = t.Any
Label = t.Union[int, np.ndarray]
RandomState = t.Union[None, int, np.random.RandomState]
