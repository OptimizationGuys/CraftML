import numpy as np
import typing as t

KwArgs = t.Dict[str, t.Any]
SerializableSingleType = t.Union[str, int, float]
SerializableType = t.Union[SerializableSingleType, t.List[SerializableSingleType], t.Sequence[SerializableSingleType]]
SerializableKwArgs = t.Dict[str, SerializableType]
Identifier = t.Union[int, str]
Object = t.Any
Label = t.Union[int, np.ndarray]
RandomState = t.Union[None, int, np.random.RandomState]
