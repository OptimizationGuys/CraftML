import numpy as np
import importlib
from functools import reduce
from ..mltypes import KwArgs
import typing as t


def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def compose(fns):  #  t.Sequence[t.Callable[[...], ...]]):
    return reduce(compose2, fns)


def forward_compose(fns):  #  t.Sequence[t.Callable[[...], ...]]) -> t.Callable[[...], ...]:
    return reduce(compose2, reversed(fns))


def ensure_numpy(array_like_object) -> np.ndarray:
    if isinstance(array_like_object, np.ndarray):
        return array_like_object
    return np.asarray(array_like_object)


def get_class(class_name: str):
    mod_parts = class_name.split('.')
    mod = importlib.import_module('.'.join(mod_parts[:-1]))
    class_obj = getattr(mod, mod_parts[-1])
    return class_obj


def initialize_class(class_name: str, arguments: KwArgs) -> object:
    return get_class(class_name)(**arguments)
