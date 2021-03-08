from dataclasses import dataclass
import typing as t
from ..mltypes import SerializableKwArgs


@dataclass
class BlockParams:
    name: str
    inputs: t.Sequence[str]
    realization_class: str
    realization_params: SerializableKwArgs

    @staticmethod
    def to_dict(params) -> t.Dict[str, t.Any]:
        return dict(name=params.name,
                    inputs=params.inputs,
                    realization_class=params.realization_class,
                    realization_params=params.realization_params)


TypeName = str
HyperParam = t.Tuple[TypeName, t.Any]


class Block:
    """
    Base class for all pipeline blocks.
    """
    def __init__(self):
        self.cached_result = None
        self.cache_id = None

    def validate_cache(self, cache_id: int) -> bool:
        return self.cache_id is not None and cache_id == self.cache_id

    def set_cached_result(self, cache_id: int, cached_result: t.Any) -> None:
        self.cache_id = cache_id
        self.cached_result = cached_result

    def cached_run(self, cache_id: int, inputs: t.Any) -> t.Any:
        if not self.validate_cache(cache_id):
            cached_result = self.run(inputs)
            self.set_cached_result(cache_id, cached_result)
        return self.cached_result

    def run(self, inputs: t.Any) -> t.Any:
        raise NotImplementedError

    def get_hyperparams(self) -> t.Dict[str, HyperParam]:
        return {}
