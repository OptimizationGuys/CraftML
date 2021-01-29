from dataclasses import dataclass
import typing as t
from ..mltypes import SerializableKwArgs


@dataclass
class BlockParams:
    name: str
    inputs: t.Sequence[str]
    realization_class: str
    realization_params: SerializableKwArgs


class Block:
    """
    Base class for all pipeline blocks.
    """
    def run(self, inputs: t.Any) -> t.Any:
        raise NotImplementedError
