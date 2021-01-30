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


class Block:
    """
    Base class for all pipeline blocks.
    """
    def run(self, inputs: t.Any) -> t.Any:
        raise NotImplementedError
