import typing as t
from .block import BlockParams, Block


Pipeline = t.Sequence[Block]


def serialize(pipeline: Pipeline) -> str:
    pass


def deserialize(serialized_pipeline: str) -> Pipeline:
    pass


def run_pipeline(pipeline: Pipeline, inputs: t.Sequence[t.Any]) -> t.Sequence[t.Any]:
    pass
