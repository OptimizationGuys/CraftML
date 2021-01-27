import typing as t
from ..block import Block
from ...mltypes import KwArgs


class IdentityBlock(Block):
    def run(self, inputs: t.Any) -> t.Any:
        return inputs


class GetIdx(Block):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def run(self, inputs: t.Sequence[t.Any]) -> t.Any:
        if len(inputs) >= self.index:
            raise IndexError(f'index {self.index} out of range input length of {len(inputs)}')
        return inputs[self.index]


class Flatten(Block):
    def run(self, inputs: t.Sequence[t.Any]) -> t.Sequence[t.Any]:
        flat_inputs = []
        for el in inputs:
            if isinstance(el, list):
                flat_inputs += el
            else:
                flat_inputs.append(el)
        return flat_inputs


class Wrapper(Block):
    def __init__(self, class_name: str, arguments: KwArgs, method_to_run: t.Optional[str] = None):
        super().__init__()
        # TODO: add class initialization
        self.object = None
        self.runnable = getattr(self.object, method_to_run)

    def run(self, inputs: t.Any) -> t.Any:
        return self.runnable(inputs)
