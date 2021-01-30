import importlib
import typing as t
from ..block import Block
from ...mltypes import KwArgs
from ...utils.common import initialize_class, get_class


class IdentityBlock(Block):
    def run(self, inputs: t.Any) -> t.Any:
        return inputs


Placeholder = IdentityBlock


class Constant(Block):
    def __init__(self, value: t.Any):
        self.value = value

    def run(self, inputs: None = None) -> t.Any:
        return self.value


class GetIdx(Block):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def run(self, inputs: t.Sequence[t.Any]) -> t.Any:
        if len(inputs) <= self.index:
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


class Initializer(Block):
    def __init__(self, class_name: str, arguments: KwArgs):
        self.class_init = get_class(class_name)
        self.arguments = arguments

    def run(self, inputs: t.Optional[t.Union[KwArgs, t.Sequence[t.Any]]] = None) -> object:
        cur_kwargs = self.arguments
        if inputs is not None:
            if isinstance(inputs, dict):
                cur_args = []
                cur_kwargs.update(inputs)
            elif isinstance(inputs, list) or isinstance(inputs, tuple):
                cur_args = inputs
            else:
                cur_args = [inputs]
        return self.class_init(*cur_args, **cur_kwargs)


class Apply(Block):
    def __init__(self, method_to_run: str = '__call__'):
        self.method_to_run = method_to_run

    def run(self, inputs: t.Tuple[t.Callable[[t.Any], t.Any], t.Any]) -> t.Any:
        fn = getattr(inputs[0], self.method_to_run)
        return fn(inputs[1])


class Wrapper(Block):
    def __init__(self, class_name: str, arguments: KwArgs, method_to_run: t.Optional[str] = None):
        super().__init__()
        self.object = initialize_class(class_name, arguments)
        if method_to_run is None:
            method_to_run = '__call__'
        if method_to_run == 'id':
            self.runnable = lambda _: self.object
        else:
            self.runnable = getattr(self.object, method_to_run)

    def run(self, inputs: t.Any) -> t.Any:
        return self.runnable(inputs)
