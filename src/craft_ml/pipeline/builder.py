import json
from . import blocks as block_provider
import typing as t
from .block import BlockParams, Block


class Pipeline:
    def __init__(self, serialized_pipeline: str):
        self.block_params: t.List[BlockParams] = []
        self.blocks: t.List[Block] = []
        self.cached_outputs: t.Dict[str, t.Any] = {}
        self.names = set()
        objects = json.loads(serialized_pipeline)
        for cur_object in objects:
            cur_params = BlockParams(name=cur_object['name'],
                                     inputs=cur_object['inputs'],
                                     realization_class=cur_object['realization_class'],
                                     realization_params=cur_object['realization_params'])
            if cur_params.name in self.names:
                raise RuntimeError(f'Duplicated block with a name {cur_params.name}')
            self.names.update(cur_params.name)
            block_class = getattr(block_provider, cur_params.realization_class)
            cur_block = block_class(**cur_params.realization_params)
            self.block_params.append(cur_params)
            self.blocks.append(cur_block)
            self.cached_outputs[cur_params.name] = None

    def find_block(self, name: str) -> t.Tuple[BlockParams, Block, t.Any]:
        found_idx = None
        for cur_idx, cur_params in enumerate(self.block_params):
            if cur_params.name == name:
                found_idx = cur_idx
        if found_idx is None:
            raise IndexError(f'Could not find block {name} in the pipeline.')
        return self.block_params[found_idx], self.blocks[found_idx], self.cached_outputs[name]

    def get_output_names(self) -> t.List[str]:
        return list(self.names)

    def get_placeholders(self) -> t.List[str]:
        inputs = set(sum([cur_params.inputs for cur_params in self.block_params], []))
        return list(inputs - self.names)

    def serialize(self) -> str:
        serializable_params = []
        for cur_params in self.block_params:
            serializable_params.append(
                dict(name=cur_params.name,
                     inputs=cur_params.inputs,
                     realization_class=cur_params.realization_class,
                     realization_params=cur_params.realization_params)
            )
        return json.dumps(serializable_params)

    def get_output(self, name: str):
        if name not in self.cached_outputs:
            raise IndexError(f'Could not found block with a name {name}')
        return self.cached_outputs[name]

    def run_pipeline(self, inputs: t.Optional[t.Dict[str, t.Any]] = None) -> t.Any:
        if inputs is not None:
            for input_name, input_value in inputs.items():
                self.cached_outputs[input_name] = input_value
        for block_idx, (cur_block, cur_params) in enumerate(zip(self.blocks, self.block_params)):
            needed_inputs = [self.cached_outputs[cur_input] for cur_input in cur_params.inputs]
            if None in needed_inputs:
                raise RuntimeError(f'Some inputs for block {cur_params.name} on index {block_idx} '
                                   f'are calculated in the future')
            if len(needed_inputs) == 0:
                needed_inputs = None
            elif len(needed_inputs) == 1:
                needed_inputs = needed_inputs[0]
            else:
                needed_inputs = tuple(needed_inputs)
            res = cur_block.run(needed_inputs)
            self.cached_outputs[cur_params.name] = res
        return self.cached_outputs[self.block_params[-1].name]
