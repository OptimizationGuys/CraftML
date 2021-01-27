import json
from . import blocks as block_provider
import typing as t
from .block import BlockParams, Block


class Pipeline:
    def __init__(self, serialized_pipeline: str):
        self.block_params: t.List[BlockParams] = []
        self.blocks: t.List[Block] = []
        self.cached_outputs: t.List[t.Any] = []
        objects = json.loads(serialized_pipeline)
        for cur_object in objects:
            cur_params = BlockParams(name=cur_object.name,
                                     inputs=cur_object.inputs,
                                     realization_class=cur_object.realization_class,
                                     realization_params=cur_object.realization_params)
            block_class = getattr(block_provider, cur_params.realization_class)
            cur_block = block_class(**cur_params.realization_params)
            self.block_params.append(block_class)
            self.blocks.append(cur_block)
            self.cached_outputs.append(None)

    def serialize(self) -> str:
        # TODO: add serialization
        pass

    def run_pipeline(self, inputs: t.Any) -> t.Any:
        for block_idx, (cur_block, cur_params) in enumerate(zip(self.blocks, self.block_params)):
            needed_inputs = [self.cached_outputs[cur_input] for cur_input in cur_params.inputs]
            if None in needed_inputs:
                raise RuntimeError(f'Some inputs for block {cur_params.name} on index {block_idx} '
                                   f'are calculated in the future')
            self.cached_outputs[block_idx] = cur_block.run(needed_inputs)
        return self.cached_outputs[-1]
