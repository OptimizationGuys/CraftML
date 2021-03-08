import json
from . import blocks as block_provider
import typing as t
from .block import BlockParams, Block


class Pipeline:
    def __init__(self, objects: t.List[t.Dict[str, t.Any]]):
        self.block_params: t.List[BlockParams] = []
        self.blocks: t.OrderedDict[str, Block] = []
        self.name_id_map = dict()
        for cur_idx, cur_object in enumerate(objects):
            cur_params = BlockParams(name=cur_object['name'],
                                     inputs=cur_object['inputs'],
                                     realization_class=cur_object['realization_class'],
                                     realization_params=cur_object['realization_params'])
            if cur_params.name in self.names:
                raise RuntimeError(f'Duplicated block with a name {cur_params.name}')
            self.name_id_map[cur_params.name] = cur_idx
            self.names.update(cur_params.name)
            block_class = getattr(block_provider, cur_params.realization_class)
            cur_block = block_class(**cur_params.realization_params)
            self.block_params.append(cur_params)
            self.blocks[cur_params.name] = cur_block
        inputs = set(sum([cur_params.inputs for cur_params in self.block_params], []))
        self.placeholder_names = list(inputs - set(self.name_id_map.keys()))

    @staticmethod
    def deserialize(serialized_pipeline: str):
        return Pipeline(json.loads(serialized_pipeline))

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
        return self.placeholder_names

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

    def get_output(self, name: str, inputs: t.Optional[t.Dict[str, t.Any]] = None) -> t.Hashable:
        if name not in self.cached_outputs:
            raise IndexError(f'Could not found block with a name {name}')
        if inputs is not None:
            # TODO: place inputs, check cache for invalidation
            pass
        # TODO: run pipeline


    def run_pipeline(self, inputs: t.Optional[t.Dict[str, t.Any]] = None) -> t.Hashable:
        return self.get_output(self.block_params[-1].name, inputs)
