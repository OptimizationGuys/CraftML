from pathlib import Path
import json
from .pipeline.block import BlockParams
from .pipeline.builder import Pipeline
import typing as t


def default_pipeline() -> t.List[t.Dict[str, t.Any]]:
    train_csv = '/home/andrey/tmp/train.csv'
    test_csv = '/home/andrey/tmp/test.csv'
    train_path_block = BlockParams(name='train_path',
                                   inputs=[],
                                   realization_class='Constant',
                                   realization_params={'value': train_csv})
    test_path_block = BlockParams(name='test_path',
                                  inputs=[],
                                  realization_class='Constant',
                                  realization_params={'value': test_csv})
    dataset_block = BlockParams(name='pandas_data',
                                inputs=['train_path', 'test_path'],
                                realization_class='PandasLoader',
                                realization_params={})
    training_data_raw = BlockParams(name='training_data_raw',
                                    inputs=['pandas_data'],
                                    realization_class='GetIdx',
                                    realization_params={'index': 0})
    testing_data_raw = BlockParams(name='testing_data_raw',
                                   inputs=['pandas_data'],
                                   realization_class='GetIdx',
                                   realization_params={'index': 1})
    training_data = BlockParams(name='training_data',
                                inputs=['training_data_raw'],
                                realization_class='TablePreprocess',
                                realization_params={'fn_name': 'craft_ml.processing.preprocess.fillna'})
    testing_data = BlockParams(name='testing_data',
                               inputs=['testing_data_raw'],
                               realization_class='TablePreprocess',
                               realization_params={'fn_names': [
                                   'craft_ml.processing.preprocess.findna',
                                   'craft_ml.processing.preprocess.fillna'
                               ]})
    classifier_model = BlockParams(name='classifier',
                                   inputs=[],
                                   realization_class='Wrapper',
                                   realization_params=dict(
                                       class_name='sklearn.neighbors.KNeighborsClassifier',
                                       arguments=dict(
                                           n_neighbors=5,
                                           n_jobs=-1
                                       ),
                                       method_to_run='id'
                                   ))
    training_block = BlockParams(name='training_block',
                                 inputs=['classifier', 'training_data'],
                                 realization_class='TrainModel',
                                 realization_params=dict(
                                     use_wrapper='craft_ml.processing.model.SklearnClassifier'
                                 ))
    prediction_block = BlockParams(name='prediction_block',
                                   inputs=['training_block', 'testing_data'],
                                   realization_class='InferenceModel',
                                   realization_params={}
                                   )
    return list(map(BlockParams.to_dict, [
        train_path_block, test_path_block,
        dataset_block, training_data_raw,
        testing_data_raw, training_data,
        testing_data, classifier_model,
        training_block, prediction_block
    ]))


def run_app():
    blocks = default_pipeline()
    blocks_str = json.dumps(blocks)
    print(blocks_str)
    pipeline = Pipeline(blocks_str)
    print('Loading OK')
    print(pipeline.run_pipeline())
