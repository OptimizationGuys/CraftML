from pathlib import Path
import json
from .pipeline.block import BlockParams
from .pipeline.builder import Pipeline
import typing as t


def default_pipeline() -> t.List[t.Dict[str, t.Any]]:
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
                                realization_params={'fn_names': [
                                   'craft_ml.processing.preprocess.drop_index',
                                   'craft_ml.processing.preprocess.findna',
                                   'craft_ml.processing.preprocess.fillna'
                               ]})
    testing_data = BlockParams(name='testing_data',
                               inputs=['testing_data_raw'],
                               realization_class='TablePreprocess',
                               realization_params={'fn_names': [
                                   'craft_ml.processing.preprocess.drop_index',
                                   'craft_ml.processing.preprocess.findna',
                                   'craft_ml.processing.preprocess.fillna'
                               ]})
    categorizer = BlockParams(name='categorizer',
                              inputs=[],
                              realization_class='Wrapper',
                              realization_params=dict(
                                  class_name='craft_ml.processing.transform.ToCategory',
                                  arguments={},
                                  method_to_run='id'
                              ))
    train_categorizer = BlockParams(name='train_categorizer',
                                    inputs=['categorizer', 'training_data'],
                                    realization_class='TrainModel',
                                    realization_params={})
    process_train = BlockParams(name='process_train',
                                inputs=['train_categorizer', 'training_data'],
                                realization_class='InferenceModel',
                                realization_params={})
    process_test = BlockParams(name='process_test',
                               inputs=['train_categorizer', 'testing_data'],
                               realization_class='InferenceModel',
                               realization_params={})
    dropper = BlockParams(name='dropper',
                          inputs=[],
                          realization_class='Wrapper',
                          realization_params=dict(
                              class_name='craft_ml.processing.transform.DropStrings',
                              arguments={},
                              method_to_run='id'
                          ))
    train_dropper = BlockParams(name='train_dropper',
                                inputs=['dropper', 'training_data'],
                                realization_class='TrainModel',
                                realization_params={})
    drop_train = BlockParams(name='drop_train',
                             inputs=['train_dropper', 'process_train'],
                             realization_class='InferenceModel',
                             realization_params={})
    drop_test = BlockParams(name='drop_test',
                            inputs=['train_dropper', 'process_test'],
                            realization_class='InferenceModel',
                            realization_params={})
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
                                 inputs=['classifier', 'drop_train'],
                                 realization_class='TrainModel',
                                 realization_params=dict(
                                     use_wrapper='craft_ml.processing.model.SklearnClassifier'
                                 ))
    prediction_block = BlockParams(name='prediction_block',
                                   inputs=['training_block', 'drop_test'],
                                   realization_class='InferenceModel',
                                   realization_params={}
                                   )
    return list(map(BlockParams.to_dict, [
        dataset_block, training_data_raw,
        testing_data_raw, training_data,
        testing_data, categorizer,
        train_categorizer, process_train,
        process_test,
        dropper, train_dropper,
        drop_train, drop_test,
        classifier_model,
        training_block, prediction_block
    ]))
