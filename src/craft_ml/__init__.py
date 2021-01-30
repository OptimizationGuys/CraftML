from pathlib import Path
import json
from .pipeline.block import BlockParams
from .pipeline.builder import Pipeline
import typing as t


def loading_pipeline() -> t.List[t.Dict[str, t.Any]]:
    dataset_block = BlockParams(name='pandas_data',
                                inputs=['train_path', 'test_path', 'target_column'],
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
    return list(map(BlockParams.to_dict, [
        dataset_block, training_data_raw, testing_data_raw
    ]))


def preprocessing_pipeline() -> t.List[t.Dict[str, t.Any]]:
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
    process_train = BlockParams(name='categorized_train',
                                inputs=['train_categorizer', 'training_data'],
                                realization_class='InferenceModel',
                                realization_params={})
    process_test = BlockParams(name='categorized_test',
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
    drop_train = BlockParams(name='process_train',
                             inputs=['train_dropper', 'categorized_train'],
                             realization_class='InferenceModel',
                             realization_params={})
    drop_test = BlockParams(name='process_test',
                            inputs=['train_dropper', 'categorized_test'],
                            realization_class='InferenceModel',
                            realization_params={})
    return list(map(BlockParams.to_dict, [
        training_data,
        testing_data, categorizer,
        train_categorizer, process_train,
        process_test,
        dropper, train_dropper,
        drop_train, drop_test
    ]))


def classifier_pipeline() -> t.List[t.Dict[str, t.Any]]:
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
    split_block = BlockParams(name='split_block',
                              inputs=['train_size'],
                              realization_class='Initializer',
                              realization_params=dict(
                                  class_name='craft_ml.data.split.TrainTestSplit',
                                  arguments={'random_state': 100, 'shuffle': True}
                              ))
    splitter = BlockParams(name='splitter',
                           inputs=['split_block', 'process_train'],
                           realization_class='Apply',
                           realization_params={'method_to_run': 'get_splits'})
    train_val_data = BlockParams(name='train_val_data',
                                 inputs=['splitter'],
                                 realization_class='NextSplit',
                                 realization_params={})
    split_train_data = BlockParams(name='split_train_data',
                                   inputs=['train_val_data'],
                                   realization_class='GetIdx',
                                   realization_params={'index': 0})
    split_val_data = BlockParams(name='split_val_data',
                                 inputs=['train_val_data'],
                                 realization_class='GetIdx',
                                 realization_params={'index': 1})
    training_block = BlockParams(name='training_block',
                                 inputs=['classifier', 'split_train_data'],
                                 realization_class='TrainModel',
                                 realization_params=dict(
                                     use_wrapper='craft_ml.processing.model.SklearnClassifier'
                                 ))
    prediction_train_block = BlockParams(name='prediction_train_block',
                                         inputs=['training_block', 'split_train_data'],
                                         realization_class='InferenceModel',
                                         realization_params={}
                                         )
    prediction_val_block = BlockParams(name='prediction_val_block',
                                       inputs=['training_block', 'split_val_data'],
                                       realization_class='InferenceModel',
                                       realization_params={}
                                       )
    prediction_test_block = BlockParams(name='prediction_test_block',
                                        inputs=['training_block', 'process_test'],
                                        realization_class='InferenceModel',
                                        realization_params={}
                                        )
    return list(map(BlockParams.to_dict, [
        classifier_model, split_block,
        splitter, train_val_data, split_train_data, split_val_data,
        training_block, prediction_train_block, prediction_val_block, prediction_test_block
    ]))


def default_pipeline() -> t.List[t.Dict[str, t.Any]]:
    return loading_pipeline() + preprocessing_pipeline() + classifier_pipeline()
