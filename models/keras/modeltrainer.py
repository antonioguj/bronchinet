
from typing import Dict, Any
import numpy as np

from tensorflow.keras.models import load_model

from common.constant import NAME_LOSSHISTORY_FILE, NAME_SAVEDMODEL_INTER_KERAS, NAME_SAVEDMODEL_LAST_KERAS, IS_SHUFFLE_TRAINDATA
from common.exceptionmanager import catch_error_exception
from common.functionutil import join_path_names
from dataloaders.batchdatagenerator import BatchDataGenerator
from models.modeltrainer import ModelTrainerBase
from models.keras.callbacks import RecordLossHistory, EarlyStopping, ModelCheckpoint


class ModelTrainer(ModelTrainerBase):

    def __init__(self):
        super(ModelTrainer, self).__init__()
        self._list_callbacks = None

    def finalise_model(self) -> None:
        self._compiled_model = self._network.get_compiled_model()

        list_metrics_funs = [imetric.renamed_compute() for imetric in self._list_metrics]
        self._compiled_model.compile(optimizer=self._optimizer,
                                     loss=self._loss.lossfun,
                                     metrics=list_metrics_funs)

    def create_callbacks(self, models_path: str, **kwargs) -> None:
        self._list_callbacks = []

        is_restart_model = kwargs['is_restart_model'] if 'is_restart_model' in kwargs.keys() else False
        is_validation_data = kwargs['is_validation_data'] if 'is_validation_data' in kwargs.keys() else True
        freq_save_check_model = kwargs['freq_save_check_model'] if 'freq_save_check_model' in kwargs.keys() else 1

        losshistory_filename = join_path_names(models_path, NAME_LOSSHISTORY_FILE)
        new_callback = RecordLossHistory(losshistory_filename, self._list_metrics,
                                         is_restart_model=is_restart_model,
                                         is_hist_validation=is_validation_data)
        self._list_callbacks.append(new_callback)

        model_filename = join_path_names(models_path, NAME_SAVEDMODEL_INTER_KERAS)
        new_callback = ModelCheckpoint(model_filename)
        self._list_callbacks.append(new_callback)

        model_filename = join_path_names(models_path, NAME_SAVEDMODEL_LAST_KERAS)
        new_callback = ModelCheckpoint(model_filename)
        self._list_callbacks.append(new_callback)

    def summary_model(self) -> None:
        self._compiled_model.summary()

    def load_model_only_weights(self, model_filename: str) -> None:
        self._compiled_model.load_weights(model_filename)

    def load_model_full(self, model_filename: str, **kwargs) -> None:
        list_metrics_funs = [imetric.renamed_compute() for imetric in self._list_metrics]
        custom_objects = dict(map(lambda fun: (fun.__name__, fun), [self._loss.lossfun] + list_metrics_funs))
        self._compiled_model = load_model(model_filename, custom_objects=custom_objects)

    def load_model_full_backward_compat(self, model_filename: str, **kwargs) -> None:
        self.load_model_full(model_filename, **kwargs)

    def save_model_only_weights(self, model_filename: str) -> None:
        pass

    def save_model_full(self, model_filename: str) -> None:
        pass

    def get_size_output_image_model(self):
        return self._compiled_model.outputs[0].shape[1:-1]
        #return self._network.get_size_output()[:-1]

    def train(self,
              train_data_loader: BatchDataGenerator,
              valid_data_loader: BatchDataGenerator = None,
              num_epochs: int = 1,
              max_steps_epoch: int = None,
              initial_epoch: int = 0
              ) -> None:
        self._compiled_model.fit_generator(generator=train_data_loader,
                                           steps_per_epoch=max_steps_epoch,
                                           epochs=num_epochs,
                                           verbose=1,
                                           callbacks=self._list_callbacks,
                                           validation_data=valid_data_loader,
                                           shuffle=IS_SHUFFLE_TRAINDATA,
                                           initial_epoch=initial_epoch)

    def predict(self, test_data_loader: BatchDataGenerator) -> np.ndarray:
        output_prediction = self._compiled_model.predict(test_data_loader.get_full_data(),
                                                         batch_size=1)
        return output_prediction