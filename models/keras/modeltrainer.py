
from typing import Dict, Any
import numpy as np

from tensorflow.keras.models import load_model

from common.constant import NAME_LOSSHISTORY_FILE, NAME_SAVEDMODELS_EPOCHS_KERAS, NAME_SAVEDMODELS_LAST_KERAS, SHUFFLETRAINDATA
from common.exception_manager import catch_error_exception
from common.function_util import join_path_names
from dataloaders.batchdatagenerator import BatchDataGenerator
from models.modeltrainer import ModelTrainerBase
from models.keras.callbacks import RecordLossHistory, EarlyStopping, ModelCheckpoint


class ModelTrainer(ModelTrainerBase):

    def __init__(self):
        super(ModelTrainer, self).__init__()
        self._list_callbacks = None

    def compile_model(self) -> None:
        list_metrics_funs = [imetric.renamed_compute() for imetric in self._list_metrics]
        self._network.compile(optimizer=self._optimizer,
                              loss=self._loss.lossfun,
                              metrics=list_metrics_funs)

    def create_callbacks(self, modelspath: str, **kwargs) -> None:
        self._list_callbacks = []

        loss_filename = join_path_names(modelspath, NAME_LOSSHISTORY_FILE)
        new_callback = RecordLossHistory(loss_filename, self._list_metrics),
        self._list_callbacks.append(new_callback)

        model_filename = join_path_names(modelspath, NAME_SAVEDMODELS_EPOCHS_KERAS)
        new_callback = ModelCheckpoint(model_filename)
        self._list_callbacks.append(new_callback)

        model_filename = join_path_names(modelspath, NAME_SAVEDMODELS_LAST_KERAS)
        new_callback = ModelCheckpoint(model_filename)
        self._list_callbacks.append(new_callback)

    def summary_model(self) -> None:
        self._network.summary()

    def load_model_only_weights(self, model_filename: str) -> None:
        self._network.load_weights(model_filename)

    def load_model_full(self, model_filename: str, **kwargs) -> None:
        list_metrics_funs = [imetric.renamed_compute() for imetric in self._list_metrics]
        custom_objects = dict(map(lambda fun: (fun.__name__, fun), [self._loss.lossfun] + list_metrics_funs))
        return load_model(model_filename, custom_objects=custom_objects)

    def save_model_only_weights(self, model_filename: str) -> None:
        pass

    def save_model_full(self, model_filename: str) -> None:
        pass

    def train(self,
              train_data_generator: BatchDataGenerator,
              valid_data_generator: BatchDataGenerator = None,
              num_epochs: int = 1,
              max_steps_epoch: int = None,
              initial_epoch: int = 0
              ) -> None:
        self._network.fit_generator(generator=train_data_generator,
                                    steps_per_epoch=max_steps_epoch,
                                    epochs=num_epochs,
                                    verbose=1,
                                    callbacks=self._list_callbacks,
                                    validation_data=valid_data_generator,
                                    shuffle=SHUFFLETRAINDATA,
                                    initial_epoch=initial_epoch)

    def predict(self, test_data_generator: BatchDataGenerator) -> np.ndarray:
        output_prediction = self._network.predict(test_data_generator.get_full_data(),
                                                  batch_size=1)
        return output_prediction