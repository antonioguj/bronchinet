
import numpy as np

from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal

from common.exceptionmanager import catch_error_exception
from common.functionutil import join_path_names
from dataloaders.batchdatagenerator import BatchDataGenerator
from models.modeltrainer import ModelTrainerBase
from models.keras.callbacks import RecordLossHistory, ModelCheckpoint

NAME_SAVEDMODEL_EPOCH = 'model_e%0.2d.hdf5'
NAME_SAVEDMODEL_LAST = 'model_last.hdf5'


class ModelTrainer(ModelTrainerBase):

    def __init__(self):
        super(ModelTrainer, self).__init__()
        self._list_callbacks = None

    def _set_manual_random_seed(self, seed: int) -> None:
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
        import random
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def finalise_model(self) -> None:
        self._built_model = self._network.get_built_model()

        list_metrics_funs = [imetric.renamed_compute() for imetric in self._list_metrics]
        self._built_model.compile(optimizer=self._optimizer,
                                  loss=self._loss.lossfun,
                                  metrics=list_metrics_funs)

    def create_callbacks(self, models_path: str, losshist_filename: str, **kwargs) -> None:
        self._list_callbacks = []

        is_validation_data = kwargs['is_validation_data'] if 'is_validation_data' in kwargs.keys() \
            else True
        freq_save_check_model = kwargs['freq_save_check_model'] if 'freq_save_check_model' in kwargs.keys() \
            else 1
        # freq_validate_model = kwargs['freq_validate_model'] if 'freq_validate_model' in kwargs.keys() \
        #     else 1
        is_restart_model = kwargs['is_restart_model'] if 'is_restart_model' in kwargs.keys() \
            else False

        losshist_filename = join_path_names(models_path, losshist_filename)
        new_callback = RecordLossHistory(losshist_filename, self._list_metrics,
                                         is_hist_validation=is_validation_data,
                                         is_restart_model=is_restart_model)
        self._list_callbacks.append(new_callback)

        model_filename = join_path_names(models_path, NAME_SAVEDMODEL_EPOCH)
        new_callback = ModelCheckpoint(model_filename, self,
                                       freq_save_model=freq_save_check_model,
                                       type_save_model='full_model',
                                       update_filename_epoch=True)
        self._list_callbacks.append(new_callback)

        model_filename = join_path_names(models_path, NAME_SAVEDMODEL_LAST)
        new_callback = ModelCheckpoint(model_filename, self,
                                       type_save_model='full_model')
        self._list_callbacks.append(new_callback)

    def summary_model(self) -> None:
        self._built_model.summary()

    def load_model_only_weights(self, model_filename: str) -> None:
        self._built_model.load_weights(model_filename)

    def load_model_weights_diff_model(self, model_filename: str, **kwargs) -> None:
        type_load_model = kwargs['type_load_model']
        if type_load_model == 'UNet_noSkipConns':
            self.load_model_weights_unet_noskipconns(model_filename)
        else:
            message = "Type of loading weights from a different model not implemented: %s..." % (type_load_model)
            catch_error_exception(message)

    def load_model_full(self, model_filename: str, **kwargs) -> None:
        custom_loss = self._loss.lossfun
        custom_metrics = [imetric.renamed_compute() for imetric in self._list_metrics]
        custom_objects = dict(map(lambda fun: (fun.__name__, fun), [custom_loss] + custom_metrics))
        self._built_model = load_model(model_filename, custom_objects=custom_objects)

    def load_model_full_backward_compat(self, model_filename: str, **kwargs) -> None:
        custom_loss = self._loss.renamed_lossfun_backward_compat()
        custom_metrics = [imetric.renamed_compute() for imetric in self._list_metrics]
        custom_objects = dict(map(lambda fun: (fun.__name__, fun), [custom_loss] + custom_metrics))
        self._built_model = load_model(model_filename, custom_objects=custom_objects)

    def save_model_only_weights(self, model_filename: str) -> None:
        self._built_model.save_weights(model_filename)

    def save_model_full(self, model_filename: str) -> None:
        self._built_model.save(model_filename)

    def train(self,
              train_data_loader: BatchDataGenerator,
              valid_data_loader: BatchDataGenerator = None,
              num_epochs: int = 1,
              max_steps_epoch: int = None,
              initial_epoch: int = 0,
              is_shuffle_data: bool = False
              ) -> None:
        self._built_model.fit_generator(generator=train_data_loader,
                                        steps_per_epoch=max_steps_epoch,
                                        epochs=num_epochs,
                                        verbose=1,
                                        callbacks=self._list_callbacks,
                                        validation_data=valid_data_loader,
                                        shuffle=is_shuffle_data,
                                        initial_epoch=initial_epoch)

    def predict(self, test_data_loader: BatchDataGenerator) -> np.ndarray:
        output_prediction = self._built_model.predict(test_data_loader.get_full_data(),
                                                      batch_size=1)
        return output_prediction

    def load_model_weights_unet_noskipconns(self, model_filename: str) -> None:
        def lossfun_dummy_max(y_true, y_pred):
            from keras import backend as K
            return K.max(K.abs(y_pred - y_true), axis=-1)

        loaded_copyfrom_model = load_model(model_filename, custom_objects={'lossfun': lossfun_dummy_max})

        next_layer_convol_after_upsample = False
        for i, model_layer in enumerate(self._built_model.layers):
            layer_name = model_layer.name

            if 'concatenate' in layer_name:
                # do nothing for 'concatenate' layers
                next_layer_convol_after_upsample = True
                pass

            elif next_layer_convol_after_upsample:
                # for the convolutional layers after 'upsample + merge', we need to tweak the loaded weights
                # for this layer, the kernel weights in UNet have larger dimension than those in UNet_noSkipConn
                # extend kernel weights with added ones, initialized randomly, for the features from skip conns

                loaded_model_layer = loaded_copyfrom_model.get_layer(layer_name)
                (loaded_weights_kernel, loaded_weights_bias) = loaded_model_layer.get_weights()

                dim_weights_kernel_featsbeg = loaded_weights_kernel.shape[3]
                dim_extend_weights_kernel_featsbeg = dim_weights_kernel_featsbeg // 2

                extend_weights_kernel_shape = loaded_weights_kernel.shape[:3] + (dim_extend_weights_kernel_featsbeg,
                                                                                 loaded_weights_kernel.shape[-1])
                weights_initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)
                extend_weights_kernel = weights_initializer(shape=extend_weights_kernel_shape)

                loaded_weights_kernel = np.concatenate([loaded_weights_kernel, extend_weights_kernel], axis=-2)

                loaded_weights = (loaded_weights_kernel, loaded_weights_bias)
                model_layer.set_weights(loaded_weights)

                next_layer_convol_after_upsample = False

            else:
                # load weights normally for all the other layers
                loaded_model_layer = loaded_copyfrom_model.get_layer(layer_name)
                model_layer.set_weights(loaded_model_layer.get_weights())
