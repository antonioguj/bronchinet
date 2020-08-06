
from typing import Tuple, List
import numpy as np

from common.constant import TYPE_DNNLIB_USED
from common.exception_manager import catch_error_exception
from dataloaders.batchdatagenerator import BatchDataGenerator
from models.networks import get_network
from models.metrics import get_metric_train
from models.optimizers import get_optimizer
if TYPE_DNNLIB_USED == 'Pytorch':
    from models.pytorch.modeltrainer import ModelTrainer
elif TYPE_DNNLIB_USED == 'Keras':
    from models.keras.modeltrainer import ModelTrainer


class ModelTrainerBase(object):

    def __init__(self):
        self._network = None
        self._loss = None
        self._optimizer = None
        self._list_metrics = None

    def create_network(self,
                       type_network: str,
                       size_image_in: Tuple[int, ...],
                       num_levels: int,
                       num_featmaps_in: int,
                       num_channels_in: int = 1,
                       num_classes_out: int = 1,
                       is_use_valid_convols: bool = False,
                       type_activate_hidden: str = 'relu',
                       type_activate_output: str = 'sigmoid',
                       is_use_dropout: bool = False,
                       dropout_rate: float = 0.2,
                       is_use_batchnormalize: bool = False,
                       is_model_half_precision: bool = False
                       ) -> None:
        self._network = get_network(type_network,
                                    size_image_in,
                                    num_levels,
                                    num_featmaps_in,
                                    num_channels_in,
                                    num_classes_out,
                                    is_use_valid_convols=is_use_valid_convols,
                                    type_activate_hidden=type_activate_hidden,
                                    type_activate_output=type_activate_output,
                                    is_use_dropout=is_use_dropout,
                                    dropout_rate=dropout_rate,
                                    is_use_batchnormalize=is_use_batchnormalize)
        if is_model_half_precision:
            if TYPE_DNNLIB_USED == 'Pytorch':
                self._network.half()
            elif TYPE_DNNLIB_USED == 'Keras':
                message = 'Networks implementation in Keras not available in Half Precision'
                catch_error_exception(message)

    def create_loss(self, type_loss: str, is_mask_to_region_interest: bool = False) -> None:
        self._loss = get_metric_train(type_loss,
                                      is_mask_exclude=is_mask_to_region_interest)

    def create_optimizer(self, type_optimizer: str, learn_rate: float) -> None:
        model_params = self._network.parameters() if TYPE_DNNLIB_USED == 'Pytorch' else None
        self._optimizer = get_optimizer(type_optimizer,
                                        learn_rate=learn_rate,
                                        model_params=model_params)

    def create_list_metrics(self, list_type_metrics: List[str], is_mask_to_region_interest: bool = False) -> None:
        self._list_metrics = []
        for iname_metric in list_type_metrics:
            new_metric = get_metric_train(iname_metric,
                                          is_masks_exclude=is_mask_to_region_interest)
            self._list_metrics.append(new_metric)

        self._num_metrics = len(self._list_metrics)

    def compile_model(self) -> None:
        raise NotImplementedError

    def create_callbacks(self, modelspath: str, **kwargs) -> None:
        raise NotImplementedError

    def summary_model(self) -> None:
        raise NotImplementedError

    def load_model_only_weights(self, model_filename: str) -> None:
        raise NotImplementedError

    def load_model_full(self, model_filename: str, **kwargs) -> None:
        raise NotImplementedError

    def save_model_only_weights(self, model_filename: str) -> None:
        raise NotImplementedError

    def save_model_full(self, model_filename: str) -> None:
        raise NotImplementedError

    def train(self,
              train_data_loader: BatchDataGenerator,
              num_epochs: int = 1,
              max_steps_epoch: int = None,
              valid_data_loader: BatchDataGenerator = None,
              initial_epoch: int = 0
              ) -> None:
        raise NotImplementedError

    def predict(self, test_data_loader: BatchDataGenerator) -> np.ndarray:
        raise NotImplementedError