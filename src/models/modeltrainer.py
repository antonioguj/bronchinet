
from typing import Tuple, List, Union
import numpy as np

from common.constant import TYPE_DNNLIB_USED
from dataloaders.batchdatagenerator import BatchDataGenerator
from models.model_manager import get_network, get_metric_train, get_optimizer


class ModelTrainerBase(object):

    def __init__(self):
        self._network = None
        self._loss = None
        self._optimizer = None
        self._list_metrics = None

    def _set_manual_random_seed(self, seed: int) -> None:
        raise NotImplementedError

    def create_network(self,
                       type_network: str,
                       size_image_in: Union[Tuple[int, int, int], Tuple[int, int]],
                       num_featmaps_in: int,
                       num_channels_in: int = 1,
                       num_classes_out: int = 1,
                       is_use_valid_convols: bool = False,
                       num_levels: int = 5,
                       type_activate_hidden: str = 'relu',
                       type_activate_output: str = 'sigmoid',
                       is_use_dropout: bool = False,
                       dropout_rate: float = 0.2,
                       is_use_batchnormalize: bool = False,
                       manual_seed: int = None
                       ) -> None:
        if manual_seed is not None:
            self._set_manual_random_seed(manual_seed)

        self._network = get_network(type_network=type_network,
                                    size_image_in=size_image_in,
                                    num_featmaps_in=num_featmaps_in,
                                    num_channels_in=num_channels_in,
                                    num_classes_out=num_classes_out,
                                    is_use_valid_convols=is_use_valid_convols,
                                    num_levels=num_levels,
                                    type_activate_hidden=type_activate_hidden,
                                    type_activate_output=type_activate_output,
                                    is_use_dropout=is_use_dropout,
                                    dropout_rate=dropout_rate,
                                    is_use_batchnormalize=is_use_batchnormalize)

    def create_loss(self, type_loss: str, is_mask_to_region_interest: bool = False,
                    weight_combined_loss: float = 1.0) -> None:
        self._loss = get_metric_train(type_metric=type_loss,
                                      is_mask_exclude=is_mask_to_region_interest,
                                      weight_combined_loss=weight_combined_loss)

    def create_optimizer(self, type_optimizer: str, learn_rate: float) -> None:
        model_params = self._network.parameters() if TYPE_DNNLIB_USED == 'Pytorch' else None
        self._optimizer = get_optimizer(type_optimizer=type_optimizer,
                                        learn_rate=learn_rate,
                                        model_params=model_params)

    def create_list_metrics(self, list_type_metrics: List[str], is_mask_to_region_interest: bool = False) -> None:
        self._list_metrics = []
        for iname_metric in list_type_metrics:
            new_metric = get_metric_train(type_metric=iname_metric,
                                          is_masks_exclude=is_mask_to_region_interest)
            self._list_metrics.append(new_metric)

        self._num_metrics = len(self._list_metrics)

    def finalise_model(self) -> None:
        raise NotImplementedError

    def create_callbacks(self, models_path: str, losshist_filename: str, **kwargs) -> None:
        raise NotImplementedError

    def summary_model(self) -> None:
        raise NotImplementedError

    def load_model_only_weights(self, model_filename: str) -> None:
        raise NotImplementedError

    def load_model_full(self, model_filename: str, **kwargs) -> None:
        raise NotImplementedError

    def load_model_full_backward_compat(self, model_filename: str, **kwargs) -> None:
        raise NotImplementedError

    def save_model_only_weights(self, model_filename: str) -> None:
        raise NotImplementedError

    def save_model_full(self, model_filename: str) -> None:
        raise NotImplementedError

    def get_size_output_image_model(self) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        return self._network.get_size_output_last_layer()

    def train(self,
              train_data_loader: BatchDataGenerator,
              valid_data_loader: BatchDataGenerator = None,
              num_epochs: int = 1,
              max_steps_epoch: int = None,
              initial_epoch: int = 0,
              is_shuffle_data: bool = False
              ) -> None:
        raise NotImplementedError

    def predict(self, test_data_loader: BatchDataGenerator) -> np.ndarray:
        raise NotImplementedError
