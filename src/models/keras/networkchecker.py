
from typing import Tuple, List, Union
import numpy as np

from tensorflow.keras import backend as K

from common.exceptionmanager import catch_error_exception
from dataloaders.batchdatagenerator import BatchDataGenerator
from models.keras.networks import UNet
from models.networkchecker import NetworkCheckerBase


class NetworkChecker(NetworkCheckerBase):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 network: UNet
                 ) -> None:
        super(NetworkChecker, self).__init__(size_image)
        self._network = network
        self._built_model = network.get_built_model()

    def _find_layer_index_from_name(self, in_layer_name: str) -> Union[int, None]:
        for idx_layer, i_layer in enumerate(self._built_model.layers):
            if i_layer.name == in_layer_name:
                return idx_layer
        return None

    def get_network_layers_names_all(self) -> List[str]:
        return [it_layer.name for it_layer in self._built_model.layers]

    def _compute_feature_maps(self, image_data_loader: BatchDataGenerator,
                              in_layer_name: str,
                              index_first_featmap: int = None,
                              max_num_featmaps: int = None
                              ) -> np.ndarray:
        # find index for "name_layer"
        index_layer = self._find_layer_index_from_name(in_layer_name)
        if not index_layer:
            message = 'Layer \'%s\' does not exist in model...' % (in_layer_name)
            catch_error_exception(message)

        model_layer_class = self._built_model.layers[index_layer]
        num_featmaps = model_layer_class.output.shape[-1].value

        if max_num_featmaps:
            # compute the limit indexes for feature maps,
            if not index_first_featmap:
                index_first_featmap = 0
            num_featmaps = min(max_num_featmaps, num_featmaps - index_first_featmap)
            index_last_featmap = index_first_featmap + num_featmaps

            # define function to retrieve the feature maps for "index_layer"
            get_feat_maps_layer_func = K.function(
                [self._built_model.input],
                [model_layer_class.output[..., index_first_featmap:index_last_featmap]])
        else:
            # define function to retrieve the feature maps for "index_layer"
            get_feat_maps_layer_func = K.function([self._built_model.input],
                                                  [model_layer_class.output])

        num_patches = len(image_data_loader)
        out_shape_featmaps = [num_patches] + list(self._size_image) + [num_featmaps]
        out_featmaps = np.zeros(out_shape_featmaps, dtype=np.float32)

        for i_img, x_image in enumerate(image_data_loader):
            # compute the feature maps for input patch
            ipatch_featmaps = get_feat_maps_layer_func([[x_image]])
            out_featmaps[i_img] = ipatch_featmaps[0].astype(np.float32)

        return out_featmaps
