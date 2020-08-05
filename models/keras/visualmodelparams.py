
from typing import Tuple, List, Union
import numpy as np

#from tensorflow.keras import backend as K

from common.exception_manager import catch_error_exception
from models.visualmodelparams import VisualModelParamsBase
from preprocessing.imagegenerator import ImageGenerator


class VisualModelParams(VisualModelParamsBase):

    def __init__(self,
                 network,
                 size_image: Tuple[int, ...]
                 ) -> None:
        super(VisualModelParams, self).__init__(network, size_image)

    def _find_layer_index_from_name(self, in_name_layer: str) -> int:
        for index_layer, name_layer in enumerate(self._network.layers):
            if (name_layer.name == in_name_layer):
                return index_layer
        return None

    def _compute_feature_maps(self,
                              in_images: Union[np.ndarray, List[np.ndarray], ImageGenerator],
                              in_name_layer: str,
                              index_first_featmap: int = None,
                              max_num_featmaps: int = None
                              ) -> np.ndarray:
        # find index for "name_layer"
        index_layer = self._find_layer_index_from_name(in_name_layer)
        if not index_layer:
            message = 'Layer \'%s\' does not exist in model...' % (in_name_layer)
            catch_error_exception(message)

        model_layer_class = self._network.layers[index_layer]
        num_featmaps = model_layer_class.output.shape[-1].value

        if max_num_featmaps:
            # compute the limit indexes for feature maps,
            if not index_first_featmap:
                index_first_featmap = 0
            num_featmaps = min(max_num_featmaps, num_featmaps - index_first_featmap)
            index_last_featmap = index_first_featmap + num_featmaps

            # define function to retrieve the feature maps for "index_layer"
            get_feat_maps_layer_func = K.function([self.model.input],
                                                  [model_layer_class.output[..., index_first_featmap:index_last_featmap]])
        else:
            # define function to retrieve the feature maps for "index_layer"
            get_feat_maps_layer_func = K.function([self._network.input],
                                                  [model_layer_class.output])

        if (self._is_list_images_patches(in_images.shape)):
            # Input: list of image patches
            num_patches = in_images.shape[0]
            out_shape_featmaps = [num_patches] + list(self._size_image) + [num_featmaps]
            out_featmaps = np.zeros(out_shape_featmaps, dtype=np.float32)

            for i, ipatch_images in enumerate(in_images):
                # compute the feature maps for input patch
                ipatch_featmaps = get_feat_maps_layer_func([[ipatch_images]])
                out_featmaps[i] = ipatch_featmaps[0].astype(np.float32)

            return out_featmaps
        else:
            # Input: one full image
            # compute the feature maps for full image
            out_featmaps = get_feat_maps_layer_func([[in_images]])

            return out_featmaps[0].astype(np.float32)