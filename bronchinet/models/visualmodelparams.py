
from typing import Tuple, List, Dict, Union
import numpy as np

from common.exceptionmanager import catch_error_exception
from preprocessing.imagegenerator import ImageGenerator


class VisualModelParamsBase(object):

    def __init__(self,
                 network,
                 size_image: Tuple[int, ...]
                 ) -> None:
        self._network = network
        self._size_image = size_image

    def _is_list_images_patches(self, in_shape_image: Tuple[int, ...]) -> bool:
        if(len(in_shape_image) == len(self._size_image) + 1):
            return False
        elif(len(in_shape_image) == len(self._size_image) + 2):
            return True
        else:
            message = 'wrong \'in_shape_image\': %s...' % (in_shape_image)
            catch_error_exception(message)

    def _compute_feature_maps(self,
                              in_images: Union[np.ndarray, List[np.ndarray], ImageGenerator],
                              in_name_layer: str,
                              index_first_featmap: int = None,
                              max_num_featmaps: int = None
                              ) -> np.ndarray:
        raise NotImplementedError

    def get_feature_maps(self,
                         in_images: Union[np.ndarray, List[np.ndarray], ImageGenerator],
                         in_name_layer: str,
                         index_first_featmap: int = None,
                         max_num_featmaps: int = None
                         ) -> np.ndarray:
        return self._compute_feature_maps(in_images, in_name_layer,
                                          index_first_featmap, max_num_featmaps)

    def get_feature_maps_list_layers(self,
                                     in_images: Union[np.ndarray, List[np.ndarray], ImageGenerator],
                                     in_list_names_layers: List[str],
                                     index_first_featmap: int = None,
                                     max_num_featmaps: int = None
                                     ) -> Dict[str, np.ndarray]:
        out_dict_featmaps = {}
        for it_layer_name in in_list_names_layers:
            out_dict_featmaps[it_layer_name] = self._compute_feature_maps(in_images, it_layer_name,
                                                                          index_first_featmap, max_num_featmaps)
        return out_dict_featmaps

    def get_feature_maps_all_layers(self,
                                    in_images: Union[np.ndarray, List[np.ndarray], ImageGenerator],
                                    index_first_featmap: int = None,
                                    max_num_featmaps: int = None
                                    ) -> Dict[str, np.ndarray]:
        in_list_names_layers = [it_layer.name for it_layer in self._network.layers]
        return self.get_feature_maps_list_layers(in_images, in_list_names_layers,
                                                 index_first_featmap, max_num_featmaps)
