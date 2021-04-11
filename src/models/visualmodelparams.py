
from typing import Tuple, List, Dict, Union
import numpy as np

from common.exceptionmanager import catch_error_exception
from models.networks import UNetBase
from preprocessing.imagegenerator import ImageGenerator


class VisualModelParamsBase(object):

    def __init__(self,
                 network: UNetBase,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]]
                 ) -> None:
        self._network = network
        self._size_image = size_image

    def _is_list_images_patches(self, shape_in_image: Tuple[int, ...]) -> bool:
        if len(shape_in_image) == len(self._size_image) + 1:
            return False
        elif len(shape_in_image) == len(self._size_image) + 2:
            return True
        else:
            message = 'wrong \'in_shape_image\': %s...' % (shape_in_image)
            catch_error_exception(message)

    def get_network_layers_names_all(self) -> List[str]:
        raise NotImplementedError

    def _compute_feature_maps(self,
                              in_images: Union[np.ndarray, List[np.ndarray], ImageGenerator],
                              in_layer_name: str,
                              index_first_featmap: int = None,
                              max_num_featmaps: int = None
                              ) -> np.ndarray:
        raise NotImplementedError

    def get_feature_maps(self,
                         in_images: Union[np.ndarray, List[np.ndarray], ImageGenerator],
                         in_layer_name: str,
                         index_first_featmap: int = None,
                         max_num_featmaps: int = None
                         ) -> np.ndarray:
        return self._compute_feature_maps(in_images, in_layer_name,
                                          index_first_featmap, max_num_featmaps)

    def get_feature_maps_list_layers(self,
                                     in_images: Union[np.ndarray, List[np.ndarray], ImageGenerator],
                                     in_list_layers_names: List[str],
                                     index_first_featmap: int = None,
                                     max_num_featmaps: int = None
                                     ) -> Dict[str, np.ndarray]:
        out_dict_featmaps = {}
        for it_layer_name in in_list_layers_names:
            out_dict_featmaps[it_layer_name] = self._compute_feature_maps(in_images, it_layer_name,
                                                                          index_first_featmap, max_num_featmaps)
        return out_dict_featmaps

    def get_feature_maps_all_layers(self,
                                    in_images: Union[np.ndarray, List[np.ndarray], ImageGenerator],
                                    index_first_featmap: int = None,
                                    max_num_featmaps: int = None
                                    ) -> Dict[str, np.ndarray]:
        in_list_layers_names = self.get_network_layers_names_all()
        return self.get_feature_maps_list_layers(in_images, in_list_layers_names,
                                                 index_first_featmap, max_num_featmaps)
