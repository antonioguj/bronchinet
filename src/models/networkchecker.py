
from typing import Tuple, List, Dict, Union
import numpy as np

from dataloaders.batchdatagenerator import BatchDataGenerator


class NetworkCheckerBase(object):

    def __init__(self, size_image: Union[Tuple[int, int, int], Tuple[int, int]]) -> None:
        self._size_image = size_image

    def get_network_layers_names_all(self) -> List[str]:
        raise NotImplementedError

    def _compute_feature_maps(self, image_data_loader: BatchDataGenerator,
                              in_layer_name: str,
                              index_first_featmap: int = None,
                              max_num_featmaps: int = None
                              ) -> np.ndarray:
        raise NotImplementedError

    def get_feature_maps(self, image_data_loader: BatchDataGenerator,
                         in_layer_name: str,
                         index_first_featmap: int = None,
                         max_num_featmaps: int = None
                         ) -> np.ndarray:
        return self._compute_feature_maps(image_data_loader, in_layer_name,
                                          index_first_featmap, max_num_featmaps)

    def get_feature_maps_list_layers(self, image_data_loader: BatchDataGenerator,
                                     in_list_layers_names: List[str],
                                     index_first_featmap: int = None,
                                     max_num_featmaps: int = None
                                     ) -> Dict[str, np.ndarray]:
        out_dict_featmaps = {}
        for it_layer_name in in_list_layers_names:
            out_dict_featmaps[it_layer_name] = self._compute_feature_maps(image_data_loader, it_layer_name,
                                                                          index_first_featmap, max_num_featmaps)
        return out_dict_featmaps

    def get_feature_maps_all_layers(self, image_data_loader: BatchDataGenerator,
                                    index_first_featmap: int = None,
                                    max_num_featmaps: int = None
                                    ) -> Dict[str, np.ndarray]:
        in_list_layers_names = self.get_network_layers_names_all()
        return self.get_feature_maps_list_layers(image_data_loader, in_list_layers_names,
                                                 index_first_featmap, max_num_featmaps)
