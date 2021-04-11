
from typing import Tuple, List, Union
import numpy as np

from common.exceptionmanager import catch_error_exception
from common.functionutil import ImagesUtil
from models.pytorch.networks import UNet
from models.visualmodelparams import VisualModelParamsBase
from preprocessing.imagegenerator import ImageGenerator


class VisualModelParams(VisualModelParamsBase):

    def __init__(self,
                 network: UNet,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]]
                 ) -> None:
        super(VisualModelParams, self).__init__(network, size_image)

    def _is_exist_name_layer_model(self, in_layer_name: str) -> bool:
        for ikey_layer_name, _ in self._network._modules.items():
            if ikey_layer_name == in_layer_name:
                return True
        return False

    def get_network_layers_names_all(self) -> List[str]:
        return [ikey_layer_name for ikey_layer_name, _ in self._network._modules.items()]

    def _compute_feature_maps(self,
                              in_images_generator: Union[np.ndarray, List[np.ndarray], ImageGenerator],
                              in_layer_name: str,
                              index_first_featmap: int = None,
                              max_num_featmaps: int = None
                              ) -> np.ndarray:
        # check that "name_layer" exists in model
        if not self._is_exist_name_layer_model(in_layer_name):
            message = 'Layer \'%s\' does not exist in model...' % (in_layer_name)
            catch_error_exception(message)

        num_featmaps = 8

        # define hook to retrieve feature maps of the desired model layer
        def hook(model, input, output):
            global out_featmaps_patch
            out_featmaps_patch = output.detach().cpu()
            # this should be eventually inside Networks.forward()
            out_featmaps_patch = out_featmaps_patch.view(-1, num_featmaps, self._size_image[0],
                                                         self._size_image[1], self._size_image[2])
            return None

        # attach hook to the corresponding module layer
        self._network._modules[in_layer_name].register_forward_hook(hook)

        num_patches = len(in_images_generator)
        out_shape_featmaps = [num_patches, num_featmaps] + list(self._size_image)
        out_featmaps = np.zeros(out_shape_featmaps, dtype=np.float32)

        self._network = self._network.eval()
        self._network.preprocess(-1)

        for i_batch, (x_batch, y_batch) in enumerate(in_images_generator):
            x_batch.cuda()
            self._network(x_batch)
            out_featmaps[i_batch] = out_featmaps_patch  # 'out_featmaps_patch' store inside the function 'hook' above

        return ImagesUtil.reshape_channels_last(out_featmaps)   # output format "channels_last"
