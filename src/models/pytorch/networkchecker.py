
from typing import Tuple, List, Union
import numpy as np

from common.exceptionmanager import catch_error_exception
from common.functionutil import ImagesUtil
from dataloaders.batchdatagenerator import BatchDataGenerator
from models.pytorch.networks import UNet
from models.networkchecker import NetworkCheckerBase


class NetworkChecker(NetworkCheckerBase):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 network: UNet
                 ) -> None:
        super(NetworkChecker, self).__init__(size_image)
        self._network = network

    def _is_exist_name_layer_model(self, in_layer_name: str) -> bool:
        for ikey_layer_name, _ in self._network._modules.items():
            if ikey_layer_name == in_layer_name:
                return True
        return False

    def get_network_layers_names_all(self) -> List[str]:
        return [ikey_layer_name for ikey_layer_name, _ in self._network._modules.items()]

    def _compute_feature_maps(self, image_data_loader: BatchDataGenerator,
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

        num_patches = len(image_data_loader)
        out_shape_featmaps = [num_patches, num_featmaps] + list(self._size_image)
        out_featmaps = np.zeros(out_shape_featmaps, dtype=np.float32)

        self._network = self._network.eval()    # switch to evaluate mode

        for i_img, x_image in enumerate(image_data_loader):
            x_image.cuda()
            self._network(x_image)
            out_featmaps[i_img] = out_featmaps_patch  # 'out_featmaps_patch' store inside the function 'hook' above

        return ImagesUtil.reshape_channels_last(out_featmaps)   # output format "channels_last"
