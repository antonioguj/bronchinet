
from typing import Tuple, List, Dict, Any

from torch.nn import Conv3d, ConvTranspose3d, MaxPool3d, Upsample, BatchNorm3d, Dropout3d, ReLU, Sigmoid, Softmax
import torch.nn as nn
import torch.nn.functional as F
import torch

from common.exception_manager import catch_error_exception
from networks.networks import UNetBase

LIST_AVAIL_NETWORKS = ['UNet3D_Original',
                       'UNet3D_General',
                       'UNet3D_Plugin',
                       ]


class UNet(UNetBase, nn.Module):

    def __init__(self,
                 size_image_in: Tuple[int, ...],
                 num_levels: int,
                 num_featmaps_in: int,
                 num_channels_in: int,
                 num_classes_out: int,
                 is_use_valid_convols: bool = False
                 ) -> None:
        super(UNet, self).__init__(size_image_in, num_levels, num_channels_in, num_classes_out, num_featmaps_in,
                                   is_use_valid_convols=is_use_valid_convols)

    @staticmethod
    def get_create_model(type_network: str, **kwargs) -> UNetBase:
        if type_network == 'UNet3D_Original':
            return UNet3D_Original(**kwargs)
        elif type_network == 'UNet3D_General':
            return UNet3D_General(**kwargs)
        elif type_network == 'UNet3D_Plugin':
            return UNet3D_Plugin(**kwargs)
        else:
            message = 'wrong input network \'type_model\': %s...' % (type_network)
            catch_error_exception(message)

    def _build_list_info_crop_where_merge(self) -> None:
        indexes_output_where_merge = [i for i, el in enumerate(self._list_opers_names_layers_all) if el == 'upsample']
        self._list_sizes_crop_where_merge = [self._list_sizes_output_all_layers[i] for i in indexes_output_where_merge][::-1]


class UNet3D_Original(UNet):

    def __init__(self,
                 size_image_in: Tuple[int, int, int],
                 num_featmaps_in: int = 16,
                 num_channels_in: int = 1,
                 num_classes_out: int = 1
                 ) -> None:
        num_levels = 5
        super(UNet3D_Original, self).__init__(size_image_in, num_levels, num_featmaps_in, num_channels_in, num_classes_out,
                                              is_use_valid_convols=False)
        self._build_model()

    def get_model_construct_input_args(self) -> List[str, Dict[str, Any]]:
        return ['UNet3D_Original', {'size_image': self._size_image_in,
                                    'num_featmaps_in': self._num_featmaps_in,
                                    'num_channels_in': self._num_channels_in,
                                    'num_classes_out': self._num_classes_out}]

    def _build_model(self) -> None:

        num_featmaps_lev1 = self._num_featmaps_in
        self.convolution_down_lev1_1 = Conv3d(self.num_channels_in, num_featmaps_lev1, kernel_size=3, padding=1)
        self.convolution_down_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3, padding=1)
        self.pooling_down_lev1 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev2 = 2 * num_featmaps_lev1
        self.convolution_down_lev2_1 = Conv3d(num_featmaps_lev1, num_featmaps_lev2, kernel_size=3, padding=1)
        self.convolution_down_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3, padding=1)
        self.pooling_down_lev2 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev3 = 2 * num_featmaps_lev2
        self.convolution_down_lev3_1 = Conv3d(num_featmaps_lev2, num_featmaps_lev3, kernel_size=3, padding=1)
        self.convolution_down_lev3_2 = Conv3d(num_featmaps_lev3, num_featmaps_lev3, kernel_size=3, padding=1)
        self.pooling_down_lev3 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev4 = 2 * num_featmaps_lev3
        self.convolution_down_lev4_1 = Conv3d(num_featmaps_lev3, num_featmaps_lev4, kernel_size=3, padding=1)
        self.convolution_down_lev4_2 = Conv3d(num_featmaps_lev4, num_featmaps_lev4, kernel_size=3, padding=1)
        self.pooling_down_lev4 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev5 = 2 * num_featmaps_lev4
        self.convolution_down_lev5_1 = Conv3d(num_featmaps_lev4, num_featmaps_lev5, kernel_size=3, padding=1)
        self.convolution_down_lev5_2 = Conv3d(num_featmaps_lev5, num_featmaps_lev5, kernel_size=3, padding=1)
        self.upsample_up_lev5 = Upsample(scale_factor=2, mode='nearest')

        num_featmaps_lev4pl5 = num_featmaps_lev4 + num_featmaps_lev5
        self.convolution_up_lev4_1 = Conv3d(num_featmaps_lev4pl5, num_featmaps_lev4, kernel_size=3, padding=1)
        self.convolution_up_lev4_2 = Conv3d(num_featmaps_lev4, num_featmaps_lev4, kernel_size=3, padding=1)
        self.upsample_up_lev4 = Upsample(scale_factor=2, mode='nearest')

        num_featmaps_lev3pl4 = num_featmaps_lev3 + num_featmaps_lev4
        self.convolution_up_lev3_1 = Conv3d(num_featmaps_lev3pl4, num_featmaps_lev3, kernel_size=3, padding=1)
        self.convolution_up_lev3_2 = Conv3d(num_featmaps_lev3, num_featmaps_lev3, kernel_size=3, padding=1)
        self.upsample_up_lev3 = Upsample(scale_factor=2, mode='nearest')

        num_featmaps_lev2pl3 = num_featmaps_lev2 + num_featmaps_lev3
        self.convolution_up_lev2_1 = Conv3d(num_featmaps_lev2pl3, num_featmaps_lev2, kernel_size=3, padding=1)
        self.convolution_up_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3, padding=1)
        self.upsample_up_lev2 = Upsample(scale_factor=2, mode='nearest')

        num_featmaps_lev1pl2 = num_featmaps_lev1 + num_featmaps_lev2
        self.convolution_up_lev1_1 = Conv3d(num_featmaps_lev1pl2, num_featmaps_lev1, kernel_size=3, padding=1)
        self.convolution_up_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3, padding=1)

        self.classification_last = Conv3d(num_featmaps_lev1, self.num_classes_out, kernel_size=1, padding=0)
        self.activation_last = Sigmoid()

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:

        hidden_next = self.convolution_down_lev1_1(input)
        hidden_next = self.convolution_down_lev1_2(hidden_next)
        skipconn_lev1 = hidden_next
        hidden_next = self.pooling_down_lev1(hidden_next)

        hidden_next = self.convolution_down_lev2_1(hidden_next)
        hidden_next = self.convolution_down_lev2_2(hidden_next)
        skipconn_lev2 = hidden_next
        hidden_next = self.pooling_down_lev2(hidden_next)

        hidden_next = self.convolution_down_lev3_1(hidden_next)
        hidden_next = self.convolution_down_lev3_2(hidden_next)
        skipconn_lev3 = hidden_next
        hidden_next = self.pooling_down_lev3(hidden_next)

        hidden_next = self.convolution_down_lev4_1(hidden_next)
        hidden_next = self.convolution_down_lev4_2(hidden_next)
        skipconn_lev4 = hidden_next
        hidden_next = self.pooling_down_lev4(hidden_next)

        hidden_next = self.convolution_down_lev5_1(hidden_next)
        hidden_next = self.convolution_down_lev5_2(hidden_next)
        hidden_next = self.upsample_up_lev5(hidden_next)

        hidden_next = torch.cat([hidden_next, skipconn_lev4], dim=1)
        hidden_next = self.convolution_up_lev4_1(hidden_next)
        hidden_next = self.convolution_up_lev4_2(hidden_next)
        hidden_next = self.upsample_up_lev4(hidden_next)

        hidden_next = torch.cat([hidden_next, skipconn_lev3], dim=1)
        hidden_next = self.convolution_up_lev3_1(hidden_next)
        hidden_next = self.convolution_up_lev3_2(hidden_next)
        hidden_next = self.upsample_up_lev3(hidden_next)

        hidden_next = torch.cat([hidden_next, skipconn_lev2], dim=1)
        hidden_next = self.convolution_up_lev2_1(hidden_next)
        hidden_next = self.convolution_up_lev2_2(hidden_next)
        hidden_next = self.upsample_up_lev2(hidden_next)

        hidden_next = torch.cat([hidden_next, skipconn_lev1], dim=1)
        hidden_next = self.convolution_up_lev1_1(hidden_next)
        hidden_next = self.convolution_up_lev1_2(hidden_next)

        output = self.activation_last(self.classification_last(hidden_next))
        return output


class UNet3D_General(UNet):
    pass


class UNet3D_Plugin(UNet):
    _num_levels_default = 5
    _num_levels_non_padded = 3
    _num_featmaps_in_default = 16
    _num_channels_in_default = 1
    _num_classes_out_default = 1
    _dropout_rate_default = 0.2
    _type_activate_hidden_default = 'relu'
    _type_activate_output_default = 'sigmoid'

    def __init__(self,
                 size_image_in: Tuple[int, int, int],
                 num_levels: int = _num_levels_default,
                 num_featmaps_in: int = _num_featmaps_in_default,
                 num_channels_in: int = _num_channels_in_default,
                 num_classes_out: int = _num_classes_out_default,
                 is_use_valid_convols: bool = False,
                 type_activate_hidden: str = _type_activate_hidden_default,
                 type_activate_output: str = _type_activate_output_default
                 ) -> None:
        super(UNet3D_Plugin, self).__init__(size_image_in, num_levels, num_featmaps_in, num_channels_in, num_classes_out,
                                            is_use_valid_convols=is_use_valid_convols)

        self._type_activate_hidden = type_activate_hidden
        self._type_activate_output = type_activate_output

        self._build_model()

    def get_model_construct_input_args(self) -> List[str, Dict[str, Any]]:
        return ['UNet3D_Plugin', {'size_image_in': self._size_image_in,
                                  'num_levels': self._num_levels,
                                  'num_featmaps_in': self._num_featmaps_in,
                                  'num_channels_in': self._num_channels_in,
                                  'num_classes_out': self._num_classes_out,
                                  'is_use_valid_convols': self._is_use_valid_convols,
                                  'type_activate_hidden': self._type_activate_hidden,
                                  'type_activate_output': self._type_activate_output}]

    def _build_model(self) -> None:
        padding_value = 0 if self._is_use_valid_convols else 1

        num_featmaps_lev1 = self.num_featmaps_in
        self.convolution_down_lev1_1 = Conv3d(self.num_channels_in, num_featmaps_lev1, kernel_size=3, padding=padding_value)
        self.convolution_down_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3, padding=padding_value)
        self.pooling_down_lev1 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev2 = 2 * num_featmaps_lev1
        self.convolution_down_lev2_1 = Conv3d(num_featmaps_lev1, num_featmaps_lev2, kernel_size=3, padding=padding_value)
        self.convolution_down_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3, padding=padding_value)
        self.pooling_down_lev2 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev3 = 2 * num_featmaps_lev2
        self.convolution_down_lev3_1 = Conv3d(num_featmaps_lev2, num_featmaps_lev3, kernel_size=3, padding=padding_value)
        self.convolution_down_lev3_2 = Conv3d(num_featmaps_lev3, num_featmaps_lev3, kernel_size=3, padding=padding_value)
        self.pooling_downlay3 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev4 = 2 * num_featmaps_lev3
        self.convolution_down_lev4_1 = Conv3d(num_featmaps_lev3, num_featmaps_lev4, kernel_size=3, padding=1)
        self.convolution_down_lev4_2 = Conv3d(num_featmaps_lev4, num_featmaps_lev4, kernel_size=3, padding=1)
        self.pooling_downlay4 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev5 = 2 * num_featmaps_lev4
        self.convolution_down_lev5_1 = Conv3d(num_featmaps_lev4, num_featmaps_lev5, kernel_size=3, padding=1)
        self.convolution_down_lev5_2 = Conv3d(num_featmaps_lev5, num_featmaps_lev5, kernel_size=3, padding=1)
        self.upsample_down_lev5 = Upsample(scale_factor=2, mode='nearest')

        num_featmaps_lev4pl5 = num_featmaps_lev4 + num_featmaps_lev5
        self.convolution_up_lev4_1 = Conv3d(num_featmaps_lev4pl5, num_featmaps_lev4, kernel_size=3, padding=1)
        self.convolution_up_lev4_2 = Conv3d(num_featmaps_lev4, num_featmaps_lev4, kernel_size=3, padding=1)
        self.upsample_up_lev4 = Upsample(scale_factor=2, mode='nearest')

        num_featmaps_lev3pl4 = num_featmaps_lev3 + num_featmaps_lev4
        self.convolution_up_lev3_1 = Conv3d(num_featmaps_lev3pl4, num_featmaps_lev3, kernel_size=3, padding=padding_value)
        self.convolution_up_lev3_2 = Conv3d(num_featmaps_lev3, num_featmaps_lev3, kernel_size=3, padding=padding_value)
        self.upsample_up_lev3 = Upsample(scale_factor=2, mode='nearest')

        num_featmaps_lev2pl3 = num_featmaps_lev2 + num_featmaps_lev3
        self.convolution_up_lev2_1 = Conv3d(num_featmaps_lev2pl3, num_featmaps_lev2, kernel_size=3, padding=padding_value)
        self.convolution_up_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3, padding=padding_value)
        self.upsample_up_lev2 = Upsample(scale_factor=2, mode='nearest')

        num_featmaps_lay1pl2 = num_featmaps_lev1 + num_featmaps_lev2
        self.convolution_up_lev1_1 = Conv3d(num_featmaps_lay1pl2, num_featmaps_lev1, kernel_size=3, padding=padding_value)
        self.convolution_up_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3, padding=padding_value)

        self.classification_last = Conv3d(num_featmaps_lev1, self.num_classes_out, kernel_size=1, padding=0)

        if self._type_activate_hidden == 'relu':
            self.activation_hidden = ReLU(inplace=True)
        else:
            message = 'Type activation hidden not existing: \'%s\'' % (self._type_activate_hidden)
            catch_error_exception(message)

        if self._type_activate_output == 'sigmoid':
            self.activation_last = Sigmoid()
        elif self._type_activate_output == 'linear':
            self.activation_last = lambda x: x
        else:
            message = 'Type activation output not existing: \'%s\' ' %(self._type_activate_output)
            catch_error_exception(message)

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:

        hidden_next = self.activation_hidden(self.convolution_down_lev1_1(input))
        hidden_next = self.activation_hidden(self.convolution_down_lev1_2(hidden_next))
        skipconn_lev1 = hidden_next
        if self._is_use_valid_convols:
            skipconn_lev1 = self.crop_image(skipconn_lev1, self._list_sizes_crop_where_merge[0])
        hidden_next = self.pooling_down_lev1(hidden_next)

        hidden_next = self.activation_hidden(self.convolution_down_lev2_1(hidden_next))
        hidden_next = self.activation_hidden(self.convolution_down_lev2_2(hidden_next))
        skipconn_lev2 = hidden_next
        if self._is_use_valid_convols:
            skipconn_lev2 = self.crop_image(skipconn_lev2, self._list_sizes_crop_where_merge[1])
        hidden_next = self.pooling_down_lev2(hidden_next)

        hidden_next = self.activation_hidden(self.convolution_down_lev3_1(hidden_next))
        hidden_next = self.activation_hidden(self.convolution_down_lev3_2(hidden_next))
        skipconn_lev3 = hidden_next
        if self._is_use_valid_convols:
            skipconn_lev3 = self.crop_image(skipconn_lev3, self._list_sizes_crop_where_merge[2])
        hidden_next = self.pooling_downlay3(hidden_next)

        hidden_next = self.activation_hidden(self.convolution_down_lev4_1(hidden_next))
        hidden_next = self.activation_hidden(self.convolution_down_lev4_2(hidden_next))
        skipconn_lev4 = hidden_next
        if self._is_use_valid_convols:
            skipconn_lev4 = self.crop_image(skipconn_lev4, self._list_sizes_crop_where_merge[3])
        hidden_next = self.pooling_downlay4(hidden_next)

        hidden_next = self.activation_hidden(self.convolution_down_lev5_1(hidden_next))
        hidden_next = self.activation_hidden(self.convolution_down_lev5_2(hidden_next))
        hidden_next = self.upsample_down_lev5(hidden_next)

        hidden_next = torch.cat([hidden_next, skipconn_lev4], dim=1)
        hidden_next = self.activation_hidden(self.convolution_up_lev4_1(hidden_next))
        hidden_next = self.activation_hidden(self.convolution_up_lev4_2(hidden_next))
        hidden_next = self.upsample_up_lev4(hidden_next)

        hidden_next = torch.cat([hidden_next, skipconn_lev3], dim=1)
        hidden_next = self.activation_hidden(self.convolution_up_lev3_1(hidden_next))
        hidden_next = self.activation_hidden(self.convolution_up_lev3_2(hidden_next))
        hidden_next = self.upsample_up_lev3(hidden_next)

        hidden_next = torch.cat([hidden_next, skipconn_lev2], dim=1)
        hidden_next = self.activation_hidden(self.convolution_up_lev2_1(hidden_next))
        hidden_next = self.activation_hidden(self.convolution_up_lev2_2(hidden_next))
        hidden_next = self.upsample_up_lev2(hidden_next)

        hidden_next = torch.cat([hidden_next, skipconn_lev1], dim=1)
        hidden_next = self.activation_hidden(self.convolution_up_lev1_1(hidden_next))
        hidden_next = self.activation_hidden(self.convolution_up_lev1_2(hidden_next))

        output = self.activation_last(self.classification_last(hidden_next))
        return output


# class Unet3D_General(NeuralNetwork):
#
#     num_layers_default = 5
#     num_featmaps_in_default = 16
#
#     num_convolutions_downlays_default = 2
#     num_convolution_uplays_default = 2
#     size_convolutionkernel_downlays_default = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
#     size_convolutionkernel_uplays_default = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
#     size_pooling_layers_default = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
#     #size_cropping_layers = [(0, 4, 4), (0, 16, 16), (0, 41, 41), (0, 90, 90)]
#
#     def __init__(self, size_image,
#                  num_channels_in= 1,
#                  num_classes_out= 1,
#                  num_layers= num_layers_default,
#                  num_featmaps_in= num_featmaps_in_default,
#                  num_featmaps_layers= None,
#                  num_convolution_downlays= num_convolutions_downlays_default,
#                  num_convolution_uplays= num_convolution_uplays_default,
#                  size_convolutionkernel_downlayers= size_convolutionkernel_downlays_default,
#                  size_convolutionkernel_uplayers= size_convolutionkernel_uplays_default,
#                  size_pooling_downlayers= size_pooling_layers_default,
#                  is_disable_convolutionpooling_zdim_lastlayer= False):
#
#         super(Unet3D_General, self).__init__(size_image, num_channels_in, num_classes_out)
#         self.num_layers = num_layers
#         if num_featmaps_layers:
#             self.num_featmaps_layers = num_featmaps_layers
#         else:
#             # Default: double featmaps after every pooling
#             self.num_featmaps_layers = [num_featmaps_in] + [0]*(self.num_layers-1)
#             for i in range(1, self.num_layers):
#                 self.num_featmaps_layers[i] = 2 * self.num_featmaps_layers[i-1]
#
#         self.num_convolution_downlays = num_convolution_downlays
#         self.num_convolution_uplays = num_convolution_uplays
#         self.size_convolutionkernel_downlayers = size_convolutionkernel_downlayers[0:self.num_layers]
#         self.size_convolutionkernel_uplayers = size_convolutionkernel_uplayers[0:self.num_layers]
#         self.size_pooling_downlayers = size_pooling_downlayers[0:self.num_layers-1]
#         self.size_upsample_uplayers = self.size_pooling_downlayers
#
#         if is_disable_convolutionpooling_zdim_lastlayer:
#             temp_size_kernel_lastlayer = self.size_convolutionkernel_downlayers[-1]
#             self.size_convolutionkernel_downlayers[-1] = (1, temp_size_kernel_lastlayer[1], temp_size_kernel_lastlayer[2])
#             temp_size_pooling_lastlayer = self.size_pooling_downlayers[-1]
#             self.size_pooling_downlayers[-1] = (1, temp_size_pooling_lastlayer[1], temp_size_pooling_lastlayer[2])
#
#         self.build_model()
#
#     def build_model(self):
#         pass
#
#     def forward(self, input):
#         pass