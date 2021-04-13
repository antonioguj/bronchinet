
from typing import Tuple, List, Dict, Union, Any

from torch.nn import Conv3d, MaxPool3d, Upsample, BatchNorm3d, Dropout3d, ReLU, Sigmoid
import torch.nn as nn
import torch

from common.exceptionmanager import catch_error_exception
from common.functionutil import ImagesUtil
from models.networks import UNetBase

LIST_AVAIL_NETWORKS = ['UNet3DOriginal',
                       'UNet3DGeneral',
                       'UNet3DPlugin',
                       ]


class UNet(UNetBase, nn.Module):

    def __init__(self,
                 size_image_in: Union[Tuple[int, int, int], Tuple[int, int]],
                 num_levels: int,
                 num_featmaps_in: int,
                 num_channels_in: int,
                 num_classes_out: int,
                 is_use_valid_convols: bool = False
                 ) -> None:
        super(UNet, self).__init__(size_image_in, num_levels, num_featmaps_in, num_channels_in,
                                   num_classes_out, is_use_valid_convols=is_use_valid_convols)
        nn.Module.__init__(self)

        self._shape_input = ImagesUtil.get_shape_channels_first(self._shape_input)
        self._shape_output = ImagesUtil.get_shape_channels_first(self._shape_output)

    def preprocess(self, *args, **kwargs) -> None:
        pass

    def get_network_input_args(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_list_info_crop_where_merge(self) -> None:
        indexes_output_where_merge = \
            [i for i, elem in enumerate(self._list_operation_names_layers_all) if elem == 'upsample']
        self._list_sizes_crop_where_merge = \
            [self._list_sizes_output_all_layers[ind] for ind in indexes_output_where_merge][::-1]

    def _crop_image_2d(self, input: torch.Tensor, size_crop: Tuple[int, int]) -> torch.Tensor:
        size_input_image = input.shape[-2:]
        limits_out_image = self._get_limits_output_crop(size_input_image, size_crop)
        return input[:, :,  # dims for input and output features
                     limits_out_image[0][0]:limits_out_image[0][1],
                     limits_out_image[1][0]:limits_out_image[1][1]]

    def _crop_image_3d(self, input: torch.Tensor, size_crop: Tuple[int, int, int]) -> torch.Tensor:
        size_input_image = input.shape[-3:]
        limits_out_image = self._get_limits_output_crop(size_input_image, size_crop)
        return input[:, :,  # dims for input and output features
                     limits_out_image[0][0]:limits_out_image[0][1],
                     limits_out_image[1][0]:limits_out_image[1][1],
                     limits_out_image[2][0]:limits_out_image[2][1]]


class UNet3DOriginal(UNet):
    _num_levels_fixed = 5

    def __init__(self,
                 size_image_in: Tuple[int, int, int],
                 num_featmaps_in: int = 16,
                 num_channels_in: int = 1,
                 num_classes_out: int = 1
                 ) -> None:
        super(UNet3DOriginal, self).__init__(size_image_in, self._num_levels_fixed, num_featmaps_in, num_channels_in,
                                             num_classes_out, is_use_valid_convols=False)
        self._build_model()

    def get_network_input_args(self) -> Dict[str, Any]:
        return {'size_image': self._size_image_in,
                'num_featmaps_in': self._num_featmaps_in,
                'num_channels_in': self._num_channels_in,
                'num_classes_out': self._num_classes_out}

    def _build_model(self) -> None:

        num_featmaps_lev1 = self._num_featmaps_in
        self._convolution_down_lev1_1 = Conv3d(self._num_channels_in, num_featmaps_lev1, kernel_size=3, padding=1)
        self._convolution_down_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3, padding=1)
        self._pooling_down_lev1 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev2 = 2 * num_featmaps_lev1
        self._convolution_down_lev2_1 = Conv3d(num_featmaps_lev1, num_featmaps_lev2, kernel_size=3, padding=1)
        self._convolution_down_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3, padding=1)
        self._pooling_down_lev2 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev3 = 2 * num_featmaps_lev2
        self._convolution_down_lev3_1 = Conv3d(num_featmaps_lev2, num_featmaps_lev3, kernel_size=3, padding=1)
        self._convolution_down_lev3_2 = Conv3d(num_featmaps_lev3, num_featmaps_lev3, kernel_size=3, padding=1)
        self._pooling_down_lev3 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev4 = 2 * num_featmaps_lev3
        self._convolution_down_lev4_1 = Conv3d(num_featmaps_lev3, num_featmaps_lev4, kernel_size=3, padding=1)
        self._convolution_down_lev4_2 = Conv3d(num_featmaps_lev4, num_featmaps_lev4, kernel_size=3, padding=1)
        self._pooling_down_lev4 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev5 = 2 * num_featmaps_lev4
        self._convolution_down_lev5_1 = Conv3d(num_featmaps_lev4, num_featmaps_lev5, kernel_size=3, padding=1)
        self._convolution_down_lev5_2 = Conv3d(num_featmaps_lev5, num_featmaps_lev5, kernel_size=3, padding=1)
        self._upsample_up_lev5 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev4pl5 = num_featmaps_lev4 + num_featmaps_lev5
        self._convolution_up_lev4_1 = Conv3d(num_feats_lev4pl5, num_featmaps_lev4, kernel_size=3, padding=1)
        self._convolution_up_lev4_2 = Conv3d(num_featmaps_lev4, num_featmaps_lev4, kernel_size=3, padding=1)
        self._upsample_up_lev4 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev3pl4 = num_featmaps_lev3 + num_featmaps_lev4
        self._convolution_up_lev3_1 = Conv3d(num_feats_lev3pl4, num_featmaps_lev3, kernel_size=3, padding=1)
        self._convolution_up_lev3_2 = Conv3d(num_featmaps_lev3, num_featmaps_lev3, kernel_size=3, padding=1)
        self._upsample_up_lev3 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev2pl3 = num_featmaps_lev2 + num_featmaps_lev3
        self._convolution_up_lev2_1 = Conv3d(num_feats_lev2pl3, num_featmaps_lev2, kernel_size=3, padding=1)
        self._convolution_up_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3, padding=1)
        self._upsample_up_lev2 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev1pl2 = num_featmaps_lev1 + num_featmaps_lev2
        self._convolution_up_lev1_1 = Conv3d(num_feats_lev1pl2, num_featmaps_lev1, kernel_size=3, padding=1)
        self._convolution_up_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3, padding=1)

        self._classification_last = Conv3d(num_featmaps_lev1, self._num_classes_out, kernel_size=1, padding=0)
        self._activation_last = Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        hidden_next = self._convolution_down_lev1_1(input)
        hidden_next = self._convolution_down_lev1_2(hidden_next)
        skipconn_lev1 = hidden_next
        hidden_next = self._pooling_down_lev1(hidden_next)

        hidden_next = self._convolution_down_lev2_1(hidden_next)
        hidden_next = self._convolution_down_lev2_2(hidden_next)
        skipconn_lev2 = hidden_next
        hidden_next = self._pooling_down_lev2(hidden_next)

        hidden_next = self._convolution_down_lev3_1(hidden_next)
        hidden_next = self._convolution_down_lev3_2(hidden_next)
        skipconn_lev3 = hidden_next
        hidden_next = self._pooling_down_lev3(hidden_next)

        hidden_next = self._convolution_down_lev4_1(hidden_next)
        hidden_next = self._convolution_down_lev4_2(hidden_next)
        skipconn_lev4 = hidden_next
        hidden_next = self._pooling_down_lev4(hidden_next)

        hidden_next = self._convolution_down_lev5_1(hidden_next)
        hidden_next = self._convolution_down_lev5_2(hidden_next)
        hidden_next = self._upsample_up_lev5(hidden_next)

        hidden_next = torch.cat([hidden_next, skipconn_lev4], dim=1)
        hidden_next = self._convolution_up_lev4_1(hidden_next)
        hidden_next = self._convolution_up_lev4_2(hidden_next)
        hidden_next = self._upsample_up_lev4(hidden_next)

        hidden_next = torch.cat([hidden_next, skipconn_lev3], dim=1)
        hidden_next = self._convolution_up_lev3_1(hidden_next)
        hidden_next = self._convolution_up_lev3_2(hidden_next)
        hidden_next = self._upsample_up_lev3(hidden_next)

        hidden_next = torch.cat([hidden_next, skipconn_lev2], dim=1)
        hidden_next = self._convolution_up_lev2_1(hidden_next)
        hidden_next = self._convolution_up_lev2_2(hidden_next)
        hidden_next = self._upsample_up_lev2(hidden_next)

        hidden_next = torch.cat([hidden_next, skipconn_lev1], dim=1)
        hidden_next = self._convolution_up_lev1_1(hidden_next)
        hidden_next = self._convolution_up_lev1_2(hidden_next)

        output = self._activation_last(self._classification_last(hidden_next))
        return output


class UNet3DGeneral(UNet):
    _num_levels_default = 5
    _num_featmaps_in_default = 16
    _num_channels_in_default = 1
    _num_classes_out_default = 1
    _dropout_rate_default = 0.2
    _type_activate_hidden_default = 'relu'
    _type_activate_output_default = 'sigmoid'
    _num_convols_levels_down_default = 2
    _num_convols_levels_up_default = 2
    _sizes_kernel_convols_levels_down_default = (3, 3, 3)
    _sizes_kernel_convols_levels_up_default = (3, 3, 3)
    _sizes_pooling_levels_default = (2, 2, 2)

    def __init__(self,
                 size_image_in: Tuple[int, int, int],
                 num_levels: int = _num_levels_default,
                 num_featmaps_in: int = _num_featmaps_in_default,
                 num_channels_in: int = _num_channels_in_default,
                 num_classes_out: int = _num_classes_out_default,
                 is_use_valid_convols: bool = False,
                 type_activate_hidden: str = _type_activate_hidden_default,
                 type_activate_output: str = _type_activate_output_default,
                 num_featmaps_levels: List[int] = None,
                 num_convols_levels_down: Union[int, Tuple[int, ...]] = _num_convols_levels_down_default,
                 num_convols_levels_up: Union[int, Tuple[int, ...]] = _num_convols_levels_up_default,
                 sizes_kernel_convols_levels_down: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] =
                 _sizes_kernel_convols_levels_down_default,
                 sizes_kernel_convols_levels_up: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] =
                 _sizes_kernel_convols_levels_up_default,
                 sizes_pooling_levels: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] =
                 _sizes_pooling_levels_default,
                 is_disable_convol_pooling_axialdim_lastlevel: bool = False,
                 is_use_dropout: bool = False,
                 dropout_rate: float = _dropout_rate_default,
                 is_use_dropout_levels_down: Union[bool, List[bool]] = True,
                 is_use_dropout_levels_up: Union[bool, List[bool]] = True,
                 is_use_batchnormalize=False,
                 is_use_batchnormalize_levels_down: Union[bool, List[bool]] = True,
                 is_use_batchnormalize_levels_up: Union[bool, List[bool]] = True
                 ) -> None:
        super(UNet, self).__init__(size_image_in, num_levels, num_featmaps_in, num_channels_in,
                                   num_classes_out, is_use_valid_convols=is_use_valid_convols)
        self._type_activate_hidden = type_activate_hidden
        self._type_activate_output = type_activate_output

        if num_featmaps_levels:
            self._num_featmaps_levels = num_featmaps_levels
        else:
            # default: double featmaps after every pooling
            self._num_featmaps_levels = [self._num_featmaps_in]
            for i in range(1, self._num_levels):
                self._num_featmaps_levels[i] = 2 * self._num_featmaps_levels[i - 1]

        if type(num_convols_levels_down) == int:
            self._num_convols_levels_down = [num_convols_levels_down] * self._num_levels
        else:
            self._num_convols_levels_down = num_convols_levels_down
        if type(num_convols_levels_up) == int:
            self._num_convols_levels_up = [num_convols_levels_up] * (self._num_levels - 1)
        else:
            self._num_convols_levels_up = num_convols_levels_up

        if type(sizes_kernel_convols_levels_down) == tuple:
            self._sizes_kernel_convols_levels_down = [sizes_kernel_convols_levels_down] * self._num_levels
        else:
            self._sizes_kernel_convols_levels_down = sizes_kernel_convols_levels_down
        if type(sizes_kernel_convols_levels_up) == tuple:
            self._sizes_kernel_convols_levels_up = [sizes_kernel_convols_levels_up] * (self._num_levels - 1)
        else:
            self._sizes_kernel_convols_levels_up = sizes_kernel_convols_levels_up

        if type(sizes_pooling_levels) == tuple:
            self._sizes_pooling_levels = [sizes_pooling_levels] * self._num_levels
        else:
            self._sizes_pooling_levels = sizes_pooling_levels
        self._sizes_upsample_levels = self._sizes_pooling_levels[:-1]

        if is_disable_convol_pooling_axialdim_lastlevel:
            size_kernel_convol_lastlevel = self._sizes_kernel_convols_levels_down[-1]
            self._sizes_kernel_convols_levels_down[-1] = (1, size_kernel_convol_lastlevel[1],
                                                          size_kernel_convol_lastlevel[2])
            size_pooling_lastlevel = self._sizes_pooling_levels[-1]
            self._sizes_pooling_levels[-1] = (1, size_pooling_lastlevel[1], size_pooling_lastlevel[2])

        self._is_use_dropout = is_use_dropout
        if is_use_dropout:
            self._dropout_rate = dropout_rate

            if type(is_use_dropout_levels_down) == bool:
                self._is_use_dropout_levels_down = [is_use_dropout_levels_down] * self._num_levels
            else:
                self._is_use_dropout_levels_down = is_use_dropout_levels_down
            if type(is_use_dropout_levels_up) == bool:
                self._is_use_dropout_levels_up = [is_use_dropout_levels_up] * (self._num_levels - 1)
            else:
                self._is_use_dropout_levels_up = is_use_dropout_levels_up

        self._is_use_batchnormalize = is_use_batchnormalize
        if is_use_batchnormalize:
            if type(is_use_batchnormalize_levels_down) == bool:
                self._is_use_batchnormalize_levels_down = [is_use_batchnormalize_levels_down] * self._num_levels
            else:
                self._is_use_batchnormalize_levels_down = is_use_batchnormalize_levels_down
            if type(is_use_batchnormalize_levels_up) == bool:
                self._is_use_batchnormalize_levels_up = [is_use_batchnormalize_levels_up] * (self._num_levels - 1)
            else:
                self._is_use_batchnormalize_levels_up = is_use_batchnormalize_levels_up

        self._build_model()

    def get_network_input_args(self) -> Dict[str, Any]:
        return {'size_image_in': self._size_image_in,
                'num_levels': self._num_levels,
                'num_featmaps_in': self._num_featmaps_in,
                'num_channels_in': self._num_channels_in,
                'num_classes_out': self._num_classes_out,
                'is_use_valid_convols': self._is_use_valid_convols}

    def _build_model(self) -> None:
        val_padding_convols = 0 if self._is_use_valid_convols else 1

        self._convolutions_levels_down = [[] for i in range(self._num_levels)]
        self._convolutions_levels_up = [[] for i in range(self._num_levels - 1)]
        self._poolings_levels_down = []
        self._upsamples_levels_up = []
        self._batchnormalize_levels_down = [[] for i in range(self._num_levels)]
        self._batchnormalize_levels_up = [[] for i in range(self._num_levels - 1)]

        # ENCODING LAYERS
        for i_lev in range(self._num_levels):
            num_featmaps_in_level = self._num_channels_in if i_lev == 0 else self._num_featmaps_levels[i_lev - 1]
            num_featmaps_out_level = self._num_featmaps_levels[i_lev]

            for i_con in range(self._num_convols_levels_down[i_lev]):
                num_featmaps_in_convol = num_featmaps_in_level if i_con else num_featmaps_in_level
                num_featmaps_out_convol = num_featmaps_out_level

                new_convolution = Conv3d(num_featmaps_in_convol, num_featmaps_out_convol,
                                         kernel_size=self._sizes_kernel_convols_levels_down[i_lev],
                                         padding=val_padding_convols)
                self._convolutions_levels_down[i_lev].append(new_convolution)

                if self._is_use_batchnormalize and self._is_use_batchnormalize_levels_down[i_lev]:
                    new_batchnormalize = BatchNorm3d(num_featmaps_out_convol)
                    self._batchnormalize_levels_down[i_lev].append(new_batchnormalize)

            if (i_lev != self._num_levels - 1):
                new_pooling = MaxPool3d(kernel_size=self._sizes_pooling_levels[i_lev], padding=0)
                self._poolings_levels_down.append(new_pooling)

        # DECODING LAYERS
        for i_lev in range(self._num_levels - 2, -1, -1):
            num_featmaps_in_level = self._num_featmaps_levels[i_lev - 1] + self._num_featmaps_levels[i_lev]
            num_featmaps_out_level = self._num_featmaps_levels[i_lev]

            new_upsample = Upsample(scale_factor=self._sizes_upsample_levels[i_lev], mode='nearest')
            self._upsamples_levels_up.append(new_upsample)

            for i_con in range(self._num_convols_levels_up[i_lev]):
                num_featmaps_in_convol = num_featmaps_in_level if i_con else num_featmaps_in_level
                num_featmaps_out_convol = num_featmaps_out_level

                new_convolution = Conv3d(num_featmaps_in_convol, num_featmaps_out_convol,
                                         kernel_size=self._sizes_kernel_convols_levels_up[i_lev],
                                         padding=val_padding_convols)
                self._convolutions_levels_up[i_lev].append(new_convolution)

                if self._is_use_batchnormalize and self._is_use_batchnormalize_levels_up[i_lev]:
                    new_batchnormalize = BatchNorm3d(num_featmaps_out_convol)
                    self._batchnormalize_levels_up[i_lev].append(new_batchnormalize)

        self._classification_last = Conv3d(self._num_featmaps_in, self._num_classes_out, kernel_size=1, padding=0)

        if self._is_use_dropout:
            self._dropout_all_levels = Dropout3d(self._dropout_rate, inplace=True)

        if self._type_activate_hidden == 'relu':
            self._activation_hidden = ReLU(inplace=True)
        elif self._type_activate_hidden == 'none':
            def func_activation_none(input: torch.Tensor) -> torch.Tensor:
                return input
            self._activation_hidden = func_activation_none
        else:
            message = 'Type activation hidden not existing: \'%s\'' % (self._type_activate_hidden)
            catch_error_exception(message)

        if self._type_activate_output == 'sigmoid':
            self._activation_last = Sigmoid()
        elif self._type_activate_output == 'linear':
            def func_activation_linear(input: torch.Tensor) -> torch.Tensor:
                return input
            self._activation_last = func_activation_linear
        else:
            message = 'Type activation output not existing: \'%s\' ' % (self._type_activate_output)
            catch_error_exception(message)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_next = input
        list_skipconn_levels = []

        # ENCODING LAYERS
        for i_lev in range(self._num_levels):
            for i_con in range(self._num_convols_levels_down[i_lev]):
                hidden_next = self._activation_hidden(self._convolutions_levels_down[i_lev][i_con](hidden_next))

                if self._is_use_batchnormalize and self._is_use_batchnormalize_levels_down[i_lev]:
                    hidden_next = self._batchnormalize_levels_down[i_lev][i_con](hidden_next)

            if self._is_use_dropout and self._is_use_dropout_levels_down[i_lev]:
                hidden_next = self._dropout_all_levels(hidden_next)

            if (i_lev != self._num_levels - 1):
                list_skipconn_levels.append(hidden_next)
                hidden_next = self._poolings_levels_down[i_lev](hidden_next)

        # DECODING LAYERS
        for i_lev in range(self._num_levels - 2, -1, -1):
            hidden_next = self._upsamples_levels_up[i_lev](hidden_next)

            skipconn_thislev = list_skipconn_levels[i_lev]
            if self._is_use_valid_convols:
                skipconn_thislev = self._crop_image_3d(skipconn_thislev, self._list_sizes_crop_where_merge[3])
            hidden_next = torch.cat([hidden_next, skipconn_thislev], dim=1)

            for i_con in range(self._num_convols_levels_up[i_lev]):
                hidden_next = self._activation_hidden(self._convolutions_levels_up[i_lev][i_con](hidden_next))

                if self._is_use_batchnormalize and self._is_use_batchnormalize_levels_up[i_lev]:
                    hidden_next = self._batchnormalize_levels_up[i_lev][i_con](hidden_next)

            if self._is_use_dropout and self._is_use_dropout_levels_up[i_lev]:
                hidden_next = self._dropout_all_levels(hidden_next)

        output = self._activation_last(self._classification_last(hidden_next))
        return output


class UNet3DPlugin(UNet):
    _num_levels_fixed = 5
    _num_featmaps_in_default = 16
    _num_channels_in_default = 1
    _num_classes_out_default = 1
    _dropout_rate_default = 0.2
    _type_activate_hidden_default = 'relu'
    _type_activate_output_default = 'sigmoid'

    def __init__(self,
                 size_image_in: Tuple[int, int, int],
                 num_featmaps_in: int = _num_featmaps_in_default,
                 num_channels_in: int = _num_channels_in_default,
                 num_classes_out: int = _num_classes_out_default,
                 is_use_valid_convols: bool = False
                 ) -> None:
        super(UNet3DPlugin, self).__init__(size_image_in, self._num_levels_fixed, num_featmaps_in, num_channels_in,
                                           num_classes_out, is_use_valid_convols=is_use_valid_convols)
        self._type_activate_hidden = self._type_activate_hidden_default
        self._type_activate_output = self._type_activate_output_default

        self._build_model()

    def get_network_input_args(self) -> Dict[str, Any]:
        return {'size_image_in': self._size_image_in,
                'num_featmaps_in': self._num_featmaps_in,
                'num_channels_in': self._num_channels_in,
                'num_classes_out': self._num_classes_out,
                'is_use_valid_convols': self._is_use_valid_convols}

    def _build_model(self) -> None:
        val_padding = 0 if self._is_use_valid_convols else 1

        num_featmaps_lev1 = self._num_featmaps_in
        self._convolution_down_lev1_1 = Conv3d(self._num_channels_in, num_featmaps_lev1, kernel_size=3,
                                               padding=val_padding)
        self._convolution_down_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3, padding=val_padding)
        self._pooling_down_lev1 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev2 = 2 * num_featmaps_lev1
        self._convolution_down_lev2_1 = Conv3d(num_featmaps_lev1, num_featmaps_lev2, kernel_size=3, padding=val_padding)
        self._convolution_down_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3, padding=val_padding)
        self._pooling_down_lev2 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev3 = 2 * num_featmaps_lev2
        self._convolution_down_lev3_1 = Conv3d(num_featmaps_lev2, num_featmaps_lev3, kernel_size=3, padding=val_padding)
        self._convolution_down_lev3_2 = Conv3d(num_featmaps_lev3, num_featmaps_lev3, kernel_size=3, padding=val_padding)
        self._pooling_down_lev3 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev4 = 2 * num_featmaps_lev3
        self._convolution_down_lev4_1 = Conv3d(num_featmaps_lev3, num_featmaps_lev4, kernel_size=3, padding=1)
        self._convolution_down_lev4_2 = Conv3d(num_featmaps_lev4, num_featmaps_lev4, kernel_size=3, padding=1)
        self._pooling_down_lev4 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev5 = 2 * num_featmaps_lev4
        self._convolution_down_lev5_1 = Conv3d(num_featmaps_lev4, num_featmaps_lev5, kernel_size=3, padding=1)
        self._convolution_down_lev5_2 = Conv3d(num_featmaps_lev5, num_featmaps_lev5, kernel_size=3, padding=1)
        self._upsample_up_lev5 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev4pl5 = num_featmaps_lev4 + num_featmaps_lev5
        self._convolution_up_lev4_1 = Conv3d(num_feats_lev4pl5, num_featmaps_lev4, kernel_size=3, padding=1)
        self._convolution_up_lev4_2 = Conv3d(num_featmaps_lev4, num_featmaps_lev4, kernel_size=3, padding=1)
        self._upsample_up_lev4 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev3pl4 = num_featmaps_lev3 + num_featmaps_lev4
        self._convolution_up_lev3_1 = Conv3d(num_feats_lev3pl4, num_featmaps_lev3, kernel_size=3, padding=val_padding)
        self._convolution_up_lev3_2 = Conv3d(num_featmaps_lev3, num_featmaps_lev3, kernel_size=3, padding=val_padding)
        self._upsample_up_lev3 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev2pl3 = num_featmaps_lev2 + num_featmaps_lev3
        self._convolution_up_lev2_1 = Conv3d(num_feats_lev2pl3, num_featmaps_lev2, kernel_size=3, padding=val_padding)
        self._convolution_up_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3, padding=val_padding)
        self._upsample_up_lev2 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lay1pl2 = num_featmaps_lev1 + num_featmaps_lev2
        self._convolution_up_lev1_1 = Conv3d(num_feats_lay1pl2, num_featmaps_lev1, kernel_size=3, padding=val_padding)
        self._convolution_up_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3, padding=val_padding)

        self._classification_last = Conv3d(num_featmaps_lev1, self._num_classes_out, kernel_size=1, padding=0)

        if self._type_activate_hidden == 'relu':
            self._activation_hidden = ReLU(inplace=True)
        elif self._type_activate_hidden == 'linear':
            def func_activation_linear(input: torch.Tensor) -> torch.Tensor:
                return input
            self._activation_hidden = func_activation_linear
        else:
            message = 'Type activation hidden not existing: \'%s\'' % (self._type_activate_hidden)
            catch_error_exception(message)

        if self._type_activate_output == 'sigmoid':
            self._activation_last = Sigmoid()
        elif self._type_activate_output == 'linear':
            def func_activation_linear(input: torch.Tensor) -> torch.Tensor:
                return input
            self._activation_last = func_activation_linear
        else:
            message = 'Type activation output not existing: \'%s\' ' % (self._type_activate_output)
            catch_error_exception(message)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        hidden_next = self._activation_hidden(self._convolution_down_lev1_1(input))
        hidden_next = self._activation_hidden(self._convolution_down_lev1_2(hidden_next))
        skipconn_lev1 = hidden_next
        hidden_next = self._pooling_down_lev1(hidden_next)

        hidden_next = self._activation_hidden(self._convolution_down_lev2_1(hidden_next))
        hidden_next = self._activation_hidden(self._convolution_down_lev2_2(hidden_next))
        skipconn_lev2 = hidden_next
        hidden_next = self._pooling_down_lev2(hidden_next)

        hidden_next = self._activation_hidden(self._convolution_down_lev3_1(hidden_next))
        hidden_next = self._activation_hidden(self._convolution_down_lev3_2(hidden_next))
        skipconn_lev3 = hidden_next
        hidden_next = self._pooling_down_lev3(hidden_next)

        hidden_next = self._activation_hidden(self._convolution_down_lev4_1(hidden_next))
        hidden_next = self._activation_hidden(self._convolution_down_lev4_2(hidden_next))
        skipconn_lev4 = hidden_next
        hidden_next = self._pooling_down_lev4(hidden_next)

        hidden_next = self._activation_hidden(self._convolution_down_lev5_1(hidden_next))
        hidden_next = self._activation_hidden(self._convolution_down_lev5_2(hidden_next))
        hidden_next = self._upsample_up_lev5(hidden_next)

        if self._is_use_valid_convols:
            skipconn_lev4 = self._crop_image_3d(skipconn_lev4, self._list_sizes_crop_where_merge[3])
        hidden_next = torch.cat([hidden_next, skipconn_lev4], dim=1)
        hidden_next = self._activation_hidden(self._convolution_up_lev4_1(hidden_next))
        hidden_next = self._activation_hidden(self._convolution_up_lev4_2(hidden_next))
        hidden_next = self._upsample_up_lev4(hidden_next)

        if self._is_use_valid_convols:
            skipconn_lev3 = self._crop_image_3d(skipconn_lev3, self._list_sizes_crop_where_merge[2])
        hidden_next = torch.cat([hidden_next, skipconn_lev3], dim=1)
        hidden_next = self._activation_hidden(self._convolution_up_lev3_1(hidden_next))
        hidden_next = self._activation_hidden(self._convolution_up_lev3_2(hidden_next))
        hidden_next = self._upsample_up_lev3(hidden_next)

        if self._is_use_valid_convols:
            skipconn_lev2 = self._crop_image_3d(skipconn_lev2, self._list_sizes_crop_where_merge[1])
        hidden_next = torch.cat([hidden_next, skipconn_lev2], dim=1)
        hidden_next = self._activation_hidden(self._convolution_up_lev2_1(hidden_next))
        hidden_next = self._activation_hidden(self._convolution_up_lev2_2(hidden_next))
        hidden_next = self._upsample_up_lev2(hidden_next)

        if self._is_use_valid_convols:
            skipconn_lev1 = self._crop_image_3d(skipconn_lev1, self._list_sizes_crop_where_merge[0])
        hidden_next = torch.cat([hidden_next, skipconn_lev1], dim=1)
        hidden_next = self._activation_hidden(self._convolution_up_lev1_1(hidden_next))
        hidden_next = self._activation_hidden(self._convolution_up_lev1_2(hidden_next))

        output = self._activation_last(self._classification_last(hidden_next))
        return output
