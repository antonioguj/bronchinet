
from typing import Tuple, List, Dict, Any

BoundBoxNDType = Tuple[Tuple[int, int], ...]

from common.constant import TYPE_DNNLIB_USED
from common.exceptionmanager import catch_error_exception
if TYPE_DNNLIB_USED == 'Pytorch':
    from models.pytorch.networks import UNet, UNet3D_Original, UNet3D_General, UNet3D_Plugin, LIST_AVAIL_NETWORKS
elif TYPE_DNNLIB_USED == 'Keras':
    from models.keras.networks import UNet, UNet3D_Original, UNet3D_General, UNet3D_Plugin, LIST_AVAIL_NETWORKS


class NeuralNetwork(object):

    def __init__(self,
                 size_input: Tuple[int, ...],
                 size_output: Tuple[int, ...]
                 ) -> None:
        self._size_input = size_input
        self._size_output = size_output

    def get_size_input(self) -> Tuple[int, ...]:
        return self._size_input

    def get_size_output(self) -> Tuple[int, ...]:
        return self._size_output

    def count_model_params(self) -> int:
        raise NotImplementedError

    def preprocess(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def _build_model(self) -> None:
        raise NotImplementedError


class ConvNetBase(NeuralNetwork):

    def __init__(self,
                 size_image_in: Tuple[int, ...],
                 num_featmaps_in: int,
                 num_channels_in: int,
                 num_classes_out: int,
                 is_use_valid_convols: bool = False
                 ) -> None:
        self._size_image_in = size_image_in
        self._num_featmaps_in = num_featmaps_in
        self._num_channels_in = num_channels_in
        self._num_classes_out = num_classes_out
        self._is_use_valid_convols = is_use_valid_convols

        if self._is_use_valid_convols:
            self._list_opers_names_layers_all = []
            self._build_list_opers_names_layers()
            self._build_list_sizes_output_all_layers()

        size_input = self._size_image_in + (self._num_channels_in,)

        size_output = self._get_size_output_last_layer() + (self._num_classes_out,)

        super(ConvNetBase, self).__init__(size_input, size_output)

    def get_model_construct_input_args(self) -> List[str, Dict[str, Any]]:
        raise NotImplementedError

    def _build_list_opers_names_layers(self) -> None:
        raise NotImplementedError

    def _get_size_output_last_layer(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def _get_size_output_layer(self, size_input: Tuple[int, ...], oper_name: str) -> Tuple[int, ...]:
        if oper_name == 'convolution':
            if self._is_use_valid_convols:
                return self._get_size_output_valid_convolution(size_input)
            else:
                return size_input
        elif oper_name == 'convolution_padded':
            return size_input
        elif oper_name == 'pooling':
            return self._get_size_output_pooling(size_input)
        elif oper_name == 'upsample':
            return self._get_size_output_upsample(size_input)
        elif oper_name == 'convol_classify':
            return size_input
        else:
            raise NotImplementedError

    def _get_size_output_group_layers(self, level_begin: int = 0, level_end: int = None) -> Tuple[int, ...]:
        if not level_end:
            level_end = len(self._list_opers_names_layers_all)

        if level_end < level_begin or \
            level_begin >= len(self._list_opers_names_layers_all) or \
            level_end >= len(self._list_opers_names_layers_all):
            message = 'Problem with input \'level_begin\' (%s) or \'level_end\' (%s)' % (level_begin, level_end)
            catch_error_exception(message)

        in_list_opers_names_layers = self._list_opers_names_layers_all[level_begin:level_end]

        if level_begin == 0:
            size_input = self._size_image_in
        else:
            size_input = self._get_size_output_group_layers(level_begin=0, level_end=level_begin)

        size_next = size_input
        for oper_name in in_list_opers_names_layers:
            size_next = self._get_size_output_layer(size_next, oper_name)

        return size_next

    def _build_list_sizes_output_all_layers(self) -> List[Tuple[int, ...]]:
        self._list_sizes_output_all_layers = []
        size_next = self._size_image_in
        for oper_name in self._list_opers_names_layers_all:
            size_next = self._get_size_output_layer(size_next, oper_name)
            self._list_sizes_output_all_layers.append(size_next)

        return self._list_sizes_output_all_layers

    @staticmethod
    def _get_size_output_valid_convolution(size_input: Tuple[int, ...], size_kernel: int = 3) -> Tuple[int, ...]:
        return tuple([elem - size_kernel + 1 for elem in size_input])

    @staticmethod
    def _get_size_output_pooling(size_input: Tuple[int, ...], size_pool: int = 2) -> Tuple[int, ...]:
        return tuple([elem // size_pool for elem in size_input])

    @staticmethod
    def _get_size_output_upsample(size_input: Tuple[int, ...], size_upsample: int = 2) -> Tuple[int, ...]:
        return tuple([elem * size_upsample for elem in size_input])


class UNetBase(ConvNetBase):
    _num_levels_non_padded = 3

    def __init__(self,
                 size_image_in: Tuple[int, ...],
                 num_levels: int,
                 num_featmaps_in: int,
                 num_channels_in: int,
                 num_classes_out: int,
                 is_use_valid_convols: bool = False
                 ) -> None:
        self._num_levels = num_levels
        super(UNetBase, self).__init__(size_image_in, num_featmaps_in, num_channels_in, num_classes_out,
                                       is_use_valid_convols=is_use_valid_convols)

        if self._is_use_valid_convols:
            self._build_list_info_crop_where_merge()

    def _get_size_output_last_layer(self) -> Tuple[int, ...]:
        if self._is_use_valid_convols:
            return self._get_size_output_group_layers(level_begin=0, level_end=len(self._list_opers_names_layers_all))
        else:
            return self._size_image_in

    def _build_list_opers_names_layers(self) -> None:
        if self._num_levels == 1:
            self._list_opers_names_layers_all = ['convolution'] * 4 + ['convol_classify']

        elif self._is_use_valid_convols and self._num_levels > self._num_levels_non_padded:
            # Assume that last convolutions have padding, to avoid large reduction of image dims
            num_levels_with_padding = self._num_levels - self._num_levels_non_padded - 1
            self._list_opers_names_layers_all = self._num_levels_non_padded * ['convolution'] * 2 + ['pooling'] + \
                                                num_levels_with_padding * ['convolution_padded'] * 2 + ['pooling'] + \
                                                ['convolution_padded'] * 2 + \
                                                num_levels_with_padding * ['upsample'] + ['convolution_padded'] * 2 + \
                                                self._num_levels_non_padded * ['upsample'] + ['convolution'] * 2 + \
                                                ['convol_classify']
        else:
            self._list_opers_names_layers_all = (self._num_levels - 1) * ['convolution'] * 2 + ['pooling'] + \
                                                ['convolution'] * 2 + \
                                                (self._num_levels - 1) * ['upsample'] + ['convolution'] * 2 + \
                                                ['convol_classify']

    def _build_list_info_crop_where_merge(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _get_limits_output_crop_1d(size_input: int, size_crop: int) -> Tuple[int, int]:
        coord_begin = int((size_input - size_crop) / 2)
        coord_end = coord_begin + size_crop
        return (coord_begin, coord_end)

    @classmethod
    def _get_limits_output_crop(cls, size_input: Tuple[int, ...], size_crop: Tuple[int, ...]) -> BoundBoxNDType:
        return tuple([cls._get_limits_output_crop_1d(elem_si, elem_sc) for (elem_si, elem_sc) in zip(size_input, size_crop)])

    @staticmethod
    def _get_size_borders_output_crop(size_input: Tuple[int, ...], size_crop: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple([int((elem_si - elem_sc) / 2) for (elem_si, elem_sc) in zip(size_input, size_crop)])



def get_network(type_network: str,
                size_image_in: Tuple[int, ...],
                num_levels: int = 5,
                num_featmaps_in: int = 16,
                num_channels_in: int = 1,
                num_classes_out: int = 1,
                is_use_valid_convols: bool = False,
                **kwargs) -> UNet:
    if type_network == 'UNet_Original':
        return UNet3D_Original(size_image_in,
                               num_featmaps_in=num_featmaps_in,
                               num_channels_in=num_channels_in,
                               num_classes_out=num_classes_out)

    elif type_network == 'UNet_General':
        type_activate_hidden = kwargs['type_activate_hidden'] if 'type_activate_hidden' in kwargs.keys() else 'relu'
        type_activate_output = kwargs['type_activate_output'] if 'type_activate_output' in kwargs.keys() else 'sigmoid'
        num_featmaps_levels = kwargs['num_featmaps_levels'] if 'num_featmaps_levels' in kwargs.keys() else None
        is_use_dropout = kwargs['is_use_dropout'] if 'is_use_dropout' in kwargs.keys() else False
        dropout_rate = kwargs['dropout_rate'] if 'dropout_rate' in kwargs.keys() else 0.2
        is_use_batchnormalize = kwargs['is_use_batchnormalize'] if 'is_use_batchnormalize' in kwargs.keys() else False

        return UNet3D_General(size_image_in,
                              num_levels,
                              num_featmaps_in=num_featmaps_in,
                              num_channels_in=num_channels_in,
                              num_classes_out=num_classes_out,
                              is_use_valid_convols=is_use_valid_convols,
                              type_activate_hidden=type_activate_hidden,
                              type_activate_output=type_activate_output,
                              num_featmaps_levels=num_featmaps_levels,
                              is_use_dropout=is_use_dropout,
                              dropout_rate=dropout_rate,
                              is_use_batchnormalize=is_use_batchnormalize)

    elif type_network == 'UNet_Plugin':
        return UNet3D_Plugin(size_image_in,
                             num_levels,
                             num_featmaps_in=num_featmaps_in,
                             num_channels_in=num_channels_in,
                             num_classes_out=num_classes_out,
                             is_use_valid_convols=is_use_valid_convols)
    else:
        message = 'Choice Network not found. Networks available: %s' % (', '.join(LIST_AVAIL_NETWORKS))
        catch_error_exception(message)