
from typing import Tuple, List, Dict, Any

from tensorflow.keras.layers import Input, concatenate, Dropout, BatchNormalization
from tensorflow.keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, Cropping3D, Conv3DTranspose
from tensorflow.keras.models import Model, load_model

from common.exception_manager import catch_error_exception
from networks.metrics import MetricBase
from networks.keras.metrics import Metric as Metric_train
from networks.networks import UNetBase

LIST_AVAIL_NETWORKS = ['UNet3D_Original',
                       'UNet3D_General',
                       'UNet3D_Plugin',
                       ]


class UNet(UNetBase):

    def __init__(self,
                 size_image_in: Tuple[int, ...],
                 num_levels: int,
                 num_featmaps_in: int,
                 num_channels_in: int,
                 num_classes_out: int,
                 is_use_valid_convols: bool = False
                 ) -> None:
        super(UNet, self).__init__(size_image_in,
                                   num_levels,
                                   num_channels_in,
                                   num_classes_out,
                                   num_featmaps_in,
                                   is_use_valid_convols=is_use_valid_convols)

    @staticmethod
    def get_load_saved_model(model_saved_path: str, custom_objects: Any = None):
        return load_model(model_saved_path, custom_objects=custom_objects)

    def _build_model_and_compile(self, optimizer, loss: Metric_train, list_metrics: List[MetricBase]):
        return self._build_model().compile(optimizer=optimizer,
                                           loss=loss,
                                           metrics=list_metrics)

    def _build_list_info_crop_where_merge(self) -> None:
        indexes_output_where_pooling = [(i-1) for i, el in enumerate(self._list_opers_names_layers_all) if el == 'pooling']
        indexes_output_where_merge   = [i for i, el in enumerate(self._list_opers_names_layers_all) if el == 'upsample']
        self._list_sizes_borders_crop_where_merge = []
        for i_pool, i_merge in zip(indexes_output_where_pooling, indexes_output_where_merge[::-1]):
            size_borders_crop_where_merge = self._get_size_borders_output_crop(self._list_sizes_output_all_layers[i_pool],
                                                                               self._list_sizes_output_all_layers[i_merge])
            self._list_sizes_borders_crop_where_merge.append(size_borders_crop_where_merge)


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

    def _build_model(self) -> None:
        input_layer = Input((self._size_image_in) + (self._num_channels_in,))

        num_featmaps_lev1 = self._num_featmaps_in
        convolution_down_lev1_1 = Convolution3D(num_featmaps_lev1, kernel_size=(3,3,3), activation='relu', padding='same')(input_layer)
        convolution_down_lev1_2 = Convolution3D(num_featmaps_lev1, kernel_size=(3,3,3), activation='relu', padding='same')(convolution_down_lev1_1)
        pooling_down_lev1 = MaxPooling3D(pool_size=(2,2,2))(convolution_down_lev1_2)

        num_featmaps_lev2 = 2 * num_featmaps_lev1
        convolution_down_lev2_1 = Convolution3D(num_featmaps_lev2, kernel_size=(3,3,3), activation='relu', padding='same')(pooling_down_lev1)
        convolution_down_lev2_2 = Convolution3D(num_featmaps_lev2, kernel_size=(3,3,3), activation='relu', padding='same')(convolution_down_lev2_1)
        pooling_down_lev2 = MaxPooling3D(pool_size=(2,2,2))(convolution_down_lev2_2)

        num_featmaps_lev3 = 2 * num_featmaps_lev2
        convolution_down_lev3_1 = Convolution3D(num_featmaps_lev3, kernel_size=(3,3,3), activation='relu', padding='same')(pooling_down_lev2)
        convolution_down_lev3_2 = Convolution3D(num_featmaps_lev3, kernel_size=(3,3,3), activation='relu', padding='same')(convolution_down_lev3_1)
        pooling_down_lev3 = MaxPooling3D(pool_size=(2,2,2))(convolution_down_lev3_2)

        num_featmaps_lev4 = 2 * num_featmaps_lev3
        convolution_down_lev4_1 = Convolution3D(num_featmaps_lev4, kernel_size=(3,3,3), activation='relu', padding='same')(pooling_down_lev3)
        convolution_down_lev4_2 = Convolution3D(num_featmaps_lev4, kernel_size=(3,3,3), activation='relu', padding='same')(convolution_down_lev4_1)
        pooling_down_lev4 = MaxPooling3D(pool_size=(2,2,2))(convolution_down_lev4_2)

        num_featmaps_lev5 = 2 * num_featmaps_lev4
        convolution_down_lev5_1 = Convolution3D(num_featmaps_lev5, kernel_size=(3,3,3), activation='relu', padding='same')(pooling_down_lev4)
        convolution_down_lev5_2 = Convolution3D(num_featmaps_lev5, kernel_size=(3,3,3), activation='relu', padding='same')(convolution_down_lev5_1)
        upsample_up_lev5 = UpSampling3D(size=(2,2,2))(convolution_down_lev5_2)

        upsample_up_lev5 = concatenate([upsample_up_lev5, convolution_down_lev4_2], axis=-1)
        convolution_up_lev4_1 = Convolution3D(num_featmaps_lev4, kernel_size=(3,3,3), activation='relu', padding='same')(upsample_up_lev5)
        convolution_up_lev4_2 = Convolution3D(num_featmaps_lev4, kernel_size=(3,3,3), activation='relu', padding='same')(convolution_up_lev4_1)
        upsample_up_lev4 = UpSampling3D(size=(2,2,2))(convolution_up_lev4_2)

        upsample_up_lev4 = concatenate([upsample_up_lev4, convolution_down_lev3_2], axis=-1)
        convolution_up_lev3_1 = Convolution3D(num_featmaps_lev3, kernel_size=(3,3,3), activation='relu', padding='same')(upsample_up_lev4)
        convolution_up_lev3_2 = Convolution3D(num_featmaps_lev3, kernel_size=(3,3,3), activation='relu', padding='same')(convolution_up_lev3_1)
        upsample_up_lev3 = UpSampling3D(size=(2,2,2))(convolution_up_lev3_2)

        upsample_up_lev3 = concatenate([upsample_up_lev3, convolution_down_lev2_2], axis=-1)
        convolution_up_lev2_1 = Convolution3D(num_featmaps_lev2, kernel_size=(3,3,3), activation='relu', padding='same')(upsample_up_lev3)
        convolution_up_lev2_2 = Convolution3D(num_featmaps_lev2, kernel_size=(3,3,3), activation='relu', padding='same')(convolution_up_lev2_1)
        upsample_up_lev2 = UpSampling3D(size=(2,2,2))(convolution_up_lev2_2)

        upsample_up_lev2 = concatenate([upsample_up_lev2, convolution_down_lev1_2], axis=-1)
        convolution_up_lev1_1 = Convolution3D(num_featmaps_lev1, kernel_size=(3,3,3), activation='relu', padding='same')(upsample_up_lev2)
        convolution_up_lev1_2 = Convolution3D(num_featmaps_lev1, kernel_size=(3,3,3), activation='relu', padding='same')(convolution_up_lev1_1)

        output_layer = Convolution3D(self._num_classes_out, kernel_size=(1,1,1), activation='sigmoid')(convolution_up_lev1_2)

        output_model = Model(inputs=input_layer, outputs=output_layer)
        return output_model


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

    def _build_model(self) -> None:
        type_padding = 'valid' if self._is_use_valid_convols else 'same'

        input_layer = Input((self._size_image_in) + (self._num_channels_in,))

        num_featmaps_lev1 = self._num_featmaps_in
        hidden_next = Convolution3D(num_featmaps_lev1, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding=type_padding)(input_layer)
        hidden_next = Convolution3D(num_featmaps_lev1, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding=type_padding)(hidden_next)
        skipconn_lev1 = hidden_next
        if self._is_use_valid_convols:
            skipconn_lev1 = Cropping3D(cropping=self._list_sizes_borders_crop_where_merge[0])(skipconn_lev1)
        hidden_next = MaxPooling3D(pool_size=(2,2,2))(hidden_next)

        num_featmaps_lev2 = 2 * num_featmaps_lev1
        hidden_next = Convolution3D(num_featmaps_lev2, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding=type_padding)(hidden_next)
        hidden_next = Convolution3D(num_featmaps_lev2, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding=type_padding)(hidden_next)
        skipconn_lev2 = hidden_next
        if self._is_use_valid_convols:
            skipconn_lev2 = Cropping3D(cropping= self._list_sizes_borders_crop_where_merge[1])(skipconn_lev2)
        hidden_next = MaxPooling3D(pool_size=(2,2,2))(hidden_next)

        num_featmaps_lev3 = 2 * num_featmaps_lev2
        hidden_next = Convolution3D(num_featmaps_lev3, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding=type_padding)(hidden_next)
        hidden_next = Convolution3D(num_featmaps_lev3, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding=type_padding)(hidden_next)
        skipconn_lev3 = hidden_next
        if self._is_use_valid_convols:
            skipconn_lev3 = Cropping3D(cropping= self._list_sizes_borders_crop_where_merge[2])(skipconn_lev3)
        hidden_next = MaxPooling3D(pool_size=(2,2,2))(hidden_next)

        num_featmaps_lev4 = 2 * num_featmaps_lev3
        hidden_next = Convolution3D(num_featmaps_lev4, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding='same')(hidden_next)
        hidden_next = Convolution3D(num_featmaps_lev4, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding='same')(hidden_next)
        skipconn_lev4 = hidden_next
        if self._is_use_valid_convols:
            skipconn_lev4 = Cropping3D(cropping= self._list_sizes_borders_crop_where_merge[3])(skipconn_lev4)
        hidden_next = MaxPooling3D(pool_size=(2,2,2))(hidden_next)

        num_featmaps_lev5 = 2 * num_featmaps_lev4
        hidden_next = Convolution3D(num_featmaps_lev5, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding='same')(hidden_next)
        hidden_next = Convolution3D(num_featmaps_lev5, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding='same')(hidden_next)
        hidden_next = UpSampling3D(size=(2,2,2))(hidden_next)
        
        hidden_next = concatenate([hidden_next, skipconn_lev4], axis=-1)
        hidden_next = Convolution3D(num_featmaps_lev4, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding='same')(hidden_next)
        hidden_next = Convolution3D(num_featmaps_lev4, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding='same')(hidden_next)
        hidden_next = UpSampling3D(size=(2,2,2))(hidden_next)
        
        hidden_next = concatenate([hidden_next, skipconn_lev3], axis=-1)
        hidden_next = Convolution3D(num_featmaps_lev3, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding=type_padding)(hidden_next)
        hidden_next = Convolution3D(num_featmaps_lev3, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding=type_padding)(hidden_next)
        hidden_next = UpSampling3D(size=(2,2,2))(hidden_next)
        
        hidden_next = concatenate([hidden_next, skipconn_lev2], axis=-1)
        hidden_next = Convolution3D(num_featmaps_lev2, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding=type_padding)(hidden_next)
        hidden_next = Convolution3D(num_featmaps_lev2, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding=type_padding)(hidden_next)
        hidden_next = UpSampling3D(size=(2,2,2))(hidden_next)
        
        hidden_next = concatenate([hidden_next, skipconn_lev1], axis=-1)
        hidden_next = Convolution3D(num_featmaps_lev1, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding=type_padding)(hidden_next)
        hidden_next = Convolution3D(num_featmaps_lev1, kernel_size=(3,3,3), activation=self._type_activate_hidden, padding=type_padding)(hidden_next)

        output_layer = Convolution3D(self._num_classes_out, kernel_size=(1,1,1), activation=self._type_activate_output)(hidden_next)

        output_model = Model(inputs=input_layer, outputs=output_layer)
        return output_model



# class Unet3D_General_Extended(Unet3D_General):
#     num_convols_downlevels_default = 2
#     num_convols_uplevels_default   = 2
#     size_convolfilter_downlevels_default= [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
#     size_convolfilter_uplevels_default  = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
#     size_pooling_levels_default         = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
#     #size_cropping_levels_default        = [(0, 4, 4), (0, 16, 16), (0, 41, 41), (0, 90, 90)]
#
#     where_dropout_downlevels_default       = [False, False, False, False, True]
#     where_dropout_uplevels_default         = [True, True, True, True]
#     where_batchnormalize_downlevels_default= [False, False, False, False, True]
#     where_batchnormalize_uplevels_default  = [True, True, True, True]
#
#
#     def __init__(self, size_image,
#                  num_levels= Unet3D_General.num_levels_default,
#                  num_channels_in= Unet3D_General.num_channels_in_default,
#                  num_classes_out= Unet3D_General.num_classes_out_default,
#                  num_featmaps_in= Unet3D_General.num_featmaps_in_default,
#                  isUse_valid_convols= False,
#                  num_featmaps_levels= None,
#                  num_convols_downlevels= num_convols_downlevels_default,
#                  num_convols_uplevels= num_convols_uplevels_default,
#                  size_convolfilter_downlevels= size_convolfilter_downlevels_default,
#                  size_convolfilter_uplevels= size_convolfilter_uplevels_default,
#                  size_pooling_downlevels= size_pooling_levels_default,
#                  type_activate_hidden= Unet3D_General.type_activate_hidden_default,
#                  type_activate_output= Unet3D_General.type_activate_output_default,
#                  is_disable_convol_pooling_zdim_lastlevel= False,
#                  isuse_dropout= False,
#                  dropout_rate= Unet3D_General.dropout_rate_default,
#                  where_dropout_downlevels= where_dropout_downlevels_default,
#                  where_dropout_uplevels= where_dropout_uplevels_default,
#                  isuse_batchnormalize= False,
#                  where_batchnormalize_downlevels= where_batchnormalize_downlevels_default,
#                  where_batchnormalize_uplevels= where_batchnormalize_uplevels_default):
#         super(Unet3D_General_Extended, self).__init__(size_image,
#                                              num_levels,
#                                              num_channels_in,
#                                              num_classes_out,
#                                              num_featmaps_in,
#                                              isUse_valid_convols)
#
#         if num_featmaps_levels:
#             self.num_featmaps_levels = num_featmaps_levels
#         else:
#             # Default: double featmaps after every pooling
#             self.num_featmaps_levels = [num_featmaps_in] + [0]*(self.num_levels-1)
#             for i in range(1, self.num_levels):
#                 self.num_featmaps_levels[i] = 2 * self.num_featmaps_levels[i-1]
#
#         self.num_convols_downlevels      = num_convols_downlevels
#         self.num_convols_uplevels        = num_convols_uplevels
#         self.size_convolfilter_downlevels= size_convolfilter_downlevels[0:self.num_levels]
#         self.size_convolfilter_uplevels  = size_convolfilter_uplevels[0:self.num_levels]
#         self.size_pooling_downlevels     = size_pooling_downlevels[0:self.num_levels-1]
#         self.size_upsample_uplevels      = self.size_pooling_downlevels
#
#         if is_disable_convol_pooling_zdim_lastlevel:
#             temp_size_filter_lastlevel = self.size_convolfilter_downlevels[-1]
#             self.size_convolfilter_downlevels[-1] = (1, temp_size_filter_lastlevel[1], temp_size_filter_lastlevel[2])
#             temp_size_pooling_lastlevel = self.size_pooling_downlevels[-1]
#             self.size_pooling_downlevels[-1] = (1, temp_size_pooling_lastlevel[1], temp_size_pooling_lastlevel[2])
#
#         self.type_activate_hidden = type_activate_hidden
#         self.type_activate_output = type_activate_output
#
#         self.isuse_dropout = isuse_dropout
#         if isuse_dropout:
#             self.dropout_rate            = dropout_rate
#             self.where_dropout_downlevels= where_dropout_downlevels
#             self.where_dropout_uplevels  = where_dropout_uplevels
#
#         self.isuse_batchnormalize = isuse_batchnormalize
#         if isuse_batchnormalize:
#             self.where_batchnormalize_downlevels = where_batchnormalize_downlevels
#             self.where_batchnormalize_uplevels   = where_batchnormalize_uplevels
#
#
#     def build_model(self):
#         inputlayer = Input((self.size_image[0], self.size_image[1], self.size_image[2], self.num_channels_in))
#         hiddenlayer_next = inputlayer
#
#         list_hiddenlayer_skipconn = []
#         # ENCODING LAYERS
#         for i in range(self.num_levels):
#             for j in range(self.num_convols_downlevels):
#                 hiddenlayer_next = Convolution3D(self.num_featmaps_levels[i],
#                                                  self.size_convolfilter_downlevels[i],
#                                                  activation=self.type_activate_hidden,
#                                                  padding=self.type_padding_convol)(hiddenlayer_next)
#             #endfor
#
#             if self.isuse_dropout and self.where_dropout_downlevels[i]:
#                 hiddenlayer_next = Dropout(self.dropout_rate)(hiddenlayer_next)
#             if self.isuse_batchnormalize and self.where_batchnormalize_downlevels[i]:
#                 hiddenlayer_next = BatchNormalization()(hiddenlayer_next)
#             if i!=self.num_levels-1:
#                 list_hiddenlayer_skipconn.append(hiddenlayer_next)
#                 hiddenlayer_next = MaxPooling3D(pool_size=self.size_pooling_downlevels[i])(hiddenlayer_next)
#         #endfor
#
#         # DECODING LAYERS
#         for i in range(self.num_levels-2,-1,-1):
#             hiddenlayer_next = UpSampling3D(size=self.size_upsample_uplevels[i])(hiddenlayer_next)
#             hiddenlayer_next = concatenate([hiddenlayer_next, list_hiddenlayer_skipconn[i]], axis=-1)
#
#             for j in range(self.num_convols_downlevels):
#                 hiddenlayer_next = Convolution3D(self.num_featmaps_levels[i],
#                                                  self.size_convolfilter_uplevels[i],
#                                                  activation=self.type_activate_hidden,
#                                                  padding=self.type_padding_convol)(hiddenlayer_next)
#             #endfor
#
#             if self.isuse_dropout and self.where_dropout_uplevels[i]:
#                 hiddenlayer_next = Dropout(self.dropout_rate)(hiddenlayer_next)
#             if self.isuse_batchnormalize and self.where_batchnormalize_uplevels[i]:
#                 hiddenlayer_next = BatchNormalization()(hiddenlayer_next)
#         #endfor
#
#         outputlayer = Convolution3D(self.num_classes_out, (1, 1, 1), activation=self.type_activate_output)(hiddenlayer_next)
#
#         output_model = Model(inputs=inputlayer, outputs=outputlayer)
#
#         return output_model