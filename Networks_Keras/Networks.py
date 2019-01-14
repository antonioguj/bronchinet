#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from keras.layers import Input, merge, concatenate, Dropout, BatchNormalization
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, Cropping3D, Conv3DTranspose
from keras.models import Model, load_model


class NeuralNetwork(object):

    def get_model(self):
        pass

    def get_model_and_compile(self, optimizer, lossfunction, metrics):
        return self.get_model().compile(optimizer=optimizer,
                                        loss=lossfunction,
                                        metrics=metrics )
    @staticmethod
    def get_load_saved_model(model_saved_path, custom_objects=None):
        return load_model(model_saved_path, custom_objects=custom_objects)


class Unet3D_Original(NeuralNetwork):

    def __init__(self, size_image,
                 num_channels_in=1,
                 num_classes_out=1):
        self.size_image      = size_image
        self.num_channels_in = num_channels_in
        self.num_classes_out = num_classes_out
        self.num_featmaps_base = 16

    def get_model(self):

        inputlayer = Input((self.size_image[0], self.size_image[1], self.size_image[2], self.num_channels_in))

        num_featmaps_lay1   = self.num_featmaps_base
        hiddenlayer_down1_2 = Convolution3D(num_featmaps_lay1, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputlayer)
        hiddenlayer_down1_3 = Convolution3D(num_featmaps_lay1, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down1_2)
        hiddenlayer_down2_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down1_3)

        num_featmaps_lay2   = 2 * num_featmaps_lay1
        hiddenlayer_down2_2 = Convolution3D(num_featmaps_lay2, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down2_1)
        hiddenlayer_down2_3 = Convolution3D(num_featmaps_lay2, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down2_2)
        hiddenlayer_down3_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down2_3)

        num_featmaps_lay3   = 2 * num_featmaps_lay2
        hiddenlayer_down3_2 = Convolution3D(num_featmaps_lay3, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down3_1)
        hiddenlayer_down3_3 = Convolution3D(num_featmaps_lay3, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down3_2)
        hiddenlayer_down4_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down3_3)

        num_featmaps_lay4   = 2 * num_featmaps_lay3
        hiddenlayer_down4_2 = Convolution3D(num_featmaps_lay4, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down4_1)
        hiddenlayer_down4_3 = Convolution3D(num_featmaps_lay4, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down4_2)
        hiddenlayer_down5_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down4_3)

        num_featmaps_lay5   = 2 * num_featmaps_lay4
        hiddenlayer_down5_2 = Convolution3D(num_featmaps_lay5, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down5_1)
        hiddenlayer_down5_3 = Convolution3D(num_featmaps_lay5, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down5_2)

        hiddenlayer_up4_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_down5_3)
        hiddenlayer_up4_1 = merge([hiddenlayer_up4_1, hiddenlayer_down4_3], mode='concat', concat_axis=-1)
        hiddenlayer_up4_2 = Convolution3D(num_featmaps_lay4, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up4_1)
        hiddenlayer_up4_3 = Convolution3D(num_featmaps_lay4, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up4_2)

        hiddenlayer_up3_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_up4_3)
        hiddenlayer_up3_1 = merge([hiddenlayer_up3_1, hiddenlayer_down3_3], mode='concat', concat_axis=-1)
        hiddenlayer_up3_2 = Convolution3D(num_featmaps_lay3, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up3_1)
        hiddenlayer_up3_3 = Convolution3D(num_featmaps_lay3, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up3_2)

        hiddenlayer_up2_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_up3_3)
        hiddenlayer_up2_1 = merge([hiddenlayer_up2_1, hiddenlayer_down2_3], mode='concat', concat_axis=-1)
        hiddenlayer_up2_2 = Convolution3D(num_featmaps_lay2, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up2_1)
        hiddenlayer_up2_3 = Convolution3D(num_featmaps_lay2, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up2_2)

        hiddenlayer_up1_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_up2_3)
        hiddenlayer_up1_1 = merge([hiddenlayer_up1_1, hiddenlayer_down1_3], mode='concat', concat_axis=-1)
        hiddenlayer_up1_2 = Convolution3D(num_featmaps_lay1, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up1_1)
        hiddenlayer_up1_3 = Convolution3D(num_featmaps_lay1, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up1_2)

        outputlayer = Convolution3D(self.num_classes_out, kernel_size=(1, 1, 1), activation='sigmoid')(hiddenlayer_up1_3)

        out_model = Model(input=inputlayer, output=outputlayer)

        return out_model


class Unet3D_General(NeuralNetwork):

    num_layers_default = 5
    num_featmaps_base_default = 16

    num_convols_downlayers_default = 2
    num_convols_uplayers_default   = 2
    size_convolfilter_downlayers_default= [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    size_convolfilter_uplayers_default  = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    size_pooling_layers_default         = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    #size_cropping_layers = [(0, 4, 4), (0, 16, 16), (0, 41, 41), (0, 90, 90)]

    type_activate_hidden_default = 'relu'
    type_activate_output_default = 'sigmoid'
    type_padding_convol_default  = 'same'

    dropout_rate_default = 0.2

    where_dropout_downlayers_default       = [False, False, False, False, True]
    where_dropout_uplayers_default         = [True, True, True, True]
    where_batchnormalize_downlayers_default= [False, False, False, False, True]
    where_batchnormalize_uplayers_default  = [True, True, True, True]


    def __init__(self, size_image,
                 num_channels_in=1,
                 num_classes_out=1,
                 num_layers=num_layers_default,
                 num_featmaps_base=num_featmaps_base_default,
                 num_featmaps_layers=None,
                 num_convols_downlayers=num_convols_downlayers_default,
                 num_convols_uplayers=num_convols_uplayers_default,
                 size_convolfilter_downlayers=size_convolfilter_downlayers_default,
                 size_convolfilter_uplayers=size_convolfilter_uplayers_default,
                 size_pooling_downlayers=size_pooling_layers_default,
                 type_activate_hidden=type_activate_hidden_default,
                 type_activate_output=type_activate_output_default,
                 type_padding_convol=type_padding_convol_default,
                 is_disable_convol_pooling_zdim_lastlayer=False,
                 isuse_dropout=False,
                 dropout_rate=dropout_rate_default,
                 where_dropout_downlayers=where_dropout_downlayers_default,
                 where_dropout_uplayers=where_dropout_uplayers_default,
                 isuse_batchnormalize=False,
                 where_batchnormalize_downlayers=where_batchnormalize_downlayers_default,
                 where_batchnormalize_uplayers=where_batchnormalize_uplayers_default):

        self.size_image      = size_image
        self.num_channels_in = num_channels_in
        self.num_classes_out = num_classes_out

        self.num_layers = num_layers
        if num_featmaps_layers:
            self.num_featmaps_layers = num_featmaps_layers
        else:
            # Default: double featmaps after every pooling
            self.num_featmaps_layers = [num_featmaps_base] + [0]*(self.num_layers-1)
            for i in range(1, self.num_layers):
                self.num_featmaps_layers[i] = 2 * self.num_featmaps_layers[i-1]

        self.num_convols_downlayers      = num_convols_downlayers
        self.num_convols_uplayers        = num_convols_uplayers
        self.size_convolfilter_downlayers= size_convolfilter_downlayers
        self.size_convolfilter_uplayers  = size_convolfilter_uplayers
        self.size_pooling_downlayers     = size_pooling_downlayers
        self.size_upsample_uplayers      = self.size_pooling_downlayers

        if is_disable_convol_pooling_zdim_lastlayer:
            temp_size_filter_lastlayer = self.size_convolfilter_downlayers[-1]
            self.size_convolfilter_downlayers[-1] = (1, temp_size_filter_lastlayer[1], temp_size_filter_lastlayer[2])

            temp_size_pooling_lastlayer = self.size_pooling_downlayers[-1]
            self.size_pooling_downlayers[-1] = (1, temp_size_pooling_lastlayer[1], temp_size_pooling_lastlayer[2])

        self.type_activate_hidden = type_activate_hidden
        self.type_activate_output = type_activate_output
        self.type_padding_convol  = type_padding_convol

        self.isuse_dropout = isuse_dropout
        if isuse_dropout:
            self.dropout_rate            = dropout_rate
            self.where_dropout_downlayers= where_dropout_downlayers
            self.where_dropout_uplayers  = where_dropout_uplayers

        self.isuse_batchnormalize = isuse_batchnormalize
        if isuse_batchnormalize:
            self.where_batchnormalize_downlayers = where_batchnormalize_downlayers
            self.where_batchnormalize_uplayers   = where_batchnormalize_uplayers


    def get_model(self):

        inputlayer = Input((self.size_image[0], self.size_image[1], self.size_image[2], self.num_channels_in))

        list_hiddenlayer_toskipconnect = []
        hiddenlayer_next = inputlayer

        # ENCODING LAYERS
        for i in range(self.num_layers):
            for j in range(self.num_convols_downlayers):
                hiddenlayer_next = Convolution3D(self.num_featmaps_layers[i],
                                                 self.size_convolfilter_downlayers[i],
                                                 activation=self.type_activate_hidden,
                                                 padding=self.type_padding_convol)(hiddenlayer_next)
            #endfor

            if self.isuse_dropout and self.where_dropout_downlayers[i]:
                hiddenlayer_next = Dropout(self.dropout_rate)(hiddenlayer_next)

            if self.isuse_batchnormalize and self.where_batchnormalize_downlayers[i]:
                hiddenlayer_next = BatchNormalization()(hiddenlayer_next)

            if i!=self.num_layers-1:
                list_hiddenlayer_toskipconnect.append(hiddenlayer_next)

                hiddenlayer_next = MaxPooling3D(pool_size=self.size_pooling_downlayers[i])(hiddenlayer_next)
        #endfor

        # DECODING LAYERS
        for i in range(self.num_layers-2,-1,-1):
            hiddenlayer_next = UpSampling3D(size=self.size_upsample_uplayers[i])(hiddenlayer_next)
            hiddenlayer_next = merge([hiddenlayer_next, list_hiddenlayer_toskipconnect[i]], mode='concat', concat_axis=-1)

            for j in range(self.num_convols_downlayers):
                hiddenlayer_next = Convolution3D(self.num_featmaps_layers[i],
                                                 self.size_convolfilter_uplayers[i],
                                                 activation=self.type_activate_hidden,
                                                 padding=self.type_padding_convol)(hiddenlayer_next)
            #endfor

            if self.isuse_dropout and self.where_dropout_uplayers[i]:
                hiddenlayer_next = Dropout(self.dropout_rate)(hiddenlayer_next)

            if self.isuse_batchnormalize and self.where_batchnormalize_uplayers[i]:
                hiddenlayer_next = BatchNormalization()(hiddenlayer_next)
        #endfor

        outputlayer = Convolution3D(self.num_classes_out, (1, 1, 1), activation=self.type_activate_output)(hiddenlayer_next)

        out_model = Model(input=inputlayer, output=outputlayer)

        return out_model


class Unet3D_Tailored(NeuralNetwork):

    def __init__(self, size_image,
                 num_channels_in=1,
                 num_classes_out=1):
        self.size_image      = size_image
        self.num_channels_in = num_channels_in
        self.num_classes_out = num_classes_out
        self.num_featmaps_base = 16
        self.dropout_rate      = 0.5

    def get_model(self):

        inputlayer = Input((self.size_image[0], self.size_image[1], self.size_image[2], self.num_channels_in))

        num_featmaps_lay1   = self.num_featmaps_base
        hiddenlayer_down1_2 = Convolution3D(num_featmaps_lay1, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputlayer)
        hiddenlayer_down1_2 = BatchNormalization()(hiddenlayer_down1_2)
        hiddenlayer_down1_3 = Convolution3D(num_featmaps_lay1, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down1_2)
        hiddenlayer_down1_3 = BatchNormalization()(hiddenlayer_down1_3)
        hiddenlayer_down2_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down1_3)

        num_featmaps_lay2   = 2 * num_featmaps_lay1
        hiddenlayer_down2_2 = Convolution3D(num_featmaps_lay2, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down2_1)
        hiddenlayer_down2_2 = BatchNormalization()(hiddenlayer_down2_2)
        hiddenlayer_down2_3 = Convolution3D(num_featmaps_lay2, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down2_2)
        hiddenlayer_down2_3 = BatchNormalization()(hiddenlayer_down2_3)
        hiddenlayer_down3_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down2_3)

        num_featmaps_lay3   = 2 * num_featmaps_lay2
        hiddenlayer_down3_2 = Convolution3D(num_featmaps_lay3, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down3_1)
        hiddenlayer_down3_2 = BatchNormalization()(hiddenlayer_down3_2)
        hiddenlayer_down3_3 = Convolution3D(num_featmaps_lay3, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down3_2)
        hiddenlayer_down3_3 = BatchNormalization()(hiddenlayer_down3_3)
        hiddenlayer_down4_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down3_3)

        num_featmaps_lay4   = 2 * num_featmaps_lay3
        hiddenlayer_down4_2 = Convolution3D(num_featmaps_lay4, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down4_1)
        hiddenlayer_down4_2 = BatchNormalization()(hiddenlayer_down4_2)
        hiddenlayer_down4_3 = Convolution3D(num_featmaps_lay4, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down4_2)
        hiddenlayer_down4_3 = BatchNormalization()(hiddenlayer_down4_3)
        hiddenlayer_down5_1 = MaxPooling3D(pool_size=(1, 2, 2))(hiddenlayer_down4_3)

        num_featmaps_lay5   = 2 * num_featmaps_lay4
        hiddenlayer_down5_2 = Convolution3D(num_featmaps_lay5, kernel_size=(1, 3, 3), activation='relu', padding='same')(hiddenlayer_down5_1)
        hiddenlayer_down5_2 = BatchNormalization()(hiddenlayer_down5_2)
        hiddenlayer_down5_3 = Convolution3D(num_featmaps_lay5, kernel_size=(1, 3, 3), activation='relu', padding='same')(hiddenlayer_down5_2)
        hiddenlayer_down5_3 = BatchNormalization()(hiddenlayer_down5_3)

        hiddenlayer_up4_1 = UpSampling3D(size=(1, 2, 2))(hiddenlayer_down5_3)
        hiddenlayer_up4_1 = merge([hiddenlayer_up4_1, hiddenlayer_down4_3], mode='concat', concat_axis=-1)
        hiddenlayer_up4_2 = Convolution3D(num_featmaps_lay4, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up4_1)
        hiddenlayer_up4_2 = BatchNormalization()(hiddenlayer_up4_2)
        hiddenlayer_up4_3 = Convolution3D(num_featmaps_lay4, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up4_2)
        hiddenlayer_up4_3 = BatchNormalization()(hiddenlayer_up4_3)

        hiddenlayer_up3_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_up4_3)
        hiddenlayer_up3_1 = merge([hiddenlayer_up3_1, hiddenlayer_down3_3], mode='concat', concat_axis=-1)
        hiddenlayer_up3_2 = Convolution3D(num_featmaps_lay3, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up3_1)
        hiddenlayer_up3_2 = BatchNormalization()(hiddenlayer_up3_2)
        hiddenlayer_up3_3 = Convolution3D(num_featmaps_lay3, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up3_2)
        hiddenlayer_up3_3 = BatchNormalization()(hiddenlayer_up3_3)

        hiddenlayer_up2_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_up3_3)
        hiddenlayer_up2_1 = merge([hiddenlayer_up2_1, hiddenlayer_down2_3], mode='concat', concat_axis=-1)
        hiddenlayer_up2_2 = Convolution3D(num_featmaps_lay2, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up2_1)
        hiddenlayer_up2_2 = BatchNormalization()(hiddenlayer_up2_2)
        hiddenlayer_up2_3 = Convolution3D(num_featmaps_lay2, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up2_2)
        hiddenlayer_up2_3 = BatchNormalization()(hiddenlayer_up2_3)

        hiddenlayer_up1_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_up2_3)
        hiddenlayer_up1_1 = merge([hiddenlayer_up1_1, hiddenlayer_down1_3], mode='concat', concat_axis=-1)
        hiddenlayer_up1_2 = Convolution3D(num_featmaps_lay1, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up1_1)
        hiddenlayer_up1_2 = BatchNormalization()(hiddenlayer_up1_2)
        hiddenlayer_up1_3 = Convolution3D(num_featmaps_lay1, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up1_2)
        hiddenlayer_up1_3 = BatchNormalization()(hiddenlayer_up1_3)

        hiddenlayer_up1_4 = Dropout(self.dropout_rate)(hiddenlayer_up1_3)

        outputlayer = Convolution3D(self.num_classes_out, kernel_size=(1, 1, 1), activation='sigmoid')(hiddenlayer_up1_4)

        out_model = Model(input=inputlayer, output=outputlayer)

        return out_model


# all available networks
def DICTAVAILMODELS3D(size_image,
                      num_channels_in=1,
                      num_classes_out=1,
                      tailored_build_model=False,
                      num_layers=5,
                      num_featmaps_base=16,
                      type_network='classification',
                      type_activate_hidden='relu',
                      type_activate_output='sigmoid',
                      type_padding_convol='same',
                      is_disable_convol_pooling_lastlayer=False,
                      isuse_dropout=False,
                      isuse_batchnormalize=False):

    if tailored_build_model:
        return Unet3D_Tailored(size_image,
                               num_channels_in=num_channels_in)
    else:
        if type_network == 'regression':
            type_activate_output = 'linear'

        return Unet3D_General(size_image,
                              num_channels_in=num_channels_in,
                              num_classes_out=num_classes_out,
                              num_layers=num_layers,
                              num_featmaps_base=num_featmaps_base,
                              type_activate_hidden=type_activate_hidden,
                              type_activate_output=type_activate_output,
                              type_padding_convol=type_padding_convol,
                              is_disable_convol_pooling_zdim_lastlayer=is_disable_convol_pooling_lastlayer,
                              isuse_dropout=isuse_dropout,
                              isuse_batchnormalize=isuse_batchnormalize)