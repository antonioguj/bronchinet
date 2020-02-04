#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.ErrorMessages import *
from tensorflow.keras.layers import Input, concatenate, Dropout, BatchNormalization
from tensorflow.keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, Cropping3D, Conv3DTranspose
from tensorflow.keras.models import Model, load_model



class NeuralNetwork(object):

    def __init__(self, size_image,
                 num_levels,
                 num_channels_in,
                 num_classes_out,
                 num_featmaps_in,
                 isUse_valid_convols=False):
        super(NeuralNetwork, self).__init__()
        self.size_image = size_image
        self.num_levels = num_levels
        self.num_channels_in = num_channels_in
        self.num_classes_out = num_classes_out
        self.num_featmaps_in = num_featmaps_in

        self.isUse_valid_convols = isUse_valid_convols
        if self.isUse_valid_convols:
            self.gen_list_module_operations()
            self.gen_list_sizes_output_valid()
            self.gen_list_sizes_crop_merge()

        if self.isUse_valid_convols:
            self.size_output = self.get_size_output_valid(self.size_image)
        else:
            self.size_output = self.size_image


    @staticmethod
    def get_create_model(type_model, dict_input_args):
        if type_model == 'Unet3D_Original':
            return Unet3D_Original(**dict_input_args)
        elif type_model == 'Unet3D_General':
            return Unet3D_General(**dict_input_args)

    def get_size_input(self):
        return [self.num_channels_in] + list(self.size_image)

    def get_size_output(self):
        return [self.num_classes_out] + list(self.size_output)

    def count_model_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def preprocess(self, *args, **kwargs):
        pass

    def build_model(self):
        return NotImplemented

    def build_model_and_compile(self, optimizer, lossfunction, metrics):
        return self.build_model().compile(optimizer=optimizer,
                                          loss=lossfunction,
                                          metrics=metrics)
    @staticmethod
    def get_load_saved_model(model_saved_path, custom_objects=None):
        return load_model(model_saved_path, custom_objects=custom_objects)

    def gen_list_module_operations(self):
        if self.num_levels == 1:
            self.list_module_opers = ['convol', 'convol', 'convol', 'convol', 'conv_last']
        elif self.num_levels <= self.num_levels_non_padded:
            self.list_module_opers = (self.num_levels-1) * ['convol', 'convol', 'pool'] + \
                                                           ['convol', 'convol'] + \
                                     (self.num_levels-1) * ['upsam', 'convol', 'convol'] + \
                                                           ['conv_last']
        else:
            #Assume that last convolutions have padding, to avoid large reduction of image dims
            num_levels_with_padding = self.num_levels - self.num_levels_non_padded - 1
            self.list_module_opers = self.num_levels_non_padded * ['convol', 'convol', 'pool'] + \
                                     num_levels_with_padding    * ['convol_pad', 'convol_pad', 'pool'] + \
                                                                  ['convol_pad', 'convol_pad'] + \
                                     num_levels_with_padding    * ['upsam', 'convol_pad', 'convol_pad'] + \
                                     self.num_levels_non_padded * ['upsam', 'convol', 'convol'] + \
                                                                  ['conv_last']

    def gen_list_sizes_crop_merge(self):
        indexes_list_sizes_output_where_pooling = [(i-1) for i,el in enumerate(self.list_module_opers) if el=='pool']
        indexes_list_sizes_output_where_merge   = [i for i,el in enumerate(self.list_module_opers) if el=='upsam']
        self.list_sizes_border_crop_merge = [self.get_size_border_crop(self.list_sizes_output[i], self.list_sizes_output[j]) for i,j in zip(indexes_list_sizes_output_where_pooling,
                                                                                                                                            indexes_list_sizes_output_where_merge[::-1])]

    def gen_list_sizes_output_valid(self):
        self.list_sizes_output = []
        size_last = self.size_image
        for i, oper in enumerate(self.list_module_opers):
            if oper=='convol':
                size_last = self.get_size_output_valid_after_convolution(size_last)
            elif oper=='convol_pad':
                size_last = size_last
            elif oper=='pool':
                size_last = self.get_size_output_valid_after_pooling(size_last)
            elif oper=='upsam':
                size_last = self.get_size_output_valid_after_upsample(size_last)
            elif oper=='conv_last':
                size_last = self.get_size_output_valid_after_convolution(size_last, size_kernel=(1,1,1))
            else:
                return NotImplemented
            self.list_sizes_output.append(size_last)
        #endfor

    def get_size_output_valid(self, size_input, level_beg=None, level_end=None):
        if level_beg and level_end:
            this_list_module_opers = self.list_module_opers[level_beg:level_end]
        else:
            this_list_module_opers = self.list_module_opers

        size_last = size_input
        for i, oper in enumerate(this_list_module_opers):
            if oper=='convol':
                size_last = self.get_size_output_valid_after_convolution(size_last)
            elif oper=='convol_pad':
                size_last = size_last
            elif oper=='pool':
                size_last = self.get_size_output_valid_after_pooling(size_last)
            elif oper=='upsam':
                size_last = self.get_size_output_valid_after_upsample(size_last)
            elif oper=='conv_last':
                size_last = self.get_size_output_valid_after_convolution(size_last, size_kernel=(1,1,1))
            else:
                return NotImplemented
        #endfor
        return size_last

    @staticmethod
    def get_size_output_valid_after_convolution(size_input, size_kernel=(3,3,3)):
        output_size = tuple([s_i-s_k+1 for s_i, s_k in zip(size_input, size_kernel)])
        return output_size

    @staticmethod
    def get_size_output_valid_after_pooling(size_input, size_pool=(2,2,2)):
        output_size = tuple([s_i//s_p for s_i, s_p in zip(size_input, size_pool)])
        return output_size

    @staticmethod
    def get_size_output_valid_after_upsample(size_input, size_upsample=(2,2,2)):
        output_size = tuple([s_i*s_u for s_i, s_u in zip(size_input, size_upsample)])
        return output_size

    @staticmethod
    def get_size_border_crop(size_input, size_crop):
        dx_border = (size_input[0] - size_crop[0]) / 2
        dy_border = (size_input[1] - size_crop[1]) / 2
        dz_border = (size_input[2] - size_crop[2]) / 2
        return (dx_border, dy_border, dz_border)

    @staticmethod
    def get_output_lims_crop(size_input, size_crop):
        z_beg = (size_input[0] - size_crop[0]) / 2
        x_beg = (size_input[1] - size_crop[1]) / 2
        y_beg = (size_input[2] - size_crop[2]) / 2
        output_lims = ((z_beg, z_beg + size_crop[0]),
                       (x_beg, x_beg + size_crop[1]),
                       (y_beg, y_beg + size_crop[2]))
        return output_lims



class Unet3D_Original(NeuralNetwork):

    def __init__(self, size_image,
                 num_channels_in=1,
                 num_classes_out=1,
                 num_featmaps_in=16):
        super(Unet3D_Original, self).__init__(size_image,
                                              5,
                                              num_channels_in,
                                              num_classes_out,
                                              num_featmaps_in,
                                              isUse_valid_convols=False)
        self.build_model()


    def build_model(self):
        inputlayer = Input((self.size_image[0], self.size_image[1], self.size_image[2], self.num_channels_in))

        num_featmaps_lev1   = self.num_featmaps_in
        hiddenlayer_down1_2 = Convolution3D(num_featmaps_lev1, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputlayer)
        hiddenlayer_down1_3 = Convolution3D(num_featmaps_lev1, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down1_2)
        hiddenlayer_down2_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down1_3)

        num_featmaps_lev2   = 2 * num_featmaps_lev1
        hiddenlayer_down2_2 = Convolution3D(num_featmaps_lev2, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down2_1)
        hiddenlayer_down2_3 = Convolution3D(num_featmaps_lev2, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down2_2)
        hiddenlayer_down3_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down2_3)

        num_featmaps_lev3   = 2 * num_featmaps_lev2
        hiddenlayer_down3_2 = Convolution3D(num_featmaps_lev3, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down3_1)
        hiddenlayer_down3_3 = Convolution3D(num_featmaps_lev3, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down3_2)
        hiddenlayer_down4_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down3_3)

        num_featmaps_lev4   = 2 * num_featmaps_lev3
        hiddenlayer_down4_2 = Convolution3D(num_featmaps_lev4, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down4_1)
        hiddenlayer_down4_3 = Convolution3D(num_featmaps_lev4, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down4_2)
        hiddenlayer_down5_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down4_3)

        num_featmaps_lev5   = 2 * num_featmaps_lev4
        hiddenlayer_down5_2 = Convolution3D(num_featmaps_lev5, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down5_1)
        hiddenlayer_down5_3 = Convolution3D(num_featmaps_lev5, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_down5_2)

        hiddenlayer_up4_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_down5_3)
        hiddenlayer_up4_1 = concatenate([hiddenlayer_up4_1, hiddenlayer_down4_3], axis=-1)
        hiddenlayer_up4_2 = Convolution3D(num_featmaps_lev4, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up4_1)
        hiddenlayer_up4_3 = Convolution3D(num_featmaps_lev4, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up4_2)

        hiddenlayer_up3_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_up4_3)
        hiddenlayer_up3_1 = concatenate([hiddenlayer_up3_1, hiddenlayer_down3_3], axis=-1)
        hiddenlayer_up3_2 = Convolution3D(num_featmaps_lev3, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up3_1)
        hiddenlayer_up3_3 = Convolution3D(num_featmaps_lev3, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up3_2)

        hiddenlayer_up2_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_up3_3)
        hiddenlayer_up2_1 = concatenate([hiddenlayer_up2_1, hiddenlayer_down2_3], axis=-1)
        hiddenlayer_up2_2 = Convolution3D(num_featmaps_lev2, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up2_1)
        hiddenlayer_up2_3 = Convolution3D(num_featmaps_lev2, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up2_2)

        hiddenlayer_up1_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_up2_3)
        hiddenlayer_up1_1 = concatenate([hiddenlayer_up1_1, hiddenlayer_down1_3], axis=-1)
        hiddenlayer_up1_2 = Convolution3D(num_featmaps_lev1, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up1_1)
        hiddenlayer_up1_3 = Convolution3D(num_featmaps_lev1, kernel_size=(3, 3, 3), activation='relu', padding='same')(hiddenlayer_up1_2)

        outputlayer = Convolution3D(self.num_classes_out, kernel_size=(1, 1, 1), activation='sigmoid')(hiddenlayer_up1_3)

        output_model = Model(inputs=inputlayer, outputs=outputlayer)

        return output_model



class Unet3D_General(NeuralNetwork):
    num_levels_default = 5
    num_levels_non_padded = 3
    num_channels_in_default = 1
    num_classes_out_default = 1
    num_featmaps_in_default = 16

    dropout_rate_default = 0.2

    type_activate_hidden_default = 'relu'
    type_activate_output_default = 'sigmoid'

    def __init__(self, size_image,
                 num_levels= num_levels_default,
                 num_channels_in= num_channels_in_default,
                 num_classes_out= num_classes_out_default,
                 num_featmaps_in= num_featmaps_in_default,
                 isUse_valid_convols= False,
                 type_activate_hidden= type_activate_hidden_default,
                 type_activate_output= type_activate_output_default):
        super(Unet3D_General, self).__init__(size_image,
                                             num_levels,
                                             num_channels_in,
                                             num_classes_out,
                                             num_featmaps_in,
                                             isUse_valid_convols)
        self.type_activate_hidden = type_activate_hidden
        self.type_activate_output = type_activate_output


    def build_model(self):
        if self.isUse_valid_convols:
           type_padding = 'valid'
        else:
           type_padding = 'same'

        inputlayer = Input((self.size_image[0], self.size_image[1], self.size_image[2], self.num_channels_in))

        num_featmaps_lev1   = self.num_featmaps_in
        hiddenlayer_down1_2 = Convolution3D(num_featmaps_lev1, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding=type_padding)(inputlayer)
        hiddenlayer_down1_3 = Convolution3D(num_featmaps_lev1, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding=type_padding)(hiddenlayer_down1_2)
        if self.isUse_valid_convols:
            hiddenlayer_skipconn_lev1 = Cropping3D(cropping= self.list_sizes_border_crop_merge[0])(hiddenlayer_down1_3)
        else:
            hiddenlayer_skipconn_lev1 = hiddenlayer_down1_3
        hiddenlayer_down2_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down1_3)

        num_featmaps_lev2   = 2 * num_featmaps_lev1
        hiddenlayer_down2_2 = Convolution3D(num_featmaps_lev2, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding=type_padding)(hiddenlayer_down2_1)
        hiddenlayer_down2_3 = Convolution3D(num_featmaps_lev2, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding=type_padding)(hiddenlayer_down2_2)
        if self.isUse_valid_convols:
            hiddenlayer_skipconn_lev2 = Cropping3D(cropping= self.list_sizes_border_crop_merge[1])(hiddenlayer_down2_3)
        else:
            hiddenlayer_skipconn_lev2 = hiddenlayer_down2_3
        hiddenlayer_down3_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down2_3)

        num_featmaps_lev3   = 2 * num_featmaps_lev2
        hiddenlayer_down3_2 = Convolution3D(num_featmaps_lev3, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding=type_padding)(hiddenlayer_down3_1)
        hiddenlayer_down3_3 = Convolution3D(num_featmaps_lev3, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding=type_padding)(hiddenlayer_down3_2)
        if self.isUse_valid_convols:
            hiddenlayer_skipconn_lev3 = Cropping3D(cropping= self.list_sizes_border_crop_merge[2])(hiddenlayer_down3_3)
        else:
            hiddenlayer_skipconn_lev3 = hiddenlayer_down3_3
        hiddenlayer_down4_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down3_3)

        num_featmaps_lev4   = 2 * num_featmaps_lev3
        hiddenlayer_down4_2 = Convolution3D(num_featmaps_lev4, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding='same')(hiddenlayer_down4_1)
        hiddenlayer_down4_3 = Convolution3D(num_featmaps_lev4, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding='same')(hiddenlayer_down4_2)
        if self.isUse_valid_convols:
            hiddenlayer_skipconn_lev4 = Cropping3D(cropping= self.list_sizes_border_crop_merge[3])(hiddenlayer_down4_3)
        else:
            hiddenlayer_skipconn_lev4 = hiddenlayer_down4_3
        hiddenlayer_down5_1 = MaxPooling3D(pool_size=(2, 2, 2))(hiddenlayer_down4_3)

        num_featmaps_lev5   = 2 * num_featmaps_lev4
        hiddenlayer_down5_2 = Convolution3D(num_featmaps_lev5, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding='same')(hiddenlayer_down5_1)
        hiddenlayer_down5_3 = Convolution3D(num_featmaps_lev5, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding='same')(hiddenlayer_down5_2)

        hiddenlayer_up4_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_down5_3)
        hiddenlayer_up4_1 = concatenate([hiddenlayer_up4_1, hiddenlayer_skipconn_lev4], axis=-1)
        hiddenlayer_up4_2 = Convolution3D(num_featmaps_lev4, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding='same')(hiddenlayer_up4_1)
        hiddenlayer_up4_3 = Convolution3D(num_featmaps_lev4, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding='same')(hiddenlayer_up4_2)

        hiddenlayer_up3_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_up4_3)
        hiddenlayer_up3_1 = concatenate([hiddenlayer_up3_1, hiddenlayer_skipconn_lev3], axis=-1)
        hiddenlayer_up3_2 = Convolution3D(num_featmaps_lev3, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding=type_padding)(hiddenlayer_up3_1)
        hiddenlayer_up3_3 = Convolution3D(num_featmaps_lev3, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding=type_padding)(hiddenlayer_up3_2)

        hiddenlayer_up2_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_up3_3)
        hiddenlayer_up2_1 = concatenate([hiddenlayer_up2_1, hiddenlayer_skipconn_lev2], axis=-1)
        hiddenlayer_up2_2 = Convolution3D(num_featmaps_lev2, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding=type_padding)(hiddenlayer_up2_1)
        hiddenlayer_up2_3 = Convolution3D(num_featmaps_lev2, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding=type_padding)(hiddenlayer_up2_2)

        hiddenlayer_up1_1 = UpSampling3D(size=(2, 2, 2))(hiddenlayer_up2_3)
        hiddenlayer_up1_1 = concatenate([hiddenlayer_up1_1, hiddenlayer_skipconn_lev1], axis=-1)
        hiddenlayer_up1_2 = Convolution3D(num_featmaps_lev1, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding=type_padding)(hiddenlayer_up1_1)
        hiddenlayer_up1_3 = Convolution3D(num_featmaps_lev1, kernel_size=(3, 3, 3), activation=self.type_activate_hidden, padding=type_padding)(hiddenlayer_up1_2)

        outputlayer = Convolution3D(self.num_classes_out, kernel_size=(1, 1, 1), activation=self.type_activate_output)(hiddenlayer_up1_3)

        output_model = Model(inputs=inputlayer, outputs=outputlayer)

        return output_model



class Unet3D_General_EXTENDED(Unet3D_General):
    num_convols_downlevels_default = 2
    num_convols_uplevels_default   = 2
    size_convolfilter_downlevels_default= [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    size_convolfilter_uplevels_default  = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    size_pooling_levels_default         = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    #size_cropping_levels_default        = [(0, 4, 4), (0, 16, 16), (0, 41, 41), (0, 90, 90)]

    where_dropout_downlevels_default       = [False, False, False, False, True]
    where_dropout_uplevels_default         = [True, True, True, True]
    where_batchnormalize_downlevels_default= [False, False, False, False, True]
    where_batchnormalize_uplevels_default  = [True, True, True, True]


    def __init__(self, size_image,
                 num_levels= Unet3D_General.num_levels_default,
                 num_channels_in= Unet3D_General.num_channels_in_default,
                 num_classes_out= Unet3D_General.num_classes_out_default,
                 num_featmaps_in= Unet3D_General.num_featmaps_in_default,
                 isUse_valid_convols= False,
                 num_featmaps_levels= None,
                 num_convols_downlevels= num_convols_downlevels_default,
                 num_convols_uplevels= num_convols_uplevels_default,
                 size_convolfilter_downlevels= size_convolfilter_downlevels_default,
                 size_convolfilter_uplevels= size_convolfilter_uplevels_default,
                 size_pooling_downlevels= size_pooling_levels_default,
                 type_activate_hidden= Unet3D_General.type_activate_hidden_default,
                 type_activate_output= Unet3D_General.type_activate_output_default,
                 is_disable_convol_pooling_zdim_lastlevel= False,
                 isuse_dropout= False,
                 dropout_rate= Unet3D_General.dropout_rate_default,
                 where_dropout_downlevels= where_dropout_downlevels_default,
                 where_dropout_uplevels= where_dropout_uplevels_default,
                 isuse_batchnormalize= False,
                 where_batchnormalize_downlevels= where_batchnormalize_downlevels_default,
                 where_batchnormalize_uplevels= where_batchnormalize_uplevels_default):
        super(Unet3D_General, self).__init__(size_image,
                                             num_levels,
                                             num_channels_in,
                                             num_classes_out,
                                             num_featmaps_in,
                                             isUse_valid_convols)

        if num_featmaps_levels:
            self.num_featmaps_levels = num_featmaps_levels
        else:
            # Default: double featmaps after every pooling
            self.num_featmaps_levels = [num_featmaps_in] + [0]*(self.num_levels-1)
            for i in range(1, self.num_levels):
                self.num_featmaps_levels[i] = 2 * self.num_featmaps_levels[i-1]

        self.num_convols_downlevels      = num_convols_downlevels
        self.num_convols_uplevels        = num_convols_uplevels
        self.size_convolfilter_downlevels= size_convolfilter_downlevels[0:self.num_levels]
        self.size_convolfilter_uplevels  = size_convolfilter_uplevels[0:self.num_levels]
        self.size_pooling_downlevels     = size_pooling_downlevels[0:self.num_levels-1]
        self.size_upsample_uplevels      = self.size_pooling_downlevels

        if is_disable_convol_pooling_zdim_lastlevel:
            temp_size_filter_lastlevel = self.size_convolfilter_downlevels[-1]
            self.size_convolfilter_downlevels[-1] = (1, temp_size_filter_lastlevel[1], temp_size_filter_lastlevel[2])
            temp_size_pooling_lastlevel = self.size_pooling_downlevels[-1]
            self.size_pooling_downlevels[-1] = (1, temp_size_pooling_lastlevel[1], temp_size_pooling_lastlevel[2])

        self.type_activate_hidden = type_activate_hidden
        self.type_activate_output = type_activate_output

        self.isuse_dropout = isuse_dropout
        if isuse_dropout:
            self.dropout_rate            = dropout_rate
            self.where_dropout_downlevels= where_dropout_downlevels
            self.where_dropout_uplevels  = where_dropout_uplevels

        self.isuse_batchnormalize = isuse_batchnormalize
        if isuse_batchnormalize:
            self.where_batchnormalize_downlevels = where_batchnormalize_downlevels
            self.where_batchnormalize_uplevels   = where_batchnormalize_uplevels


    def build_model(self):
        inputlayer = Input((self.size_image[0], self.size_image[1], self.size_image[2], self.num_channels_in))
        hiddenlayer_next = inputlayer

        list_hiddenlayer_skipconn = []
        # ENCODING LAYERS
        for i in range(self.num_levels):
            for j in range(self.num_convols_downlevels):
                hiddenlayer_next = Convolution3D(self.num_featmaps_levels[i],
                                                 self.size_convolfilter_downlevels[i],
                                                 activation=self.type_activate_hidden,
                                                 padding=self.type_padding_convol)(hiddenlayer_next)
            #endfor

            if self.isuse_dropout and self.where_dropout_downlevels[i]:
                hiddenlayer_next = Dropout(self.dropout_rate)(hiddenlayer_next)
            if self.isuse_batchnormalize and self.where_batchnormalize_downlevels[i]:
                hiddenlayer_next = BatchNormalization()(hiddenlayer_next)
            if i!=self.num_levels-1:
                list_hiddenlayer_skipconn.append(hiddenlayer_next)
                hiddenlayer_next = MaxPooling3D(pool_size=self.size_pooling_downlevels[i])(hiddenlayer_next)
        #endfor

        # DECODING LAYERS
        for i in range(self.num_levels-2,-1,-1):
            hiddenlayer_next = UpSampling3D(size=self.size_upsample_uplevels[i])(hiddenlayer_next)
            hiddenlayer_next = concatenate([hiddenlayer_next, list_hiddenlayer_skipconn[i]], axis=-1)

            for j in range(self.num_convols_downlevels):
                hiddenlayer_next = Convolution3D(self.num_featmaps_levels[i],
                                                 self.size_convolfilter_uplevels[i],
                                                 activation=self.type_activate_hidden,
                                                 padding=self.type_padding_convol)(hiddenlayer_next)
            #endfor

            if self.isuse_dropout and self.where_dropout_uplevels[i]:
                hiddenlayer_next = Dropout(self.dropout_rate)(hiddenlayer_next)
            if self.isuse_batchnormalize and self.where_batchnormalize_uplevels[i]:
                hiddenlayer_next = BatchNormalization()(hiddenlayer_next)
        #endfor

        outputlayer = Convolution3D(self.num_classes_out, (1, 1, 1), activation=self.type_activate_output)(hiddenlayer_next)

        output_model = Model(inputs=inputlayer, outputs=outputlayer)

        return output_model



# all available networks
def DICTAVAILMODELS3D(size_image,
                      num_levels= 5,
                      num_channels_in= 1,
                      num_classes_out= 1,
                      num_featmaps_in= 16,
                      isUse_valid_convols= False,
                      type_network= 'classification',
                      type_activate_hidden= 'relu'):
                      #is_disable_convol_pooling_lastlevel= False,
                      #isuse_dropout= False,
                      #isuse_batchnormalize= False):
    if type_network == 'classification':
        type_activate_output = 'sigmoid'
    elif type_network == 'regression':
        type_activate_output = 'linear'
    else:
        message = 'type network not existing: \'%s\'' %(type_network)
        CatchErrorException(message)

    return Unet3D_General(size_image,
                          num_levels=num_levels,
                          num_channels_in=num_channels_in,
                          num_classes_out=num_classes_out,
                          num_featmaps_in=num_featmaps_in,
                          isUse_valid_convols= isUse_valid_convols,
                          type_activate_hidden= type_activate_hidden,
                          type_activate_output= type_activate_output)
