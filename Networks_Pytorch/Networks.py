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
from torch.nn import Conv3d, ConvTranspose3d, MaxPool3d, Upsample, BatchNorm3d, Dropout3d, ReLU, Sigmoid, Softmax
import torch.nn as nn
import torch.nn.functional as F
import torch



class NeuralNetwork(nn.Module):

    def __init__(self, size_image,
                 num_levels,
                 num_channels_in,
                 num_classes_out,
                 num_featmaps_in,
                 isUse_valid_convols= False):
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
        # if type_model == 'Unet3D_Original':
        #     return Unet3D_Original(**dict_input_args)
        # elif type_model == 'Unet3D_General':
        #     return Unet3D_General(**dict_input_args)
        return Unet3D_General(**dict_input_args)

    def get_size_input(self):
        return [self.num_channels_in] + list(self.size_image)

    def get_size_output(self):
        return [self.num_classes_out] + list(self.size_output)

    def count_model_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_arch_desc(self):
        return NotImplemented

    def build_model(self):
        return NotImplemented

    def preprocess(self, *args, **kwargs):
        pass

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
        indexes_list_sizes_output_where_merge = [i for i,el in enumerate(self.list_module_opers) if el=='upsam']
        self.list_sizes_crop_merge = [self.list_sizes_output[i] for i in indexes_list_sizes_output_where_merge][::-1]

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
    def get_output_lims_crop(size_input, size_crop):
        z_begin = int( (size_input[0] - size_crop[0]) / 2)
        x_begin = int( (size_input[1] - size_crop[1]) / 2)
        y_begin = int( (size_input[2] - size_crop[2]) / 2)
        output_lims = ((z_begin, z_begin + size_crop[0]),
                       (x_begin, x_begin + size_crop[1]),
                       (y_begin, y_begin + size_crop[2]))
        return output_lims

    @classmethod
    def crop_image(cls, input, size_crop):
        size_input = input.shape[-3:]
        output_lims = cls.get_output_lims_crop(size_input, size_crop)
        return input[:,:, output_lims[0][0]:output_lims[0][1],
                          output_lims[1][0]:output_lims[1][1],
                          output_lims[2][0]:output_lims[2][1]]



class Unet3D_Original(NeuralNetwork):

    def __init__(self, size_image,
                 num_channels_in= 1,
                 num_classes_out= 1,
                 num_featmaps_in= 16):
        super(Unet3D_Original, self).__init__(size_image,
                                              5,
                                              num_channels_in,
                                              num_classes_out,
                                              num_featmaps_in,
                                              isUse_valid_convols=False)
        self.build_model()

    def get_arch_desc(self):
        return ['Unet3D_Original', {'size_image': self.size_image,
                                    'num_channels_in': self.num_channels_in,
                                    'num_classes_out': self.num_classes_out,
                                    'num_featmaps_in': self.num_featmaps_in}]


    def build_model(self):

        num_featmaps_lay1 = self.num_featmaps_in
        self.convolution_downlay1_1 = Conv3d(self.num_channels_in, num_featmaps_lay1, kernel_size= 3, padding= 1)
        self.convolution_downlay1_2 = Conv3d(num_featmaps_lay1, num_featmaps_lay1, kernel_size= 3, padding= 1)
        self.pooling_downlay1 = MaxPool3d(kernel_size= 2, padding= 0)

        num_featmaps_lay2 = 2 * num_featmaps_lay1
        self.convolution_downlay2_1 = Conv3d(num_featmaps_lay1, num_featmaps_lay2, kernel_size= 3, padding= 1)
        self.convolution_downlay2_2 = Conv3d(num_featmaps_lay2, num_featmaps_lay2, kernel_size= 3, padding= 1)
        self.pooling_downlay2 = MaxPool3d(kernel_size= 2, padding= 0)

        num_featmaps_lay3 = 2 * num_featmaps_lay2
        self.convolution_downlay3_1 = Conv3d(num_featmaps_lay2, num_featmaps_lay3, kernel_size= 3, padding= 1)
        self.convolution_downlay3_2 = Conv3d(num_featmaps_lay3, num_featmaps_lay3, kernel_size= 3, padding= 1)
        self.pooling_downlay3 = MaxPool3d(kernel_size= 2, padding= 0)

        num_featmaps_lay4 = 2 * num_featmaps_lay3
        self.convolution_downlay4_1 = Conv3d(num_featmaps_lay3, num_featmaps_lay4, kernel_size= 3, padding= 1)
        self.convolution_downlay4_2 = Conv3d(num_featmaps_lay4, num_featmaps_lay4, kernel_size= 3, padding= 1)
        self.pooling_downlay4 = MaxPool3d(kernel_size= 2, padding= 0)

        num_featmaps_lay5 = 2 * num_featmaps_lay4
        self.convolution_downlay5_1 = Conv3d(num_featmaps_lay4, num_featmaps_lay5, kernel_size= 3, padding= 1)
        self.convolution_downlay5_2 = Conv3d(num_featmaps_lay5, num_featmaps_lay5, kernel_size= 3, padding= 1)
        self.upsample_uplay5 = Upsample(scale_factor= 2, mode= 'nearest')

        num_featmaps_lay4pl5 = num_featmaps_lay4 + num_featmaps_lay5
        self.convolution_uplay4_1 = Conv3d(num_featmaps_lay4pl5, num_featmaps_lay4, kernel_size= 3, padding= 1)
        self.convolution_uplay4_2 = Conv3d(num_featmaps_lay4, num_featmaps_lay4, kernel_size= 3, padding= 1)
        self.upsample_uplay4 = Upsample(scale_factor= 2, mode= 'nearest')

        num_featmaps_lay3pl4 = num_featmaps_lay3 + num_featmaps_lay4
        self.convolution_uplay3_1 = Conv3d(num_featmaps_lay3pl4, num_featmaps_lay3, kernel_size= 3, padding= 1)
        self.convolution_uplay3_2 = Conv3d(num_featmaps_lay3, num_featmaps_lay3, kernel_size= 3, padding= 1)
        self.upsample_uplay3 = Upsample(scale_factor= 2, mode= 'nearest')

        num_featmaps_lay2pl3 = num_featmaps_lay2 + num_featmaps_lay3
        self.convolution_uplay2_1 = Conv3d(num_featmaps_lay2pl3, num_featmaps_lay2, kernel_size= 3, padding= 1)
        self.convolution_uplay2_2 = Conv3d(num_featmaps_lay2, num_featmaps_lay2, kernel_size= 3, padding= 1)
        self.upsample_uplay2 = Upsample(scale_factor= 2, mode= 'nearest')

        num_featmaps_lay1pl2 = num_featmaps_lay1 + num_featmaps_lay2
        self.convolution_uplay1_1 = Conv3d(num_featmaps_lay1pl2, num_featmaps_lay1, kernel_size= 3, padding= 1)
        self.convolution_uplay1_2 = Conv3d(num_featmaps_lay1, num_featmaps_lay1, kernel_size= 3, padding= 1)

        self.classification_layer = Conv3d(num_featmaps_lay1, self.num_classes_out, kernel_size= 1, padding= 0)
        self.activation_layer = Sigmoid()

    def forward(self, input):

        hiddenlay_down1_1 = self.convolution_downlay1_1(input)
        hiddenlay_down1_2 = self.convolution_downlay1_2(hiddenlay_down1_1)
        hiddenlay_down2_1 = self.pooling_downlay1(hiddenlay_down1_2)

        hiddenlay_down2_2 = self.convolution_downlay2_1(hiddenlay_down2_1)
        hiddenlay_down2_3 = self.convolution_downlay2_2(hiddenlay_down2_2)
        hiddenlay_down3_1 = self.pooling_downlay2(hiddenlay_down2_3)

        hiddenlay_down3_2 = self.convolution_downlay3_1(hiddenlay_down3_1)
        hiddenlay_down3_3 = self.convolution_downlay3_2(hiddenlay_down3_2)
        hiddenlay_down4_1 = self.pooling_downlay3(hiddenlay_down3_3)

        hiddenlay_down4_2 = self.convolution_downlay4_1(hiddenlay_down4_1)
        hiddenlay_down4_3 = self.convolution_downlay4_2(hiddenlay_down4_2)
        hiddenlay_down5_1 = self.pooling_downlay4(hiddenlay_down4_3)

        hiddenlay_down5_2 = self.convolution_downlay5_1(hiddenlay_down5_1)
        hiddenlay_down5_3 = self.convolution_downlay5_2(hiddenlay_down5_2)
        hiddenlay_up4_1 = self.upsample_uplay5(hiddenlay_down5_3)

        hiddenlay_up4_2 = torch.cat([hiddenlay_up4_1, hiddenlay_down4_3], dim=1)
        hiddenlay_up4_3 = self.convolution_uplay4_1(hiddenlay_up4_2)
        hiddenlay_up4_4 = self.convolution_uplay4_2(hiddenlay_up4_3)
        hiddenlay_up3_1 = self.upsample_uplay4(hiddenlay_up4_4)

        hiddenlay_up3_2 = torch.cat([hiddenlay_up3_1, hiddenlay_down3_3], dim=1)
        hiddenlay_up3_3 = self.convolution_uplay3_1(hiddenlay_up3_2)
        hiddenlay_up3_4 = self.convolution_uplay3_2(hiddenlay_up3_3)
        hiddenlay_up2_1 = self.upsample_uplay3(hiddenlay_up3_4)

        hiddenlay_up2_2 = torch.cat([hiddenlay_up2_1, hiddenlay_down2_3], dim=1)
        hiddenlay_up2_3 = self.convolution_uplay2_1(hiddenlay_up2_2)
        hiddenlay_up2_4 = self.convolution_uplay2_2(hiddenlay_up2_3)
        hiddenlay_up1_1 = self.upsample_uplay2(hiddenlay_up2_4)

        hiddenlay_up1_2 = torch.cat([hiddenlay_up1_1, hiddenlay_down1_2], dim=1)
        hiddenlay_up1_3 = self.convolution_uplay1_1(hiddenlay_up1_2)
        hiddenlay_up1_4 = self.convolution_uplay1_2(hiddenlay_up1_3)

        classifier = self.classification_layer(hiddenlay_up1_4)
        output = self.activation_layer(classifier)

        return output



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
                 type_activate_hidden=type_activate_hidden_default,
                 type_activate_output=type_activate_output_default):
        super(Unet3D_General, self).__init__(size_image,
                                             num_levels,
                                             num_channels_in,
                                             num_classes_out,
                                             num_featmaps_in,
                                             isUse_valid_convols)
        self.type_activate_hidden = type_activate_hidden
        self.type_activate_output = type_activate_output

        self.build_model()

    def get_arch_desc(self):
        return ['Unet3D_General', {'size_image': self.size_image,
                                   'num_levels': self.num_levels,
                                   'num_channels_in': self.num_channels_in,
                                   'num_classes_out': self.num_classes_out,
                                   'num_featmaps_in': self.num_featmaps_in,
                                   'isUse_valid_convols': self.isUse_valid_convols,
                                   'type_activate_hidden': self.type_activate_hidden,
                                   'type_activate_output': self.type_activate_output}]


    def build_model(self):
        if self.isUse_valid_convols:
           padding_val = 0
        else:
           padding_val = 1

        num_featmaps_lay1 = self.num_featmaps_in
        self.convolution_downlay1_1 = Conv3d(self.num_channels_in, num_featmaps_lay1, kernel_size= 3, padding= padding_val)
        self.convolution_downlay1_2 = Conv3d(num_featmaps_lay1, num_featmaps_lay1, kernel_size= 3, padding= padding_val)
        self.pooling_downlay1 = MaxPool3d(kernel_size= 2, padding= 0)

        num_featmaps_lay2 = 2 * num_featmaps_lay1
        self.convolution_downlay2_1 = Conv3d(num_featmaps_lay1, num_featmaps_lay2, kernel_size= 3, padding= padding_val)
        self.convolution_downlay2_2 = Conv3d(num_featmaps_lay2, num_featmaps_lay2, kernel_size= 3, padding= padding_val)
        self.pooling_downlay2 = MaxPool3d(kernel_size= 2, padding= 0)

        num_featmaps_lay3 = 2 * num_featmaps_lay2
        self.convolution_downlay3_1 = Conv3d(num_featmaps_lay2, num_featmaps_lay3, kernel_size= 3, padding= padding_val)
        self.convolution_downlay3_2 = Conv3d(num_featmaps_lay3, num_featmaps_lay3, kernel_size= 3, padding= padding_val)
        self.pooling_downlay3 = MaxPool3d(kernel_size= 2, padding= 0)

        num_featmaps_lay4 = 2 * num_featmaps_lay3
        self.convolution_downlay4_1 = Conv3d(num_featmaps_lay3, num_featmaps_lay4, kernel_size= 3, padding= 1)
        self.convolution_downlay4_2 = Conv3d(num_featmaps_lay4, num_featmaps_lay4, kernel_size= 3, padding= 1)
        self.pooling_downlay4 = MaxPool3d(kernel_size= 2, padding=0)

        num_featmaps_lay5 = 2 * num_featmaps_lay4
        self.convolution_downlay5_1 = Conv3d(num_featmaps_lay4, num_featmaps_lay5, kernel_size= 3, padding= 1)
        self.convolution_downlay5_2 = Conv3d(num_featmaps_lay5, num_featmaps_lay5, kernel_size= 3, padding= 1)
        self.upsample_downlay5 = Upsample(scale_factor= 2, mode='nearest')

        num_featmaps_lay4pl5 = num_featmaps_lay4 + num_featmaps_lay5
        self.convolution_uplay4_1 = Conv3d(num_featmaps_lay4pl5, num_featmaps_lay4, kernel_size= 3, padding= 1)
        self.convolution_uplay4_2 = Conv3d(num_featmaps_lay4, num_featmaps_lay4, kernel_size= 3, padding= 1)
        self.upsample_uplay4 = Upsample(scale_factor= 2, mode= 'nearest')

        num_featmaps_lay3pl4 = num_featmaps_lay3 + num_featmaps_lay4
        self.convolution_uplay3_1 = Conv3d(num_featmaps_lay3pl4, num_featmaps_lay3, kernel_size= 3, padding= padding_val)
        self.convolution_uplay3_2 = Conv3d(num_featmaps_lay3, num_featmaps_lay3, kernel_size= 3, padding= padding_val)
        self.upsample_uplay3 = Upsample(scale_factor= 2, mode= 'nearest')

        num_featmaps_lay2pl3 = num_featmaps_lay2 + num_featmaps_lay3
        self.convolution_uplay2_1 = Conv3d(num_featmaps_lay2pl3, num_featmaps_lay2, kernel_size= 3, padding= padding_val)
        self.convolution_uplay2_2 = Conv3d(num_featmaps_lay2, num_featmaps_lay2, kernel_size= 3, padding= padding_val)
        self.upsample_uplay2 = Upsample(scale_factor= 2, mode= 'nearest')

        num_featmaps_lay1pl2 = num_featmaps_lay1 + num_featmaps_lay2
        self.convolution_uplay1_1 = Conv3d(num_featmaps_lay1pl2, num_featmaps_lay1, kernel_size= 3, padding= padding_val)
        self.convolution_uplay1_2 = Conv3d(num_featmaps_lay1, num_featmaps_lay1, kernel_size= 3, padding= padding_val)

        self.classification_layer = Conv3d(num_featmaps_lay1, self.num_classes_out, kernel_size= 1, padding= 0)

        if self.type_activate_hidden == 'relu':
            self.activation_hidden = ReLU(inplace=True)
        else:
            message = 'type activation hidden not existing: \'%s\'' %(self.type_activate_hidden)
            CatchErrorException(message)

        if self.type_activate_output == 'sigmoid':
            self.activation_output = Sigmoid()
        elif self.type_activate_output == 'linear':
            self.activation_output = lambda x: x
        else:
            message = 'type activation output not existing: \'%s\'' %(self.type_activate_output)
            CatchErrorException(message)


    def forward(self, input):

        hiddenlayer_next = self.activation_hidden(self.convolution_downlay1_1(input))
        hiddenlayer_next = self.activation_hidden(self.convolution_downlay1_2(hiddenlayer_next))
        if self.isUse_valid_convols:
            hiddenlayer_skipconn_lev1 = self.crop_image(hiddenlayer_next, self.list_sizes_crop_merge[0])
        else:
            hiddenlayer_skipconn_lev1 = hiddenlayer_next
        hiddenlayer_next = self.pooling_downlay1(hiddenlayer_next)

        hiddenlayer_next = self.activation_hidden(self.convolution_downlay2_1(hiddenlayer_next))
        hiddenlayer_next = self.activation_hidden(self.convolution_downlay2_2(hiddenlayer_next))
        if self.isUse_valid_convols:
            hiddenlayer_skipconn_lev2 = self.crop_image(hiddenlayer_next, self.list_sizes_crop_merge[1])
        else:
            hiddenlayer_skipconn_lev2 = hiddenlayer_next
        hiddenlayer_next = self.pooling_downlay2(hiddenlayer_next)

        hiddenlayer_next = self.activation_hidden(self.convolution_downlay3_1(hiddenlayer_next))
        hiddenlayer_next = self.activation_hidden(self.convolution_downlay3_2(hiddenlayer_next))
        if self.isUse_valid_convols:
            hiddenlayer_skipconn_lev3 = self.crop_image(hiddenlayer_next, self.list_sizes_crop_merge[2])
        else:
            hiddenlayer_skipconn_lev3 = hiddenlayer_next
        hiddenlayer_next = self.pooling_downlay3(hiddenlayer_next)

        hiddenlayer_next = self.activation_hidden(self.convolution_downlay4_1(hiddenlayer_next))
        hiddenlayer_next = self.activation_hidden(self.convolution_downlay4_2(hiddenlayer_next))
        if self.isUse_valid_convols:
            hiddenlayer_skipconn_lev4 = self.crop_image(hiddenlayer_next, self.list_sizes_crop_merge[3])
        else:
            hiddenlayer_skipconn_lev4 = hiddenlayer_next
        hiddenlayer_next = self.pooling_downlay4(hiddenlayer_next)

        hiddenlayer_next = self.activation_hidden(self.convolution_downlay5_1(hiddenlayer_next))
        hiddenlayer_next = self.activation_hidden(self.convolution_downlay5_2(hiddenlayer_next))
        hiddenlayer_next = self.upsample_downlay5(hiddenlayer_next)

        hiddenlayer_next = torch.cat([hiddenlayer_next, hiddenlayer_skipconn_lev4], dim=1)
        hiddenlayer_next = self.activation_hidden(self.convolution_uplay4_1(hiddenlayer_next))
        hiddenlayer_next = self.activation_hidden(self.convolution_uplay4_2(hiddenlayer_next))
        hiddenlayer_next = self.upsample_uplay4(hiddenlayer_next)

        hiddenlayer_next = torch.cat([hiddenlayer_next, hiddenlayer_skipconn_lev3], dim=1)
        hiddenlayer_next = self.activation_hidden(self.convolution_uplay3_1(hiddenlayer_next))
        hiddenlayer_next = self.activation_hidden(self.convolution_uplay3_2(hiddenlayer_next))
        hiddenlayer_next = self.upsample_uplay3(hiddenlayer_next)

        hiddenlayer_next = torch.cat([hiddenlayer_next, hiddenlayer_skipconn_lev2], dim=1)
        hiddenlayer_next = self.activation_hidden(self.convolution_uplay2_1(hiddenlayer_next))
        hiddenlayer_next = self.activation_hidden(self.convolution_uplay2_2(hiddenlayer_next))
        hiddenlayer_next = self.upsample_uplay2(hiddenlayer_next)

        hiddenlayer_next = torch.cat([hiddenlayer_next, hiddenlayer_skipconn_lev1], dim=1)
        hiddenlayer_next = self.activation_hidden(self.convolution_uplay1_1(hiddenlayer_next))
        hiddenlayer_next = self.activation_hidden(self.convolution_uplay1_2(hiddenlayer_next))

        output = self.classification_layer(hiddenlayer_next)
        output = self.activation_output(output)

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



# all available networks
def DICTAVAILMODELS3D(size_image,
                      num_levels= 5,
                      num_channels_in= 1,
                      num_classes_out= 1,
                      num_featmaps_in= 16,
                      isUse_valid_convols= False,
                      type_network= 'classification',
                      type_activate_hidden= 'relu'):
                      #is_disable_convol_pooling_lastlayer= False,
                      #isuse_dropout= False,
                      #isuse_batchnormalize= False):
    if type_network == 'classification':
        type_activate_output = 'sigmoid'
    elif type_network == 'regression':
        type_activate_output = 'linear'
    else:
        message = 'type network not existing: \'%s\'' % (type_network)
        CatchErrorException(message)

    return Unet3D_General(size_image,
                          num_levels=num_levels,
                          num_channels_in=num_channels_in,
                          num_classes_out=num_classes_out,
                          num_featmaps_in=num_featmaps_in,
                          isUse_valid_convols= isUse_valid_convols,
                          type_activate_hidden=type_activate_hidden,
                          type_activate_output=type_activate_output)
