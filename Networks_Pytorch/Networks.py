#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from torch.nn import Conv3d, ConvTranspose3d, MaxPool3d, Upsample, BatchNorm3d, Dropout3d, ReLU, Sigmoid, Softmax
import torch.nn as nn
import torch.nn.functional as F
import torch


class NeuralNetwork(nn.Module):

    def __init__(self, size_image,
                 num_channels_in= 1,
                 num_classes_out= 1,
                 num_featmaps_in= 16):
        super(NeuralNetwork, self).__init__()
        self.size_image = size_image
        self.num_channels_in = num_channels_in
        self.num_classes_out = num_classes_out
        self.num_featmaps_in = num_featmaps_in

    @staticmethod
    def get_create_model(type_model, dict_input_args):
        if type_model == 'Unet3D_Original':
            return Unet3D_Original(**dict_input_args)
        elif type_model == 'Unet3D_Tailored':
            return Unet3D_Tailored(**dict_input_args)

    def get_size_input(self):
        return [self.num_channels_in] + list(self.size_image)

    def get_size_output(self):
        return [self.num_classes_out] + list(self.size_image)

    def count_model_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_arch_desc(self):
        return NotImplemented

    def build_model(self):
        return NotImplemented

    def preprocess(self, *args, **kwargs):
        pass


class Unet3D_Original(NeuralNetwork):

    def __init__(self, size_image,
                 num_channels_in= 1,
                 num_classes_out= 1,
                 num_featmaps_in= 16):
        super(Unet3D_Original, self).__init__(size_image,
                                              num_channels_in,
                                              num_classes_out,
                                              num_featmaps_in)
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

    num_layers_default = 5
    num_featmaps_in_default = 16

    num_convolutions_downlays_default = 2
    num_convolution_uplays_default = 2
    size_convolutionkernel_downlays_default = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    size_convolutionkernel_uplays_default = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    size_pooling_layers_default = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    #size_cropping_layers = [(0, 4, 4), (0, 16, 16), (0, 41, 41), (0, 90, 90)]

    def __init__(self, size_image,
                 num_channels_in= 1,
                 num_classes_out= 1,
                 num_layers= num_layers_default,
                 num_featmaps_in= num_featmaps_in_default,
                 num_featmaps_layers= None,
                 num_convolution_downlays= num_convolutions_downlays_default,
                 num_convolution_uplays= num_convolution_uplays_default,
                 size_convolutionkernel_downlayers= size_convolutionkernel_downlays_default,
                 size_convolutionkernel_uplayers= size_convolutionkernel_uplays_default,
                 size_pooling_downlayers= size_pooling_layers_default,
                 is_disable_convolutionpooling_zdim_lastlayer= False):

        super(Unet3D_General, self).__init__(size_image, num_channels_in, num_classes_out)
        self.num_layers = num_layers
        if num_featmaps_layers:
            self.num_featmaps_layers = num_featmaps_layers
        else:
            # Default: double featmaps after every pooling
            self.num_featmaps_layers = [num_featmaps_in] + [0]*(self.num_layers-1)
            for i in range(1, self.num_layers):
                self.num_featmaps_layers[i] = 2 * self.num_featmaps_layers[i-1]

        self.num_convolution_downlays = num_convolution_downlays
        self.num_convolution_uplays = num_convolution_uplays
        self.size_convolutionkernel_downlayers = size_convolutionkernel_downlayers[0:self.num_layers]
        self.size_convolutionkernel_uplayers = size_convolutionkernel_uplayers[0:self.num_layers]
        self.size_pooling_downlayers = size_pooling_downlayers[0:self.num_layers-1]
        self.size_upsample_uplayers = self.size_pooling_downlayers

        if is_disable_convolutionpooling_zdim_lastlayer:
            temp_size_kernel_lastlayer = self.size_convolutionkernel_downlayers[-1]
            self.size_convolutionkernel_downlayers[-1] = (1, temp_size_kernel_lastlayer[1], temp_size_kernel_lastlayer[2])
            temp_size_pooling_lastlayer = self.size_pooling_downlayers[-1]
            self.size_pooling_downlayers[-1] = (1, temp_size_pooling_lastlayer[1], temp_size_pooling_lastlayer[2])

        self.build_model()

    def build_model(self):
        pass

    def forward(self, input):
        pass


class Unet3D_Tailored(NeuralNetwork):

    def __init__(self, size_image,
                 num_channels_in= 1,
                 num_classes_out= 1,
                 num_featmaps_in= 16):
        super(Unet3D_Tailored, self).__init__(size_image,
                                              num_channels_in,
                                              num_classes_out,
                                              num_featmaps_in)
        self.dropout_rate = 0.2

        self.build_model()

    def get_arch_desc(self):
        return ['Unet3D_Tailored', {'size_image': self.size_image,
                                    'num_channels_in': self.num_channels_in,
                                    'num_classes_out': self.num_classes_out,
                                    'num_featmaps_in': self.num_featmaps_in}]

    def build_model(self):

        num_featmaps_lay1 = self.num_featmaps_in
        self.convolution_downlay1_1 = Conv3d(self.num_channels_in, num_featmaps_lay1, kernel_size= 3, padding= 1)
        self.activation_downlay1_1 = ReLU(inplace=True)
        # self.batchnorm_downlay1_1 = BatchNorm3d(num_featmaps_lay1)
        self.convolution_downlay1_2 = Conv3d(num_featmaps_lay1, num_featmaps_lay1, kernel_size= 3, padding= 1)
        self.activation_downlay1_2 = ReLU(inplace=True)
        # self.batchnorm_downlay1_2 = BatchNorm3d(num_featmaps_lay1)
        # self.dropout_downlay1 = Dropout3d(p=self.dropout_rate)
        self.pooling_downlay1 = MaxPool3d(kernel_size= 2, padding= 0)

        num_featmaps_lay2 = 2 * num_featmaps_lay1
        self.convolution_downlay2_1 = Conv3d(num_featmaps_lay1, num_featmaps_lay2, kernel_size= 3, padding= 1)
        self.activation_downlay2_1 = ReLU(inplace=True)
        # self.batchnorm_downlay2_1 = BatchNorm3d(num_featmaps_lay2)
        self.convolution_downlay2_2 = Conv3d(num_featmaps_lay2, num_featmaps_lay2, kernel_size= 3, padding= 1)
        self.activation_downlay2_2 = ReLU(inplace=True)
        # self.batchnorm_downlay2_2 = BatchNorm3d(num_featmaps_lay2)
        # self.dropout_downlay2 = Dropout3d(p =self.dropout_rate)
        self.pooling_downlay2 = MaxPool3d(kernel_size= 2, padding= 0)

        num_featmaps_lay3 = 2 * num_featmaps_lay2
        self.convolution_downlay3_1 = Conv3d(num_featmaps_lay2, num_featmaps_lay3, kernel_size= 3, padding= 1)
        self.activation_downlay3_1 = ReLU(inplace=True)
        # self.batchnorm_downlay3_1 = BatchNorm3d(num_featmaps_lay3)
        self.convolution_downlay3_2 = Conv3d(num_featmaps_lay3, num_featmaps_lay3, kernel_size= 3, padding= 1)
        self.activation_downlay3_2 = ReLU(inplace=True)
        # self.batchnorm_downlay3_2 = BatchNorm3d(num_featmaps_lay3)
        # self.dropout_downlay3 = Dropout3d(p=self.dropout_rate)
        self.pooling_downlay3 = MaxPool3d(kernel_size= 2, padding= 0)

        num_featmaps_lay4 = 2 * num_featmaps_lay3
        self.convolution_downlay4_1 = Conv3d(num_featmaps_lay3, num_featmaps_lay4, kernel_size= 3, padding= 1)
        self.activation_downlay4_1 = ReLU(inplace=True)
        # self.batchnorm_downlay4_1 = BatchNorm3d(num_featmaps_lay4)
        self.convolution_downlay4_2 = Conv3d(num_featmaps_lay4, num_featmaps_lay4, kernel_size= 3, padding= 1)
        self.activation_downlay4_2 = ReLU(inplace=True)
        # self.batchnorm_downlay4_2 = BatchNorm3d(num_featmaps_lay4)
        # self.dropout_downlay4 = Dropout3d(p=self.dropout_rate)
        self.pooling_downlay4 = MaxPool3d(kernel_size= (1,2,2), padding= 0)
        #self.pooling_downlay4 = MaxPool3d(kernel_size= 2, padding=0)

        num_featmaps_lay5 = 2 * num_featmaps_lay4
        self.convolution_downlay5_1 = Conv3d(num_featmaps_lay4, num_featmaps_lay5, kernel_size= (1,3,3), padding= (0,1,1))
        #self.convolution_downlay5_1 = Conv3d(num_featmaps_lay4, num_featmaps_lay5, kernel_size=3, padding=1)
        self.activation_downlay5_1 = ReLU(inplace=True)
        # self.batchnorm_downlay5_1 = BatchNorm3d(num_featmaps_lay5)
        self.convolution_downlay5_2 = Conv3d(num_featmaps_lay5, num_featmaps_lay5, kernel_size= (1,3,3), padding= (0,1,1))
        #self.convolution_downlay5_2 = Conv3d(num_featmaps_lay5, num_featmaps_lay5, kernel_size= 3, padding= 1)
        self.activation_downlay5_2 = ReLU(inplace=True)
        # self.batchnorm_downlay5_2 = BatchNorm3d(num_featmaps_lay5)
        # self.dropout_downlay5 = Dropout3d(p=self.dropout_rate)
        self.upsample_downlay5 = Upsample(scale_factor= (1,2,2), mode= 'nearest')
        #self.upsample_downlay5 = Upsample(scale_factor= 2, mode='nearest')
        #self.upsample_downlay5 = ConvTranspose3d(num_featmaps_lay5, num_featmaps_lay5, kernel_size= (1,2,2), stride= (1,2,2), padding=0)
        #self.upsample_downlay5 = ConvTranspose3d(num_featmaps_lay5, num_featmaps_lay5, kernel_size= 2, stride= 2, padding=0)

        num_featmaps_lay4pl5 = num_featmaps_lay4 + num_featmaps_lay5
        #num_featmaps_lay4pl5 = num_featmaps_lay5
        self.convolution_uplay4_1 = Conv3d(num_featmaps_lay4pl5, num_featmaps_lay4, kernel_size= 3, padding= 1)
        self.activation_uplay4_1 = ReLU(inplace=True)
        # self.batchnorm_uplay4_1 = BatchNorm3d(num_featmaps_lay4)
        self.convolution_uplay4_2 = Conv3d(num_featmaps_lay4, num_featmaps_lay4, kernel_size= 3, padding= 1)
        self.activation_uplay4_2 = ReLU(inplace=True)
        # self.batchnorm_uplay4_2 = BatchNorm3d(num_featmaps_lay4)
        # self.dropout_uplay4 = Dropout3d(p=self.dropout_rate)
        self.upsample_uplay4 = Upsample(scale_factor= 2, mode= 'nearest')
        #self.upsample_uplay4 = ConvTranspose3d(num_featmaps_lay4, num_featmaps_lay4, kernel_size=2, stride=2, padding=0)

        num_featmaps_lay3pl4 = num_featmaps_lay3 + num_featmaps_lay4
        #num_featmaps_lay3pl4 = num_featmaps_lay4
        self.convolution_uplay3_1 = Conv3d(num_featmaps_lay3pl4, num_featmaps_lay3, kernel_size= 3, padding= 1)
        self.activation_uplay3_1 = ReLU(inplace=True)
        # self.batchnorm_uplay3_1 = BatchNorm3d(num_featmaps_lay3)
        self.convolution_uplay3_2 = Conv3d(num_featmaps_lay3, num_featmaps_lay3, kernel_size= 3, padding= 1)
        self.activation_uplay3_2 = ReLU(inplace=True)
        # self.batchnorm_uplay3_2 = BatchNorm3d(num_featmaps_lay3)
        # self.dropout_uplay3 = Dropout3d(p=self.dropout_rate)
        self.upsample_uplay3 = Upsample(scale_factor= 2, mode= 'nearest')
        #self.upsample_uplay3 = ConvTranspose3d(num_featmaps_lay3, num_featmaps_lay3, kernel_size=2, stride=2, padding=0)

        num_featmaps_lay2pl3 = num_featmaps_lay2 + num_featmaps_lay3
        #num_featmaps_lay2pl3 = num_featmaps_lay3
        self.convolution_uplay2_1 = Conv3d(num_featmaps_lay2pl3, num_featmaps_lay2, kernel_size= 3, padding= 1)
        self.activation_uplay2_1 = ReLU(inplace=True)
        # self.batchnorm_uplay2_1 = BatchNorm3d(num_featmaps_lay2)
        self.convolution_uplay2_2 = Conv3d(num_featmaps_lay2, num_featmaps_lay2, kernel_size= 3, padding= 1)
        self.activation_uplay2_2 = ReLU(inplace=True)
        # self.batchnorm_uplay2_2 = BatchNorm3d(num_featmaps_lay2)
        # self.dropout_uplay2 = Dropout3d(p=self.dropout_rate)
        self.upsample_uplay2 = Upsample(scale_factor= 2, mode= 'nearest')
        #self.upsample_uplay2 = ConvTranspose3d(num_featmaps_lay2, num_featmaps_lay2, kernel_size=2, stride=2, padding=0)

        num_featmaps_lay1pl2 = num_featmaps_lay1 + num_featmaps_lay2
        #num_featmaps_lay1pl2 = num_featmaps_lay2
        self.convolution_uplay1_1 = Conv3d(num_featmaps_lay1pl2, num_featmaps_lay1, kernel_size= 3, padding= 1)
        self.activation_uplay1_1 = ReLU(inplace=True)
        # self.batchnorm_uplay1_1 = BatchNorm3d(num_featmaps_lay1)
        self.convolution_uplay1_2 = Conv3d(num_featmaps_lay1, num_featmaps_lay1, kernel_size= 3, padding= 1)
        self.activation_uplay1_2 = ReLU(inplace=True)
        # self.batchnorm_uplay1_2 = BatchNorm3d(num_featmaps_lay1)
        # self.dropout_uplay1 = Dropout3d(p=self.dropout_rate)
        self.classification_layer = Conv3d(num_featmaps_lay1, self.num_classes_out, kernel_size= 1, padding= 0)

        self.activation_output = Sigmoid()

    def forward(self, input):

        hiddenlayer_next = self.convolution_downlay1_1(input)
        hiddenlayer_next = self.activation_downlay1_1(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_downlay1_1(hiddenlayer_next)
        hiddenlayer_next = self.convolution_downlay1_2(hiddenlayer_next)
        hiddenlayer_next = self.activation_downlay1_2(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_downlay1_2(hiddenlayer_next)
        # hiddenlayer_next = self.dropout_downlay1(hiddenlayer_next)
        hiddenlayer_skipconn_lev1 = hiddenlayer_next
        hiddenlayer_next = self.pooling_downlay1(hiddenlayer_next)

        hiddenlayer_next = self.convolution_downlay2_1(hiddenlayer_next)
        hiddenlayer_next = self.activation_downlay2_1(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_downlay2_1(hiddenlayer_next)
        hiddenlayer_next = self.convolution_downlay2_2(hiddenlayer_next)
        hiddenlayer_next = self.activation_downlay2_2(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_downlay2_2(hiddenlayer_next)
        # hiddenlayer_next = self.dropout_downlay2(hiddenlayer_next)
        hiddenlayer_skipconn_lev2 = hiddenlayer_next
        hiddenlayer_next = self.pooling_downlay2(hiddenlayer_next)

        hiddenlayer_next = self.convolution_downlay3_1(hiddenlayer_next)
        hiddenlayer_next = self.activation_downlay3_1(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_downlay3_1(hiddenlayer_next)
        hiddenlayer_next = self.convolution_downlay3_2(hiddenlayer_next)
        hiddenlayer_next = self.activation_downlay3_2(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_downlay3_2(hiddenlayer_next)
        # hiddenlayer_next = self.dropout_downlay3(hiddenlayer_next)
        hiddenlayer_skipconn_lev3 = hiddenlayer_next
        hiddenlayer_next = self.pooling_downlay3(hiddenlayer_next)

        hiddenlayer_next = self.convolution_downlay4_1(hiddenlayer_next)
        hiddenlayer_next = self.activation_downlay4_1(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_downlay4_1(hiddenlayer_next)
        hiddenlayer_next = self.convolution_downlay4_2(hiddenlayer_next)
        hiddenlayer_next = self.activation_downlay4_2(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_downlay4_2(hiddenlayer_next)
        # hiddenlayer_next = self.dropout_downlay4(hiddenlayer_next)
        hiddenlayer_skipconn_lev4 = hiddenlayer_next
        hiddenlayer_next = self.pooling_downlay4(hiddenlayer_next)

        hiddenlayer_next = self.convolution_downlay5_1(hiddenlayer_next)
        hiddenlayer_next = self.activation_downlay5_1(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_downlay5_1(hiddenlayer_next)
        hiddenlayer_next = self.convolution_downlay5_2(hiddenlayer_next)
        hiddenlayer_next = self.activation_downlay5_2(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_downlay5_2(hiddenlayer_next)
        # hiddenlayer_next = self.dropout_downlay5(hiddenlayer_next)
        hiddenlayer_next = self.upsample_downlay5(hiddenlayer_next)

        hiddenlayer_next = torch.cat([hiddenlayer_next, hiddenlayer_skipconn_lev4], dim=1)
        hiddenlayer_next = self.convolution_uplay4_1(hiddenlayer_next)
        hiddenlayer_next = self.activation_uplay4_1(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_uplay4_1(hiddenlayer_next)
        hiddenlayer_next = self.convolution_uplay4_2(hiddenlayer_next)
        hiddenlayer_next = self.activation_uplay4_2(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_uplay4_2(hiddenlayer_next)
        # hiddenlayer_next = self.dropout_uplay4(hiddenlayer_next)
        hiddenlayer_next = self.upsample_uplay4(hiddenlayer_next)

        hiddenlayer_next = torch.cat([hiddenlayer_next, hiddenlayer_skipconn_lev3], dim=1)
        hiddenlayer_next = self.convolution_uplay3_1(hiddenlayer_next)
        hiddenlayer_next = self.activation_uplay3_1(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_uplay3_1(hiddenlayer_next)
        hiddenlayer_next = self.convolution_uplay3_2(hiddenlayer_next)
        hiddenlayer_next = self.activation_uplay3_2(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_uplay3_2(hiddenlayer_next)
        # hiddenlayer_next = self.dropout_uplay3(hiddenlayer_next)
        hiddenlayer_next = self.upsample_uplay3(hiddenlayer_next)

        hiddenlayer_next = torch.cat([hiddenlayer_next, hiddenlayer_skipconn_lev2], dim=1)
        hiddenlayer_next = self.convolution_uplay2_1(hiddenlayer_next)
        hiddenlayer_next = self.activation_uplay2_1(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_uplay2_1(hiddenlayer_next)
        hiddenlayer_next = self.convolution_uplay2_2(hiddenlayer_next)
        hiddenlayer_next = self.activation_uplay2_2(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_uplay2_2(hiddenlayer_next)
        # hiddenlayer_next = self.dropout_uplay2(hiddenlayer_next)
        hiddenlayer_next = self.upsample_uplay2(hiddenlayer_next)

        hiddenlayer_next = torch.cat([hiddenlayer_next, hiddenlayer_skipconn_lev1], dim=1)
        hiddenlayer_next = self.convolution_uplay1_1(hiddenlayer_next)
        hiddenlayer_next = self.activation_uplay1_1(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_uplay1_1(hiddenlayer_next)
        hiddenlayer_next = self.convolution_uplay1_2(hiddenlayer_next)
        hiddenlayer_next = self.activation_uplay1_2(hiddenlayer_next)
        # hiddenlayer_next = self.batchnorm_uplay1_2(hiddenlayer_next)
        # hiddenlayer_next = self.dropout_uplay1(hiddenlayer_next)

        output = self.classification_layer(hiddenlayer_next)
        output = self.activation_output(output)

        return output


# all available networks
def DICTAVAILMODELS3D(size_image,
                      num_channels_in= 1,
                      num_classes_out= 1,
                      num_featmaps_in= 16,
                      num_layers= 5,
                      tailored_build_model= False,
                      type_network= 'classification',
                      type_activate_hidden= 'relu',
                      type_activate_output= 'sigmoid',
                      type_padding_convol= 'same',
                      is_disable_convol_pooling_lastlayer= False,
                      isuse_dropout= False,
                      isuse_batchnormalize= False):
    if tailored_build_model:
        return Unet3D_Tailored(size_image,
                               num_channels_in= num_channels_in,
                               num_classes_out= num_classes_out,
                               num_featmaps_in=num_featmaps_in)
    else:
        return Unet3D_General(size_image,
                              num_channels_in=num_channels_in,
                              num_classes_out=num_classes_out,
                              num_layers=num_layers,
                              num_featmaps_in=num_featmaps_in)
