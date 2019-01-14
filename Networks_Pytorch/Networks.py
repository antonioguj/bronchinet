#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from torch.nn import Conv3d, MaxPool3d, Upsample
import torch


class NeuralNetwork(object):

    def build_model(self):
        pass

    def forward(self, input):
        pass


class Unet3D_Original(NeuralNetwork):

    def __init__(self, size_image,
                 num_channels_in=1,
                 num_classes_out=1):
        self.size_image      = size_image
        self.num_channels_in = num_channels_in
        self.num_classes_out = num_classes_out
        self.num_featmaps_base = 16

    def build_model(self):

        num_featmaps_lay1        = self.num_featmaps_base
        self.convolut_downlay1_1 = Conv3d(self.num_channels_in, num_featmaps_lay1, kernel_size=3, stride=1, padding=1)
        self.convolut_downlay1_2 = Conv3d(num_featmaps_lay1,    num_featmaps_lay1, kernel_size=3, stride=1, padding=1)
        self.pooling_downlay1    = MaxPool3d(kernel_size=2, stride=2, padding=0)

        num_featmaps_lay2        = 2 * num_featmaps_lay1
        self.convolut_downlay2_1 = Conv3d(num_featmaps_lay1, num_featmaps_lay2, kernel_size=3, stride=1, padding=1)
        self.convolut_downlay2_2 = Conv3d(num_featmaps_lay2, num_featmaps_lay2, kernel_size=3, stride=1, padding=1)
        self.pooling_downlay2    = MaxPool3d(kernel_size=2, stride=2, padding=0)

        num_featmaps_lay3        = 2 * num_featmaps_lay2
        self.convolut_downlay3_1 = Conv3d(num_featmaps_lay2, num_featmaps_lay3, kernel_size=3, stride=1, padding=1)
        self.convolut_downlay3_2 = Conv3d(num_featmaps_lay3, num_featmaps_lay3, kernel_size=3, stride=1, padding=1)
        self.pooling_downlay3    = MaxPool3d(kernel_size=2, stride=2, padding=0)

        num_featmaps_lay4        = 2 * num_featmaps_lay3
        self.convolut_downlay4_1 = Conv3d(num_featmaps_lay3, num_featmaps_lay4, kernel_size=3, stride=1, padding=1)
        self.convolut_downlay4_2 = Conv3d(num_featmaps_lay4, num_featmaps_lay4, kernel_size=3, stride=1, padding=1)
        self.pooling_downlay4    = MaxPool3d(kernel_size=2, stride=2, padding=0)

        num_featmaps_lay5        = 2 * num_featmaps_lay4
        self.convolut_downlay5_1 = Conv3d(num_featmaps_lay4, num_featmaps_lay5, kernel_size=3, stride=1, padding=1)
        self.convolut_downlay5_2 = Conv3d(num_featmaps_lay5, num_featmaps_lay5, kernel_size=3, stride=1, padding=1)
        self.upsample_uplay5     = Upsample(scale_factor=2, mode='nearest')

        self.convolut_uplay4_1   = Conv3d(num_featmaps_lay5, num_featmaps_lay4, kernel_size=3, stride=1, padding=1)
        self.convolut_uplay4_2   = Conv3d(num_featmaps_lay4, num_featmaps_lay4, kernel_size=3, stride=1, padding=1)
        self.upsample_uplay4     = Upsample(scale_factor=2, mode='nearest')

        self.convolut_uplay3_1   = Conv3d(num_featmaps_lay4, num_featmaps_lay3, kernel_size=3, stride=1, padding=1)
        self.convolut_uplay3_2   = Conv3d(num_featmaps_lay3, num_featmaps_lay3, kernel_size=3, stride=1, padding=1)
        self.upsample_uplay3     = Upsample(scale_factor=2, mode='nearest')

        self.convolut_uplay2_1   = Conv3d(num_featmaps_lay3, num_featmaps_lay2, kernel_size=3, stride=1, padding=1)
        self.convolut_uplay2_2   = Conv3d(num_featmaps_lay2, num_featmaps_lay2, kernel_size=3, stride=1, padding=1)
        self.upsample_uplay2     = Upsample(scale_factor=2, mode='nearest')

        self.convolut_uplay1_1   = Conv3d(num_featmaps_lay2, num_featmaps_lay1, kernel_size=3, stride=1, padding=1)
        self.convolut_uplay1_2   = Conv3d(num_featmaps_lay1, num_featmaps_lay1, kernel_size=3, stride=1, padding=1)
        self.convolut_lastlayer  = Conv3d(num_featmaps_lay1, self.num_classes_out, kernel_size=1, stride=1, padding=0)

    def forward(self, input):

        hidden_down1_1 = self.convolut_downlay1_1(input)
        hidden_down1_2 = self.convolut_downlay1_2(hidden_down1_1)
        hidden_down2_1 = self.pooling_downlay1   (hidden_down1_2)

        hidden_down2_2 = self.convolut_downlay2_1(hidden_down2_1)
        hidden_down2_3 = self.convolut_downlay2_2(hidden_down2_2)
        hidden_down3_1 = self.pooling_downlay2   (hidden_down2_3)

        hidden_down3_2 = self.convolut_downlay3_1(hidden_down3_1)
        hidden_down3_3 = self.convolut_downlay3_2(hidden_down3_2)
        hidden_down4_1 = self.pooling_downlay3   (hidden_down3_3)

        hidden_down4_2 = self.convolut_downlay4_1(hidden_down4_1)
        hidden_down4_3 = self.convolut_downlay4_2(hidden_down4_2)
        hidden_down5_1 = self.pooling_downlay4   (hidden_down4_3)

        hidden_down5_2 = self.convolut_downlay5_1(hidden_down5_1)
        hidden_down5_3 = self.convolut_downlay5_2(hidden_down5_2)
        hidden_up4_1   = self.upsample_uplay5    (hidden_down5_3)

        hidden_up4_2   = torch.cat([hidden_up4_1, hidden_down4_3])
        hidden_up4_3   = self.convolut_uplay4_1  (hidden_up4_2)
        hidden_up4_4   = self.convolut_uplay4_2  (hidden_up4_3)
        hidden_up3_1   = self.upsample_uplay4    (hidden_up4_4)

        hidden_up3_2   = torch.cat([hidden_up3_1, hidden_down3_3])
        hidden_up3_3   = self.convolut_uplay3_1  (hidden_up3_2)
        hidden_up3_4   = self.convolut_uplay3_2  (hidden_up3_3)
        hidden_up2_1   = self.upsample_uplay3    (hidden_up3_4)

        hidden_up2_2   = torch.cat([hidden_up2_1, hidden_down2_3])
        hidden_up2_3   = self.convolut_uplay2_1  (hidden_up2_2)
        hidden_up2_4   = self.convolut_uplay2_2  (hidden_up2_3)
        hidden_up1_1   = self.upsample_uplay2    (hidden_up2_4)

        hidden_up1_2   = torch.cat([hidden_up1_1, hidden_down1_2])
        hidden_up1_3   = self.convolut_uplay1_1  (hidden_up1_2)
        hidden_up1_4   = self.convolut_uplay1_2  (hidden_up1_3)
        output         = self.convolut_lastlayer (hidden_up1_4)

        return output


class Unet3D_General(NeuralNetwork):

    num_layers_default = 5
    num_featmaps_base_default = 16

    num_convols_downlayers_default = 2
    num_convols_uplayers_default   = 2
    size_convolfilter_downlayers_default= [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    size_convolfilter_uplayers_default  = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    size_pooling_layers_default         = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    #size_cropping_layers = [(0, 4, 4), (0, 16, 16), (0, 41, 41), (0, 90, 90)]


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
                 is_disable_convol_pooling_zdim_lastlayer=False):

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

    def build_model(self):
        pass

    def forward(self, input):
        pass


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

    return Unet3D_Original(size_image,
                           num_channels_in=num_channels_in)