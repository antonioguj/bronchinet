#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

import torch.nn as nn
import torch


class Unet3D_Original(NeuralNetwork):

    num_layers = 5
    num_featmaps_base = 16

    size_convolfilter = (3, 3, 3)
    size_pooling = (2, 2, 2)

    def __init__(self, ):
        self.size_image = size_image

    def get_model(self):

        inputlayer = Input((self.size_image[0], self.size_image[1], self.size_image[2], 1))