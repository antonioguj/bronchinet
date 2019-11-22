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
import numpy as np


def is_image_array_without_channels(size_image, in_array_shape):
    return len(in_array_shape) == len(size_image)


def get_num_channels_array(size_image, in_array_shape):
    if is_image_array_without_channels(size_image, in_array_shape):
        return None
    else:
        return in_array_shape[-1]


def update_seed_with_index(seed, index):
    if seed is not None:
        return seed + index
    else:
        return None
