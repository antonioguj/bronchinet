#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
from CommonUtil.ErrorMessages import *
import numpy as np


class BaseImageReconstructor(object):

    def __init__(self, size_image):
        self.size_image = size_image

    def check_shape_predict_data(self, predict_data_shape):
        pass

    def is_images_array_without_channels(self, in_array_shape):
        return len(in_array_shape) == len(self.size_image)

    def get_num_channels_array(self, in_array_shape):
        if self.is_images_array_without_channels(in_array_shape):
            return None
        else:
            return in_array_shape[-1]

    def get_reconstructed_image_sample_array(self, image_sample_array):

        if self.get_num_channels_array(image_sample_array.shape) == 1:
            return np.squeeze(image_sample_array, axis=-1)
        else:
            return self.get_reconstructed_image_sample_multiclass(image_sample_array)

    def get_reconstructed_image_sample_multiclass(self, image_sample_array):

        new_image_sample_array = np.ndarray(self.size_image_sample, dtype=image_sample_array.dtype)

        if len(self.size_image) == 2:
            for i in range(self.size_image[0]):
                for j in range(self.size_image[1]):
                    index_argmax = np.argmax(image_sample_array[i, j, :])
                    new_image_sample_array[i, j] = index_argmax
                # endfor
            # endfor
        elif len(self.size_image) == 3:
            for i in range(self.size_image[0]):
                for j in range(self.size_image[1]):
                    for k in range(self.size_image[2]):
                        index_argmax = np.argmax(image_sample_array[i, j, k, :])
                        new_image_sample_array[i, j, k] = index_argmax
                    # endfor
                # endfor
            # endfor
        else:
            message = "wrong shape of input images..." %(self.size_image)
            CatchErrorException(message)

        return new_image_sample_array

    def compute(self, predict_data):
        pass