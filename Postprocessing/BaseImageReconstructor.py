#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
from Common.ErrorMessages import *
from Postprocessing.FilteringValidUnetOutput import *
import numpy as np


class BaseImageReconstructor(object):

    def __init__(self, size_image,
                 size_total_image,
                 isfilterImages=False,
                 prop_valid_outUnet=None,
                 is_onehotmulticlass=False):
        self.size_image         = size_image
        self.size_total_image   = size_total_image
        self.isfilterImages     = isfilterImages
        self.is_onehotmulticlass= is_onehotmulticlass

        if isfilterImages:
            size_valid_outUnet = tuple([int(prop_valid_outUnet * elem) for elem in size_image])

            print("Filter output probability maps, ith size of valid convs Unet: %s..." % (str(size_valid_outUnet)))

            self.filterImages_calculator = FilteringValidUnetOutput3D(IMAGES_DIMS_Z_X_Y, size_valid_outUnet)
        else:
            self.filterImages_calculator = None


    def complete_init_data(self, in_array_shape):
        pass

    def check_correct_shape_input_array(self, in_array_shape):
        pass

    def is_images_array_without_channels(self, in_array_shape):
        return len(in_array_shape) == len(self.size_image)

    def get_num_channels_array(self, in_array_shape):
        if self.is_images_array_without_channels(in_array_shape):
            return None
        else:
            return in_array_shape[-1]

    def get_shape_out_array(self, in_array_shape):
        num_channels = self.get_num_channels_array(in_array_shape[1:])
        return list(self.size_total_image) + [num_channels]

    def get_reshaped_in_array(self, in_array):
        num_channels = self.get_num_channels_array(in_array.shape[1:])
        if num_channels:
            return in_array
        else:
            return np.expand_dims(in_array, axis=-1)

    def get_reshaped_out_array(self, out_array):
        num_channels = self.get_num_channels_array(out_array.shape)
        if num_channels==1:
            return np.squeeze(out_array, axis=-1)
        else:
            return out_array


    def get_processed_images_array(self, images_array):
        if self.is_onehotmulticlass:
            images_array = self.get_processed_image_onehotmulticlass_array(images_array)

        if self.isfilterImages:
            images_array = self.filterImages_calculator.get_images_array(images_array)

        return images_array

        # .get_num_channels_array(images_array.shape)
        #
        # if not num_channels_array:
        #     return images_array
        # elif num_channels_array==1:
        #     return np.squeeze(images_array, axis=-1)
        # else:
        #     return images_array

    def get_processed_image_onehotmulticlass_array(self, images_array):

        new_images_array = np.ndarray(self.size_image, dtype=images_array.dtype)

        if len(self.size_image) == 2:
            for i in range(self.size_image[0]):
                for j in range(self.size_image[1]):
                    index_argmax = np.argmax(images_array[i, j, :])
                    new_images_array[i, j] = index_argmax
                # endfor
            # endfor

        elif len(self.size_image) == 3:
            for i in range(self.size_image[0]):
                for j in range(self.size_image[1]):
                    for k in range(self.size_image[2]):
                        index_argmax = np.argmax(images_array[i, j, k, :])
                        new_images_array[i, j, k] = index_argmax
                    # endfor
                # endfor
            # endfor
        else:
            message = "wrong shape of input images..." %(self.size_image)
            CatchErrorException(message)

        return new_images_array


    def compute(self, predict_data):
        pass
