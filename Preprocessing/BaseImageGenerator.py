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



class BaseImageGenerator(object):

    def __init__(self, size_image):
        self.size_image = size_image

    def complete_init_data(self, in_array_shape):
        pass

    def get_num_images(self):
        pass

    def is_images_array_without_channels(self, in_array_shape):
        return len(in_array_shape) == len(self.size_image)

    def get_num_channels_array(self, in_array_shape):
        if self.is_images_array_without_channels(in_array_shape):
            return None
        else:
            return in_array_shape[-1]

    def get_shape_out_array(self, in_array_shape):
        pass

    def get_images_array(self, images_array, index, masks_array=None, seed=None):
        pass

    def compute_images_array_all(self, images_array, masks_array=None, seed_0=None):
        pass
