#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################


class BaseImageGenerator(object):

    def __init__(self, size_image):
        self.size_image = size_image

    def complete_init_data(self, size_total):
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

    def get_image_array(self, images_array, index):
        pass

    def compute_images_array_all(self, images_array):
        pass