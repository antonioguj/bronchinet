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
from Common.FunctionsImagesUtil import *



class BaseImageGenerator(object):

    def __init__(self, size_image, num_images=1):
        self.size_image = size_image
        self.num_images  = num_images


    def get_size_image(self):
        return self.size_image

    def get_num_images(self):
        return self.num_images

    def complete_init_data(self, *args):
        return NotImplemented

    def get_image(self, in_array, in2nd_array= None, index= None, seed= None):
        return NotImplemented


    def get_shape_out_array(self, in_array_shape):
        if is_image_array_without_channels(self.size_image, in_array_shape):
            return [num_images] + list(self.size_image)
        else:
            num_channels = self.get_num_channels_array(in_array_shape)
            return [num_images] + list(self.size_image) + [num_channels]


    def compute_images_all(self, in_array, in2nd_array= None, seed_0= None):
        out_shape = self.get_shape_out_array(in_array.shape)
        out_array = np.ndarray(out_shape, dtype=in_array.dtype)

        if in2nd_array is None:
            for index in range(self.num_images):
                seed = update_seed_with_index(seed_0, index)
                out_array[index] = self.get_image(in_array, index=index, seed=seed)
            # endfor

            return out_array
        else:
            out2nd_shape = self.get_shape_out_array(in2nd_array.shape)
            out2nd_array = np.ndarray(out2nd_shape, dtype=in2nd_array.dtype)

            for index in range(self.num_images):
                seed = update_seed_with_index(seed_0, index)
                (out_array[index], out2nd_array[index]) = self.get_image(in_array, in2nd_array=in2nd_array, index=index, seed=seed)
            #endfor

            return (out_array, out2nd_array)
