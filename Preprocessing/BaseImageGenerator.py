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

    def __init__(self, size_image, num_images):
        self.size_image = size_image
        self.num_images  = num_images
        self.is_compute_gendata = True


    def get_size_image(self):
        return self.size_image

    def get_num_images(self):
        return self.num_images

    #def update_image_data(self, **kwargs):
    def update_image_data(self, in_array_shape):
        return NotImplemented

    def compute_gendata(self, **kwargs):
        return NotImplemented

    def initialize_gendata(self):
        return NotImplemented

    def get_text_description(self):
        return NotImplemented


    def get_image(self, in_array):
        return NotImplemented


    def get_image_1array(self, in_array, **kwargs):
        self.compute_gendata(**kwargs)
        out_array = self.get_image(in_array)
        self.initialize_gendata()
        return out_array


    def get_image_2arrays(self, in_array, in2nd_array, **kwargs):
        self.compute_gendata(**kwargs)
        out_array = self.get_image(in_array)
        out2nd_array = self.get_image(in2nd_array)
        self.initialize_gendata()

        return (out_array, out2nd_array)


    def get_image_multarrays(self, list_in_arrays, **kwargs):
        self.compute_gendata(**kwargs)

        list_out_arrays = []
        for in_array in list_in_arrays:
            out_array = self.get_image(in_array)
            list_out_arrays.append(out_array)
        #endfor

        self.initialize_gendata()
        return list_out_arrays


    def get_shape_output_array(self, in_array_shape):
        if is_image_array_without_channels(self.size_image, in_array_shape):
            return [self.num_images] + list(self.size_image)
        else:
            num_channels = get_num_channels_array(self.size_image, in_array_shape)
            return [self.num_images] + list(self.size_image) + [num_channels]


    def compute_images_all(self, list_in_arrays, **kwargs):
        seed_0 = kwargs['seed_0']

        list_out_arrays = []
        for in_array in list_in_arrays:
            out_shape = self.get_shape_output_array(in_array.shape)
            out_array = np.ndarray(out_shape, dtype=in_array.dtype)
            list_out_arrays.append(out_array)
        #endfor

        for index in range(self.num_images):
            seed = update_seed_with_index(seed_0, index)
            add_kwargs = {'index': index, 'seed': seed}

            self.compute_gendata(**add_kwargs)

            for i, in_array in list_in_arrays:
                list_out_arrays[i][index] = self.get_image(in_array)
            #endfor

            self.initialize_gendata()
        #endfor

        return list_out_arrays