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


    def get_image(self, in_array, **kwargs):
        return NotImplemented

    def get_image_withcheck(self, in_array, **kwargs):
        if self.is_compute_gendata:
            self.compute_gendata(**kwargs)
        return self.get_image(in_array, **kwargs)


    def get_images(self, in_array, in2nd_array, **kwargs):
        self.compute_gendata(**kwargs)

        out_array = self.get_image(in_array, **kwargs)
        out2nd_array = self.get_image(in2nd_array, **kwargs)

        self.initialize_gendata()

        return (out_array, out2nd_array)


    def get_images_prototype(self, in_array, list_inadd_array, **kwargs):
        self.compute_gendata(**kwargs)

        out_array = self.get_image(in_array, **kwargs)

        list_outadd_array = []
        for inadd_array in list_inadd_array:
            outadd_array = self.get_image(inadd_array, **kwargs)
            list_outadd_array.append(outadd_array)
        #endfor

        self.initialize_gendata()

        return (out_array, list_outadd_array)


    def get_shape_out_array(self, in_array_shape):
        if is_image_array_without_channels(self.size_image, in_array_shape):
            return [num_images] + list(self.size_image)
        else:
            num_channels = self.get_num_channels_array(in_array_shape)
            return [num_images] + list(self.size_image) + [num_channels]


    def compute_images_all(self, in_array, list_inadd_array= None, **kwargs):
        seed_0 = kwargs['seed_0']

        out_shape = self.get_shape_out_array(in_array.shape)
        out_array = np.ndarray(out_shape, dtype=in_array.dtype)

        if list_inadd_array is None:
            for index in range(self.num_images):
                seed = update_seed_with_index(seed_0, index)
                add_kwargs = {'index': index, 'seed': seed}

                self.compute_gendata(**add_kwargs)
                out_array[index] = self.get_image(in_array, **add_kwargs)
                self.initialize_gendata()
            # endfor

            return out_array

        else:
            num_inadd_array = len(list_inadd_array)

            list_outadd_array = []
            for i in range(num_inadd_array):
                outadd_shape = self.get_shape_out_array(list_inadd_array[i].shape)
                outadd_array = np.ndarray(outadd_shape, dtype=list_inadd_array[i].dtype)
                list_outadd_array.append(outadd_array)
            #endfor

            for index in range(self.num_images):
                seed = update_seed_with_index(seed_0, index)
                add_kwargs = {'index': index, 'seed': seed}

                self.compute_gendata(**add_kwargs)
                out_array[index] = self.get_image(in_array, **add_kwargs)

                for i in range(num_inadd_array):
                    list_outadd_array[i][index] = self.get_image(list_inadd_array[i], **add_kwargs)
                #endfor
                self.initialize_gendata()
            #endfor

            return (out_array, list_outadd_array)