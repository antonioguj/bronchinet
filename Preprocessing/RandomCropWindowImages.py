#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Preprocessing.BaseImageGenerator import *
from Preprocessing.OperationImages import CropImages, SetPatchInImages
import numpy as np



class RandomCropWindowImages(BaseImageGenerator):

    def __init__(self, size_image, num_images, size_full_image):
        super(RandomCropWindowImages, self).__init__(size_image, num_images)

        self.ndims = len(self.size_image)
        self.size_full_image = size_full_image

        if self.ndims==2:
            self.fun_crop_images = CropImages.compute2D
        elif self.ndims==3:
            self.fun_crop_images = CropImages.compute3D
        else:
            raise Exception('Error: self.ndims')


    def complete_init_data(self, in_array_shape, seed_0=None):
        self.size_full_image = in_array_shape[0:self.ndims]
        self.compute_search_space_crop_bounding_box_origin()


    def get_random_origin_crop_window(self, seed= None):
        if seed is not None:
            np.random.seed(seed)

        random_origin_boundbox = []
        for i in range(self.ndims):
            random_index = np.random.randint(self.self.size_full_image[i] - self.size_image[i])
            random_origin_boundbox.apppend(random_index)
        #endfor
        random_origin_boundbox = tuple(random_origin_boundbox)
        return random_origin_boundbox


    def get_crop_window_image(self, seed= None):
        origin_window = self.get_random_origin_crop_window(seed=seed)
        crop_bounding_box = []
        for i in range(self.ndims):
            limit_left  = origin_window[i]
            limit_right = origin_window[i] + self.size_image[i]
            crop_bounding_box.append((limit_left, limit_right))
        #endfor
        crop_bounding_box = tuple(crop_bounding_box)
        return crop_bounding_box


    def get_cropped_image(self, in_array, seed= None):
        crop_bounding_box = self.get_crop_window_image(seed=seed)
        return self.fun_crop_images(in_array, crop_bounding_box)


    def get_image(self, in_array, in2nd_array= None, index= None, seed= None):
        out_array = self.get_cropped_image(in_array, index)

        if in2nd_array is None:
            return out_array
        else:
            out2nd_array = self.get_cropped_image(in2nd_array, index)
            return (out_array, out2nd_array)
