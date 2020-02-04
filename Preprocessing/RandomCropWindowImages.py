#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from OperationImages.OperationImages import CropImages, SetPatchInImages
from Preprocessing.BaseImageGenerator import *
import numpy as np
np.random.seed(2017)



class RandomCropWindowImages(BaseImageGenerator):

    def __init__(self, size_image, num_images, size_full_image=0):
        super(RandomCropWindowImages, self).__init__(size_image, num_images)

        self.ndims = len(self.size_image)
        if np.isscalar(size_full_image):
            self.size_full_image = tuple([size_full_image]*self.ndims)
        else:
            self.size_full_image = size_full_image

        if self.ndims==2:
            self.fun_crop_images = CropImages.compute2D
        elif self.ndims==3:
            self.fun_crop_images = CropImages.compute3D
        else:
            raise Exception('Error: self.ndims')


    def update_image_data(self, in_array_shape, seed_0=None):
        self.size_full_image = in_array_shape[0:self.ndims]


    def compute_gendata(self, **kwargs):
        seed = kwargs['seed']
        self.crop_window_bounding_box = self.get_crop_window_image(seed)
        self.is_compute_gendata = False

    def initialize_gendata(self):
        self.is_compute_gendata = True
        self.crop_window_bounding_box = None


    def get_text_description(self):
        message  = 'Random cropping window generation of images:\n'
        message += 'Image size: \'%s\', volume image size: \'%s\'. Num random patches per volume: \'%s\'...\n' %(str(self.size_image),
                                                                                                                 str(self.size_full_image),
                                                                                                                 self.num_images)
        return message


    def get_random_origin_crop_window(self, seed= None):
        if seed is not None:
            np.random.seed(seed)

        origin_crop_window = []
        for i in range(self.ndims):
            searching_space_1d = self.size_full_image[i] - self.size_image[i]
            origin_1d = np.random.randint(searching_space_1d + 1)
            origin_crop_window.append(origin_1d)
        #endfor
        origin_crop_window = tuple(origin_crop_window)
        return origin_crop_window


    def get_crop_window_image(self, seed= None):
        origin_crop_window = self.get_random_origin_crop_window(seed=seed)
        crop_bounding_box = []
        for i in range(self.ndims):
            limit_left  = origin_crop_window[i]
            limit_right = origin_crop_window[i] + self.size_image[i]
            crop_bounding_box.append((limit_left, limit_right))
        #endfor
        crop_bounding_box = tuple(crop_bounding_box)
        return crop_bounding_box


    def get_cropped_image(self, in_array, seed= None):
        crop_bounding_box = self.get_crop_window_image(seed=seed)
        return self.fun_crop_images(in_array, crop_bounding_box)


    def get_image(self, in_array):
        return self.fun_crop_images(in_array, self.crop_window_bounding_box)
