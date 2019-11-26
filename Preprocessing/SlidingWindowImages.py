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


def get_indexes_local_2dim(index, (num_images_x, num_images_y)):
    index_y = index // num_images_x
    index_x = index % num_images_x
    return (index_x, index_y)

def get_indexes_local_3dim(index, (num_images_z, num_images_x, num_images_y)):
    num_images_xy = num_images_x * num_images_y
    index_z = index // (num_images_xy)
    index_xy = index % (num_images_xy)
    index_y = index_xy // num_images_x
    index_x = index_xy % num_images_x
    return (index_z, index_x, index_y)



class SlidingWindowImages(BaseImageGenerator):

    def __init__(self, size_image, prop_overlap, size_full_image=0):
        super(SlidingWindowImages, self).__init__(size_image, num_images=1)

        self.ndims = len(size_image)
        if np.isscalar(prop_overlap):
            self.prop_overlap = tuple([prop_overlap]*self.ndims)
        else:
            self.prop_overlap = prop_overlap
        if np.isscalar(size_full_image):
            self.size_full_image = tuple([size_full_image]*self.ndims)
        else:
            self.size_full_image = size_full_image

        self.num_images_dirs = self.get_num_images_dirs()
        self.num_images = np.prod(self.num_images_dirs)

        if self.ndims==2:
            self.fun_get_indexes_local = get_indexes_local_2dim
            self.fun_crop_images = CropImages.compute2D
            self.fun_setpatch_images = SetPatchInImages.compute2D
            self.fun_setpatch_images_byadding = SetPatchInImages.compute2D_byadding
        elif self.ndims==3:
            self.fun_get_indexes_local = get_indexes_local_3dim
            self.fun_crop_images = CropImages.compute3D
            self.fun_setpatch_images = SetPatchInImages.compute3D
            self.fun_setpatch_images_byadding = SetPatchInImages.compute3D_byadding
        else:
            raise Exception('Error: self.ndims')

        self.initialize_gendata()


    @staticmethod
    def get_num_images_1d(size_segment, prop_overlap, size_total):
        return max(int(np.ceil((size_total - prop_overlap*size_segment) /(1-prop_overlap) /size_segment)), 0)

    @staticmethod
    def get_limits_image_1d(index, size_segment, prop_overlap, size_total):
        coord_n = int(index * (1.0-prop_overlap) * size_segment)
        coord_npl1 = coord_n + size_segment
        if coord_npl1 > size_total:
            coord_npl1 = size_total
            coord_n = size_total - size_segment
        return (coord_n, coord_npl1)


    def update_image_data(self, in_array_shape):
        self.size_full_image = in_array_shape[0:self.ndims]
        self.num_images_dirs = self.get_num_images_dirs()
        self.num_images = np.prod(self.num_images_dirs)


    def compute_gendata(self, **kwargs):
        index = kwargs['index']
        self.crop_window_bounding_box = self.get_crop_window_image(index)
        self.is_compute_gendata = False

    def initialize_gendata(self):
        self.is_compute_gendata = True
        self.crop_window_bounding_box = None


    def get_text_description(self):
        message  = 'Sliding-window generation of images:\n'
        message += 'image size: \'%s\', proportion overlap: \'%s\', volume image size: \'%s\'...\n' %(str(self.size_image),
                                                                                                      str(self.prop_overlap),
                                                                                                      str(self.size_full_image))
        message += 'num images total: \'%s\', and num images in each direction: \'%s\'...\n' %(self.num_images,
                                                                                               str(self.num_images_dirs))
        limits_window_image = self.get_limits_sliding_window_image()
        for i in range(self.ndims):
            message += 'limits images in dir \'%s\': \'%s\'...\n' %(i, str(limits_window_image[i]))
        #endfor
        return message


    def get_num_images_dirs(self):
        num_images_dirs = []
        for i in range(self.ndims):
            num_images_1d = self.get_num_images_1d(self.size_image[i], self.prop_overlap[i], self.size_full_image[i])
            num_images_dirs.append(num_images_1d)
        #endfor
        return tuple(num_images_dirs)


    def get_crop_window_image(self, index):
        indexes_local = self.fun_get_indexes_local(index, self.num_images_dirs)
        crop_bounding_box = []
        for i in range(self.ndims):
            (limit_left, limit_right) = self.get_limits_image_1d(indexes_local[i], self.size_image[i],
                                                                 self.prop_overlap[i], self.size_full_image[i])
            crop_bounding_box.append((limit_left, limit_right))
        #endfor
        crop_bounding_box = tuple(crop_bounding_box)
        return crop_bounding_box


    def get_limits_sliding_window_image(self):
        limits_window_image = []
        for i in range(self.ndims):
            limits_image_1dir = [self.get_limits_image_1d(j, self.size_image[i], self.prop_overlap[i], self.size_full_image[i])
                                 for j in range(self.num_images_dirs[i])]
            limits_window_image.append(limits_image_1dir)
        #endfor
        return limits_window_image


    def get_cropped_image(self, in_array, index):
        crop_bounding_box = self.get_crop_window_image(index)
        return self.fun_crop_images(in_array, crop_bounding_box)


    def set_assign_image_patch(self, in_array, in_full_array, index):
        crop_bounding_box = self.get_crop_window_image(index)
        self.fun_setpatch_images(in_array, in_full_array, crop_bounding_box)


    def set_add_image_patch(self, in_array, in_full_array, index):
        crop_bounding_box = self.get_crop_window_image(index)
        self.fun_setpatch_images_byadding(in_array, in_full_array, crop_bounding_box)


    def get_image(self, in_array):
        return self.fun_crop_images(in_array, self.crop_window_bounding_box)



class SlicingImages(SlidingWindowImages):
    def __init__(self, size_image, size_full_image):
        super(SlicingImages, self).__init__(size_image, 0.0, size_full_image)
