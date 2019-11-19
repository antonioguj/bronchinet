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


def get_indexes_local_2dim(index, num_images_dirs):
    index_y = index // num_images_dirs[0]
    index_x = index % num_images_dirs[0]
    return (index_x, index_y)

def get_indexes_local_3dim(index, num_images_dirs):
    num_images_xy = num_images_dirs[0] * num_images_dirs[1]
    index_z = index // (num_images_xy)
    index_xy = index % (num_images_xy)
    index_y = index_xy // num_images_dirs[0]
    index_x = index_xy % num_images_dirs[0]
    return (index_z, index_x, index_y)



class SlidingWindowImages(BaseImageGenerator):

    def __init__(self, size_image, prop_overlap, size_full_image):
        super(SlidingWindowImages, self).__init__(size_image)

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


    def complete_init_data(self, in_array_shape):
        self.size_full_image = in_array_shape[0:self.ndims]
        self.num_images_dirs = self.get_num_images_dirs()
        self.num_images = np.prod(self.num_images_dirs)


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


    def get_image(self, in_array, in2nd_array= None, index= None, seed= None):
        if index is None:
            message = "\'index\' is missing in sliding-window image generator"
            CatchErrorException(message)

        out_array = self.get_cropped_image(in_array, index)

        if in2nd_array is None:
            return out_array
        else:
            out2nd_array = self.get_cropped_image(in2nd_array, index)
            return (out_array, out2nd_array)



class SlicingImages(SlidingWindowImages):
    def __init__(self, size_image, size_full_image):
        super(SlicingImages, self).__init__(size_image, 0.0, size_full_image)
