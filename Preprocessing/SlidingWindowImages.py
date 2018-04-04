#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

import numpy as np


class SlidingWindowImages(object):

    def __init__(self, size_image, prop_overlap):
        (self.size_image_z,   self.size_image_x,   self.size_image_y  ) = size_image
        (self.prop_overlap_z, self.prop_overlap_x, self.prop_overlap_y) = prop_overlap

    @staticmethod
    def get_num_images_1d(size_total, size_image, prop_overlap):

        return int(np.floor((size_total - prop_overlap*size_image) /(1-prop_overlap) /size_image))

    @staticmethod
    def get_limits_image_1d(index, size_image, prop_overlap):

        coord_n    = int(index * (1.0-prop_overlap) * size_image)
        coord_npl1 = coord_n + size_image
        return (coord_n, coord_npl1)

    @staticmethod
    def get_indexes_2d(index, num_images_x):

        index_y  = index // num_images_x
        index_x  = index % num_images_x
        return (index_x, index_y)

    @staticmethod
    def get_indexes_3d(index, (num_images_x, num_images_y)):

        num_images_xy = num_images_x * num_images_y
        index_z  = index // (num_images_xy)
        index_xy = index % (num_images_xy)
        index_y  = index_xy // num_images_x
        index_x  = index_xy % num_images_x
        return (index_x, index_y, index_z)


    def get_num_images_3d(self, (sizetotal_z, sizetotal_x, sizetotal_y)):

        num_images_x = self.get_num_images_1d(sizetotal_x, self.size_image_x, self.prop_overlap_x)
        num_images_y = self.get_num_images_1d(sizetotal_y, self.size_image_y, self.prop_overlap_y)
        num_images_z = self.get_num_images_1d(sizetotal_z, self.size_image_z, self.prop_overlap_z)

        return (num_images_x, num_images_y, num_images_z)

    def get_num_images_total(self, (sizetotal_z, sizetotal_x, sizetotal_y)):

        (num_images_x, num_images_y, num_images_z) = self.get_num_images_3d((sizetotal_z, sizetotal_x, sizetotal_y))
        return num_images_x * num_images_y * num_images_z

    def get_limits_image_3d(self, (index_x, index_y, index_z)):

        (x_left, x_right) = self.get_limits_image_1d(index_x, self.size_image_x, self.prop_overlap_x)
        (y_down, y_up   ) = self.get_limits_image_1d(index_y, self.size_image_y, self.prop_overlap_y)
        (z_back, z_front) = self.get_limits_image_1d(index_z, self.size_image_z, self.prop_overlap_z)

        return (x_left, x_right, y_down, y_up, z_back, z_front)


    def compute_1array(self, images_array):

        (num_images_x, num_images_y, num_images_z) = self.get_num_images_3d(images_array.shape)
        num_images = num_images_x * num_images_y * num_images_z

        out_images_array = np.ndarray([num_images, self.size_image_z, self.size_image_x, self.size_image_y], dtype=images_array.dtype)

        for i, index in enumerate(range(num_images)):

            (index_x, index_y, index_z) = self.get_indexes_3d(index, (num_images_x, num_images_y))

            (x_left, x_right, y_down, y_up, z_back, z_front) = self.get_limits_image_3d((index_x, index_y, index_z))

            out_images_array[i] = np.asarray(images_array[z_back:z_front, x_left:x_right, y_down:y_up], dtype=images_array.dtype)
        #endfor

        return out_images_array


    def compute_2array(self, images_array, masks_array):

        (num_images_x, num_images_y, num_images_z) = self.get_num_images_3d(images_array.shape)
        num_images = num_images_x * num_images_y * num_images_z

        out_images_array = np.ndarray([num_images, self.size_image_z, self.size_image_x, self.size_image_y], dtype=images_array.dtype)
        out_masks_array  = np.ndarray([num_images, self.size_image_z, self.size_image_x, self.size_image_y], dtype=masks_array.dtype)

        for i, index in enumerate(range(num_images)):

            (index_x, index_y, index_z) = self.get_indexes_3d(index, (num_images_x, num_images_y))

            (x_left, x_right, y_down, y_up, z_back, z_front) = self.get_limits_image_3d((index_x, index_y, index_z))

            out_images_array[i] = np.asarray(images_array[z_back:z_front, x_left:x_right, y_down:y_up], dtype=images_array.dtype)
            out_masks_array [i] = np.asarray(masks_array [z_back:z_front, x_left:x_right, y_down:y_up], dtype=masks_array.dtype)
        #endfor

        return (out_images_array, out_masks_array)


class SlicingImages(SlidingWindowImages):

    def __init__(self, size_image):
        super(SlicingImages, self).__init__(size_image, (0.0, 0.0, 0.0))