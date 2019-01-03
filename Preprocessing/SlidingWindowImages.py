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
import numpy as np


class SlidingWindowImages(BaseImageGenerator):

    def __init__(self, size_image, num_images=0):
        self.size_image = size_image
        self.num_images = num_images

        super(SlidingWindowImages, self).__init__(size_image)

    @staticmethod
    def get_num_images_1d(size_image, size_total, prop_overlap):

        return max(int(np.ceil((size_total - prop_overlap*size_image) /(1-prop_overlap) /size_image)), 0)

    @staticmethod
    def get_limits_image_1d(index, size_image, size_total, prop_overlap):

        coord_n    = int(index * (1.0-prop_overlap) * size_image)
        coord_npl1 = coord_n + size_image
        if coord_npl1 > size_total:
            coord_npl1 = size_total
            coord_n    = size_total - size_image
        return (coord_n, coord_npl1)

    def complete_init_data(self, in_array_shape):
        pass

    def get_num_images(self):
        return self.num_images

    def get_shape_out_array(self, in_array_shape):
        num_images   = self.get_num_images()
        num_channels = self.get_num_channels_array(in_array_shape)
        if num_channels:
            return [num_images] + list(self.size_image) + [num_channels]
        else:
            return [num_images] + list(self.size_image)


    def get_cropped_image(self, images_array, index):
        pass

    def get_images_array(self, images_array, index, masks_array=None, seed=None):

        out_images_array = self.get_cropped_image(images_array, index)

        if masks_array is None:
            return out_images_array
        else:
            out_masks_array = self.get_cropped_image(masks_array, index)

            return (out_images_array, out_masks_array)

    def compute_images_array_all(self, images_array, masks_array=None, seed_0=None):

        out_array_shape  = self.get_shape_out_array(images_array.shape)
        out_images_array = np.ndarray(out_array_shape, dtype=images_array.dtype)

        if masks_array is None:
            num_images = self.num_images
            for index in range(num_images):
                out_images_array[index] = self.get_images_array(images_array, index)
            #endfor

            return out_images_array
        else:
            out_array_shape = self.get_shape_out_array(masks_array.shape)
            out_masks_array = np.ndarray(out_array_shape, dtype=masks_array.dtype)

            num_images = self.num_images
            for index in range(num_images):
                (out_images_array[index], out_masks_array[index]) = self.get_images_array(images_array, index, masks_array=masks_array)
            #endfor

            return (out_images_array, out_masks_array)


class SlidingWindowImages2D(SlidingWindowImages):

    def __init__(self, size_image, prop_overlap, size_total=(0, 0)):

        (self.size_image_x,   self.size_image_y  ) = size_image
        (self.prop_overlap_x, self.prop_overlap_y) = prop_overlap
        (self.size_total_x,   self.size_total_y  ) = size_total
        (self.num_images_x,   self.num_images_y  ) = self.get_num_images_local()
        num_images_total = self.num_images_x * self.num_images_y

        super(SlidingWindowImages2D, self).__init__(size_image, num_images_total)

    def complete_init_data(self, in_array_shape):

        size_total = in_array_shape[0:2]
        (self.size_total_x, self.size_total_y) = size_total
        (self.num_images_x, self.num_images_y) = self.get_num_images_local()
        self.num_images = self.num_images_x * self.num_images_y

    def get_indexes_local(self, index):

        index_y = index // self.num_images_x
        index_x = index % self.num_images_x
        return (index_x, index_y)

    def get_num_images_local(self):

        num_images_x = self.get_num_images_1d(self.size_image_x, self.size_total_x, self.prop_overlap_x)
        num_images_y = self.get_num_images_1d(self.size_image_y, self.size_total_y, self.prop_overlap_y)

        return (num_images_x, num_images_y)

    def get_limits_image(self, index):

        (index_x, index_y) = self.get_indexes_local(index)

        (x_left, x_right) = self.get_limits_image_1d(index_x, self.size_image_x, self.size_total_x, self.prop_overlap_x)
        (y_down, y_up   ) = self.get_limits_image_1d(index_y, self.size_image_y, self.size_total_y, self.prop_overlap_y)

        return (x_left, x_right, y_down, y_up)

    def get_cropped_image(self, images_array, index):

        (x_left, x_right, y_down, y_up) = self.get_limits_image(index)

        return images_array[x_left:x_right, y_down:y_up, ...]


class SlidingWindowImages3D(SlidingWindowImages):

    def __init__(self, size_image, prop_overlap, size_total=(0, 0, 0)):

        (self.size_image_z,   self.size_image_x,   self.size_image_y  ) = size_image
        (self.prop_overlap_z, self.prop_overlap_x, self.prop_overlap_y) = prop_overlap
        (self.size_total_z,   self.size_total_x,   self.size_total_y  ) = size_total
        (self.num_images_z,   self.num_images_x,   self.num_images_y  ) = self.get_num_images_local()
        num_images_total = self.num_images_x * self.num_images_y * self.num_images_z

        super(SlidingWindowImages3D, self).__init__(size_image, num_images_total)

    def complete_init_data(self, in_array_shape):

        size_total = in_array_shape[0:3]
        (self.size_total_z, self.size_total_x, self.size_total_y) = size_total
        (self.num_images_z, self.num_images_x, self.num_images_y) = self.get_num_images_local()
        self.num_images = self.num_images_x * self.num_images_y * self.num_images_z

    def get_indexes_local(self, index):

        num_images_xy = self.num_images_x * self.num_images_y
        index_z  = index // (num_images_xy)
        index_xy = index % (num_images_xy)
        index_y  = index_xy // self.num_images_x
        index_x  = index_xy % self.num_images_x
        return (index_z, index_x, index_y)

    def get_num_images_local(self):

        num_images_x = self.get_num_images_1d(self.size_image_x, self.size_total_x, self.prop_overlap_x)
        num_images_y = self.get_num_images_1d(self.size_image_y, self.size_total_y, self.prop_overlap_y)
        num_images_z = self.get_num_images_1d(self.size_image_z, self.size_total_z, self.prop_overlap_z)

        return (num_images_z, num_images_x, num_images_y)

    def get_limits_image(self, index):

        (index_z, index_x, index_y) = self.get_indexes_local(index)

        (x_left, x_right) = self.get_limits_image_1d(index_x, self.size_image_x, self.size_total_x, self.prop_overlap_x)
        (y_down, y_up   ) = self.get_limits_image_1d(index_y, self.size_image_y, self.size_total_y, self.prop_overlap_y)
        (z_back, z_front) = self.get_limits_image_1d(index_z, self.size_image_z, self.size_total_z, self.prop_overlap_z)

        return (z_back, z_front, x_left, x_right, y_down, y_up)

    def get_cropped_image(self, images_array, index):

        (z_back, z_front, x_left, x_right, y_down, y_up) = self.get_limits_image(index)

        return images_array[z_back:z_front, x_left:x_right, y_down:y_up, ...]


class SlicingImages2D(SlidingWindowImages2D):

    def __init__(self, size_image, size_total=(0, 0)):
        super(SlicingImages2D, self).__init__(size_image, (0.0, 0.0), size_total)

class SlicingImages3D(SlidingWindowImages3D):

    def __init__(self, size_image, size_total=(0, 0, 0)):
        super(SlicingImages3D, self).__init__(size_image, (0.0, 0.0, 0.0), size_total)
