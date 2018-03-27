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

    @staticmethod
    def get_num_sliding_images(size_total, size_img, prop_overlap):

        return int(np.floor((size_total - prop_overlap*size_img)/(1 - prop_overlap)/size_img))

    @staticmethod
    def get_limits_sliding_image(index, size_img, prop_overlap):

        coord_n    = int(index * (1.0-prop_overlap) * size_img)
        coord_npl1 = coord_n + size_img
        return (coord_n, coord_npl1)

    @staticmethod
    def get_indexes_3dirs(index, (sizetotal_x, sizetotal_y)):

        sizetotal_xy = sizetotal_x * sizetotal_y
        index_z  = index // (sizetotal_xy)
        index_xy = index % (sizetotal_xy)
        index_y  = index_xy // sizetotal_x
        index_x  = index_xy % sizetotal_x
        return (index_x, index_y, index_z)


    @classmethod
    def compute_num_images(cls, (sizetotal_z, sizetotal_x, sizetotal_y),
                                (size_img_z, size_img_x, size_img_y),
                                (prop_overlap_z, prop_overlap_x, prop_overlap_y)):

        num_images_x = cls.get_num_sliding_images(sizetotal_x, size_img_x, prop_overlap_x)
        num_images_y = cls.get_num_sliding_images(sizetotal_y, size_img_y, prop_overlap_y)
        num_images_z = cls.get_num_sliding_images(sizetotal_z, size_img_z, prop_overlap_z)
        num_images   = num_images_x * num_images_y * num_images_z

        return (num_images, num_images_x, num_images_y, num_images_z)


    @classmethod
    def compute(cls, images_array,
                    (size_img_z, size_img_x, size_img_y),
                    (prop_overlap_z, prop_overlap_x, prop_overlap_y)):

        (sizetotal_z, sizetotal_x, sizetotal_y) = images_array.shape

        (num_images, num_images_x, num_images_y, num_images_z) = cls.compute_num_images((sizetotal_z, sizetotal_x, sizetotal_y),
                                                                                        (size_img_z, size_img_x, size_img_y),
                                                                                        (prop_overlap_z, prop_overlap_x, prop_overlap_y))

        out_images_array = np.ndarray([num_images, size_img_z, size_img_x, size_img_y], dtype=images_array.dtype)

        for index in range(num_images):

            (index_x, index_y, index_z) = cls.get_indexes_3dirs(index, (num_images_x, num_images_y))

            (x_left, x_right) = cls.get_limits_sliding_image(index_x, size_img_x, prop_overlap_x)
            (y_down, y_up   ) = cls.get_limits_sliding_image(index_y, size_img_y, prop_overlap_y)
            (z_back, z_front) = cls.get_limits_sliding_image(index_z, size_img_z, prop_overlap_z)

            out_images_array[index] = np.asarray(images_array[z_back:z_front, x_left:x_right, y_down:y_up], dtype=images_array.dtype)
        #endfor

        return out_images_array


    @classmethod
    def compute2(cls, images_array, masks_array,
                    (size_img_z, size_img_x, size_img_y),
                    (prop_overlap_z, prop_overlap_x, prop_overlap_y)):

        (sizetotal_z, sizetotal_x, sizetotal_y) = images_array.shape

        (num_images, num_images_x, num_images_y, num_images_z) = cls.compute_num_images((sizetotal_z, sizetotal_x, sizetotal_y),
                                                                                        (size_img_z, size_img_x, size_img_y),
                                                                                        (prop_overlap_z, prop_overlap_x, prop_overlap_y))

        out_images_array = np.ndarray([num_images, size_img_z, size_img_x, size_img_y], dtype=images_array.dtype)
        out_masks_array  = np.ndarray([num_images, size_img_z, size_img_x, size_img_y], dtype=masks_array.dtype)

        for index in range(num_images):

            (index_x, index_y, index_z) = cls.get_indexes_3dirs(index, (num_images_x, num_images_y))

            (x_left, x_right) = cls.get_limits_sliding_image(index_x, size_img_x, prop_overlap_x)
            (y_down, y_up   ) = cls.get_limits_sliding_image(index_y, size_img_y, prop_overlap_y)
            (z_back, z_front) = cls.get_limits_sliding_image(index_z, size_img_z, prop_overlap_z)

            out_images_array[index] = np.asarray(images_array[z_back:z_front, x_left:x_right, y_down:y_up], dtype=images_array.dtype)
            out_masks_array [index] = np.asarray(masks_array [z_back:z_front, x_left:x_right, y_down:y_up], dtype=masks_array.dtype)
        # endfor

        return (out_images_array, out_masks_array)