#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
from Preprocessing.BoundingBoxMasks import *
import numpy as np


class ReconstructImage(object):

    @staticmethod
    def get_num_batch_images(size_total, size_img):

        return int(np.floor((size_total)/size_img))

    @staticmethod
    def get_limits_batch_image(index, size_img, coord_0=0):

        coord_n    = coord_0 + index * size_img
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
                                (size_img_z, size_img_x, size_img_y)):

        num_images_x = cls.get_num_batch_images(sizetotal_x, size_img_x)
        num_images_y = cls.get_num_batch_images(sizetotal_y, size_img_y)
        num_images_z = cls.get_num_batch_images(sizetotal_z, size_img_z)
        num_images   = num_images_x * num_images_y * num_images_z

        return (num_images, num_images_x, num_images_y, num_images_z)


    @classmethod
    def compute(cls, masks_array, yPredict):

        (sizetotal_z, sizetotal_x, sizetotal_y) = masks_array.shape

        (num_batches, size_img_z, size_img_x, size_img_y) = yPredict.shape

        (num_images, num_images_x, num_images_y, num_images_z) = cls.compute_num_images((sizetotal_z, sizetotal_x, sizetotal_y),
                                                                                        (size_img_z, size_img_x, size_img_y))

        masks_predict = np.ndarray(masks_array.shape, dtype=masks_array.dtype)
        masks_predict[:, :, :] = 0

        for index in range(num_images):

            (index_x, index_y, index_z) = cls.get_indexes_3dirs(index, (num_images_x, num_images_y))

            (x_left, x_right) = cls.get_limits_batch_image(index_x, size_img_x)
            (y_down, y_up   ) = cls.get_limits_batch_image(index_y, size_img_y)
            (z_back, z_front) = cls.get_limits_batch_image(index_z, size_img_z)

            masks_predict[z_back:z_front, x_left:x_right, y_down:y_up] = np.asarray(yPredict[index], dtype=masks_array.dtype).reshape(size_img_z, size_img_x, size_img_y)
        #endfor

        return masks_predict


    @classmethod
    def compute_cropped(cls, masks_array, yPredict, boundingBox):

        (sizetotal_z, sizetotal_x, sizetotal_y) = BoundingBoxMasks.computeSizeBoundingBox(boundingBox)

        (num_batches, size_img_z, size_img_x, size_img_y) = yPredict.shape

        (num_images, num_images_x, num_images_y, num_images_z) = cls.compute_num_images((sizetotal_z, sizetotal_x, sizetotal_y),
                                                                                        (size_img_z, size_img_x, size_img_y))

        masks_predict = np.ndarray(masks_array.shape, dtype=masks_array.dtype)
        masks_predict[:, :, :] = 0

        for index in range(num_images):

            (index_x, index_y, index_z) = cls.get_indexes_3dirs(index, (num_images_x, num_images_y))

            (x_left, x_right) = cls.get_limits_batch_image(index_x, size_img_x, boundingBox[1][0])
            (y_down, y_up   ) = cls.get_limits_batch_image(index_y, size_img_y, boundingBox[2][0])
            (z_back, z_front) = cls.get_limits_batch_image(index_z, size_img_z, boundingBox[0][0])

            masks_predict[z_back:z_front, x_left:x_right, y_down:y_up] = np.asarray(yPredict[index], dtype=masks_array.dtype).reshape(size_img_z, size_img_x, size_img_y)
        #endfor

        return masks_predict