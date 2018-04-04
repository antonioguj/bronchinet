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

    def __init__(self, size_image):
        (self.size_image_z, self.size_image_x, self.size_image_y  ) = size_image

    @staticmethod
    def get_num_images_1d(size_total, size_image):

        return int(np.floor((size_total)/size_image))

    @staticmethod
    def get_limits_image_1d(index, size_image, coord_0):

        coord_n    = coord_0 + index * size_image
        coord_npl1 = coord_n + size_image
        return (coord_n, coord_npl1)

    @staticmethod
    def get_indexes_3d(index, (num_images_x, num_images_y)):

        num_images_xy = num_images_x * num_images_y
        index_z  = index // (num_images_xy)
        index_xy = index % (num_images_xy)
        index_y  = index_xy // num_images_x
        index_x  = index_xy % num_images_x
        return (index_x, index_y, index_z)


    def get_num_images_3d(self, (sizetotal_z, sizetotal_x, sizetotal_y)):

        num_images_x = self.get_num_images_1d(sizetotal_x, self.size_image_x)
        num_images_y = self.get_num_images_1d(sizetotal_y, self.size_image_y)
        num_images_z = self.get_num_images_1d(sizetotal_z, self.size_image_z)

        return (num_images_x, num_images_y, num_images_z)

    def get_num_images_total(self, (sizetotal_z, sizetotal_x, sizetotal_y)):

        (num_images_x, num_images_y, num_images_z) = self.get_num_images_3d((sizetotal_z, sizetotal_x, sizetotal_y))
        return num_images_x * num_images_y * num_images_z

    def get_limits_image_3d(self, (index_x, index_y, index_z), (coord_x0, coord_y0, coord_z0)=(0, 0, 0)):

        (x_left, x_right) = self.get_limits_image_1d(index_x, self.size_image_x, coord_x0)
        (y_down, y_up   ) = self.get_limits_image_1d(index_y, self.size_image_y, coord_y0)
        (z_back, z_front) = self.get_limits_image_1d(index_z, self.size_image_z, coord_z0)

        return (x_left, x_right, y_down, y_up, z_back, z_front)


    def compute(self, yPredict, masks_predict_shape):

        (num_images_x, num_images_y, num_images_z) = self.get_num_images_3d(yPredict.shape)
        num_images = num_images_x * num_images_y * num_images_z

        masks_predict = np.ndarray(masks_predict_shape, dtype=FORMATMASKDATA)
        masks_predict[:, :, :] = 0

        for index in range(num_images):

            (index_x, index_y, index_z) = self.get_indexes_3d(index, (num_images_x, num_images_y))

            (x_left, x_right, y_down, y_up, z_back, z_front) = self.get_limits_image_3d((index_x, index_y, index_z))

            masks_predict[z_back:z_front, x_left:x_right, y_down:y_up] = np.asarray(yPredict[index], dtype=FORMATMASKDATA)
        #endfor

        return masks_predict


    def compute_cropped(self, yPredict, masks_predict_shape, boundingBox):

        size_boundingBox    = BoundingBoxMasks.computeSizeBoundingBox   (boundingBox)
        coords0_boundingBox = BoundingBoxMasks.computeCoords0BoundingBox(boundingBox)

        (num_images_x, num_images_y, num_images_z) = self.get_num_images_3d(size_boundingBox)
        num_images = num_images_x * num_images_y * num_images_z

        masks_predict = np.ndarray(masks_predict_shape, dtype=FORMATMASKDATA)
        masks_predict[:, :, :] = 0

        for index in range(num_images):

            (index_x, index_y, index_z) = self.get_indexes_3d(index, (num_images_x, num_images_y))

            (x_left, x_right, y_down, y_up, z_back, z_front) = self.get_limits_image_3d((index_x, index_y, index_z), coords0_boundingBox)

            masks_predict[z_back:z_front, x_left:x_right, y_down:y_up] = np.asarray(yPredict[index], dtype=FORMATMASKDATA)
        #endfor

        return masks_predict