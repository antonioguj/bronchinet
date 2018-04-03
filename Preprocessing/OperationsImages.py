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


class CropImages(object):

    @staticmethod
    def compute2D(images_array, boundingBox):

        return images_array[:,boundingBox[0][0]:boundingBox[0][1],
                              boundingBox[1][0]:boundingBox[1][1]]

    @staticmethod
    def compute3D(images_array, boundingBox):

        return images_array[boundingBox[0][0]:boundingBox[0][1],
                            boundingBox[1][0]:boundingBox[1][1],
                            boundingBox[2][0]:boundingBox[2][1]]


class StackBatchesImages(object):

    @staticmethod
    def get_batches_image(images_array, size_batch):

        if (size_batch==1):
            return np.reshape(images_array, [images_array.shape[0], 1, images_array.shape[1], images_array.shape[2]])
        else:
            num_batches = images_array.shape[0] // size_batch
            size_outarr = num_batches * size_batch
            return np.reshape(images_array[0:size_outarr], [num_batches, size_batch, images_array.shape[1], images_array.shape[2]])

    @classmethod
    def compute_stack_slices(cls, images_array):

        return cls.get_batches_image(images_array, 1)

    @classmethod
    def compute_stack_vols(cls, images_array, size_img_z):

        return cls.get_batches_image(images_array, size_img_z)