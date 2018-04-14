#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.FunctionsUtil import *
from scipy.misc import imresize
import numpy as np


class CropImages(object):

    @staticmethod
    def compute2D(images_array, boundingBox):

        return images_array[:, boundingBox[0][0]:boundingBox[0][1], boundingBox[1][0]:boundingBox[1][1]]

    @staticmethod
    def compute3D(images_array, boundingBox):

        return images_array[boundingBox[0][0]:boundingBox[0][1], boundingBox[1][0]:boundingBox[1][1], boundingBox[2][0]:boundingBox[2][1]]


class ResizeImages(object):

    type_interp_images = 'bilinear'
    type_interp_masks  = 'nearest'

    @classmethod
    def compute2D(cls, images_array, (new_height, new_width), isMasks=False):

        if (new_height < images_array.shape[1] and
            new_width < images_array.shape[2]):
            # if resized image is smaller, prevent the generation of new numpy array, not needed
            if isMasks:
                for i, slice_image in enumerate(images_array):
                    images_array[i, 0:new_height, 0:new_width] = imresize(slice_image, (new_height, new_width), interp=cls.type_interp_masks)
                #endfor
            else:
                for i, slice_image in enumerate(images_array):
                    images_array[i, 0:new_height, 0:new_width] = imresize(slice_image, (new_height, new_width), interp=cls.type_interp_images)
                #endfor
            return images_array[:, 0:new_height, 0:new_width]
        else:
            new_images_array = np.ndarray([images_array.shape[0], new_height, new_width], dtype=images_array.dtype)
            if isMasks:
                for i, slice_image in enumerate(images_array):
                    new_images_array[i] = imresize(slice_image, (new_height, new_width), interp=cls.type_interp_masks)
                #endfor
            else:
                for i, slice_image in enumerate(images_array):
                    new_images_array[i] = imresize(slice_image, (new_height, new_width), interp=cls.type_interp_images)
                #endfor
            return new_images_array

    @classmethod
    def compute3D(cls, images_array, (new_depthZ, new_height, new_width), isMasks=False):

        if isMasks:
            return imresize(images_array, (new_depthZ, new_height, new_width), interp=cls.type_interp_masks)
        else:
            return imresize(images_array, (new_depthZ, new_height, new_width), interp=cls.type_interp_images)


class ExclusionMasks(object):

    val_exclude = -1

    @classmethod
    def compute(cls, images_array, exclude_masks_array):

        if (images_array.shape  != exclude_masks_array.shape):
            message = "size of input array, %s; not equal to size of mask, %s..."%(images_array.shape, exclude_masks_array.shape)
            CatchErrorException(message)
        else:
            return np.where(exclude_masks_array == 0, cls.val_exclude, images_array)

    @classmethod
    def computeInverse(cls, images_array, exclude_masks_array):

        if (images_array.shape  != exclude_masks_array.shape):
            message = "size of input array, %s; not equal to size of mask, %s..."%(images_array.shape, exclude_masks_array.shape)
            CatchErrorException(message)
        else:
            return np.where(exclude_masks_array==cls.val_exclude, 0, images_array)
