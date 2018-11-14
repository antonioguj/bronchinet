#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.ErrorMessages import *
from CommonUtil.FunctionsUtil import *
from scipy.misc import imresize
import numpy as np


class CropImages(object):

    @staticmethod
    def compute2D(images_array, bounding_box):

        return images_array[:, bounding_box[0][0]:bounding_box[0][1],
                               bounding_box[1][0]:bounding_box[1][1]]

    @staticmethod
    def compute3D(images_array, bounding_box):

        return images_array[bounding_box[0][0]:bounding_box[0][1],
                            bounding_box[1][0]:bounding_box[1][1],
                            bounding_box[2][0]:bounding_box[2][1]]


class IncludeImages(object):

    @staticmethod
    def compute2D(images_array, bounding_box, images_full_array):

        images_full_array[bounding_box[0][0]:bounding_box[0][1],
                          bounding_box[1][0]:bounding_box[1][1]] = images_array

    @staticmethod
    def compute3D(images_array, bounding_box, images_full_array):

        images_full_array[bounding_box[0][0]:bounding_box[0][1],
                          bounding_box[1][0]:bounding_box[1][1],
                          bounding_box[2][0]:bounding_box[2][1]] = images_array


class ExtendImages(object):

    @staticmethod
    def compute2D(images_array, bounding_box, out_array_shape, background_value=0):

        if background_value==0:
            out_images_array = np.zeros(out_array_shape, dtype=images_array.dtype)
        else:
            out_images_array = np.full(out_array_shape, background_value, dtype=images_array.dtype)

        out_images_array[bounding_box[0][0]:bounding_box[0][1],
                         bounding_box[1][0]:bounding_box[1][1]] = images_array

        return out_images_array

    @staticmethod
    def compute3D(images_array, bounding_box, out_array_shape, background_value=0):

        if background_value==0:
            out_images_array = np.zeros(out_array_shape, dtype=images_array.dtype)
        else:
            out_images_array = np.full(out_array_shape, background_value, dtype=images_array.dtype)

        out_images_array[bounding_box[0][0]:bounding_box[0][1],
                         bounding_box[1][0]:bounding_box[1][1],
                         bounding_box[2][0]:bounding_box[2][1]] = images_array

        return out_images_array


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
            if isMasks:
                return cls.fix_masks_after_resizing(images_array[:, 0:new_height, 0:new_width])
            else:
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
            if isMasks:
                return cls.fix_masks_after_resizing(new_images_array)
            else:
                return new_images_array

    @classmethod
    def compute3D(cls, images_array, (new_depthZ, new_height, new_width), isMasks=False):

        if isMasks:
            return imresize(images_array, (new_depthZ, new_height, new_width), interp=cls.type_interp_masks)
        else:
            return imresize(images_array, (new_depthZ, new_height, new_width), interp=cls.type_interp_images)

    @classmethod
    def fix_masks_after_resizing(cls, images_array):
        # for some reason, after resizing the labels in masks are corrupted
        # reassign the labels to integer values: 0 for background, and (1,2,...) for the classes
        unique_vals_masks = np.unique(images_array).tolist()

        new_images_array = np.zeros_like(images_array)

        #voxels in corners always background. Remove from list background is assigned to 0
        val_background = images_array[0,0,0]

        if val_background not in unique_vals_masks:
            message = "ResizeImages: value for background %s not found in list of labels for resized masks s..."%(val_background, unique_vals_masks)
            CatchErrorException(message)
        else:
            unique_vals_masks.remove(val_background)

        for i, value in enumerate(unique_vals_masks):
            new_images_array = np.where(images_array == value, i+1, new_images_array)

        return new_images_array


class ThresholdImages(object):

    @staticmethod
    def compute(predictions_array, threshold_value):
        return np.where(predictions_array > threshold_value, 1.0, 0.0)


class FlippingImages(object):

    @staticmethod
    def compute(images_array, axis = 0):
        if axis == 0:
            return images_array[::-1, :, :]
        elif axis == 1:
            return images_array[:, ::-1, :]
        elif axis == 2:
            return images_array[:, :, ::-1]
        else:
            return False