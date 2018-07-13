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
    def compute(predictions_array, threshold_value = 0.5):
        return np.where(predictions_array > threshold_value, 1, 0)


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


class OperationsMasks(object):

    val_mask_background     = 0
    val_mask_exclude_voxels = -1

    @classmethod
    def apply_mask_exclude_voxels(cls, images_array, masks_exclude_voxels_array):

        if (images_array.shape  != masks_exclude_voxels_array.shape):
            message = "size of input array, %s; not equal to size of mask, %s..."%(images_array.shape, masks_exclude_voxels_array.shape)
            CatchErrorException(message)
        else:
            return np.where(masks_exclude_voxels_array == cls.val_mask_background, cls.val_mask_exclude_voxels, images_array)

    @classmethod
    def apply_mask_exclude_voxels_fillzero(cls, images_array, masks_exclude_voxels_array):

        if (images_array.shape  != masks_exclude_voxels_array.shape):
            message = "size of input array, %s; not equal to size of mask, %s..."%(images_array.shape, masks_exclude_voxels_array.shape)
            CatchErrorException(message)
        else:
            return np.where(masks_exclude_voxels_array == cls.val_mask_background, cls.val_mask_background, images_array)

    @classmethod
    def reverse_mask_exclude_voxels(cls, images_array, original_images_array, masks_exclude_voxels_array):

        if (images_array.shape  != original_images_array.shape) and \
            (images_array.shape != masks_exclude_voxels_array.shape):
            message = "size of input array, %s; not equal to size of original array %s, or size of mask, %s..."%(images_array.shape,
                                                                                                                 original_images_array.shape,
                                                                                                                 masks_exclude_voxels_array.shape)
            CatchErrorException(message)
        else:
            return np.where(masks_exclude_voxels_array == cls.val_mask_background, original_images_array, images_array)

    @classmethod
    def reverse_mask_exclude_voxels_fillzero(cls, images_array, masks_exclude_voxels_array):

        if (images_array.shape  != masks_exclude_voxels_array.shape):
            message = "size of input array, %s; not equal to size of mask, %s..."%(images_array.shape, masks_exclude_voxels_array.shape)
            CatchErrorException(message)
        else:
            return np.where(masks_exclude_voxels_array == cls.val_mask_background, cls.val_mask_background, images_array)


class OperationsBinaryMasks(OperationsMasks):

    val_mask_positive = 1

    @classmethod
    def process_masks(cls, masks_array):
        # convert to binary masks (0, 1)
        return np.where(masks_array != cls.val_mask_background, cls.val_mask_positive, cls.val_mask_background)

    @classmethod
    def check_masks(cls, masks_array):
        #check that 'masks_array' only contains binary vals (0, 1)
        values_found = np.unique(masks_array)
        if len(values_found) == 2 and \
            values_found[0] == cls.val_mask_background and \
            values_found[1] == cls.val_mask_positive:
            return True
        else:
            return False

    @classmethod
    def process_masks_with_exclusion(cls, masks_array):
        # convert to binary masks (0, 1), but keep exclusion mask "-1"
        return np.where(np.logical_or(masks_array != cls.val_mask_background,
                                      masks_array != cls.val_mask_exclude_voxels),
                        cls.val_mask_positive, cls.val_mask_background)

    @classmethod
    def check_masks_with_exclusion(cls, masks_array):
        # check that 'masks_array' only contains binary vals (0, 1), and exclusion mask "-1"
        values_found = np.unique(masks_array)
        if len(values_found) == 3 and \
            values_found[0] == cls.val_mask_exclude_voxels and \
            values_found[1] == cls.val_mask_background and \
            values_found[2] == cls.val_mask_positive:
            return True
        else:
            return False


class OperationsMultiClassMasks(OperationsMasks):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def process_masks(self, masks_array):
        # process masks to contain labels (1, ..., num_classes)
        return np.where(masks_array <= self.num_classes, masks_array, self.val_mask_background)

    def check_masks(self, masks_array):
        # check that 'masks_array' only contains labels (1, ..., num_classes)
        values_found = np.unique(masks_array)
        if len(values_found) == (self.num_classes + 1) and \
            values_found[0] == self.val_mask_background and \
            values_found[self.num_classes] == self.num_classes:
            return True
        else:
            return False

    def process_masks_with_exclusion(self, masks_array):
        # process masks to contain labels (1, ..., num_classes), but keep exclusion mask "-1"
        return np.where(np.logical_or(masks_array <= self.num_classes,
                                      masks_array != self.val_mask_exclude_voxels),
                        masks_array, self.val_mask_background)

    def check_masks_with_exclusion(self, masks_array):
        # check that 'masks_array' only contains labels (1, ..., num_classes), and exclusion mask "-1"
        values_found = np.unique(masks_array)
        if len(values_found) == (self.num_classes + 2) and \
            values_found[0] == self.val_mask_exclude_voxels and \
            values_found[1] == self.val_mask_background and \
            values_found[self.num_classes+1] == self.num_classes:
            return True
        else:
            return False