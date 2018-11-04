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
import numpy as np


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

    @classmethod
    def join_two_binmasks_one_image(cls, masks_array_1, masks_array_2):
        # check there is no overlap between the two masks
        #index_binmasks_1 = np.argwhere(masks_array_1 == cls.val_mask_positive)
        #index_binmasks_2 = np.argwhere(masks_array_2 == cls.val_mask_positive)

        # check there is no overlap between the two masks
        intersect_masks = np.multiply(masks_array_1, masks_array_2)
        index_posit_intersect = np.where(intersect_masks == cls.val_mask_positive)

        if len(index_posit_intersect[0] != 0):
            message = "Found intersection in between the two masks in 'join_two_binmasks_one_image'"
            CatchErrorException(message)
        else:
            return masks_array_1 + masks_array_2


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
