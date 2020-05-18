#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.ErrorMessages import *
from Common.FunctionsUtil import *
import numpy as np



class OperationMasks(object):
    val_mask_bin   = 1
    val_background = 0

    @classmethod
    def binarise(cls, in_array):
        # convert to binary masks (0, 1)
        # return np.where(in_array != cls.val_background, cls.val_mask_bin, cls.val_background)
        return np.clip(in_array, cls.val_background, cls.val_mask_bin)

    @classmethod
    def check_binary_mask(cls, in_array):
        # check that 'in_array' only contains binary values (0, 1)
        values_found = np.unique(in_array)
        if len(values_found) == 2 and \
            values_found[0] == cls.val_background and \
            values_found[1] == cls.val_mask_bin:
            return True
        else:
            return False

    @classmethod
    def mask_exclude_regions(cls, in_array, in_masks_array, val_mask_exclude=-1):
        return np.where(in_masks_array == cls.val_background, val_mask_exclude, in_array)

    @classmethod
    def mask_exclude_regions_fillzero(cls, in_array, in_masks_array):
        #return np.where(in_masks_array == cls.val_background, cls.val_background, in_array)
        return cls.multiply_two_masks(in_array, in_masks_array)


    @classmethod
    def merge_two_masks(cls, in_array_1, in_array_2, isNot_intersect_masks=False):
        if isNot_intersect_masks:
            # check there is no overlap between the two masks
            intersect_masks = np.multiply(in_array_1, in_array_2)
            index_intersect_masks = np.where(intersect_masks == cls.val_mask_bin)
            if len(index_intersect_masks[0] != 0):
                message = 'OperationMasks:merge_two_masks: Found intersection between the two input masks'
                CatchErrorException(message)

        out_array = in_array_1 + in_array_2
        return cls.binarise(out_array)

    @classmethod
    def substract_two_masks(cls, in_array_1, in_array_2):
        out_array = (in_array_1 - in_array_2).astype(np.int8)
        return cls.binarise(out_array).astype(in_array_1.dtype)

    @classmethod
    def multiply_two_masks(cls, in_array_1, in_array_2):
        out_array = np.multiply(in_array_1, in_array_2)
        return cls.binarise(out_array)


    @classmethod
    def get_masks_with_label(cls, in_array, in_label):
        return np.where(np.isin(in_array, in_label), cls.val_mask_bin, cls.val_background).astype(in_array.dtype)

    @classmethod
    def get_masks_with_labels_list(cls, in_array, inlist_labels):
        return np.where(np.isin(in_array, inlist_labels), cls.val_mask_bin, cls.val_background).astype(in_array.dtype)

    @classmethod
    def get_list_masks_with_labels_list(cls, in_array, inlist_labels):
        return [cls.get_masks_with_label(in_array, ilabel) for ilabel in inlist_labels]

    @classmethod
    def get_list_masks_all_labels(cls, in_array):
        inlist_labels = cls.extract_labels_in_masks(in_array)
        return cls.get_list_masks_with_labels_list(in_array, inlist_labels)

    @classmethod
    def extract_labels_in_masks(cls, in_array):
        return np.delete(np.unique(in_array), cls.val_background)