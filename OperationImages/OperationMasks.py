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
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion, binary_dilation, binary_opening, binary_closing
from skimage.morphology import skeletonize_3d
from skimage.measure import label
import numpy as np



class OperationMasks(object):
    val_mask_positive = 1
    val_mask_background = 0
    val_mask_exclude_voxels = -1

    @staticmethod
    def is_images_array_without_channels(images_array_shape,
                                         masks_exclude_voxels_array_shape):
        return len(images_array_shape) == len(masks_exclude_voxels_array_shape)

    @classmethod
    def check_correct_shape_input_array(cls, images_array_shape,
                                        masks_exclude_voxels_array_shape):
        if cls.is_images_array_without_channels(images_array_shape, masks_exclude_voxels_array_shape):
            if (images_array_shape == masks_exclude_voxels_array_shape):
                return True
        else:
            if (images_array_shape[0:-1] == masks_exclude_voxels_array_shape):
                return True
        message = "size of input array, %s; not equal to size of mask, %s..." %(images_array_shape, masks_exclude_voxels_array_shape)
        CatchErrorException(message)


    @classmethod
    def template_apply_mask_to_image_with_channels_2args(cls, function_apply_mask_without_channels,
                                                         images_array,
                                                         masks_exclude_voxels_array):
        num_channels = images_array.shape[-1]
        for ichan in range(num_channels):
            images_array[..., ichan] = function_apply_mask_without_channels(images_array[..., ichan], masks_exclude_voxels_array)
        #endfor
        #return images_array

    @classmethod
    def template_apply_mask_to_image_with_channels_3args(cls, function_apply_mask_without_channels,
                                                         images_array,
                                                         original_images_array,
                                                         masks_exclude_voxels_array):
        num_channels = images_array.shape[-1]
        for ichan in range(num_channels):
            images_array[..., ichan] = function_apply_mask_without_channels(images_array[..., ichan], original_images_array, masks_exclude_voxels_array)
        # endfor
        #return images_array


    @classmethod
    def function_apply_mask_exclude_voxels(cls, images_array,
                                           masks_exclude_voxels_array):
        return np.where(masks_exclude_voxels_array == cls.val_mask_background, cls.val_mask_exclude_voxels, images_array)

    @classmethod
    def function_apply_mask_exclude_voxels_fillzero(cls, images_array,
                                                    masks_exclude_voxels_array):
        return np.where(masks_exclude_voxels_array == cls.val_mask_background, cls.val_mask_background, images_array)

    @classmethod
    def function_reverse_mask_exclude_voxels(cls, images_array,
                                             original_images_array,
                                             masks_exclude_voxels_array):
        return np.where(masks_exclude_voxels_array == cls.val_mask_background, original_images_array, images_array)

    @classmethod
    def function_reverse_mask_exclude_voxels_fillzero(cls, images_array,
                                                      masks_exclude_voxels_array):
        return np.where(masks_exclude_voxels_array == cls.val_mask_background, cls.val_mask_background, images_array)


    @classmethod
    def apply_mask_exclude_voxels(cls, images_array,
                                  masks_exclude_voxels_array):
        if cls.check_correct_shape_input_array(images_array.shape,
                                               masks_exclude_voxels_array.shape):
            if cls.is_images_array_without_channels(images_array.shape,
                                                    masks_exclude_voxels_array.shape):
                return cls.function_apply_mask_exclude_voxels(images_array,
                                                              masks_exclude_voxels_array)
            else:
                cls.template_apply_mask_to_image_with_channels_2args(cls.function_apply_mask_exclude_voxels,
                                                                     images_array,
                                                                     masks_exclude_voxels_array)
                return images_array

    @classmethod
    def apply_mask_exclude_voxels_fillzero(cls, images_array,
                                           masks_exclude_voxels_array):
        if cls.check_correct_shape_input_array(images_array.shape,
                                               masks_exclude_voxels_array.shape):
            if cls.is_images_array_without_channels(images_array.shape,
                                                    masks_exclude_voxels_array.shape):
                return cls.function_apply_mask_exclude_voxels_fillzero(images_array,
                                                                       masks_exclude_voxels_array)
            else:
                cls.template_apply_mask_to_image_with_channels_2args(cls.function_apply_mask_exclude_voxels_fillzero,
                                                                     images_array,
                                                                     masks_exclude_voxels_array)
                return images_array

    @classmethod
    def reverse_mask_exclude_voxels(cls, images_array,
                                    masks_exclude_voxels_array,
                                    original_images_array):
        if cls.check_correct_shape_input_array(images_array.shape,
                                               masks_exclude_voxels_array.shape):
            if cls.is_images_array_without_channels(images_array.shape,
                                                    masks_exclude_voxels_array.shape):
                return cls.function_reverse_mask_exclude_voxels(images_array,
                                                                masks_exclude_voxels_array,
                                                                original_images_array)
            else:
                cls.template_apply_mask_to_image_with_channels_3args(cls.function_reverse_mask_exclude_voxels,
                                                                     images_array,
                                                                     masks_exclude_voxels_array,
                                                                     original_images_array)
                return images_array

    @classmethod
    def reverse_mask_exclude_voxels_fillzero(cls, images_array,
                                             masks_exclude_voxels_array):
        if cls.check_correct_shape_input_array(images_array.shape,
                                               masks_exclude_voxels_array.shape):
            if cls.is_images_array_without_channels(images_array.shape,
                                                    masks_exclude_voxels_array.shape):
                return cls.function_reverse_mask_exclude_voxels_fillzero(images_array,
                                                                         masks_exclude_voxels_array)
            else:
                cls.template_apply_mask_to_image_with_channels_2args(cls.function_reverse_mask_exclude_voxels_fillzero,
                                                                     images_array,
                                                                     masks_exclude_voxels_array)
                return images_array



class OperationBinaryMasks(OperationMasks):

    @classmethod
    def process_masks(cls, masks_array):
        # convert to binary masks (0, 1)
        #return np.where(masks_array != cls.val_mask_background, cls.val_mask_positive, cls.val_mask_background)
        return np.clip(masks_array, cls.val_mask_background, cls.val_mask_positive)

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
    def get_masks_with_label(cls, masks_array, label):
        return np.where(np.isin(masks_array, label), cls.val_mask_positive, cls.val_mask_background).astype(masks_array.dtype)

    @classmethod
    def get_masks_with_labels(cls, masks_array, labels_list):
        return np.where(np.isin(masks_array, labels_list), cls.val_mask_positive, cls.val_mask_background).astype(masks_array.dtype)

    @classmethod
    def get_list_masks_with_labels(cls, masks_array, labels_list):
        return [cls.get_masks_with_label(masks_array, ilabel) for ilabel in labels_list]

    @classmethod
    def get_list_masks_all_labels(cls, masks_array):
        labels_list = cls.extract_labels_in_masks(masks_array)
        return cls.get_list_masks_with_labels(masks_array, labels_list)

    @classmethod
    def extract_labels_in_masks(cls, masks_array):
        return np.delete(np.unique(masks_array), 0)

    @classmethod
    def merge_two_masks(cls, masks_array_1, masks_array_2, isNot_intersect_masks=False):
        if isNot_intersect_masks:
            # check there is no overlap between the two masks
            intersect_masks = np.multiply(masks_array_1, masks_array_2)
            index_posit_intersect = np.where(intersect_masks == cls.val_mask_positive)

            if len(index_posit_intersect[0] != 0):
                message = 'OperationBinaryMasks:merge_two_masks: Found intersection between the two input masks'
                CatchErrorException(message)

        out_masks_array = masks_array_1 + masks_array_2
        return np.clip(out_masks_array, cls.val_mask_background, cls.val_mask_positive)

    @classmethod
    def substract_two_masks(cls, masks_array_1, masks_array_2):
        out_masks_array = (masks_array_1 - masks_array_2).astype(np.int8)
        return np.clip(out_masks_array, cls.val_mask_background, cls.val_mask_positive).astype(masks_array_1.dtype)



class OperationMultiClassMasks(OperationMasks):

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


class ThinningMasks(OperationMasks):
    @staticmethod
    def compute(in_array):
        # thinning masks to obtain centrelines...
        return skeletonize_3d(in_array.astype(np.uint8))
        #out_cenlines_array = skeletonize_3d(in_array)
        ## convert to binary masks (0, 1)
        #return np.where(out_cenlines_array, cls.val_mask_positive, cls.val_mask_background)


class VolumeMasks(OperationMasks):
    @staticmethod
    def compute(in_array, voxel_size=None):
        masks_sum = np.sum(in_array)
        if voxel_size:
            voxel_vol = np.prod(voxel_size)
            return masks_sum * voxel_vol
        else:
            return masks_sum


class MorphoFillHolesMasks(OperationMasks):
    @staticmethod
    def compute(in_array):
        return binary_fill_holes(in_array).astype(in_array.dtype)


class MorphoErodeMasks(OperationMasks):
    @staticmethod
    def compute(in_array, num_iters=1):
        return binary_erosion(in_array, iterations=num_iters).astype(in_array.dtype)


class MorphoDilateMasks(OperationMasks):
    @staticmethod
    def compute(in_array, num_iters=1):
        return binary_dilation(in_array, iterations=num_iters).astype(in_array.dtype)


class MorphoOpenMasks(OperationMasks):
    @staticmethod
    def compute(in_array, num_iters=1):
        return binary_opening(in_array, iterations=num_iters).astype(in_array.dtype)


class MorphoCloseMasks(OperationMasks):
    @staticmethod
    def compute(in_array, num_iters=1):
        return binary_closing(in_array, iterations=num_iters).astype(in_array.dtype)


class ConnectedRegionsMasks(OperationMasks):
    @staticmethod
    def compute(in_array, connectivity_dim=None, is_return_num_regs=False):
        if not connectivity_dim:
            connectivity_dim = in_array.ndim
        if is_return_num_regs:
            (out_array, out_num_regs) = label(in_array, connectivity=connectivity_dim, background=0, return_num=is_return_num_regs)
            return (out_array.astype(in_array.dtype), out_num_regs)
        else:
            out_array = label(in_array, connectivity=connectivity_dim, background=0, return_num=is_return_num_regs)
            return out_array.astype(in_array.dtype)


class FirstConnectedRegionMasks(OperationMasks):
    @staticmethod
    def compute(in_array, connectivity_dim=None):
        (all_regions_array, num_regs) = ConnectedRegionsMasks.compute(in_array, connectivity_dim=connectivity_dim, is_return_num_regs=True)
        # retrieve the conn. region with the largest volume
        max_vol_regs = 0.0
        out_array = None
        for ireg in range(num_regs):
            # volume = count voxels for the the conn. region with label "i+1"
            iconreg_vol = np.count_nonzero(all_regions_array == ireg+1)
            if iconreg_vol > max_vol_regs:
                # extract the conn. region with label "i+1"
                out_array = np.where(all_regions_array == ireg+1, 1, 0).astype(in_array.dtype)
                max_vol_regs = iconreg_vol
        # endfor
        return out_array