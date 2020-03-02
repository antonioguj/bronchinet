#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.FunctionsUtil import *
from skimage.transform import rescale
#from scipy.misc import imresize
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing
from skimage.measure import label
import numpy as np



class OperationImages(object):
    @staticmethod
    def check_in_array_2D_without_channels(in_array_shape):
        return len(in_array_shape) == 2

    @staticmethod
    def check_in_array_3D_without_channels(in_array_shape):
        return len(in_array_shape) == 3


class NormalizeImages(OperationImages):
    @staticmethod
    def compute_nochannels(in_array):
        max_val = np.max(in_array)
        min_val = np.min(in_array)
        return (in_array - min_val) / (max_val - min_val)

    @staticmethod
    def compute_withchannels(in_array):
        num_channels = in_array.shape[-1]
        for i in range(num_channels):
            max_val = np.max(in_array[..., i])
            min_val = np.min(in_array[..., i])
            in_array[..., i] = (in_array[..., i] - min_val) / (max_val - min_val)
        #endfor
        return in_array

    @classmethod
    def compute2D(cls, in_array):
        if (cls.check_in_array_2D_without_channels(in_array.shape)):
            return cls.compute_nochannels(in_array)
        else:
            return cls.compute_withchannels(in_array)

    @classmethod
    def compute3D(cls, in_array):
        if (cls.check_in_array_3D_without_channels(in_array.shape)):
            return cls.compute_nochannels(in_array)
        else:
            return cls.compute_withchannels(in_array)


class CropImages(OperationImages):
    @staticmethod
    def compute2D(in_array, bound_box):
        return in_array[:, bound_box[0][0]:bound_box[0][1], bound_box[1][0]:bound_box[1][1], ...]

    @staticmethod
    def compute3D(in_array, bound_box):
        return in_array[bound_box[0][0]:bound_box[0][1], bound_box[1][0]:bound_box[1][1], bound_box[2][0]:bound_box[2][1], ...]


class SetPatchInImages(OperationImages):
    @staticmethod
    def compute2D(in_array, out_array, bound_box):
        out_array[bound_box[0][0]:bound_box[0][1], bound_box[1][0]:bound_box[1][1], ...] = in_array
    @staticmethod
    def compute2D_byadding(in_array, out_array, bound_box):
        out_array[bound_box[0][0]:bound_box[0][1], bound_box[1][0]:bound_box[1][1], ...] += in_array

    @staticmethod
    def compute3D(in_array, out_array, bound_box):
        out_array[bound_box[0][0]:bound_box[0][1], bound_box[1][0]:bound_box[1][1], bound_box[2][0]:bound_box[2][1], ...] = in_array
    @staticmethod
    def compute3D_byadding(in_array, out_array, bound_box):
        out_array[bound_box[0][0]:bound_box[0][1], bound_box[1][0]:bound_box[1][1], bound_box[2][0]:bound_box[2][1], ...] += in_array


class ExtendImages(OperationImages):
    @staticmethod
    def get_init_output_array(out_array_shape, out_array_dtype, background_value=0):
        if background_value==0:
            return np.zeros(out_array_shape, dtype=out_array_dtype)
        else:
            return np.full(out_array_shape, background_value, dtype=out_array_dtype)

    @classmethod
    def compute2D(cls, in_array, bound_box, out_array_shape, background_value=None):
        if not background_value:
            background_value = in_array[0][0]
        out_array = cls.get_init_output_array(out_array_shape, in_array.dtype, background_value)
        SetPatchInImages.compute2D(in_array, out_array, bound_box)
        return out_array

    @classmethod
    def compute3D(cls, in_array, bound_box, out_array_shape, background_value=0):
        if not background_value:
            background_value = in_array[0][0][0]
        out_array = cls.get_init_output_array(out_array_shape, in_array.dtype, background_value)
        SetPatchInImages.compute3D(in_array, out_array, bound_box)
        return out_array


class CropAndExtendImages(OperationImages):
    @staticmethod
    def compute2D(in_array, crop_bound_box, extend_bound_box, out_array_shape, background_value=None):
        if not background_value:
            background_value = in_array[0][0]
        return ExtendImages.compute2D(CropImages.compute2D(in_array, crop_bound_box),
                                      extend_bound_box, out_array_shape, background_value)

    @staticmethod
    def compute3D(in_array, crop_bound_box, extend_bound_box, out_array_shape, background_value=None):
        if not background_value:
            background_value = in_array[0][0][0]
        return ExtendImages.compute3D(CropImages.compute3D(in_array, crop_bound_box),
                                      extend_bound_box, out_array_shape, background_value)


class RescaleImages(OperationImages):
    order_default = 3

    @staticmethod
    def compute2D(in_array, scale_factor, order=order_default, is_binary_mask=False, is_binarise_output=False):
        if is_binary_mask:
            out_array = rescale(in_array, scale=scale_factor, order=order,
                                preserve_range=True, multichannel=False, anti_aliasing=True)
            if is_binarise_output:
                # remove noise due to interpolation
                thres_remove_noise = 0.1
                return ThresholdImages.compute(out_array, thres_val=thres_remove_noise)
            else:
                return out_array
        else:
            return rescale(in_array, scale=scale_factor, order=order,
                           preserve_range=True, multichannel=False, anti_aliasing=True)

    @staticmethod
    def compute3D(in_array, scale_factor, order=order_default, is_binary_mask=False, is_binarise_output=False):
        return RescaleImages.compute2D(in_array, scale_factor, order, is_binary_mask, is_binarise_output)


class ThresholdImages(OperationImages):
    @staticmethod
    def compute(in_array, thres_val):
        return np.where(in_array > thres_val, 1.0, 0.0)


class FlippingImages(OperationImages):
    @staticmethod
    def compute(in_array, axis = 0):
        if axis == 0:
            return in_array[::-1, :, :]
        elif axis == 1:
            return in_array[:, ::-1, :]
        elif axis == 2:
            return in_array[:, :, ::-1]
        else:
            return False


class MorphoFillHolesImages(OperationImages):
    @staticmethod
    def compute(in_array):
        return binary_fill_holes(in_array).astype(in_array.dtype)


class MorphoErodeImages(OperationImages):
    @staticmethod
    def compute(in_array):
        return binary_erosion(in_array).astype(in_array.dtype)


class MorphoDilateImages(OperationImages):
    @staticmethod
    def compute(in_array):
        return binary_dilation(in_array).astype(in_array.dtype)


class MorphoOpenImages(OperationImages):
    @staticmethod
    def compute(in_array):
        return binary_opening(in_array).astype(in_array.dtype)


class MorphoCloseImages(OperationImages):
    @staticmethod
    def compute(in_array):
        return binary_closing(in_array).astype(in_array.dtype)


class ConnectedRegionsMasks(OperationImages):
    @staticmethod
    def compute(in_array, return_num_labels=False):
        return label(in_array, connectivity=in_array.ndim, background=0, return_num=return_num_labels)