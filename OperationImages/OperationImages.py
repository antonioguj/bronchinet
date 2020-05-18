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
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion, binary_dilation, binary_opening, binary_closing
from skimage.morphology import skeletonize_3d
from skimage.measure import label
#from scipy.misc import imresize
import numpy as np



class OperationImages(object):
    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        return NotImplemented


class NormaliseImages(OperationImages):
    @staticmethod
    def compute_nochannels(in_array):
        max_val = np.max(in_array)
        min_val = np.min(in_array)
        return (in_array - min_val) / float(max_val - min_val)

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
    def compute(cls, in_array, *args, **kwargs):
        isimg_with_channels = kwargs['isimg_with_channels'] if 'isimg_with_channels' in kwargs.keys() else False
        if isimg_with_channels:
            return cls.compute_withchannels(in_array)
        else:
            return cls.compute_nochannels(in_array)


class CropImages(OperationImages):
    @staticmethod
    def compute2D(in_array, in_bound_box):
        return in_array[:, in_bound_box[0][0]:in_bound_box[0][1], in_bound_box[1][0]:in_bound_box[1][1], ...]

    @staticmethod
    def compute3D(in_array, in_bound_box):
        return in_array[in_bound_box[0][0]:in_bound_box[0][1], in_bound_box[1][0]:in_bound_box[1][1], in_bound_box[2][0]:in_bound_box[2][1], ...]

    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        is_img2D = kwargs['is_img2D'] if 'is_img2D' in kwargs.keys() else False
        if is_img2D:
            return cls.compute2D(in_array, *args)
        else:
            return cls.compute3D(in_array, *args)


class SetPatchInImages(OperationImages):
    @staticmethod
    def compute2D(in_array, out_array, in_bound_box):
        out_array[in_bound_box[0][0]:in_bound_box[0][1], in_bound_box[1][0]:in_bound_box[1][1], ...] = in_array

    @staticmethod
    def compute2D_byadding(in_array, out_array, in_bound_box):
        out_array[in_bound_box[0][0]:in_bound_box[0][1], in_bound_box[1][0]:in_bound_box[1][1], ...] += in_array

    @staticmethod
    def compute3D(in_array, out_array, in_bound_box):
        out_array[in_bound_box[0][0]:in_bound_box[0][1], in_bound_box[1][0]:in_bound_box[1][1], in_bound_box[2][0]:in_bound_box[2][1], ...] = in_array

    @staticmethod
    def compute3D_byadding(in_array, out_array, in_bound_box):
        out_array[in_bound_box[0][0]:in_bound_box[0][1], in_bound_box[1][0]:in_bound_box[1][1], in_bound_box[2][0]:in_bound_box[2][1], ...] += in_array

    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        is_img2D        = kwargs['is_img2D'] if 'is_img2D' in kwargs.keys() else False
        iscalc_byadding = kwargs['iscalc_byadding'] if 'iscalc_byadding' in kwargs.keys() else False
        if is_img2D:
            if iscalc_byadding:
                return cls.compute2D_byadding(in_array, *args)
            else:
                return cls.compute2D(in_array, *args)
        else:
            if iscalc_byadding:
                return cls.compute3D_byadding(in_array, *args)
            else:
                return cls.compute3D(in_array, *args)


class CropAndSetPatchInImages(OperationImages):
    @staticmethod
    def compute2D(in_array, out_array, in_crop_bound_box, in_extend_bound_box):
        return SetPatchInImages.compute2D(CropImages.compute2D(in_array, in_crop_bound_box), out_array, in_extend_bound_box)

    @staticmethod
    def compute3D(in_array, out_array, in_crop_bound_box, in_extend_bound_box):
        return SetPatchInImages.compute3D(CropImages.compute3D(in_array, in_crop_bound_box), out_array, in_extend_bound_box)

    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        is_img2D = kwargs['is_img2D'] if 'is_img2D' in kwargs.keys() else False
        if is_img2D:
            return cls.compute2D(in_array, *args)
        else:
            return cls.compute3D(in_array, *args)


class ExtendImages(OperationImages):
    @staticmethod
    def get_init_output_array(out_array_shape, out_array_dtype, background_value=None):
        if background_value is None:
            return np.zeros(out_array_shape, dtype=out_array_dtype)
        else:
            return np.full(out_array_shape, background_value, dtype=out_array_dtype)

    @classmethod
    def compute2D(cls, in_array, in_bound_box, out_array_shape, background_value=None):
        if background_value is None:
            background_value = in_array[0][0]
        out_array = cls.get_init_output_array(out_array_shape, in_array.dtype, background_value)
        SetPatchInImages.compute2D(in_array, out_array, in_bound_box)
        return out_array

    @classmethod
    def compute3D(cls, in_array, in_bound_box, out_array_shape, background_value=None):
        if background_value is None:
            background_value = in_array[0][0][0]
        out_array = cls.get_init_output_array(out_array_shape, in_array.dtype, background_value)
        SetPatchInImages.compute3D(in_array, out_array, in_bound_box)
        return out_array

    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        is_img2D = kwargs['is_img2D'] if 'is_img2D' in kwargs.keys() else False
        if is_img2D:
            return cls.compute2D(in_array, *args)
        else:
            return cls.compute3D(in_array, *args)


class CropAndExtendImages(OperationImages):
    @staticmethod
    def compute2D(in_array, in_crop_bound_box, in_extend_bound_box, out_array_shape, background_value=None):
        if background_value is None:
            background_value = in_array[0][0]
        return ExtendImages.compute2D(CropImages.compute2D(in_array, in_crop_bound_box), in_extend_bound_box, out_array_shape, background_value)

    @staticmethod
    def compute3D(in_array, in_crop_bound_box, in_extend_bound_box, out_array_shape, background_value=None):
        if background_value is None:
            background_value = in_array[0][0][0]
        return ExtendImages.compute3D(CropImages.compute3D(in_array, in_crop_bound_box), in_extend_bound_box, out_array_shape, background_value)

    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        is_img2D = kwargs['is_img2D'] if 'is_img2D' in kwargs.keys() else False
        if is_img2D:
            return cls.compute2D(in_array, *args)
        else:
            return cls.compute3D(in_array, *args)


class RescaleImages(OperationImages):
    order_default = 3

    @staticmethod
    def compute_int(in_array, scale_factor, order=order_default, is_inlabels=False, is_binarise_output=False):
        if is_inlabels:
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

    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        return cls.compute_int(in_array, *args, **kwargs)


class FlippingImages(OperationImages):
    @staticmethod
    def compute2D(in_array, axis=0):
        if axis == 0:
            return in_array[::-1, :]
        elif axis == 1:
            return in_array[:, ::-1]
        else:
            return False

    @staticmethod
    def compute3D(in_array, axis=0):
        if axis == 0:
            return in_array[::-1, :, :]
        elif axis == 1:
            return in_array[:, ::-1, :]
        elif axis == 2:
            return in_array[:, :, ::-1]
        else:
            return False

    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        axis     = kwargs['axis'] if 'axis' in kwargs.keys() else 0
        is_img2D = kwargs['is_img2D'] if 'is_img2D' in kwargs.keys() else False
        if is_img2D:
            return cls.compute2D(in_array, axis=axis)
        else:
            return cls.compute3D(in_array, axis=axis)


class ThresholdImages(OperationImages):
    val_mask       = 1
    val_background = 0

    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        thres_value = args[0]
        return np.where(in_array > thres_value, cls.val_mask, cls.val_background).astype(np.uint8)


class ThinningMasks(OperationImages):
    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        # thinning masks to obtain centrelines...
        return skeletonize_3d(in_array.astype(np.uint8))
        #out_cenlines_array = skeletonize_3d(in_array)
        ## convert to binary masks (0, 1)
        #return np.where(out_cenlines_array, cls.val_mask_positive, cls.val_mask_background)


class VolumeMasks(OperationImages):
    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        voxel_size = kwargs['voxel_size'] if 'voxel_size' in kwargs.keys() else None
        masks_sum = np.sum(in_array)
        if voxel_size:
            voxel_vol = np.prod(voxel_size)
            return masks_sum * voxel_vol
        else:
            return masks_sum


class MorphoFillHolesMasks(OperationImages):
    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        return binary_fill_holes(in_array).astype(in_array.dtype)


class MorphoErodeMasks(OperationImages):
    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        num_iters = kwargs['num_iters'] if 'num_iters' in kwargs.keys() else 1
        return binary_erosion(in_array, iterations=num_iters).astype(in_array.dtype)


class MorphoDilateMasks(OperationImages):
    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        num_iters = kwargs['num_iters'] if 'num_iters' in kwargs.keys() else 1
        return binary_dilation(in_array, iterations=num_iters).astype(in_array.dtype)


class MorphoOpenMasks(OperationImages):
    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        num_iters = kwargs['num_iters'] if 'num_iters' in kwargs.keys() else 1
        return binary_opening(in_array, iterations=num_iters).astype(in_array.dtype)


class MorphoCloseMasks(OperationImages):
    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        num_iters = kwargs['num_iters'] if 'num_iters' in kwargs.keys() else 1
        return binary_closing(in_array, iterations=num_iters).astype(in_array.dtype)


class ConnectedRegionsMasks(OperationImages):
    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        connectivity_dim   = kwargs['connectivity_dim']   if 'connectivity_dim'   in kwargs.keys() else in_array.ndim
        is_return_num_regs = kwargs['is_return_num_regs'] if 'is_return_num_regs' in kwargs.keys() else False
        if is_return_num_regs:
            (out_array, out_num_regs) = label(in_array, connectivity=connectivity_dim, background=0, return_num=is_return_num_regs)
            return (out_array.astype(in_array.dtype), out_num_regs)
        else:
            out_array = label(in_array, connectivity=connectivity_dim, background=0, return_num=is_return_num_regs)
            return out_array.astype(in_array.dtype)


class FirstConnectedRegionMasks(OperationImages):
    @classmethod
    def compute(cls, in_array, *args, **kwargs):
        connectivity_dim = kwargs['connectivity_dim'] if 'connectivity_dim' in kwargs.keys() else in_array.ndim

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