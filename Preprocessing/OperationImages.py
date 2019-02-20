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
#from skimage.transform import rescale
#from scipy.misc import imresize
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


class IncludeImages(OperationImages):
    @staticmethod
    def compute2D(in_array, bound_box, out_array):
        out_array[bound_box[0][0]:bound_box[0][1], bound_box[1][0]:bound_box[1][1], ...] = in_array

    @staticmethod
    def compute3D(in_array, bound_box, out_array):
        out_array[bound_box[0][0]:bound_box[0][1], bound_box[1][0]:bound_box[1][1], bound_box[2][0]:bound_box[2][1], ...] = in_array


class ExtendImages(OperationImages):
    @staticmethod
    def compute2D(in_array, bound_box, out_array_shape, background_value=0):
        if background_value==0:
            out_array = np.zeros(out_array_shape, dtype=in_array.dtype)
        else:
            out_array = np.full(out_array_shape, background_value, dtype=in_array.dtype)
        IncludeImages.compute2D(in_array, bound_box, out_array)
        return out_array

    @staticmethod
    def compute3D(in_array, bound_box, out_array_shape, background_value=0):
        if background_value==0:
            out_array = np.zeros(out_array_shape, dtype=in_array.dtype)
        else:
            out_array = np.full(out_array_shape, background_value, dtype=in_array.dtype)
        IncludeImages.compute3D(in_array, bound_box, out_array)
        return out_array


class RescaleImages(OperationImages):
    order_default = 3

    @staticmethod
    def compute2D(in_array, scale_factor, order=order_default, is_binary_mask=False):
        if is_binary_mask:
            order = 0 # "nearest-neighbour"
        return rescale(in_array, scale=scale_factor, order=order,
                       preserve_range=True, multichannel=False, anti_aliasing=True)
        # if is_binary_mask:
        #     out_array = rescale(in_array, scale=scale_factor, order=order,
        #                         preserve_range=True, multichannel=False, anti_aliasing=True)
        #     return ThresholdImages.compute(out_array, 0.5)
        # else:
        #     return rescale(in_array, scale=scale_factor, order=order,
        #                    preserve_range=True, multichannel=False, anti_aliasing=True)

    @staticmethod
    def compute3D(in_array, scale_factor, order=order_default, is_binary_mask=False):
        return RescaleImages.compute2D(in_array, scale_factor, order, is_binary_mask)


# class RescaleImages(OperationImages):
#     type_interp_images = 'bilinear'
#     type_interp_masks  = 'nearest'
#
#     @classmethod
#     def compute2D(cls, in_array, (new_height, new_width), isMasks=False):
#         if (new_height < in_array.shape[1] and
#             new_width < in_array.shape[2]):
#             # if resized image is smaller, prevent the generation of new numpy array, not needed
#             if isMasks:
#                 for i, slice_image in enumerate(in_array):
#                     in_array[i, 0:new_height, 0:new_width] = imresize(slice_image, (new_height, new_width), interp=cls.type_interp_masks)
#                 #endfor
#             else:
#                 for i, slice_image in enumerate(in_array):
#                     in_array[i, 0:new_height, 0:new_width] = imresize(slice_image, (new_height, new_width), interp=cls.type_interp_images)
#                 #endfor
#             if isMasks:
#                 return cls.fix_masks_after_resizing(in_array[:, 0:new_height, 0:new_width])
#             else:
#                 return in_array[:, 0:new_height, 0:new_width]
#         else:
#             new_in_array = np.ndarray([in_array.shape[0], new_height, new_width], dtype=in_array.dtype)
#             if isMasks:
#                 for i, slice_image in enumerate(in_array):
#                     new_in_array[i] = imresize(slice_image, (new_height, new_width), interp=cls.type_interp_masks)
#                 #endfor
#             else:
#                 for i, slice_image in enumerate(in_array):
#                     new_in_array[i] = imresize(slice_image, (new_height, new_width), interp=cls.type_interp_images)
#                 #endfor
#             if isMasks:
#                 return cls.fix_masks_after_resizing(new_in_array)
#             else:
#                 return new_in_array
#
#     @classmethod
#     def compute3D(cls, in_array, (new_depthZ, new_height, new_width), isMasks=False):
#         if isMasks:
#             return imresize(in_array, (new_depthZ, new_height, new_width), interp=cls.type_interp_masks)
#         else:
#             return imresize(in_array, (new_depthZ, new_height, new_width), interp=cls.type_interp_images)
#
#     @classmethod
#     def fix_masks_after_resizing(cls, in_array):
#         # for some reason, after resizing the labels in masks are corrupted
#         # reassign the labels to integer values: 0 for background, and (1,2,...) for the classes
#         unique_vals_masks = np.unique(in_array).tolist()
#
#         new_in_array = np.zeros_like(in_array)
#
#         #voxels in corners always background. Remove from list background is assigned to 0
#         val_background = in_array[0,0,0]
#
#         if val_background not in unique_vals_masks:
#             message = "ResizeImages: value for background %s not found in list of labels for resized masks s..."%(val_background, unique_vals_masks)
#             CatchErrorException(message)
#         else:
#             unique_vals_masks.remove(val_background)
#
#         for i, value in enumerate(unique_vals_masks):
#             new_in_array = np.where(in_array == value, i+1, new_in_array)
#
#         return new_in_array


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
