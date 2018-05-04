#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
from Preprocessing.SlidingWindowImages import *
from Preprocessing.SlidingWindowPlusTransformImages import *
from Preprocessing.TransformationImages import *


class BaseImageGenerator(object):

    def complete_init_data(self, size_total):
        pass

    def get_num_images(self):
        pass

    def get_shape_out_array(self, in_array_shape):
        pass

    def get_image_array(self, images_array, index):
        pass

    def compute_images_array_all(self, images_array):
        pass

    @staticmethod
    def getImagesGenerator2D(use_slidingWindowImages, prop_overlap_X_Y, use_TransformationImages):

        if (use_slidingWindowImages):
            # Images Data Generator by Sliding-window...
            if (use_TransformationImages):
                # Data augmentation by random transformation to input images...
                return SlidingWindowPlusTransformImages2D(IMAGES_DIMS_X_Y,
                                                          prop_overlap_X_Y,
                                                          rotation_range=ROTATION_XY_RANGE,
                                                          height_shift_range=HEIGHT_SHIFT_RANGE,
                                                          width_shift_range=WIDTH_SHIFT_RANGE,
                                                          horizontal_flip=HORIZONTAL_FLIP,
                                                          vertical_flip=VERTICAL_FLIP)
            else:
                return SlidingWindowImages2D(IMAGES_DIMS_X_Y,
                                             prop_overlap_X_Y)
        else:
            if (use_TransformationImages):
                # Data augmentation by random transformation to input images...
                return TransformationImages2D(IMAGES_DIMS_X_Y,
                                              rotation_range=ROTATION_XY_RANGE,
                                              height_shift_range=HEIGHT_SHIFT_RANGE,
                                              width_shift_range=WIDTH_SHIFT_RANGE,
                                              horizontal_flip=HORIZONTAL_FLIP,
                                              vertical_flip=VERTICAL_FLIP)
            else:
                return None

    @staticmethod
    def getImagesGenerator3D(use_slidingWindowImages, prop_overlap_Z_X_Y, use_TransformationImages):

        if (use_slidingWindowImages):
            # Images Data Generator by Sliding-window...
            if (use_TransformationImages):
                # Data augmentation by random transformation to input images...
                return SlidingWindowPlusTransformImages3D(IMAGES_DIMS_Z_X_Y,
                                                          prop_overlap_Z_X_Y,
                                                          rotation_XY_range=ROTATION_XY_RANGE,
                                                          rotation_XZ_range=ROTATION_XZ_RANGE,
                                                          rotation_YZ_range=ROTATION_YZ_RANGE,
                                                          height_shift_range=HEIGHT_SHIFT_RANGE,
                                                          width_shift_range=WIDTH_SHIFT_RANGE,
                                                          depth_shift_range=DEPTH_SHIFT_RANGE,
                                                          horizontal_flip=HORIZONTAL_FLIP,
                                                          vertical_flip=VERTICAL_FLIP,
                                                          depthZ_flip=DEPTHZ_FLIP)
            else:
                return SlidingWindowImages3D(IMAGES_DIMS_Z_X_Y,
                                             prop_overlap_Z_X_Y)
        else:
            if (use_TransformationImages):
                # Data augmentation by random transformation to input images...
                return TransformationImages3D(IMAGES_DIMS_Z_X_Y,
                                              rotation_XY_range=ROTATION_XY_RANGE,
                                              rotation_XZ_range=ROTATION_XZ_RANGE,
                                              rotation_YZ_range=ROTATION_YZ_RANGE,
                                              height_shift_range=HEIGHT_SHIFT_RANGE,
                                              width_shift_range=WIDTH_SHIFT_RANGE,
                                              depth_shift_range=DEPTH_SHIFT_RANGE,
                                              horizontal_flip=HORIZONTAL_FLIP,
                                              vertical_flip=VERTICAL_FLIP,
                                              depthZ_flip=DEPTHZ_FLIP)
            else:
                return None