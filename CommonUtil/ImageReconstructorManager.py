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
from Postprocessing.SlidingWindowReconstructorImages import *
from Postprocessing.SlidingWindowPlusTransformReconstructorImages import *
from Postprocessing.TransformationReconstructorImages import *


def getImagesReconstructor2D(use_slidingWindowImages, size_total_image, prop_overlap_X_Y, use_TransformationImages=False, num_trans_per_sample=1):

    if (use_slidingWindowImages):
        # images reconstruction by sliding-window...
        if (use_TransformationImages):
            # reconstruction from several random transformation of images...
            return SlidingWindowPlusTransformReconstructorImages2D(IMAGES_DIMS_X_Y,
                                                                   size_total_image,
                                                                   TransformationImages2D(IMAGES_DIMS_X_Y,
                                                                                          rotation_range=ROTATION_XY_RANGE,
                                                                                          height_shift_range=HEIGHT_SHIFT_RANGE,
                                                                                          width_shift_range=WIDTH_SHIFT_RANGE,
                                                                                          horizontal_flip=HORIZONTAL_FLIP,
                                                                                          vertical_flip=VERTICAL_FLIP),
                                                                   num_trans_per_sample,
                                                                   prop_overlap_X_Y)
        else:
            return SlidingWindowReconstructorImages2D(IMAGES_DIMS_X_Y,
                                                      size_total_image,
                                                      prop_overlap_X_Y)
    else:
        if (use_TransformationImages):
            return SlicingPlusTransformReconstructorImages2D(IMAGES_DIMS_X_Y,
                                                             size_total_image,
                                                             TransformationImages2D(IMAGES_DIMS_X_Y,
                                                                                    rotation_range=ROTATION_XY_RANGE,
                                                                                    height_shift_range=HEIGHT_SHIFT_RANGE,
                                                                                    width_shift_range=WIDTH_SHIFT_RANGE,
                                                                                    horizontal_flip=HORIZONTAL_FLIP,
                                                                                    vertical_flip=VERTICAL_FLIP),
                                                             num_trans_per_sample)
        else:
            return SlicingReconstructorImages2D(IMAGES_DIMS_X_Y,
                                                size_total_image)


def getImagesReconstructor3D(use_slidingWindowImages, size_total_image, prop_overlap_Z_X_Y, use_TransformationImages=False, num_trans_per_sample=1):

    if (use_slidingWindowImages):
        # images reconstruction by sliding-window...
        if (use_TransformationImages):
            # reconstruction from several random transformation of images...
            return SlidingWindowPlusTransformReconstructorImages3D(IMAGES_DIMS_X_Y,
                                                                   size_total_image,
                                                                   TransformationImages3D(IMAGES_DIMS_X_Y,
                                                                                          rotation_XY_range=ROTATION_XY_RANGE,
                                                                                          rotation_XZ_range=ROTATION_XZ_RANGE,
                                                                                          rotation_YZ_range=ROTATION_YZ_RANGE,
                                                                                          height_shift_range=HEIGHT_SHIFT_RANGE,
                                                                                          width_shift_range=WIDTH_SHIFT_RANGE,
                                                                                          depth_shift_range=DEPTH_SHIFT_RANGE,
                                                                                          horizontal_flip=HORIZONTAL_FLIP,
                                                                                          vertical_flip=VERTICAL_FLIP,
                                                                                          depthZ_flip=DEPTHZ_FLIP),
                                                                   num_trans_per_sample,
                                                                   prop_overlap_Z_X_Y)
        else:
            return SlidingWindowReconstructorImages3D(IMAGES_DIMS_Z_X_Y,
                                                      size_total_image,
                                                      prop_overlap_Z_X_Y)
    else:
        if (use_TransformationImages):
            # data augmentation by random transformation to input images...
            return SlicingPlusTransformReconstructorImages3D(IMAGES_DIMS_X_Y,
                                                             size_total_image,
                                                             TransformationImages3D(IMAGES_DIMS_X_Y,
                                                                                    rotation_XY_range=ROTATION_XY_RANGE,
                                                                                    rotation_XZ_range=ROTATION_XZ_RANGE,
                                                                                    rotation_YZ_range=ROTATION_YZ_RANGE,
                                                                                    height_shift_range=HEIGHT_SHIFT_RANGE,
                                                                                    width_shift_range=WIDTH_SHIFT_RANGE,
                                                                                    depth_shift_range=DEPTH_SHIFT_RANGE,
                                                                                    horizontal_flip=HORIZONTAL_FLIP,
                                                                                    vertical_flip=VERTICAL_FLIP,
                                                                                    depthZ_flip=DEPTHZ_FLIP),
                                                             num_trans_per_sample)
        else:
            return SlicingReconstructorImages3D(IMAGES_DIMS_Z_X_Y,
                                                size_total_image)