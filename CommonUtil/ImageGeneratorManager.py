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


def getImagesDataGenerator2D(use_slidingWindowImages,
                             prop_overlap_X_Y,
                             use_TransformationImages,
                             use_ElasticDeformationImages):

    if (use_slidingWindowImages):
        # images data generator by sliding-window...
        if (use_ElasticDeformationImages):
            # data augmentation by elastic deformation of input images...
            return SlidingWindowPlusElasticDeformationImages2D(IMAGES_DIMS_Z_X_Y,
                                                               prop_overlap_X_Y)
        elif (use_TransformationImages):
            # data augmentation by random transformation of input images...
            return SlidingWindowPlusTransformImages2D(IMAGES_DIMS_X_Y,
                                                      prop_overlap_X_Y,
                                                      rotation_range=ROTATION_XY_RANGE,
                                                      height_shift_range=HEIGHT_SHIFT_RANGE,
                                                      width_shift_range=WIDTH_SHIFT_RANGE,
                                                      horizontal_flip=HORIZONTAL_FLIP,
                                                      vertical_flip=VERTICAL_FLIP,
                                                      zoom_range=ZOOM_RANGE)
        else:
            return SlidingWindowImages2D(IMAGES_DIMS_X_Y,
                                         prop_overlap_X_Y)
    else:
        if (use_ElasticDeformationImages):
            return SlicingPlusElasticDeformationImages2D(IMAGES_DIMS_X_Y)
        if (use_TransformationImages):
            return SlicingPlusTransformImages2D(IMAGES_DIMS_X_Y,
                                                rotation_range=ROTATION_XY_RANGE,
                                                height_shift_range=HEIGHT_SHIFT_RANGE,
                                                width_shift_range=WIDTH_SHIFT_RANGE,
                                                horizontal_flip=HORIZONTAL_FLIP,
                                                vertical_flip=VERTICAL_FLIP,
                                                zoom_range=ZOOM_RANGE)
        else:
            return SlicingImages2D(IMAGES_DIMS_X_Y)


def getImagesDataGenerator3D(use_slidingWindowImages,
                             prop_overlap_Z_X_Y,
                             use_TransformationImages,
                             use_ElasticDeformationImages):

    if (use_slidingWindowImages):
        # images data generator by sliding-window...
        if (use_ElasticDeformationImages):
            # data augmentation by elastic deformation of input images...
            return SlidingWindowPlusElasticDeformationImages3D(IMAGES_DIMS_Z_X_Y,
                                                               prop_overlap_Z_X_Y)
        elif (use_TransformationImages):
            # data augmentation by random transformation of input images...
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
                                                      depthZ_flip=DEPTHZ_FLIP,
                                                      zoom_range=ZOOM_RANGE)
        else:
            return SlidingWindowImages3D(IMAGES_DIMS_Z_X_Y,
                                         prop_overlap_Z_X_Y)
    else:
        if (use_ElasticDeformationImages):
            return SlicingPlusElasticDeformationImages3D(IMAGES_DIMS_Z_X_Y)
        elif (use_TransformationImages):
            return SlicingPlusTransformImages3D(IMAGES_DIMS_Z_X_Y,
                                                rotation_XY_range=ROTATION_XY_RANGE,
                                                rotation_XZ_range=ROTATION_XZ_RANGE,
                                                rotation_YZ_range=ROTATION_YZ_RANGE,
                                                height_shift_range=HEIGHT_SHIFT_RANGE,
                                                width_shift_range=WIDTH_SHIFT_RANGE,
                                                depth_shift_range=DEPTH_SHIFT_RANGE,
                                                horizontal_flip=HORIZONTAL_FLIP,
                                                vertical_flip=VERTICAL_FLIP,
                                                depthZ_flip=DEPTHZ_FLIP,
                                                zoom_range=ZOOM_RANGE)
        else:
            return SlicingImages3D(IMAGES_DIMS_Z_X_Y)


def getImagesVolumeTransformator2D(size_image,
                                   use_TransformationImages,
                                   use_ElasticDeformationImages,
                                   type_elastic_deformation='gridwise'):

    if (use_ElasticDeformationImages):
        if type_elastic_deformation == 'pixelwise':
            return ElasticDeformationPixelwiseImages2D(size_image)
        else:
            return ElasticDeformationGridwiseImages2D(size_image)
    elif (use_TransformationImages):
        return TransformationImages2D(size_image,
                                      rotation_range=ROTATION_XY_RANGE,
                                      width_shift_range=WIDTH_SHIFT_RANGE,
                                      height_shift_range=HEIGHT_SHIFT_RANGE,
                                      horizontal_flip=HORIZONTAL_FLIP,
                                      vertical_flip=VERTICAL_FLIP,
                                      zoom_range=ZOOM_RANGE)
    else:
        return False


def getImagesVolumeTransformator3D(size_image,
                                   use_TransformationImages,
                                   use_ElasticDeformationImages,
                                   type_elastic_deformation='gridwise'):

    if (use_ElasticDeformationImages):
        if type_elastic_deformation == 'pixelwise':
            return ElasticDeformationPixelwiseImages3D(size_image)
        else:
            return ElasticDeformationGridwiseImages3D(size_image)
    elif (use_TransformationImages):
        return TransformationImages3D(size_image,
                                      rotation_XY_range=ROTATION_XY_RANGE,
                                      rotation_XZ_range=ROTATION_XZ_RANGE,
                                      rotation_YZ_range=ROTATION_YZ_RANGE,
                                      width_shift_range=WIDTH_SHIFT_RANGE,
                                      height_shift_range=HEIGHT_SHIFT_RANGE,
                                      depth_shift_range=DEPTH_SHIFT_RANGE,
                                      horizontal_flip=HORIZONTAL_FLIP,
                                      vertical_flip=VERTICAL_FLIP,
                                      depthZ_flip=DEPTHZ_FLIP,
                                      zoom_range=ZOOM_RANGE)
    else:
        return False