#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
from Preprocessing.SlidingWindowImages import *
from Preprocessing.SlidingWindowPlusTransformImages import *
from Preprocessing.TransformationImages import *



def getImagesDataGenerator2D(size_in_images,
                             use_slidingWindowImages,
                             slidewin_propOverlap,
                             use_TransformationImages,
                             use_ElasticDeformationImages):

    if (use_slidingWindowImages):
        # images data generator by sliding-window...
        if (use_ElasticDeformationImages):
            # data augmentation by elastic deformation of input images...
            return SlidingWindowPlusElasticDeformationImages2D(size_in_images,
                                                               slidewin_propOverlap)
        elif (use_TransformationImages):
            # data augmentation by random transformation of input images...
            return SlidingWindowPlusTransformImages2D(size_in_images,
                                                      slidewin_propOverlap,
                                                      rotation_range=ROTATION_XY_RANGE,
                                                      height_shift_range=HEIGHT_SHIFT_RANGE,
                                                      width_shift_range=WIDTH_SHIFT_RANGE,
                                                      horizontal_flip=HORIZONTAL_FLIP,
                                                      vertical_flip=VERTICAL_FLIP,
                                                      zoom_range=ZOOM_RANGE)
        else:
            return SlidingWindowImages2D(size_in_images,
                                         slidewin_propOverlap)
    else:
        if (use_ElasticDeformationImages):
            return SlicingPlusElasticDeformationImages2D(size_in_images)
        if (use_TransformationImages):
            return SlicingPlusTransformImages2D(size_in_images,
                                                rotation_range=ROTATION_XY_RANGE,
                                                height_shift_range=HEIGHT_SHIFT_RANGE,
                                                width_shift_range=WIDTH_SHIFT_RANGE,
                                                horizontal_flip=HORIZONTAL_FLIP,
                                                vertical_flip=VERTICAL_FLIP,
                                                zoom_range=ZOOM_RANGE)
        else:
            return SlicingImages2D(size_in_images)



def getImagesDataGenerator3D(size_in_images,
                             use_slidingWindowImages,
                             slidewin_propOverlap,
                             use_TransformationImages,
                             use_ElasticDeformationImages):

    if (use_slidingWindowImages):
        # images data generator by sliding-window...
        if (use_ElasticDeformationImages):
            # data augmentation by elastic deformation of input images...
            return SlidingWindowPlusElasticDeformationImages3D(size_in_images,
                                                               slidewin_propOverlap)
        elif (use_TransformationImages):
            # data augmentation by random transformation of input images...
            return SlidingWindowPlusTransformImages3D(size_in_images,
                                                      slidewin_propOverlap,
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
            return SlidingWindowImages3D(size_in_images,
                                         slidewin_propOverlap)
    else:
        if (use_ElasticDeformationImages):
            return SlicingPlusElasticDeformationImages3D(size_in_images)
        elif (use_TransformationImages):
            return SlicingPlusTransformImages3D(size_in_images,
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
            return SlicingImages3D(size_in_images)



def getImagesVolumeTransformator2D(size_in_images,
                                   use_TransformationImages,
                                   use_ElasticDeformationImages,
                                   type_elastic_deformation='gridwise'):

    if (use_ElasticDeformationImages):
        if type_elastic_deformation == 'pixelwise':
            return ElasticDeformationPixelwiseImages2D(size_in_images)
        else:
            return ElasticDeformationGridwiseImages2D(size_in_images)
    elif (use_TransformationImages):
        return TransformationImages2D(size_in_images,
                                      rotation_range=ROTATION_XY_RANGE,
                                      width_shift_range=WIDTH_SHIFT_RANGE,
                                      height_shift_range=HEIGHT_SHIFT_RANGE,
                                      horizontal_flip=HORIZONTAL_FLIP,
                                      vertical_flip=VERTICAL_FLIP,
                                      zoom_range=ZOOM_RANGE)
    else:
        return False



def getImagesVolumeTransformator3D(size_in_images,
                                   use_TransformationImages,
                                   use_ElasticDeformationImages,
                                   type_elastic_deformation='gridwise'):

    if (use_ElasticDeformationImages):
        if type_elastic_deformation == 'pixelwise':
            return ElasticDeformationPixelwiseImages3D(size_in_images)
        else:
            return ElasticDeformationGridwiseImages3D(size_in_images)
    elif (use_TransformationImages):
        return TransformationImages3D(size_in_images,
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
