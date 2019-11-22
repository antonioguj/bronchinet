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
from Postprocessing.SlidingWindowReconstructorImages import *
from Postprocessing.SlidingWindowPlusTransformReconstructorImages import *
from Postprocessing.TransformationReconstructorImages import *



def getImagesReconstructor2D(size_in_images,
                             use_slidingWindowImages,
                             slidewin_propOverlap,
                             size_total_image=(0, 0),
                             use_transformationImages=False,
                             num_trans_per_sample=1,
                             isUse_valid_convs= False,
                             size_output_model= None,
                             isfilter_valid_outUnet=False,
                             prop_valid_outUnet=None,
                             is_onehotmulticlass=False):

    if (use_slidingWindowImages):
        # images reconstruction by sliding-window...
        if (use_transformationImages):
            # reconstruction from several random transformation of images...
            return SlidingWindowPlusTransformReconstructorImages2D(size_in_images,
                                                                   slidewin_propOverlap,
                                                                   TransformationRigidImages2D(size_in_images,
                                                                                               rotation_range=ROTATION_XY_RANGE,
                                                                                               height_shift_range=HEIGHT_SHIFT_RANGE,
                                                                                               width_shift_range=WIDTH_SHIFT_RANGE,
                                                                                               horizontal_flip=HORIZONTAL_FLIP,
                                                                                               vertical_flip=VERTICAL_FLIP),
                                                                   num_trans_per_sample=num_trans_per_sample,
                                                                   size_total_image=size_total_image,
                                                                   isUse_valid_convs=isUse_valid_convs,
                                                                   size_output_model=size_output_model,
                                                                   isfilter_valid_outUnet=isfilter_valid_outUnet,
                                                                   prop_valid_outUnet=prop_valid_outUnet,
                                                                   is_onehotmulticlass=is_onehotmulticlass)
        else:
            return SlidingWindowReconstructorImages2D(size_in_images,
                                                      slidewin_propOverlap,
                                                      size_total_image=size_total_image,
                                                      isUse_valid_convs=isUse_valid_convs,
                                                      size_output_model=size_output_model,
                                                      isfilter_valid_outUnet=isfilter_valid_outUnet,
                                                      prop_valid_outUnet=prop_valid_outUnet,
                                                      is_onehotmulticlass=is_onehotmulticlass)
    else:
        if (use_transformationImages):
            return SlicingPlusTransformReconstructorImages2D(size_in_images,
                                                             TransformationRigidImages2D(size_in_images,
                                                                                         rotation_range=ROTATION_XY_RANGE,
                                                                                         height_shift_range=HEIGHT_SHIFT_RANGE,
                                                                                         width_shift_range=WIDTH_SHIFT_RANGE,
                                                                                         horizontal_flip=HORIZONTAL_FLIP,
                                                                                         vertical_flip=VERTICAL_FLIP),
                                                             num_trans_per_sample=num_trans_per_sample,
                                                             size_total_image=size_total_image,
                                                             isUse_valid_convs=isUse_valid_convs,
                                                             size_output_model=size_output_model,
                                                             isfilter_valid_outUnet=isfilter_valid_outUnet,
                                                             prop_valid_outUnet=prop_valid_outUnet,
                                                             is_onehotmulticlass=is_onehotmulticlass)
        else:
            return SlicingReconstructorImages2D(size_in_images,
                                                size_total_image=size_total_image,
                                                isUse_valid_convs=isUse_valid_convs,
                                                size_output_model=size_output_model,
                                                isfilter_valid_outUnet=isfilter_valid_outUnet,
                                                prop_valid_outUnet=prop_valid_outUnet,
                                                is_onehotmulticlass=is_onehotmulticlass)



def getImagesReconstructor3D(size_in_images,
                             use_slidingWindowImages,
                             slidewin_propOverlap,
                             size_total_image=(0, 0, 0),
                             use_transformationImages=False,
                             num_trans_per_sample=1,
                             isUse_valid_convs=False,
                             size_output_model=None,
                             isfilter_valid_outUnet=False,
                             prop_valid_outUnet=None,
                             is_onehotmulticlass=False):

    if (use_slidingWindowImages):
        # images reconstruction by sliding-window...
        if (use_transformationImages):
            # reconstruction from several random transformation of images...
            return SlidingWindowPlusTransformReconstructorImages3D(size_in_images,
                                                                   slidewin_propOverlap,
                                                                   TransformationRigidImages3D(size_in_images,
                                                                                               rotation_XY_range=ROTATION_XY_RANGE,
                                                                                               rotation_XZ_range=ROTATION_XZ_RANGE,
                                                                                               rotation_YZ_range=ROTATION_YZ_RANGE,
                                                                                               height_shift_range=HEIGHT_SHIFT_RANGE,
                                                                                               width_shift_range=WIDTH_SHIFT_RANGE,
                                                                                               depth_shift_range=DEPTH_SHIFT_RANGE,
                                                                                               horizontal_flip=HORIZONTAL_FLIP,
                                                                                               vertical_flip=VERTICAL_FLIP,
                                                                                               depthZ_flip=DEPTHZ_FLIP),
                                                                   num_trans_per_sample=num_trans_per_sample,
                                                                   size_total_image=size_total_image,
                                                                   isUse_valid_convs=isUse_valid_convs,
                                                                   size_output_model=size_output_model,
                                                                   isfilter_valid_outUnet=isfilter_valid_outUnet,
                                                                   prop_valid_outUnet=prop_valid_outUnet,
                                                                   is_onehotmulticlass=is_onehotmulticlass)
        else:
            return SlidingWindowReconstructorImages3D(size_in_images,
                                                      slidewin_propOverlap,
                                                      size_total_image=size_total_image,
                                                      isUse_valid_convs=isUse_valid_convs,
                                                      size_output_model=size_output_model,
                                                      isfilter_valid_outUnet=isfilter_valid_outUnet,
                                                      prop_valid_outUnet=prop_valid_outUnet,
                                                      is_onehotmulticlass=is_onehotmulticlass)
    else:
        if (use_transformationImages):
            # data augmentation by random transformation to input images...
            return SlicingPlusTransformReconstructorImages3D(size_in_images,
                                                             TransformationRigidImages3D(size_in_images,
                                                                                         rotation_XY_range=ROTATION_XY_RANGE,
                                                                                         rotation_XZ_range=ROTATION_XZ_RANGE,
                                                                                         rotation_YZ_range=ROTATION_YZ_RANGE,
                                                                                         height_shift_range=HEIGHT_SHIFT_RANGE,
                                                                                         width_shift_range=WIDTH_SHIFT_RANGE,
                                                                                         depth_shift_range=DEPTH_SHIFT_RANGE,
                                                                                         horizontal_flip=HORIZONTAL_FLIP,
                                                                                         vertical_flip=VERTICAL_FLIP,
                                                                                         depthZ_flip=DEPTHZ_FLIP),
                                                             num_trans_per_sample=num_trans_per_sample,
                                                             size_total_image=size_total_image,
                                                             isUse_valid_convs=isUse_valid_convs,
                                                             size_output_model=size_output_model,
                                                             isfilter_valid_outUnet=isfilter_valid_outUnet,
                                                             prop_valid_outUnet=prop_valid_outUnet,
                                                             is_onehotmulticlass=is_onehotmulticlass)
        else:
            return SlicingReconstructorImages3D(size_in_images,
                                                size_total_image=size_total_image,
                                                isUse_valid_convs=isUse_valid_convs,
                                                size_output_model=size_output_model,
                                                isfilter_valid_outUnet=isfilter_valid_outUnet,
                                                prop_valid_outUnet=prop_valid_outUnet,
                                                is_onehotmulticlass=is_onehotmulticlass)
