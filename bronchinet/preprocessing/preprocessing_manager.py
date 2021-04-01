
from typing import Tuple

from common.constant import TRANS_ROTATION_XY_RANGE, TRANS_ROTATION_XZ_RANGE, TRANS_ROTATION_YZ_RANGE, \
                            TRANS_HEIGHT_SHIFT_RANGE, TRANS_WIDTH_SHIFT_RANGE, \
                            TRANS_DEPTH_SHIFT_RANGE, TRANS_HORIZONTAL_FLIP, TRANS_VERTICAL_FLIP, \
                            TRANS_AXIALDIR_FLIP,  TRANS_ZOOM_RANGE, TRANS_FILL_MODE_TRANSFORM, \
                            TYPE_TRANSFORM_ELASTICDEFORM_IMAGES
from common.exceptionmanager import catch_error_exception
from preprocessing.imagegenerator import ImageGenerator, NullGenerator, CombinedImagesGenerator
from preprocessing.randomwindowimages import RandomWindowImages
from preprocessing.slidingwindowimages import SlidingWindowImages
from preprocessing.transformrigidimages import TransformRigidImages2D, TransformRigidImages3D
from preprocessing.elasticdeformimages import ElasticDeformGridwiseImages, ElasticDeformPixelwiseImages, \
                                              ElasticDeformGridwiseImagesGijs


def get_images_generator(size_images: Tuple[int, ...],
                         use_sliding_window_images: bool,
                         prop_overlap_slide_window: Tuple[int, ...],
                         use_random_window_images: bool,
                         num_random_patches_epoch: int,
                         use_transform_rigid_images: bool,
                         use_transform_elasticdeform_images: bool,
                         size_volume_image: Tuple[int, ...] = (0,)
                         ) -> ImageGenerator:
    list_images_generators = []

    if (use_sliding_window_images):
        # generator of image patches by sliding-window...
        new_images_generator = SlidingWindowImages(size_images,
                                                   prop_overlap_slide_window,
                                                   size_volume_image)
        list_images_generators.append(new_images_generator)

    elif (use_random_window_images):
        # generator of image patches by random cropping window...
        new_images_generator = RandomWindowImages(size_images,
                                                  num_random_patches_epoch,
                                                  size_volume_image)
        list_images_generators.append(new_images_generator)

    if use_transform_rigid_images:
        # generator of images by random rigid transformations of input images...
        ndims = len(size_images)
        if ndims == 2:
            new_images_generator = TransformRigidImages2D(size_images,
                                                          rotation_range=TRANS_ROTATION_XY_RANGE,
                                                          height_shift_range=TRANS_HEIGHT_SHIFT_RANGE,
                                                          width_shift_range=TRANS_WIDTH_SHIFT_RANGE,
                                                          horizontal_flip=TRANS_HORIZONTAL_FLIP,
                                                          vertical_flip=TRANS_VERTICAL_FLIP,
                                                          zoom_range=TRANS_ZOOM_RANGE,
                                                          fill_mode=TRANS_FILL_MODE_TRANSFORM)
            list_images_generators.append(new_images_generator)
        elif ndims == 3:
            new_images_generator = TransformRigidImages3D(size_images,
                                                          rotation_xy_range=TRANS_ROTATION_XY_RANGE,
                                                          rotation_xz_range=TRANS_ROTATION_XZ_RANGE,
                                                          rotation_yz_range=TRANS_ROTATION_YZ_RANGE,
                                                          height_shift_range=TRANS_HEIGHT_SHIFT_RANGE,
                                                          width_shift_range=TRANS_WIDTH_SHIFT_RANGE,
                                                          depth_shift_range=TRANS_DEPTH_SHIFT_RANGE,
                                                          horizontal_flip=TRANS_HORIZONTAL_FLIP,
                                                          vertical_flip=TRANS_VERTICAL_FLIP,
                                                          axialdir_flip=TRANS_AXIALDIR_FLIP,
                                                          zoom_range=TRANS_ZOOM_RANGE,
                                                          fill_mode=TRANS_FILL_MODE_TRANSFORM)
            list_images_generators.append(new_images_generator)
        else:
            message = 'Wrong value of \'ndims\': %s' % (ndims)
            catch_error_exception(message)

    if use_transform_elasticdeform_images:
        if TYPE_TRANSFORM_ELASTICDEFORM_IMAGES == 'gridwise':
            new_images_generator = ElasticDeformGridwiseImages(size_images,
                                                               fill_mode=TRANS_FILL_MODE_TRANSFORM)
            list_images_generators.append(new_images_generator)

        elif TYPE_TRANSFORM_ELASTICDEFORM_IMAGES == 'pixelwise':
            new_images_generator = ElasticDeformPixelwiseImages(size_images,
                                                                fill_mode=TRANS_FILL_MODE_TRANSFORM)
            list_images_generators.append(new_images_generator)

        elif TYPE_TRANSFORM_ELASTICDEFORM_IMAGES == 'gridwiseGijs':
            new_images_generator = ElasticDeformGridwiseImagesGijs(size_images,
                                                                   fill_mode=TRANS_FILL_MODE_TRANSFORM)
            list_images_generators.append(new_images_generator)

        else:
            message = 'Wrong value for type of Elastic Deformations: %s' % (TYPE_TRANSFORM_ELASTICDEFORM_IMAGES)
            catch_error_exception(message)

    num_created_images_generators = len(list_images_generators)

    if num_created_images_generators == 0:
        return NullGenerator()
    elif num_created_images_generators == 1:
        return list_images_generators[0]
    else:
        # num_created_images_generators > 1:
        # combination of single image generators
        return CombinedImagesGenerator(list_images_generators)
