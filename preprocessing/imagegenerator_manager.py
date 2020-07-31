
from typing import Tuple

from common.constant import ROTATION_XY_RANGE, ROTATION_XZ_RANGE, ROTATION_YZ_RANGE, HEIGHT_SHIFT_RANGE, WIDTH_SHIFT_RANGE, \
                            DEPTH_SHIFT_RANGE, HORIZONTAL_FLIP, VERTICAL_FLIP, AXIALDIR_FLIP,  ZOOM_RANGE, FILL_MODE_TRANSFORM, \
                            TYPETRANSFORMELASTICDEFORMATION
from common.exception_manager import catch_error_exception
from preprocessing.imagegenerator import ImageGenerator, NullGenerator, CombinedImagesGenerator
from preprocessing.randomwindowimages import RandomWindowImages
from preprocessing.slidingwindowimages import SlidingWindowImages
from preprocessing.transformrigidimages import TransformRigidImages2D, TransformRigidImages3D
from preprocessing.elasticdeformimages import ElasticDeformGridwiseImages, ElasticDeformPixelwiseImages


def get_images_generator(size_images: Tuple[int, ...],
                         use_sliding_window_images: bool,
                         slide_window_prop_overlap: Tuple[int, ...],
                         use_random_window_images: bool,
                         num_random_patches_epoch: int,
                         use_transform_rigid_images: bool,
                         use_elasticdeform_images: bool,
                         size_volume_image: Tuple[int, ...] = (0,)
                         ) -> ImageGenerator:
    list_images_generators = []

    if (use_sliding_window_images):
        # generator of image patches by sliding-window...
        new_images_generator = SlidingWindowImages(size_images,
                                                   slide_window_prop_overlap,
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
        if ndims==2:
            new_images_generator = TransformRigidImages2D(size_images,
                                                          rotation_range=ROTATION_XY_RANGE,
                                                          height_shift_range=HEIGHT_SHIFT_RANGE,
                                                          width_shift_range=WIDTH_SHIFT_RANGE,
                                                          horizontal_flip=HORIZONTAL_FLIP,
                                                          vertical_flip=VERTICAL_FLIP,
                                                          zoom_range=ZOOM_RANGE,
                                                          fill_mode=FILL_MODE_TRANSFORM)
            list_images_generators.append(new_images_generator)
        elif ndims==3:
            new_images_generator = TransformRigidImages3D(size_images,
                                                          rotation_XY_range=ROTATION_XY_RANGE,
                                                          rotation_XZ_range=ROTATION_XZ_RANGE,
                                                          rotation_YZ_range=ROTATION_YZ_RANGE,
                                                          height_shift_range=HEIGHT_SHIFT_RANGE,
                                                          width_shift_range=WIDTH_SHIFT_RANGE,
                                                          depth_shift_range=DEPTH_SHIFT_RANGE,
                                                          horizontal_flip=HORIZONTAL_FLIP,
                                                          vertical_flip=VERTICAL_FLIP,
                                                          axialdir_flip=AXIALDIR_FLIP,
                                                          zoom_range=ZOOM_RANGE,
                                                          fill_mode=FILL_MODE_TRANSFORM)
            list_images_generators.append(new_images_generator)
        else:
            message = 'Wrong value of \'ndims\': %s' % (ndims)
            catch_error_exception(message)

    if use_elasticdeform_images:
        if TYPETRANSFORMELASTICDEFORMATION == 'pixelwise':
            new_images_generator = ElasticDeformPixelwiseImages(size_images)
            list_images_generators.append(new_images_generator)

        else: #TYPETRANSFORMELASTICDEFORMATION == 'gridwise'
            new_images_generator = ElasticDeformGridwiseImages(size_images)
            list_images_generators.append(new_images_generator)


    num_created_images_generators = len(list_images_generators)

    if num_created_images_generators == 0:
        return NullGenerator()
    elif num_created_images_generators == 1:
        return list_images_generators[0]
    else: #num_created_images_generators > 1:
        # combination of single image generators
        return CombinedImagesGenerator(list_images_generators)