
from typing import Tuple, Dict, Union, Any

from common.exceptionmanager import catch_error_exception
from preprocessing.imagegenerator import ImageGenerator, NullGenerator, CombinedImagesGenerator
from preprocessing.randomwindowimages import RandomWindowImages, FixedCentralWindowImages
from preprocessing.slidingwindowimages import SlidingWindowImages, SlicingImages
from preprocessing.transformrigidimages import TransformRigidImages2D, TransformRigidImages3D
from preprocessing.elasticdeformimages import ElasticDeformGridwiseImagesImproved as ElasticDeformImages

LIST_AVAIL_GENERATE_PATCHES = ['slide_window', 'slicing', 'random_window', 'fixed_window']
LIST_AVAIL_TRANSFORM_IMAGES = ['rigid_trans', 'elastic_deform']


def get_image_generator(size_images: Union[Tuple[int, int, int], Tuple[int, int]],
                        is_generate_patches: bool,
                        type_generate_patches: str,
                        prop_overlap_slide_images: Union[Tuple[float, float, float], Tuple[float, float]],
                        num_random_images: int,
                        is_transform_images: bool,
                        type_transform_images: str,
                        trans_rotation_range: Union[Tuple[float, float, float], float],
                        trans_shift_range: Union[Tuple[float, float, float], Tuple[float, float]],
                        trans_flip_dirs: Union[Tuple[bool, bool, bool], Tuple[bool, bool]],
                        trans_zoom_range: Union[float, Tuple[float, float]],
                        trans_fill_mode: str,
                        size_volume_images: Union[Tuple[int, int, int], Tuple[int, int]] = (0, 0, 0)
                        ) -> ImageGenerator:
    if is_generate_patches:
        if type_generate_patches == 'slide_window':
            # generate patches by sliding-window...
            image_patch_generator = SlidingWindowImages(size_images,
                                                        prop_overlap_slide_images,
                                                        size_volume_images)

        elif type_generate_patches == 'slicing':
            # generate patches by slicing...
            image_patch_generator = SlicingImages(size_images,
                                                  size_volume_images)

        elif type_generate_patches == 'random_window':
            # generate patches by random cropping window...
            image_patch_generator = RandomWindowImages(size_images,
                                                       num_random_images,
                                                       size_volume_images)

        elif type_generate_patches == 'fixed_window':
            # generate patches by the central cropping window...
            image_patch_generator = FixedCentralWindowImages(size_images,
                                                             size_volume_images)

        else:
            message = 'Type Generate Patches not found: \'%s\'. Options available: \'%s\'' \
                      % (type_generate_patches, ', '.join(LIST_AVAIL_GENERATE_PATCHES))
            catch_error_exception(message)
            image_patch_generator = None
    else:
        image_patch_generator = None

    # ----------------------

    if is_transform_images:
        if type_transform_images == 'rigid_trans':
            # generate images by random rigid transformations...
            ndims = len(size_images)
            if ndims == 2:
                image_transform_generator = TransformRigidImages2D(size_images,
                                                                   rotation_range=trans_rotation_range[0],
                                                                   height_shift_range=trans_shift_range[0],
                                                                   width_shift_range=trans_shift_range[1],
                                                                   horizontal_flip=trans_flip_dirs[0],
                                                                   vertical_flip=trans_flip_dirs[1],
                                                                   zoom_range=trans_zoom_range,
                                                                   fill_mode=trans_fill_mode)
            elif ndims == 3:
                image_transform_generator = TransformRigidImages3D(size_images,
                                                                   rotation_xy_range=trans_rotation_range[0],
                                                                   rotation_xz_range=trans_rotation_range[1],
                                                                   rotation_yz_range=trans_rotation_range[2],
                                                                   height_shift_range=trans_shift_range[0],
                                                                   width_shift_range=trans_shift_range[1],
                                                                   depth_shift_range=trans_shift_range[2],
                                                                   horizontal_flip=trans_flip_dirs[0],
                                                                   vertical_flip=trans_flip_dirs[1],
                                                                   axialdir_flip=trans_flip_dirs[2],
                                                                   zoom_range=trans_zoom_range,
                                                                   fill_mode=trans_fill_mode)
            else:
                message = 'get_image_generator:__init__: wrong \'ndims\': %s' % (ndims)
                catch_error_exception(message)
                image_transform_generator = None

        elif type_transform_images == 'elastic_deform':
            # generate images by elastic deformations...
            image_transform_generator = ElasticDeformImages(size_images, fill_mode=trans_fill_mode)

        else:
            message = 'Type Transform Images not found: \'%s\'. Options available: \'%s\'' \
                      % (type_transform_images, ', '.join(LIST_AVAIL_TRANSFORM_IMAGES))
            catch_error_exception(message)
            image_transform_generator = None
    else:
        image_transform_generator = None

    # ----------------------

    if image_patch_generator is None and image_transform_generator is None:
        return NullGenerator()

    elif image_patch_generator is not None and image_transform_generator is not None:
        # combination of two image generators
        return CombinedImagesGenerator([image_patch_generator, image_transform_generator])

    elif image_patch_generator is not None:
        return image_patch_generator

    elif image_transform_generator is not None:
        return image_transform_generator


def fill_missing_trans_rigid_params(in_trans_params: Union[Dict[str, Any], None]) -> Dict[str, Any]:
    rotation_range_default = (0.0, 0.0, 0.0)
    shift_range_default = (0.0, 0.0, 0.0)
    flip_dirs_default = (False, False, False)
    zoom_range_default = 0.0
    fill_mode_default = 'nearest'

    if in_trans_params is None:
        in_trans_params = {}

    in_trans_params_keys = in_trans_params.keys()

    if 'rotation_range' not in in_trans_params_keys:
        in_trans_params['rotation_range'] = rotation_range_default
    if 'shift_range' not in in_trans_params_keys:
        in_trans_params['shift_range'] = shift_range_default
    if 'flip_dirs' not in in_trans_params_keys:
        in_trans_params['flip_dirs'] = flip_dirs_default
    if 'zoom_range' not in in_trans_params_keys:
        in_trans_params['zoom_range'] = zoom_range_default
    if 'fill_mode' not in in_trans_params_keys:
        in_trans_params['fill_mode'] = fill_mode_default

    return in_trans_params
