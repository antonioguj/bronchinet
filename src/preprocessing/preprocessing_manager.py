
from typing import Tuple, Dict, Union, Any

from common.exceptionmanager import catch_error_exception
from preprocessing.imagegenerator import ImageGenerator, NullGenerator, CombinedImagesGenerator
from preprocessing.randomwindowimages import RandomWindowImages
from preprocessing.slidingwindowimages import SlidingWindowImages
from preprocessing.transformrigidimages import TransformRigidImages2D, TransformRigidImages3D
from preprocessing.elasticdeformimages import ElasticDeformGridwiseImages, ElasticDeformPixelwiseImages, \
    ElasticDeformGridwiseImagesGijs


def get_images_generator(size_images: Union[Tuple[int, int, int], Tuple[int, int]],
                         is_sliding_window: bool,
                         prop_overlap_slide_images: Union[Tuple[float, float, float], Tuple[float, float]],
                         is_random_window: bool,
                         num_random_images: int,
                         is_transform_rigid: bool,
                         trans_rotation_range: Union[Tuple[float, float, float], float],
                         trans_shift_range: Union[Tuple[float, float, float], Tuple[float, float]],
                         trans_flip_dirs: Union[Tuple[bool, bool, bool], Tuple[bool, bool]],
                         trans_zoom_range: Union[float, Tuple[float, float]],
                         trans_fill_mode: str,
                         is_transform_elastic: bool,
                         type_trans_elastic: str,
                         size_volume_images: Union[Tuple[int, int, int], Tuple[int, int]] = (0, 0, 0)
                         ) -> ImageGenerator:
    list_images_generators = []

    if is_sliding_window:
        # generator of image patches by sliding-window...
        new_images_generator = SlidingWindowImages(size_images,
                                                   prop_overlap_slide_images,
                                                   size_volume_images)
        list_images_generators.append(new_images_generator)

    elif is_random_window:
        # generator of image patches by random cropping window...
        new_images_generator = RandomWindowImages(size_images,
                                                  num_random_images,
                                                  size_volume_images)
        list_images_generators.append(new_images_generator)

    if is_transform_rigid:
        # generator of images by random rigid transformations of input images...
        ndims = len(size_images)
        if ndims == 2:
            new_images_generator = TransformRigidImages2D(size_images,
                                                          rotation_range=trans_rotation_range[0],
                                                          height_shift_range=trans_shift_range[0],
                                                          width_shift_range=trans_shift_range[1],
                                                          horizontal_flip=trans_flip_dirs[0],
                                                          vertical_flip=trans_flip_dirs[1],
                                                          zoom_range=trans_zoom_range,
                                                          fill_mode=trans_fill_mode)
            list_images_generators.append(new_images_generator)

        elif ndims == 3:
            new_images_generator = TransformRigidImages3D(size_images,
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
            list_images_generators.append(new_images_generator)

        else:
            message = 'get_images_generator:__init__: wrong \'ndims\': %s' % (ndims)
            catch_error_exception(message)

    if is_transform_elastic:
        if type_trans_elastic == 'gridwise':
            new_images_generator = ElasticDeformGridwiseImages(size_images, fill_mode=trans_fill_mode)
            list_images_generators.append(new_images_generator)

        elif type_trans_elastic == 'pixelwise':
            new_images_generator = ElasticDeformPixelwiseImages(size_images, fill_mode=trans_fill_mode)
            list_images_generators.append(new_images_generator)

        elif type_trans_elastic == 'gridwiseGijs':
            new_images_generator = ElasticDeformGridwiseImagesGijs(size_images, fill_mode=trans_fill_mode)
            list_images_generators.append(new_images_generator)

        else:
            message = 'Wrong value for type of Elastic Deformations: %s' % (type_trans_elastic)
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
