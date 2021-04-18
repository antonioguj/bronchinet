
from typing import Tuple, Dict, Union, Any

from common.exceptionmanager import catch_error_exception
from postprocessing.imagereconstructor import ImageReconstructorWithGenerator
from preprocessing.filteringbordersimages import FilteringBordersImages2D, FilteringBordersImages3D
from preprocessing.preprocessing_manager import get_images_generator, fill_missing_trans_rigid_params


def get_images_reconstructor(size_images: Union[Tuple[int, int, int], Tuple[int, int]],
                             is_sliding_window: bool,
                             prop_overlap_slide_images: Union[Tuple[float, float, float], Tuple[float, float]],
                             is_random_window: bool,
                             num_random_images: int,
                             is_transform_rigid: bool,
                             trans_rigid_params: Union[Dict[str, Any], None],
                             is_transform_elastic: bool,
                             type_trans_elastic: str,
                             size_volume_images: Union[Tuple[int, int, int], Tuple[int, int]] = (0, 0, 0),
                             is_nnet_validconvs: bool = False,
                             size_output_images: Union[Tuple[int, int, int], Tuple[int, int]] = None,
                             is_filter_output_images: bool = False,
                             size_filter_output_images: Union[Tuple[int, int, int], Tuple[int, int]] = None
                             ) -> ImageReconstructorWithGenerator:

    trans_rigid_params = fill_missing_trans_rigid_params(trans_rigid_params)

    images_generator = get_images_generator(size_images,
                                            is_sliding_window=is_sliding_window,
                                            prop_overlap_slide_images=prop_overlap_slide_images,
                                            is_random_window=is_random_window,
                                            num_random_images=num_random_images,
                                            is_transform_rigid=is_transform_rigid,
                                            trans_rotation_range=trans_rigid_params['rotation_range'],
                                            trans_shift_range=trans_rigid_params['shift_range'],
                                            trans_flip_dirs=trans_rigid_params['flip_dirs'],
                                            trans_zoom_range=trans_rigid_params['zoom_range'],
                                            trans_fill_mode=trans_rigid_params['fill_mode'],
                                            is_transform_elastic=is_transform_elastic,
                                            type_trans_elastic=type_trans_elastic,
                                            size_volume_images=size_volume_images)

    if is_filter_output_images:
        ndims = len(size_images)
        if ndims == 2:
            filter_image_generator = FilteringBordersImages2D(size_images,
                                                              size_filter_output_images)
        elif ndims == 3:
            filter_image_generator = FilteringBordersImages3D(size_images,
                                                              size_filter_output_images)
        else:
            message = 'get_images_reconstructor:__init__: wrong \'ndims\': %s' % (ndims)
            catch_error_exception(message)
            filter_image_generator = None
    else:
        filter_image_generator = None

    if not (is_transform_rigid or is_transform_elastic):
        # reconstructor of images following the sliding-window generator of input patches
        images_reconstructor = ImageReconstructorWithGenerator(size_images,
                                                               images_generator,
                                                               size_volume_image=size_volume_images,
                                                               is_nnet_validconvs=is_nnet_validconvs,
                                                               size_output_image=size_output_images,
                                                               is_filter_output_image=is_filter_output_images,
                                                               filter_image_generator=filter_image_generator)
    else:
        message = 'Image Reconstructor with Image Transformations not implemented yet'
        catch_error_exception(message)
        images_reconstructor = None

    return images_reconstructor
