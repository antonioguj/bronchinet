
from typing import Tuple, Dict, Union, Any

from common.exceptionmanager import catch_error_exception
from postprocessing.imagereconstructor import ImageReconstructorWithGenerator
from preprocessing.filteringbordersimages import FilteringBordersImages2D, FilteringBordersImages3D
from preprocessing.preprocessing_manager import get_image_generator, fill_missing_trans_rigid_params


def get_image_reconstructor(size_images: Union[Tuple[int, int, int], Tuple[int, int]],
                            is_reconstruct_patches: bool,
                            type_reconstruct_patches: str,
                            prop_overlap_slide_images: Union[Tuple[float, float, float], Tuple[float, float]],
                            num_random_images: int,
                            is_transform_images: bool,
                            type_transform_images: str,
                            trans_rigid_params: Union[Dict[str, Any], None],
                            size_volume_images: Union[Tuple[int, int, int], Tuple[int, int]] = (0, 0, 0),
                            is_nnet_validconvs: bool = False,
                            size_output_images: Union[Tuple[int, int, int], Tuple[int, int]] = None,
                            is_filter_output_images: bool = False,
                            size_filter_output_images: Union[Tuple[int, int, int], Tuple[int, int]] = None
                            ) -> ImageReconstructorWithGenerator:

    trans_rigid_params = fill_missing_trans_rigid_params(trans_rigid_params)

    if type_reconstruct_patches not in ['slide_window', 'slicing']:
        message = 'Image Reconstructor only implemented with Image Generator as \'slide_window\', \'slicing\''
        catch_error_exception(message)
        image_generator = None
    else:
        image_generator = get_image_generator(size_images,
                                              is_generate_patches=is_reconstruct_patches,
                                              type_generate_patches=type_reconstruct_patches,
                                              prop_overlap_slide_images=prop_overlap_slide_images,
                                              num_random_images=num_random_images,
                                              is_transform_images=is_transform_images,
                                              type_transform_images=type_transform_images,
                                              trans_rotation_range=trans_rigid_params['rotation_range'],
                                              trans_shift_range=trans_rigid_params['shift_range'],
                                              trans_flip_dirs=trans_rigid_params['flip_dirs'],
                                              trans_zoom_range=trans_rigid_params['zoom_range'],
                                              trans_fill_mode=trans_rigid_params['fill_mode'],
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
            message = 'get_image_reconstructor:__init__: wrong \'ndims\': %s' % (ndims)
            catch_error_exception(message)
            filter_image_generator = None
    else:
        filter_image_generator = None

    if is_transform_images:
        message = 'Image Reconstructor with Image Transformation not implemented yet'
        catch_error_exception(message)
    else:
        return ImageReconstructorWithGenerator(size_images,
                                               image_generator,
                                               size_volume_image=size_volume_images,
                                               is_nnet_validconvs=is_nnet_validconvs,
                                               size_output_image=size_output_images,
                                               is_filter_output_image=is_filter_output_images,
                                               filter_image_generator=filter_image_generator)
