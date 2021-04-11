
from typing import Tuple, Dict, Union, Any

from common.exceptionmanager import catch_error_exception
from postprocessing.imagereconstructor import ImageReconstructor, ImageReconstructorWithTransformation
from preprocessing.filternnetoutput_validconvs import FilteringNnetOutputValidConvs2D, FilteringNnetOutputValidConvs3D
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
                             num_trans_per_sample: int = 1,
                             size_volume_images: Union[Tuple[int, int, int], Tuple[int, int]] = (0, 0, 0),
                             is_nnet_validconvs: bool = False,
                             size_output_images: Union[Tuple[int, int, int], Tuple[int, int]] = None,
                             is_filter_output_nnet: bool = False,
                             prop_filter_output_nnet: float = None,
                             ) -> ImageReconstructor:
    if not is_sliding_window:
        message = 'Image Reconstructor without Sliding-window generation of Image patches not implemented yet'
        catch_error_exception(message)

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

    if is_filter_output_nnet:
        ndims = len(size_images)
        if ndims == 2:
            size_filter_output_nnet = (int(prop_filter_output_nnet * size_images[0]),
                                       int(prop_filter_output_nnet * size_images[1]))
            print("Filter output probability maps of Nnet, with final output size: \'%s\'..."
                  % (str(size_filter_output_nnet)))

            filter_image_generator = FilteringNnetOutputValidConvs2D(size_images, size_filter_output_nnet)

        elif ndims == 3:
            size_filter_output_nnet = (int(prop_filter_output_nnet * size_images[0]),
                                       int(prop_filter_output_nnet * size_images[1]),
                                       int(prop_filter_output_nnet * size_images[2]))
            print("Filter output probability maps of Nnet, with final output size: \'%s\'..."
                  % (str(size_filter_output_nnet)))

            filter_image_generator = FilteringNnetOutputValidConvs3D(size_images, size_filter_output_nnet)
        else:
            message = 'get_images_reconstructor:__init__: wrong \'ndims\': %s...' % (ndims)
            catch_error_exception(message)
            filter_image_generator = None
    else:
        filter_image_generator = None

    if not (is_transform_rigid or is_transform_elastic):
        # reconstructor of images following the sliding-window generator of input patches
        images_reconstructor = ImageReconstructor(size_images,
                                                  images_generator,
                                                  size_volume_image=size_volume_images,
                                                  is_nnet_validconvs=is_nnet_validconvs,
                                                  size_output_image=size_output_images,
                                                  is_filter_output_nnet=is_filter_output_nnet,
                                                  filter_image_generator=filter_image_generator)
    else:
        # reconstructor of images accounting for transformations during testing (PROTOTYPE, NOT TESTED YET)
        images_reconstructor = ImageReconstructorWithTransformation(size_images,
                                                                    images_generator,
                                                                    num_trans_per_patch=num_trans_per_sample,
                                                                    size_volume_image=size_volume_images,
                                                                    is_nnet_validconvs=is_nnet_validconvs,
                                                                    size_output_image=size_output_images,
                                                                    is_filter_output_nnet=is_filter_output_nnet,
                                                                    filter_image_generator=filter_image_generator)
    return images_reconstructor
