
from typing import Tuple

from common.exceptionmanager import catch_error_exception
from postprocessing.imagereconstructor import ImageReconstructor, ImageReconstructorWithTransformation
from preprocessing.filternnetoutput_validconvs import FilteringNnetOutputValidConvs2D, FilteringNnetOutputValidConvs3D
from preprocessing.preprocessing_manager import get_images_generator


def get_images_reconstructor(size_images: Tuple[int, ...],
                             use_sliding_window_images: bool,
                             prop_overlap_slide_window: Tuple[int, ...],
                             use_random_window_images: bool,
                             num_random_patches_epoch: int,
                             use_transform_rigid_images: bool = False,
                             use_transform_elastic_images: bool = False,
                             size_volume_image: Tuple[int, ...] = (0, 0, 0),
                             is_nnet_validconvs: bool = False,
                             size_output_image: Tuple[int, ...] = None,
                             is_filter_output_nnet: bool = False,
                             prop_filter_output_nnet: Tuple[float, ...] = None,
                             num_trans_per_sample: int = 1
                             ) -> ImageReconstructor:
    images_generator = get_images_generator(size_images,
                                            use_sliding_window_images=use_sliding_window_images,
                                            prop_overlap_slide_window=prop_overlap_slide_window,
                                            use_random_window_images=use_random_window_images,
                                            num_random_patches_epoch=num_random_patches_epoch,
                                            use_transform_rigid_images=use_transform_rigid_images,
                                            use_transform_elastic_images=use_transform_elastic_images,
                                            size_volume_image=size_volume_image)

    if is_filter_output_nnet:
        size_filter_output_nnet = tuple([int(prop_filter_output_nnet * elem) for elem in size_images])
        print('Filter output probability maps of Nnet, with final output size: \'%s\'...'
              % (str(size_filter_output_nnet)))

        ndims = len(size_images)
        if ndims == 2:
            filter_image_generator = FilteringNnetOutputValidConvs2D(size_images, size_filter_output_nnet)
        elif ndims == 3:
            filter_image_generator = FilteringNnetOutputValidConvs3D(size_images, size_filter_output_nnet)
        else:
            message = 'get_images_reconstructor:__init__: wrong \'ndims\': %s...' % (ndims)
            catch_error_exception(message)
    else:
        filter_image_generator = None

    if not use_sliding_window_images and not use_random_window_images:
        message = 'Image Reconstructor without Sliding-window generation of Image patches not implemented yet'
        catch_error_exception(message)

    if not use_transform_rigid_images:
        # reconstructor of images following the sliding-window generator of input patches
        images_reconstructor = ImageReconstructor(size_images,
                                                  images_generator,
                                                  size_volume_image=size_volume_image,
                                                  is_nnet_validconvs=is_nnet_validconvs,
                                                  size_output_image=size_output_image,
                                                  is_filter_output_nnet=is_filter_output_nnet,
                                                  filter_image_generator=filter_image_generator)
    else:
        # reconstructor of images accounting for transformations during testing (PROTOTYPE, NOT TESTED YET)
        images_reconstructor = ImageReconstructorWithTransformation(size_images,
                                                                    images_generator,
                                                                    num_trans_per_patch=num_trans_per_sample,
                                                                    size_volume_image=size_volume_image,
                                                                    is_nnet_validconvs=is_nnet_validconvs,
                                                                    size_output_image=size_output_image,
                                                                    is_filter_output_nnet=is_filter_output_nnet,
                                                                    filter_image_generator=filter_image_generator)
    return images_reconstructor
