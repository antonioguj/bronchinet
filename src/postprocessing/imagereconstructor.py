
from typing import Tuple, Union
import numpy as np

from common.exceptionmanager import catch_error_exception, catch_warning_exception
from common.functionutil import ImagesUtil
from imageoperators.boundingboxes import BoundingBoxes, BoundBox3DType, BoundBox2DType
from imageoperators.imageoperator import ExtendImage, SetImageInVolume, CropImage
from preprocessing.filteringbordersimages import FilteringBordersImages
from preprocessing.slidingwindowimages import SlidingWindowImages
from preprocessing.randomwindowimages import RandomWindowImages


class ImageReconstructorGeneral(object):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 size_volume_image: Union[Tuple[int, int, int], Tuple[int, int]] = (0, 0, 0),
                 type_combine_patches: str = 'average',
                 ) -> None:
        self._size_image = size_image
        self._ndims = len(size_image)
        self._size_volume_image = size_volume_image
        self._type_combine_patches = type_combine_patches

        if self._ndims == 2:
            self._func_add_images_volume = SetImageInVolume._compute_adding2d
            self._func_crop_images = CropImage._compute2d
        elif self._ndims == 3:
            self._func_add_images_volume = SetImageInVolume._compute_adding3d
            self._func_crop_images = CropImage._compute3d
        else:
            message = 'ImageReconstructorGeneral:__init__: wrong \'ndims\': %s' % (self._ndims)
            catch_error_exception(message)

        if self._type_combine_patches not in ['average', 'max']:
            message = '\'type_combine_patches\' must be either \'average\' or \'max\'... Received: \'%s\'' \
                      % (self._type_combine_patches)
            catch_error_exception(message)

        self._initialize_data()

    def _initialize_data(self) -> None:
        self._size_volume_image = None
        self._reconstructed_image = None
        self._reconstructed_factor_overlap = None

    def get_reconstructed_image(self) -> np.ndarray:
        return self._reconstructed_image

    def get_reconstructed_factor_overlap(self) -> np.ndarray:
        return self._reconstructed_factor_overlap

    def initialize_recons_data(self, in_shape_image_volume: Tuple[int, ...]) -> None:
        self._size_volume_image = in_shape_image_volume[0:self._ndims]

    def initialize_recons_array(self, in_image_patch_example: np.ndarray) -> None:
        shape_volume_image = self._get_shape_output_image(in_image_patch_example.shape, self._size_volume_image)
        self._reconstructed_image = np.zeros(shape_volume_image, dtype=in_image_patch_example.dtype)

        if self._type_combine_patches == 'average':
            # normalizing factor to account for the overlap between the sliding-window patches
            self._reconstructed_factor_overlap = np.zeros(self._size_volume_image, dtype=np.float32)

    def include_image_patch(self, in_image: np.ndarray,
                            in_setadd_boundbox: Union[BoundBox3DType, BoundBox2DType]
                            ) -> None:
        if self._type_combine_patches == 'average':
            self._include_image_patch_type_average(in_image, in_setadd_boundbox)

        elif self._type_combine_patches == 'max':
            self._include_image_patch_type_max(in_image, in_setadd_boundbox)

    def include_image_patch_with_checks(self, in_image: np.ndarray,
                                        in_setadd_boundbox: Union[BoundBox3DType, BoundBox2DType]
                                        ) -> None:
        size_setadd_boundbox = BoundingBoxes.get_size_boundbox(in_setadd_boundbox)

        if not BoundingBoxes.is_boundbox_inside_image_size(in_setadd_boundbox, self._size_volume_image):
            print("Set-add bounding-box is not contained in the size of reconstructed image: \'%s\' > \'%s\'. "
                  "Crop images before adding patch..." % (str(size_setadd_boundbox), str(self._size_volume_image)))

            (in_crop_boundbox, in_setadd_boundbox) = \
                BoundingBoxes.calc_boundboxes_crop_extend_image_reverse(in_setadd_boundbox, self._size_volume_image)
            print("Crop input image to bounding-box: \'%s\', and then Set in reconstructed image with "
                  "bounding-box: \'%s\'.." % (str(in_crop_boundbox), str(in_setadd_boundbox)))

            in_image = self._func_crop_images(in_image, in_crop_boundbox)

        self.include_image_patch(in_image, in_setadd_boundbox)

    def _include_image_patch_type_average(self, in_image: np.ndarray,
                                          in_setadd_boundbox: Union[BoundBox3DType, BoundBox2DType]
                                          ) -> None:
        # set input image patch in reconstructed image
        self._func_add_images_volume(in_image, self._reconstructed_image, in_setadd_boundbox)

        # set new patch of 'ones' in the full-size 'factor_overlap' array
        factor_overlap_patch = self._get_factor_overlap_patch(in_image.shape)
        self._func_add_images_volume(factor_overlap_patch, self._reconstructed_factor_overlap, in_setadd_boundbox)

    def _include_image_patch_type_max(self, in_image: np.ndarray,
                                      in_setadd_boundbox: Union[BoundBox3DType, BoundBox2DType]
                                      ) -> None:
        # compare input image with a patch from reconstructed image at the same location (bounding-box)
        patch_recons_image_same_boundbox = self._func_crop_images(self._reconstructed_image, in_setadd_boundbox)
        # compute the element-wise maximum of the two images
        in_image_calcmax_recons_image = np.maximum(in_image, patch_recons_image_same_boundbox)
        # set element-wise maximum image in reconstructed image
        self._func_add_images_volume(in_image_calcmax_recons_image, self._reconstructed_image, in_setadd_boundbox)

    def _get_factor_overlap_patch(self, in_shape_image_patch: Tuple[int, ...]) -> np.ndarray:
        size_input_patch = in_shape_image_patch[0:self._ndims]
        return np.ones(size_input_patch, dtype=np.float32)

    def finalize_recons_array(self) -> None:
        out_shape_recons_image = self._reconstructed_image.shape

        if self._type_combine_patches == 'average':
            # compute the inverse of full-size 'factor_overlap' array, now containing the number of per-voxel overlaps
            # - prevent the division by zero where there were no patches processed, by setting a very large overlap
            val_infinity_overlap = 1.0e+010
            self._reconstructed_factor_overlap = np.where(self._reconstructed_factor_overlap == 0.0,
                                                          val_infinity_overlap,
                                                          self._reconstructed_factor_overlap)
            self._reconstructed_factor_overlap = np.reciprocal(self._reconstructed_factor_overlap)

            # compute the reconstructed image by multiplying voxel-wise with the 'factor_overlap'
            if ImagesUtil.is_without_channels(self._size_image, out_shape_recons_image):
                self._reconstructed_image = np.multiply(self._reconstructed_image, self._reconstructed_factor_overlap)
            else:
                # reconstructed image with several channels -> multiply each channel with the 'factor_overlap'
                if self._ndims == 2:
                    self._reconstructed_image = \
                        np.einsum('ijk,ij->ijk', self._reconstructed_image, self._reconstructed_factor_overlap)
                elif self._ndims == 3:
                    self._reconstructed_image = \
                        np.einsum('ijkl,ijk->ijkl', self._reconstructed_image, self._reconstructed_factor_overlap)

        self._reconstructed_image = self._get_reshaped_output_image(self._reconstructed_image)

    def _get_shape_output_image(self, in_shape_image: Tuple[int, ...],
                                out_size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                                ) -> Tuple[int, ...]:
        if ImagesUtil.is_without_channels(self._size_image, in_shape_image):
            return tuple(out_size_image)
        else:
            num_channels = ImagesUtil.get_num_channels(self._size_image, in_shape_image)
            return tuple(out_size_image) + (num_channels,)

    def _get_reshaped_input_image(self, in_image: np.ndarray) -> np.ndarray:
        if ImagesUtil.is_without_channels(self._size_image, in_image.shape[1:]):
            return in_image
        else:
            return np.expand_dims(in_image, axis=-1)

    def _get_reshaped_output_image(self, in_image: np.ndarray) -> np.ndarray:
        num_channels = ImagesUtil.get_num_channels(self._size_image, in_image.shape)
        if num_channels == 1:
            return np.squeeze(in_image, axis=-1)
        else:
            return in_image


class ImageReconstructorWithGenerator(ImageReconstructorGeneral):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 image_patch_generator: Union[SlidingWindowImages, RandomWindowImages],
                 size_volume_image: Union[Tuple[int, int, int], Tuple[int, int]] = (0, 0, 0),
                 is_nnet_validconvs: bool = False,
                 size_output_image: Union[Tuple[int, int, int], Tuple[int, int]] = None,
                 is_filter_output_image: bool = False,
                 filter_image_generator: Union[FilteringBordersImages, None] = None,
                 type_combine_patches: str = 'average'
                 ) -> None:
        super(ImageReconstructorWithGenerator, self).__init__(size_image, size_volume_image,
                                                              type_combine_patches)
        self._image_patch_generator = image_patch_generator

        self._is_nnet_validconvs = is_nnet_validconvs
        if is_nnet_validconvs and size_output_image and (size_image != size_output_image):
            self._size_output_image = size_output_image

            self._extend_boundbox_out_nnet_validconvs = \
                BoundingBoxes.calc_boundbox_centered_image_fitimg(self._size_output_image, self._size_image)

            if self._ndims == 2:
                self._func_extend_images = ExtendImage._compute2d
            elif self._ndims == 3:
                self._func_extend_images = ExtendImage._compute3d
            else:
                message = 'ImageReconstructorWithGenerator:__init__: wrong \'ndims\': %s' % (self._ndims)
                catch_error_exception(message)
        else:
            self._is_nnet_validconvs = False

            if not is_filter_output_image:
                message = 'For networks with non-valid convols, better to filter the output to reduce border effects'
                catch_warning_exception(message)

        self._is_filter_output_nnet = is_filter_output_image
        if is_filter_output_image:
            self._filter_image_generator = filter_image_generator

        self._initialize_data()

    def _initialize_data(self) -> None:
        super(ImageReconstructorWithGenerator, self)._initialize_data()
        self._num_patches_total = None

    def initialize_recons_data(self, in_shape_image_volume: Tuple[int, ...]) -> None:
        self._size_volume_image = in_shape_image_volume[0:self._ndims]
        self._image_patch_generator.update_image_data(in_shape_image_volume)
        self._num_patches_total = self._image_patch_generator.get_num_images()

    def get_include_image_patch(self, in_image: np.ndarray, index: int) -> None:
        in_setadd_boundbox = self._image_patch_generator._get_crop_boundbox_image(index)
        in_processed_image = self._get_processed_image_patch(in_image)
        super(ImageReconstructorWithGenerator, self).include_image_patch(in_processed_image, in_setadd_boundbox)

    def _get_processed_image_patch(self, in_image: np.ndarray) -> np.ndarray:
        if self._is_nnet_validconvs:
            size_output_image = self._get_shape_output_image(in_image.shape, self._size_image)
            out_image = self._func_extend_images(in_image, self._extend_boundbox_out_nnet_validconvs,
                                                 size_output_image, value_backgrnd=0.0)
        else:
            out_image = in_image

        if self._is_filter_output_nnet:
            out_image = self._filter_image_generator._get_image(out_image)

        return out_image

    def _get_factor_overlap_patch(self, in_shape_image_patch: Tuple[int, ...]) -> np.ndarray:
        if self._is_nnet_validconvs:
            shape_factor_overlap_patch = self._size_output_image
        else:
            shape_factor_overlap_patch = self._size_image

        out_factor_overlap_patch = np.ones(shape_factor_overlap_patch, dtype=np.float32)

        if self._is_nnet_validconvs:
            return self._func_extend_images(out_factor_overlap_patch, self._extend_boundbox_out_nnet_validconvs,
                                            self._size_image, value_backgrnd=0.0)

        if self._is_filter_output_nnet:
            out_factor_overlap_patch = self._filter_image_generator._get_image(out_factor_overlap_patch)

        return out_factor_overlap_patch

    def compute_full(self, in_images_all: np.ndarray) -> np.ndarray:
        if not self._check_correct_shape_input_image(in_images_all.shape):
            message = "Wrong shape of input data to be reconstructed: \'%s\' " % (str(in_images_all.shape))
            catch_error_exception(message)

        self.initialize_recons_array(in_images_all[0])

        for index in range(self._num_patches_total):
            self.get_include_image_patch(in_images_all[index], index)

        self.finalize_recons_array()

        return self._reconstructed_image

    def _check_correct_shape_input_image(self, in_shape_image: Tuple[int, ...]) -> bool:
        check1 = len(in_shape_image) == self._ndims + 2
        check2 = in_shape_image[0] == self._num_patches_total
        if self._is_nnet_validconvs:
            check3 = in_shape_image[1:-2] != self._size_output_image
        else:
            check3 = in_shape_image[1:-2] != self._size_image

        return check1 and check2 and check3


# class ImageReconstructorWithTransformation(ImageReconstructorWithGenerator):
#     # WATCH OUT -> PROTOTYPE OF RECONSTRUCTOR WITH TRANSFORMATION AT TESTING TIME. NOT TESTED YET !!!
#     def __init__(self,
#                  size_image: Union[Tuple[int, int, int], Tuple[int, int]],
#                  image_transform_generator: TransformRigidImages,
#                  num_trans_per_patch: int = 1,
#                  size_volume_image: Union[Tuple[int, int, int], Tuple[int, int]] = (0, 0, 0),
#                  is_nnet_validconvs: bool = False,
#                  size_output_image: Union[Tuple[int, int, int], Tuple[int, int]] = None,
#                  is_filter_output_nnet: bool = False,
#                  filter_image_generator: Union[FilterNnetOutputValidConvs, None] = None
#                  ) -> None:
#         super(ImageReconstructorWithTransformation, self).__init__(size_image,
#                                                                    image_transform_generator,
#                                                                    size_volume_image=size_volume_image,
#                                                                    is_nnet_validconvs=is_nnet_validconvs,
#                                                                    size_output_image=size_output_image,
#                                                                    is_filter_output_nnet=is_filter_output_nnet,
#                                                                    filter_image_generator=filter_image_generator)
#
#         # NEED SOMETHING LIKE THIS TO SET UP THE SAME TRANSFORMATION FOR ALL REPEATED TRANSFORMS PER PATCH
#         self._image_transform_generator = image_transform_generator
#         self._num_trans_per_patch = num_trans_per_patch
#
#     def _get_processed_image_patch(self, in_image: np.ndarray) -> np.ndarray:
#         # SEARCH FOR NUMPY FUNCTION TO DO SUMATION DIRECTLY
#         sumrun_images = in_image[0]
#         for i in range(1, self._num_trans_per_patch):
#             sumrun_images += self._image_transform_generator._get_inverse_transformed_image(in_image[i])
#
#         out_image = np.divide(sumrun_images, self._num_trans_per_patch)
#
#         # Apply rest of processing operations from parent class
#         out_image = super(ImageReconstructorWithTransformation, self)._get_processed_image_patch(out_image)
#         return out_image
#
#     def _check_correct_shape_input_image(self, in_shape_image: Tuple[int, ...]) -> bool:
#         check1 = len(in_shape_image) == self._ndims + 3
#         check2 = in_shape_image[0] == self._num_patches_total
#         check3 = in_shape_image[1] == self._num_trans_per_patch
#         if self._is_nnet_validconvs:
#             check4 = in_shape_image[2:-2] != self._size_output_image
#         else:
#             check4 = in_shape_image[2:-2] != self._size_image
#
#         return check1 and check2 and check3 and check4
