
from typing import Tuple
import numpy as np

from common.exceptionmanager import catch_error_exception
from common.functionutil import ImagesUtil
from imageoperators.boundingboxes import BoundingBoxes
from imageoperators.imageoperator import ExtendImage
from preprocessing.imagegenerator import ImageGenerator
from preprocessing.transformrigidimages import TransformRigidImages


class ImageReconstructor(object):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 image_generator: ImageGenerator,
                 size_volume_image: Tuple[int, ...] = (0,),
                 is_output_nnet_validconvs: bool = False,
                 size_output_image: Tuple[int, ...] = None,
                 is_filter_output_nnet: bool = False,
                 filter_image_generator: ImageGenerator = None
                 ) -> None:
        self._size_image = size_image
        self._ndims = len(size_image)
        self._image_generator = image_generator

        if np.isscalar(size_volume_image):
            self._size_volume_image = tuple([size_volume_image] * self._ndims)
        else:
            self._size_volume_image = size_volume_image

        self._is_output_nnet_validconvs = is_output_nnet_validconvs
        if is_output_nnet_validconvs and size_output_image and (size_image != size_output_image):
            self._size_output_image = size_output_image
            self._valid_output_bounding_box = BoundingBoxes.compute_bounding_box_centered_image_fit_image(self._size_output_image,
                                                                                                          self._size_image)
            if self._ndims==2:
                self._func_extend_image_patch = ExtendImage._compute2D
            elif self._ndims==3:
                self._func_extend_image_patch = ExtendImage._compute3D
            else:
                message = 'ImageReconstructor:__init__: wrong \'ndims\': %s...' % (self._ndims)
                catch_error_exception(message)
        else:
            self._is_output_nnet_validconvs = False

        self._is_filter_output_nnet = is_filter_output_nnet
        if is_filter_output_nnet:
            self._filter_image_generator = filter_image_generator

        self._num_patches_total = self._image_generator.get_num_images()

    def update_image_data(self, in_shape_image: Tuple[int, ...], is_compute_normfact: bool = True) -> None:
        self._update_image_data_step1(in_shape_image)
        if is_compute_normfact:
            self._update_image_data_step2()

    def _update_image_data_step1(self, in_shape_image: Tuple[int, ...]) -> None:
        self._size_volume_image = in_shape_image[0:self._ndims]
        self._image_generator.update_image_data(in_shape_image)
        self._num_patches_total = self._image_generator.get_num_images()

    def _update_image_data_step2(self) -> None:
        self._normfact_overlap_image_patches = self._compute_normfact_overlap_image_patches()

    def _get_processed_image_patch(self, in_image: np.ndarray) -> np.ndarray:
        if self._is_output_nnet_validconvs:
            out_shape_image = self._get_shape_output_image(in_image.shape, self._size_image)
            out_image = self._func_extend_image_patch(in_image, self._valid_output_bounding_box, out_shape_image, background_value=0)
        else:
            out_image = in_image

        if self._is_filter_output_nnet:
            out_image = self._filter_image_generator._get_image(out_image)

        return out_image

    def compute(self, in_image_patches: np.ndarray) -> np.ndarray:
        if not self._check_correct_shape_input_image(in_image_patches.shape):
            message = "Wrong shape of input predictions data: \'%s\' " % (str(in_image_patches.shape))
            catch_error_exception(message)

        out_shape_image = self._get_shape_output_image(in_image_patches.shape, self._size_volume_image)
        out_reconstructed_image = np.zeros(out_shape_image, dtype=in_image_patches.dtype)

        for index in range(self._num_patches_total):
            in_image_patch = self._get_processed_image_patch(in_image_patches[index])
            self._image_generator.set_add_image_patch(in_image_patch, out_reconstructed_image, index)

        out_reconstructed_image = self._multiply_matrixes_with_channels(out_reconstructed_image,
                                                                        self._normfact_overlap_image_patches)
        return self._get_reshaped_output_image(out_reconstructed_image)

    def _multiply_matrixes_with_channels(self, matrix_1_withchannels: np.ndarray, matrix_2: np.ndarray) -> np.ndarray:
        if self._ndims==2:
            return np.einsum('ijk,ij->ijk', matrix_1_withchannels, matrix_2)
        elif self._ndims==3:
            return np.einsum('ijkl,ijk->ijkl', matrix_1_withchannels, matrix_2)
        else:
            return None

    def _get_shape_output_image(self, in_shape_image: Tuple[int, ...], out_size_image: Tuple[int, ...]) -> Tuple[int, ...]:
        if ImagesUtil.is_image_without_channels(self._size_image, in_shape_image):
            return tuple(out_size_image)
        else:
            num_channels = ImagesUtil.get_num_channels_image(self._size_image, in_shape_image)
            return tuple(out_size_image) + (num_channels,)

    def _get_reshaped_input_image(self, in_image: np.ndarray) -> np.ndarray:
        if ImagesUtil.is_image_without_channels(self._size_image, in_image.shape[1:]):
            return in_image
        else:
            return np.expand_dims(in_image, axis=-1)

    def _get_reshaped_output_image(self, in_image: np.ndarray) -> np.ndarray:
        num_channels = ImagesUtil.get_num_channels_image(self._size_image, in_image.shape)
        if num_channels==1:
            return np.squeeze(in_image, axis=-1)
        else:
            return in_image

    def _check_correct_shape_input_image(self, in_shape_image: Tuple[int, ...]) -> bool:
        check1 = len(in_shape_image) == self._ndims + 2
        check2 = in_shape_image[0] == self._num_patches_total
        if self._is_output_nnet_validconvs:
            check3 = in_shape_image[1:-2] != self._size_output_image
        else:
            check3 = in_shape_image[1:-2] != self._size_image

        return check1 and check2 and check3

    def compute_overlap_image_patches(self) -> np.ndarray:
        # compute normalizing factor to account for how many times the sliding-window batches image overlap
        out_shape_image = self._size_volume_image
        out_overlap_patches = np.zeros(out_shape_image, dtype=np.float32)

        if self._is_output_nnet_validconvs:
            weight_sample_shape = self._size_output_image
        else:
            weight_sample_shape = self._size_image

        weight_sample = np.ones(weight_sample_shape, dtype=np.float32)
        weight_sample = self._get_processed_image_patch(weight_sample)

        for index in range(self._num_patches_total):
            self._image_generator.set_add_image_patch(weight_sample, out_overlap_patches, index)

        return out_overlap_patches

    def _compute_normfact_overlap_image_patches(self) -> np.ndarray:
        out_normfact_overlap_patches = self.compute_overlap_image_patches()

        # set to very large overlap to avoid division by zero in those parts where there was no batch extracted
        max_toler = 1.0e+010
        out_normfact_overlap_patches = np.where(out_normfact_overlap_patches==0.0, max_toler, out_normfact_overlap_patches)
        out_normfact_overlap_patches = np.reciprocal(out_normfact_overlap_patches)

        return out_normfact_overlap_patches


class ImageReconstructorWithTransformation(ImageReconstructor):
    # PROTOTYPE OF RECONSTRUCTOR WITH TRANSFORMATION AT TESTING TIME. NIT TESTED YET
    def __init__(self,
                 size_image: Tuple[int, ...],
                 image_transform_generator: TransformRigidImages,
                 num_trans_per_patch: int = 1,
                 size_volume_image: Tuple[int, ...] = (0,),
                 is_output_nnet_validconvs: bool = False,
                 size_output_image: Tuple[int, ...] = None,
                 is_filter_output_nnet: bool = False,
                 filter_image_generator: ImageGenerator = None
                 ) -> None:
        super(ImageReconstructor, self).__init__(size_image,
                                                 image_transform_generator,
                                                 size_volume_image=size_volume_image,
                                                 is_output_nnet_validconvs=is_output_nnet_validconvs,
                                                 size_output_image=size_output_image,
                                                 is_filter_output_unet=is_filter_output_nnet,
                                                 filter_image_generator=filter_image_generator)

        # NEED SOMETHING LIKE THIS TO SET UP THE SAME TRANSFORMATION FOR ALL REPEATED TRANSFORMS PER PATCH
        self._image_transform_generator = image_transform_generator
        self._image_transform_generator.initialize_fixed_seed_0()
        self._num_trans_per_patch = num_trans_per_patch

    def _get_processed_image_patch(self, in_image: np.ndarray) -> np.ndarray:
        # SEARCH FOR NUMPY FUNCTION TO DO SUMATION DIRECTLY
        sumrun_in_images = in_image[0]
        for i in range(1, self._num_trans_per_patch):
            sumrun_in_images += self._image_transform_generator._get_inverse_transformed_image(in_image[i])

        out_image = np.divide(sumrun_in_images, self._num_trans_per_patch)

        # Apply rest of processing operations from parent class
        out_image = super(ImageReconstructorWithTransformation, self)._get_processed_image_patch(out_image)
        return out_image

    def _check_correct_shape_input_image(self, in_shape_image: Tuple[int, ...]) -> bool:
        check1 = len(in_shape_image) == self._ndims + 3
        check2 = in_shape_image[0] == self._num_patches_total
        check3 = in_shape_image[1] == self._num_trans_per_patch
        if self._is_output_nnet_validconvs:
            check4 = in_shape_image[2:-2] != self._size_output_image
        else:
            check4 = in_shape_image[2:-2] != self._size_image

        return check1 and check2 and check3 and check4