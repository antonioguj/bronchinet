#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.ErrorMessages import *
from Common.FunctionsImagesUtil import *
from OperationImages.OperationImages import ExtendImages
import numpy as np



class ImageReconstructor(object):

    def __init__(self, size_image,
                 images_generator,
                 size_full_image= 0,
                 is_outputUnet_validconvs= False,
                 size_output_image= None,
                 is_filter_output_unet= False,
                 filter_images_generator= None):
        self.size_image = size_image
        self.ndims = len(size_image)
        self.images_generator = images_generator

        if np.isscalar(size_full_image):
            self.size_full_image = tuple([size_full_image]*self.ndims)
        else:
            self.size_full_image = size_full_image

        self.is_outputUnet_validconvs = is_outputUnet_validconvs
        if is_outputUnet_validconvs and size_output_image \
            (size_image == size_output_image):
            self.size_output_image = size_output_image
            self.valid_output_bounding_box = BoundingBoxes.compute_bounding_box_centered_image_fit_image(self.size_output_image,
                                                                                                         self.size_image)
            if self.ndims==2:
                self.fun_extend_image_patch = ExtendImages.compute2D
            elif self.ndims==3:
                self.fun_extend_image_patch = ExtendImages.compute3D
            else:
                raise Exception('Error: self.ndims')
        else:
            self.is_outputUnet_validconvs = False

        self.is_filter_output_unet = is_filter_output_unet
        if is_filter_output_unet:
            self.filter_images_generator = filter_images_generator

        self.num_patches_total = self.images_generator.get_num_images()


    def update_image_data(self, in_array_shape, is_compute_normfact=True):
        self.update_image_data_step1(in_array_shape)
        if is_compute_normfact:
            self.update_image_data_step2()

    def update_image_data_step1(self, in_array_shape):
        self.size_full_image = in_array_shape[0:self.ndims]
        self.images_generator.update_image_data(in_array_shape)
        self.num_patches_total = self.images_generator.get_num_images()

    def update_image_data_step2(self):
        self.normfact_overlap_image_patches_array = self.compute_normfact_overlap_image_patches()


    def check_correct_shape_input_array(self, in_array_shape):
        check1 = len(in_array_shape) == self.ndims + 2
        check2 = in_array_shape[0] == self.num_patches_total
        if self.is_outputUnet_validconvs:
            check3 = in_array_shape[1:-2] != self.size_output_image
        else:
            check3 = in_array_shape[1:-2] != self.size_image
        return check1 and check2 and check3


    def multiply_matrixes_with_channels(self, matrix_1_withchannels, matrix_2):
        if self.ndims==2:
            return np.einsum('ijk,ij->ijk', matrix_1_withchannels, matrix_2)
        elif self.ndims==3:
            return np.einsum('ijkl,ijk->ijkl', matrix_1_withchannels, matrix_2)
        else:
            return NotImplemented


    def get_shape_output_array(self, in_array_shape):
        if is_image_array_without_channels(self.size_image, in_array_shape):
            return list(self.size_full_image)
        else:
            num_channels = get_num_channels_array(self.size_image, in_array_shape)
            return list(self.size_full_image) + [num_channels]


    def get_reshaped_input_array(self, in_array):
        if is_image_array_without_channels(self.size_image, in_array.shape[1:]):
            return in_array
        else:
            return np.expand_dims(in_array, axis=-1)

    def get_reshaped_output_array(self, in_array):
        num_channels = get_num_channels_array(self.size_image, in_array.shape)
        if num_channels==1:
            return np.squeeze(in_array, axis=-1)
        else:
            return in_array


    def get_processed_image_patch_array(self, in_array):
        if self.is_outputUnet_validconvs:
            out_array = self.fun_extend_image_patch(in_array, self.valid_output_bounding_box, self.size_image)
        else:
            out_array = in_array
        if self.is_filter_output_unet:
            out_array = filter_images_generator.get_image(out_array)
        return out_array


    def compute(self, in_array):
        if not self.check_correct_shape_input_array(in_array.shape):
            message = "Wrong shape of input predictions data array..." % (in_array.shape)
            CatchErrorException(message)

        out_array_shape = self.get_shape_output_array(in_array.shape)
        out_reconstructed_array = np.zeros(out_array_shape, dtype=in_array.dtype)

        for index in range(self.num_patches_total):
            in_patch_array = self.get_processed_image_patch_array(in_array[index])
            self.images_generator.set_add_image_patch(in_patch_array, out_reconstructed_array, index)
        # endfor

        out_reconstructed_array = self.multiply_matrixes_with_channels(out_reconstructed_array,
                                                                       self.normfact_overlap_image_patches_array)
        return self.get_reshaped_output_array(out_reconstructed_array)


    def compute_normfact_overlap_image_patches(self):
        # compute normalizing factor to account for how many times the sliding-window batches image overlap
        out_array_shape = self.size_full_image
        normfact_overlap_image_patches_array = np.zeros(out_array_shape, dtype=np.float32)

        if self.is_outputUnet_validconvs:
            weight_sample_array = np.ones(self.size_output_image, dtype=np.float32)
        else:
            weight_sample_array = np.ones(self.size_image, dtype=np.float32)

        weight_sample_array = self.get_processed_image_patch_array(weight_sample_array)

        for index in range(self.num_patches_total):
            self.images_generator.set_add_image_patch(weight_sample_array, normfact_overlap_image_patches_array, index)
        # endfor

        # set to very large overlap to avoid division by zero in
        # those parts where there was no batch extracted
        max_toler = 1.0e+010
        normfact_overlap_image_patches_array = np.where(normfact_overlap_image_patches_array==0.0,
                                                        max_toler, normfact_overlap_image_patches_array)

        normfact_overlap_image_patches_array = np.reciprocal(normfact_overlap_image_patches_array)

        return normfact_overlap_image_patches_array


    # def check_filling_overlap_image_samples(self):
    #     if self.is_outputUnet_validconvs:
    #         fill_sample_array = np.ones(self.size_output_image, dtype=np.int8)
    #     else:
    #         fill_sample_array = np.ones(self.size_image, dtype=np.int8)
    #     fill_total_array = np.zeros(self.size_full_image, dtype=np.int8)
    #
    #     for index in range(self.num_samples_total):
    #         self.get_includedadded_image_sample(fill_sample_array, fill_total_array, index)
    #     # endfor
    #
    #     if self.is_outputUnet_validconvs:
    #         #account for border effects and remove the image borders
    #         limits_border_effects = self.get_limits_border_effects(self.size_full_image)
    #         fill_unique_values = np.unique(CropImages.compute3D(fill_total_array, limits_border_effects))
    #     else:
    #         fill_unique_values = np.unique(fill_total_array)
    #
    #     if 0 in fill_unique_values:
    #         message = "Found \'0\' in check filling overlap matrix: the sliding-window does not cover some areas..."
    #         CatchWarningException(message)
    #     print("Found num of overlaps in check filling overlap matrix: \'%s\'" %(fill_unique_values))
    #
    #     return fill_total_array


    # def get_processed_image_onehotmulticlass_array(self, images_array):
    #     new_images_array = np.ndarray(self.size_image, dtype=images_array.dtype)
    #     if len(self.size_image) == 2:
    #         for i in range(self.size_image[0]):
    #             for j in range(self.size_image[1]):
    #                 index_argmax = np.argmax(images_array[i, j, :])
    #                 new_images_array[i, j] = index_argmax
    #             # endfor
    #         # endfor
    #     elif len(self.size_image) == 3:
    #         for i in range(self.size_image[0]):
    #             for j in range(self.size_image[1]):
    #                 for k in range(self.size_image[2]):
    #                     index_argmax = np.argmax(images_array[i, j, k, :])
    #                     new_images_array[i, j, k] = index_argmax
    #                 # endfor
    #             # endfor
    #         # endfor
    #     else:
    #         message = "wrong shape of input images..." %(self.size_image)
    #         CatchErrorException(message)
    #
    #     return new_images_array



class ImageReconstructorWithTransformation(ImageReconstructor):
    # PROTOTYPE OF RECONSTRUCTOR WITH TRANSFORMATION AT TESTING TIME. NIT TESTED YET

    def __init__(self, size_image,
                 images_transform_generator,
                 num_trans_per_patch= 1,
                 size_full_image= 0,
                 is_outputUnet_validconvs= False,
                 size_output_image= None,
                 is_filter_output_unet= False,
                 filter_images_generator= None):
        super(ImageReconstructor, self).__init__(size_image,
                                                 images_transform_generator,
                                                 size_full_image=size_full_image,
                                                 is_outputUnet_validconvs=is_outputUnet_validconvs,
                                                 size_output_image=size_output_image,
                                                 is_filter_output_unet=is_filter_output_unet,
                                                 filter_images_generator=filter_images_generator)

        # NEED SOMETHING LIKE THIS TO SET UP THE SAME TRANSFORMATION FOR ALL REPEATED TRANSFORMS PER PATCH
        self.images_transform_generator.initialize_fixed_seed_0()
        self.num_trans_per_patch = num_trans_per_patch


    def check_correct_shape_input_array(self, in_array_shape):
        check1 = len(in_array_shape) == self.ndims + 3
        check2 = in_array_shape[0] == self.num_patches_total
        check3 = in_array_shape[1] == self.num_trans_per_patch
        if self.is_outputUnet_validconvs:
            check4 = in_array_shape[2:-2] != self.size_output_image
        else:
            check4 = in_array_shape[2:-2] != self.size_image
        return check1 and check2 and check3 and check4


    def get_processed_image_patch_array(self, in_array):
        # SEARCH FOR NUMPY FUNCTION TO DO SUMATION DIRECTLY
        out_sumin_array = in_array[0]
        for i in range(1,self.num_trans_per_patch):
            out_sumin_array += images_transform_generator.get_inverse_transformed_image_array(in_array[i])
        #endfor
        out_array = np.divide(out_sumin_array, self.num_trans_per_patch)

        # apply rest of processing operations from parent class
        out_array = super(ImageReconstructorWithTransformation, self).get_processed_image_patch_array(out_array)
        return out_array
