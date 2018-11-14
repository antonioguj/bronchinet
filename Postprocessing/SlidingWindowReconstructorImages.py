#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Preprocessing.SlidingWindowImages import *
from Postprocessing.BaseImageReconstructor import *


class SlidingWindowReconstructorImages(BaseImageReconstructor):

    def __init__(self, size_image_sample,
                 size_total_image,
                 num_samples_total,
                 isfilterImages=False,
                 prop_valid_outUnet=None,
                 is_onehotmulticlass=False):

        self.size_total_image  = size_total_image
        self.num_samples_total = num_samples_total

        super(SlidingWindowReconstructorImages, self).__init__(size_image_sample,
                                                               isfilterImages=isfilterImages,
                                                               prop_valid_outUnet=prop_valid_outUnet,
                                                               is_onehotmulticlass=is_onehotmulticlass)
        self.compute_normfact_overlap_images_samples()


    @staticmethod
    def multiply_matrixes_with_channels(matrix_1_withchannels, matrix_2):
        pass

    def complete_init_data(self, in_array_shape):
        self.complete_init_data_step1(in_array_shape)

        self.complete_init_data_step2()

    def complete_init_data_step1(self, in_array_shape):
        pass

    def complete_init_data_step2(self):
        self.compute_normfact_overlap_images_samples()


    def check_correct_shape_input_array(self, in_array_shape):

        check1 = len(in_array_shape) == len(self.size_total_image) + 2
        check2 = in_array_shape[0] == self.num_samples_total
        check3 = in_array_shape[1:-2] != self.size_total_image

        return check1 and check2 and check3

    def adding_reconstructed_images_sample_array(self, index, images_sample_array):
        pass

    def set_calc_reconstructed_images_array(self, input_array):
        self.reconstructed_images_array = input_array


    def compute_normfact_overlap_images_samples(self):
        # compute normalizing factor to account for how many times the sliding-window batches image overlap

        weight_overlap_images_samples_total_array = np.zeros(self.size_total_image, dtype=np.float32)

        self.set_calc_reconstructed_images_array(weight_overlap_images_samples_total_array)

        if self.isfilterImages:
            weight_sample_array = self.filterImages_calculator.get_filter_func_outUnet_array()
        else:
            weight_sample_array = np.ones(self.size_image, dtype=np.float32)

        for index in range(self.num_samples_total):
            self.adding_reconstructed_images_sample_array(index, weight_sample_array)
        # endfor

        # set to very small toler. to avoid division by zero in
        # those parts where there was no batch extracted
        weight_overlap_images_samples_total_array += 1.0e-010

        self.normfact_overlap_images_samples_array = np.divide(np.ones(self.size_total_image, dtype=np.float32),
                                                               weight_overlap_images_samples_total_array)

    def compute(self, in_images_array):

        if not self.check_correct_shape_input_array(in_images_array.shape):
            message = "wrong shape of input predictions data array..." % (in_images_array.shape)
            CatchErrorException(message)

        out_array_shape = self.get_shape_out_array(in_images_array.shape)

        out_reconstructed_images_array = np.zeros(out_array_shape, dtype=in_images_array.dtype)

        self.set_calc_reconstructed_images_array(out_reconstructed_images_array)

        for index in range(self.num_samples_total):
            images_sample_array = self.get_processed_images_array(in_images_array[index])

            self.adding_reconstructed_images_sample_array(index, images_sample_array)
        # endfor

        return self.get_reshaped_out_array(self.multiply_matrixes_with_channels(out_reconstructed_images_array,
                                                                                self.normfact_overlap_images_samples_array))


class SlidingWindowReconstructorImages2D(SlidingWindowReconstructorImages):

    def __init__(self, size_image_sample,
                 prop_overlap,
                 size_total_image=(0, 0),
                 isfilterImages=False,
                 prop_valid_outUnet=None,
                 is_onehotmulticlass=False):

        self.slidingWindow_generator = SlidingWindowImages2D(size_image_sample,
                                                             prop_overlap,
                                                             size_total=size_total_image)
        self.complete_init_data_step1(size_total_image)

        super(SlidingWindowReconstructorImages2D, self).__init__(size_image_sample,
                                                                 size_total_image=size_total_image,
                                                                 num_samples_total=self.num_samples_total,
                                                                 isfilterImages=isfilterImages,
                                                                 prop_valid_outUnet=prop_valid_outUnet,
                                                                 is_onehotmulticlass=is_onehotmulticlass)
        self.complete_init_data_step2()


    @staticmethod
    def multiply_matrixes_with_channels(matrix_1_withchannels, matrix_2):
        return np.einsum('ijk,ij->ijk', matrix_1_withchannels, matrix_2)

    def complete_init_data_step1(self, in_array_shape):

        self.size_total_image = in_array_shape[0:2]
        self.slidingWindow_generator.complete_init_data(self.size_total_image)
        self.num_samples_total = self.slidingWindow_generator.get_num_images()

    def adding_reconstructed_images_sample_array(self, index, images_sample_array):

        (x_left, x_right, y_down, y_up) = self.slidingWindow_generator.get_limits_image(index)

        self.reconstructed_images_array[x_left:x_right, y_down:y_up, ...] += images_sample_array


class SlidingWindowReconstructorImages3D(SlidingWindowReconstructorImages):

    def __init__(self, size_image_sample,
                 prop_overlap,
                 size_total_image=(0, 0, 0),
                 isfilterImages=False,
                 prop_valid_outUnet=None,
                 is_onehotmulticlass=False):

        self.slidingWindow_generator = SlidingWindowImages3D(size_image_sample,
                                                             prop_overlap,
                                                             size_total=size_total_image)
        self.complete_init_data_step1(size_total_image)

        super(SlidingWindowReconstructorImages3D, self).__init__(size_image_sample,
                                                                 size_total_image=size_total_image,
                                                                 num_samples_total=self.num_samples_total,
                                                                 isfilterImages=isfilterImages,
                                                                 prop_valid_outUnet=prop_valid_outUnet,
                                                                 is_onehotmulticlass=is_onehotmulticlass)
        self.complete_init_data_step2()


    @staticmethod
    def multiply_matrixes_with_channels(matrix_1_withchannels, matrix_2):
        return np.einsum('ijkl,ijk->ijkl', matrix_1_withchannels, matrix_2)

    def complete_init_data_step1(self, in_array_shape):

        self.size_total_image = in_array_shape[0:3]
        self.slidingWindow_generator.complete_init_data(self.size_total_image)
        self.num_samples_total = self.slidingWindow_generator.get_num_images()

    def adding_reconstructed_images_sample_array(self, index, image_sample_array):

        (z_back, z_front, x_left, x_right, y_down, y_up) = self.slidingWindow_generator.get_limits_image(index)

        self.reconstructed_images_array[z_back:z_front, x_left:x_right, y_down:y_up, ...] += image_sample_array


class SlicingReconstructorImages2D(SlidingWindowReconstructorImages2D):

    def __init__(self, size_image_sample,
                 size_total_image=(0, 0),
                 isfilterImages=False,
                 prop_valid_outUnet=None,
                 is_onehotmulticlass=False):
        super(SlicingReconstructorImages2D, self).__init__(size_image_sample,
                                                           prop_overlap=(0.0, 0.0),
                                                           size_total_image=size_total_image,
                                                           isfilterImages=isfilterImages,
                                                           prop_valid_outUnet=prop_valid_outUnet,
                                                           is_onehotmulticlass=is_onehotmulticlass)

class SlicingReconstructorImages3D(SlidingWindowReconstructorImages3D):

    def __init__(self, size_image_sample,
                 size_total_image=(0, 0, 0),
                 isfilterImages=False,
                 prop_valid_outUnet=None,
                 is_onehotmulticlass=False):
        super(SlicingReconstructorImages3D, self).__init__(size_image_sample,
                                                           prop_overlap=(0.0, 0.0, 0.0),
                                                           size_total_image=size_total_image,
                                                           isfilterImages=isfilterImages,
                                                           prop_valid_outUnet=prop_valid_outUnet,
                                                           is_onehotmulticlass=is_onehotmulticlass)