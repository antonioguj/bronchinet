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
from Preprocessing.BoundingBoxes import *
from Preprocessing.OperationImages import *
from DataLoaders.FileReaders import *



class SlidingWindowReconstructorImages(BaseImageReconstructor):

    def __init__(self, size_image_sample,
                 size_total_image,
                 isUse_valid_convs= False,
                 size_output_model= None,
                 isfilter_valid_outUnet= False,
                 prop_valid_outUnet= None,
                 is_onehotmulticlass= False):
        super(SlidingWindowReconstructorImages, self).__init__(size_image_sample,
                                                               size_total_image= size_total_image,
                                                               isUse_valid_convs= isUse_valid_convs,
                                                               size_output_model= size_output_model,
                                                               isfilter_valid_outUnet= isfilter_valid_outUnet,
                                                               prop_valid_outUnet= prop_valid_outUnet,
                                                               is_onehotmulticlass= is_onehotmulticlass)
        self.ndims = len(self.size_image)
        # initialize generator data
        self.num_samples_total = 0
        self.slidingWindow_generator = 0

    def complete_init_data(self, in_array_shape, is_compute_normfact=True):
        self.complete_init_data_step1(in_array_shape)
        if is_compute_normfact:
            self.complete_init_data_step2()

    def complete_init_data_step1(self, in_array_shape):
        self.size_total_image = in_array_shape[0:self.ndims]
        self.slidingWindow_generator.update_image_data(in_array_shape)
        self.num_samples_total = self.slidingWindow_generator.get_num_images()

    def complete_init_data_step2(self):
        self.compute_normfact_overlap_images_samples()

    def check_correct_shape_input_array(self, in_array_shape):
        check1 = len(in_array_shape) == self.ndims + 2
        check2 = in_array_shape[0] == self.num_samples_total
        check3 = in_array_shape[1:-2] != self.size_total_image
        return check1 and check2 and check3

    def multiply_matrixes_with_channels(self, matrix_1_withchannels, matrix_2):
        if self.ndims==2:
            return np.einsum('ijk,ij->ijk', matrix_1_withchannels, matrix_2)
        elif self.ndims==3:
            return np.einsum('ijkl,ijk->ijkl', matrix_1_withchannels, matrix_2)
        else:
            return NotImplemented

    def get_limits_image_sample(self, index):
        limits_input_image = self.slidingWindow_generator.get_crop_window_image(index)
        #limits_input_image = [[limits_input_image[2*i], limits_input_image[2*i+1]] for i in range(self.ndims)]
        if self.isUse_valid_convs:
            limits_output_sample = BoundingBoxes.compute_bounding_box_centered_bounding_box_fit_image(limits_input_image,
                                                                                                      self.size_output_model)
        else:
            limits_output_sample = limits_input_image
        return limits_output_sample

    def get_includedadded_image_sample(self, sample_array, images_array, index):
        limits_image_sample = self.get_limits_image_sample(index)
        SetPatchInImages.compute3D_byadding(sample_array, images_array, limits_image_sample)

    def get_limits_border_effects(self, size_image_total):
        voxels_border_effects = tuple([(s_i - s_o) / 2 for (s_i, s_o) in zip(self.size_image, self.size_output_model)])
        limits_border_effects = tuple([(v_b_e, s_t_i - v_b_e) for (v_b_e, s_t_i) in zip(voxels_border_effects, size_image_total)])
        return limits_border_effects


    def compute(self, in_images_array):
        if not self.check_correct_shape_input_array(in_images_array.shape):
            message = "wrong shape of input predictions data array..." % (in_images_array.shape)
            CatchErrorException(message)

        out_array_shape = self.get_shape_out_array(in_images_array.shape)
        out_reconstructed_images_array = np.zeros(out_array_shape, dtype=in_images_array.dtype)

        for index in range(self.num_samples_total):
            images_sample_array = self.get_processed_images_array(in_images_array[index])
            self.get_includedadded_image_sample(images_sample_array, out_reconstructed_images_array, index)
        # endfor

        return self.get_reshaped_out_array(self.multiply_matrixes_with_channels(out_reconstructed_images_array,
                                                                                self.normfact_overlap_images_samples_array))


    def compute_normfact_overlap_images_samples(self):
        # compute normalizing factor to account for how many times the sliding-window batches image overlap
        self.normfact_overlap_images_samples_array = np.zeros(self.size_total_image, dtype=np.float32)

        if self.isUse_valid_convs:
            weight_sample_array = np.ones(self.size_output_model, dtype=np.float32)
        else:
            if self.isfilter_valid_outUnet:
                weight_sample_array = self.filterImages_calculator.get_filter_func_outUnet_array()
            else:
                weight_sample_array = np.ones(self.size_image, dtype=np.float32)

        for index in range(self.num_samples_total):
            self.get_includedadded_image_sample(weight_sample_array, self.normfact_overlap_images_samples_array, index)
        # endfor

        # set to very large overlap to avoid division by zero in
        # those parts where there was no batch extracted
        max_toler = 1.0e+010
        self.normfact_overlap_images_samples_array = np.where(self.normfact_overlap_images_samples_array==0.0,
                                                              max_toler, self.normfact_overlap_images_samples_array)
        self.normfact_overlap_images_samples_array = np.reciprocal(self.normfact_overlap_images_samples_array)


    def check_filling_overlap_image_samples(self):
        if self.isUse_valid_convs:
            fill_sample_array = np.ones(self.size_output_model, dtype=np.int8)
        else:
            fill_sample_array = np.ones(self.size_image, dtype=np.int8)
        fill_total_array = np.zeros(self.size_total_image, dtype=np.int8)

        for index in range(self.num_samples_total):
            self.get_includedadded_image_sample(fill_sample_array, fill_total_array, index)
        # endfor

        if self.isUse_valid_convs:
            #account for border effects and remove the image borders
            limits_border_effects = self.get_limits_border_effects(self.size_total_image)
            fill_unique_values = np.unique(CropImages.compute3D(fill_total_array, limits_border_effects))
        else:
            fill_unique_values = np.unique(fill_total_array)

        if 0 in fill_unique_values:
            message = "Found \'0\' in check filling overlap matrix: the sliding-window does not cover some areas..."
            CatchWarningException(message)
        print("Found num of overlaps in check filling overlap matrix: \'%s\'" %(fill_unique_values))
        return fill_total_array



class SlidingWindowReconstructorImages2D(SlidingWindowReconstructorImages):

    def __init__(self, size_image_sample,
                 prop_overlap,
                 size_total_image= (0,0),
                 isfilter_valid_outUnet= False,
                 prop_valid_outUnet= None,
                 isUse_valid_convs=False,
                 size_output_model=None,
                 is_onehotmulticlass= False):
        super(SlidingWindowReconstructorImages2D, self).__init__(size_image_sample,
                                                                 size_total_image= size_total_image,
                                                                 isfilter_valid_outUnet= isfilter_valid_outUnet,
                                                                 prop_valid_outUnet= prop_valid_outUnet,
                                                                 isUse_valid_convs=isUse_valid_convs,
                                                                 size_output_model=size_output_model,
                                                                 is_onehotmulticlass= is_onehotmulticlass)
        self.slidingWindow_generator = SlidingWindowImages(size_image_sample,
                                                           prop_overlap,
                                                           size_full_image= size_total_image)
        self.complete_init_data(size_total_image)


class SlidingWindowReconstructorImages3D(SlidingWindowReconstructorImages):

    def __init__(self, size_image_sample,
                 prop_overlap,
                 size_total_image= (0,0,0),
                 isfilter_valid_outUnet= False,
                 prop_valid_outUnet= None,
                 isUse_valid_convs=False,
                 size_output_model=None,
                 is_onehotmulticlass= False):
        super(SlidingWindowReconstructorImages3D, self).__init__(size_image_sample,
                                                                 size_total_image= size_total_image,
                                                                 isfilter_valid_outUnet= isfilter_valid_outUnet,
                                                                 prop_valid_outUnet= prop_valid_outUnet,
                                                                 isUse_valid_convs=isUse_valid_convs,
                                                                 size_output_model=size_output_model,
                                                                 is_onehotmulticlass= is_onehotmulticlass)
        self.slidingWindow_generator = SlidingWindowImages(size_image_sample,
                                                           prop_overlap,
                                                           size_full_image= size_total_image)
        self.complete_init_data(size_total_image)


class SlicingReconstructorImages2D(SlidingWindowReconstructorImages2D):

    def __init__(self, size_image_sample,
                 size_total_image= (0,0),
                 isfilter_valid_outUnet= False,
                 prop_valid_outUnet= None,
                 isUse_valid_convs=False,
                 size_output_model=None,
                 is_onehotmulticlass= False):
        super(SlicingReconstructorImages2D, self).__init__(size_image_sample,
                                                           prop_overlap= (0.0,0.0),
                                                           size_total_image= size_total_image,
                                                           isfilter_valid_outUnet= isfilter_valid_outUnet,
                                                           prop_valid_outUnet= prop_valid_outUnet,
                                                           isUse_valid_convs=isUse_valid_convs,
                                                           size_output_model=size_output_model,
                                                           is_onehotmulticlass= is_onehotmulticlass)

class SlicingReconstructorImages3D(SlidingWindowReconstructorImages3D):

    def __init__(self, size_image_sample,
                 size_total_image= (0,0,0),
                 isfilter_valid_outUnet= False,
                 prop_valid_outUnet= None,
                 isUse_valid_convs=False,
                 size_output_model=None,
                 is_onehotmulticlass= False):
        super(SlicingReconstructorImages3D, self).__init__(size_image_sample,
                                                           prop_overlap= (0.0,0.0,0.0),
                                                           size_total_image= size_total_image,
                                                           isfilter_valid_outUnet= isfilter_valid_outUnet,
                                                           prop_valid_outUnet= prop_valid_outUnet,
                                                           isUse_valid_convs=isUse_valid_convs,
                                                           size_output_model=size_output_model,
                                                           is_onehotmulticlass= is_onehotmulticlass)
