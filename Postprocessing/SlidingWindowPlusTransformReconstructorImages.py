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
from Preprocessing.TransformationImages import *
from Postprocessing.SlidingWindowReconstructorImages import *



class SlidingWindowPlusTransformReconstructorImages(SlidingWindowReconstructorImages):

    def __init__(self, size_image_sample,
                 transformImages_generator,
                 num_trans_per_sample,
                 size_total_image,
                 isUse_valid_convs= False,
                 size_output_model= None,
                 isfilter_valid_outUnet= False,
                 prop_valid_outUnet= None,
                 is_onehotmulticlass= False):
        self.transformImages_generator = transformImages_generator
        # Important! seed the initial seed in transformation images
        # inverse transformation must be the same ones as those applied to get predict data
        self.transformImages_generator.initialize_fixed_seed_0()
        self.num_trans_per_sample = num_trans_per_sample

        super(SlidingWindowPlusTransformReconstructorImages, self).__init__(size_image_sample,
                                                                            size_total_image= size_total_image,
                                                                            isUse_valid_convs= isUse_valid_convs,
                                                                            size_output_model= size_output_model,
                                                                            isfilter_valid_outUnet= isfilter_valid_outUnet,
                                                                            prop_valid_outUnet= prop_valid_outUnet,
                                                                            is_onehotmulticlass= is_onehotmulticlass)
        # take into account average over various transformations
        self.normfact_overlap_images_samples_array = np.divide(self.normfact_overlap_images_samples_array,
                                                               self.num_trans_per_sample)

    def check_shape_predict_data(self, in_array_shape):
        check1 = (len(in_array_shape) == len(self.size_total_image) + 3)
        check2 = in_array_shape[0] == self.num_trans_per_sample
        check3 = in_array_shape[1] == self.num_samples_total
        check4 = in_array_shape[1:-2] != self.size_total_image
        return check1 and check2 and check3 and check4

    def get_transformed_image_array(self, image_array):
        return self.transformImages_generator.get_inverse_transformed_image(image_array)

    def compute(self, in_images_array):
        if not self.check_correct_shape_input_array(in_images_array.shape):
            message = "wrong shape of input predictions data array..." %(in_images_array.shape)
            CatchErrorException(message)

        out_array_shape = self.get_shape_out_array(in_images_array.shape)
        out_reconstructed_images_array = np.zeros(out_array_shape, dtype=in_images_array.dtype)

        for i in range(self.num_trans_per_sample):
            for index in range(self.num_samples_total):
                images_sample_array = self.get_processed_images_array(self.get_transformed_image_array(in_images_array[index]))
                self.slidingWindow_generator.set_add_image_patch(images_sample_array,
                                                                 out_reconstructed_images_array, index)
            # endfor
        #endfor
        return self.get_reshaped_out_array(self.multiply_matrixes_with_channels(out_reconstructed_images_array,
                                                                                self.normfact_overlap_images_samples_array))



class SlidingWindowPlusTransformReconstructorImages2D(SlidingWindowPlusTransformReconstructorImages):

    def __init__(self, size_image_sample,
                 prop_overlap,
                 transformImages_generator,
                 num_trans_per_sample,
                 size_total_image= (0,0),
                 isUse_valid_convs= False,
                 size_output_model= None,
                 isfilter_valid_outUnet= False,
                 prop_valid_outUnet= None,
                 is_onehotmulticlass= False):
        self.slidingWindow_generator = SlidingWindowImages2D(size_image_sample,
                                                             prop_overlap,
                                                             size_full_image= size_total_image)
        super(SlidingWindowPlusTransformReconstructorImages2D, self).__init__(size_image_sample,
                                                                              transformImages_generator,
                                                                              num_trans_per_sample,
                                                                              size_total_image= size_total_image,
                                                                              isUse_valid_convs= isUse_valid_convs,
                                                                              size_output_model= size_output_model,
                                                                              isfilter_valid_outUnet= isfilter_valid_outUnet,
                                                                              prop_valid_outUnet= prop_valid_outUnet,
                                                                              is_onehotmulticlass= is_onehotmulticlass)


class SlidingWindowPlusTransformReconstructorImages3D(SlidingWindowPlusTransformReconstructorImages):

    def __init__(self, size_image_sample,
                 prop_overlap,
                 transformImages_generator,
                 num_trans_per_sample,
                 size_total_image= (0,0,0),
                 isUse_valid_convs= False,
                 size_output_model= None,
                 isfilter_valid_outUnet= False,
                 prop_valid_outUnet= None,
                 is_onehotmulticlass= False):
        self.slidingWindow_generator = SlidingWindowImages3D(size_image_sample,
                                                             prop_overlap,
                                                             size_full_image= size_total_image)
        super(SlidingWindowPlusTransformReconstructorImages3D, self).__init__(size_image_sample,
                                                                              transformImages_generator,
                                                                              num_trans_per_sample,
                                                                              size_total_image= size_total_image,
                                                                              isUse_valid_convs= isUse_valid_convs,
                                                                              size_output_model= size_output_model,
                                                                              isfilter_valid_outUnet= isfilter_valid_outUnet,
                                                                              prop_valid_outUnet= prop_valid_outUnet,
                                                                              is_onehotmulticlass= is_onehotmulticlass)


class SlicingPlusTransformReconstructorImages2D(SlidingWindowPlusTransformReconstructorImages2D):

    def __init__(self, size_image_sample,
                 transformImages_generator,
                 num_trans_per_sample,
                 size_total_image= (0,0),
                 isUse_valid_convs= False,
                 size_output_model= None,
                 isfilter_valid_outUnet= False,
                 prop_valid_outUnet= None,
                 is_onehotmulticlass= False):
        super(SlicingPlusTransformReconstructorImages2D, self).__init__(size_image_sample,
                                                                        (0.0,0.0),
                                                                        transformImages_generator,
                                                                        num_trans_per_sample,
                                                                        size_total_image= size_total_image,
                                                                        isUse_valid_convs= isUse_valid_convs,
                                                                        size_output_model= size_output_model,
                                                                        isfilter_valid_outUnet= isfilter_valid_outUnet,
                                                                        prop_valid_outUnet= prop_valid_outUnet,
                                                                        is_onehotmulticlass= is_onehotmulticlass)

class SlicingPlusTransformReconstructorImages3D(SlidingWindowPlusTransformReconstructorImages3D):

    def __init__(self, size_image_sample,
                 transformImages_generator,
                 num_trans_per_sample,
                 size_total_image= (0,0,0),
                 isUse_valid_convs= False,
                 size_output_model= None,
                 isfilter_valid_outUnet= False,
                 prop_valid_outUnet= None,
                 is_onehotmulticlass= False):
        super(SlicingPlusTransformReconstructorImages3D, self).__init__(size_image_sample,
                                                                        (0.0,0.0,0.0),
                                                                        transformImages_generator,
                                                                        num_trans_per_sample,
                                                                        size_total_image= size_total_image,
                                                                        isUse_valid_convs= isUse_valid_convs,
                                                                        size_output_model= size_output_model,
                                                                        isfilter_valid_outUnet= isfilter_valid_outUnet,
                                                                        prop_valid_outUnet= prop_valid_outUnet,
                                                                        is_onehotmulticlass= is_onehotmulticlass)
