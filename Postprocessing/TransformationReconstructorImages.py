#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Postprocessing.BaseImageReconstructor import *



class TransformationReconstructorImages(BaseImageReconstructor):

    def __init__(self, size_image_sample,
                 transformImages_generator,
                 num_samples_total,
                 isUse_valid_convs= False,
                 size_output_model= None,
                 isfilter_valid_outUnet= False,
                 prop_valid_outUnet= None,
                 is_onehotmulticlass= False):
        self.transformImages_generator = transformImages_generator
        # Important! seed the initial seed in transformation images
        # inverse transformation must be the same ones as those applied to get predict data
        self.transformImages_generator.initialize_fixed_seed_0()
        self.num_samples_total = num_samples_total

        super(TransformationReconstructorImages, self).__init__(size_image_sample,
                                                                size_total_image= size_image_sample,
                                                                isUse_valid_convs= isUse_valid_convs,
                                                                size_output_model= size_output_model,
                                                                isfilter_valid_outUnet= isfilter_valid_outUnet,
                                                                prop_valid_outUnet= prop_valid_outUnet,
                                                                is_onehotmulticlass= is_onehotmulticlass)

    def check_correct_shape_input_array(self, in_array_shape):
        check1 = len(in_array_shape) == len(self.size_image) + 2
        check2 = in_array_shape[0] == self.num_samples_total
        check3 = in_array_shape[1:-2] != self.size_image
        return check1 and check2 and check3

    def get_transformed_image_array(self, image_array):
        return self.transformImages_generator.get_inverse_transformed_image_array(image_array)

    def compute(self, in_images_array):
        if not self.check_correct_shape_input_array(in_images_array.shape):
            message = "wrong shape of input predictions data array..." % (in_images_array.shape)
            CatchErrorException(message)

        out_array_shape = self.get_shape_out_array(in_images_array.shape)
        out_reconstructed_images_array = np.zeros(out_array_shape, dtype=in_images_array.dtype)

        # compute reconstructed image by computing the average of various transformation of patches
        for index in range(self.num_samples_total):
            images_sample_array = self.get_processed_images_array(self.get_transformed_image_array(in_images_array[index]))
            out_reconstructed_images_array += images_sample_array
        # endfor

        # compute average over various transformations
        return self.get_reshaped_out_array(np.divide(out_reconstructed_images_array, self.num_samples_total))
