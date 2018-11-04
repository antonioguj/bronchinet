#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Postprocessing.FilteringValidUnetOutput import *
from Postprocessing.BaseImageReconstructor import *


class TransformationReconstructorImages(BaseImageFilteredReconstructor):

    def __init__(self, size_image_sample, num_samples_total, transformImages_generator, size_outUnet_sample=None):

        self.size_image_sample = size_image_sample
        self.num_samples_total = num_samples_total

        self.transformImages_generator = transformImages_generator

        # Important! seed the initial seed in transformation images
        # inverse transformation must be the same ones as those applied to get predict data
        self.transformImages_generator.initialize_fixed_seed_0()

        if size_outUnet_sample and size_outUnet_sample != size_image_sample:
            if len(size_image_sample)==2:
                filterImages_calculator = FilteringValidUnetOutput2D(size_image_sample, size_outUnet_sample)
            elif len(size_image_sample)==3:
                filterImages_calculator = FilteringValidUnetOutput3D(size_image_sample, size_outUnet_sample)
        else:
            filterImages_calculator = None

        super(TransformationReconstructorImages, self).__init__(size_image_sample, filterImages_calculator)


    def check_shape_predict_data(self, predict_data_shape):

        return (len(predict_data_shape) == len(self.size_image_sample) + 2) and \
               (predict_data_shape[0] == self.num_samples_total) and \
               (predict_data_shape[1:-2] != self.size_image_sample)

    def get_transformed_image_sample_array(self, image_sample_array):

        return self.transformImages_generator.get_inverse_transformed_image(image_sample_array)

    def compute(self, predict_data):

        if not self.check_shape_predict_data(predict_data.shape):
            message = "wrong shape of input predictions array..." % (predict_data.shape)
            CatchErrorException(message)

        predict_full_array = np.zeros(self.size_image_sample, dtype=FORMATPROBABILITYDATA)

        # compute reconstructed image by computing the average of various transformation of patches
        #
        for index in range(self.num_samples_total):
            predict_full_array += self.get_processed_image_sample_array(self.get_transformed_image_sample_array(predict_data[index]))
        # endfor

        # compute average over various transformations
        return np.divide(predict_full_array, self.num_samples_total)