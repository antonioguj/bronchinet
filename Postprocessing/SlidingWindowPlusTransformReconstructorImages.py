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

    def __init__(self, size_image_sample, size_total_image, num_samples_total, transformImages_generator, num_trans_per_sample, filterImages_calculator=None):

        self.transformImages_generator = transformImages_generator

        # Important! seed the initial seed in transformation images
        # inverse transformation must be the same ones as those applied to get predict data
        self.transformImages_generator.initialize_fixed_seed_0()

        self.num_trans_per_sample = num_trans_per_sample

        super(SlidingWindowPlusTransformReconstructorImages, self).__init__(size_image_sample, size_total_image, num_samples_total, filterImages_calculator)

        # take into account average over various transformations
        self.factor_overlap_images_samples_per_voxel = np.divide(self.factor_overlap_images_samples_per_voxel,
                                                                 self.num_trans_per_sample)


    def check_shape_predict_data(self, predict_data_shape):

        return (len(predict_data_shape) == len(self.size_total_image) + 3) and \
               (predict_data_shape[0] == self.num_trans_per_sample) and \
               (predict_data_shape[1] == self.num_samples_total) and \
               (predict_data_shape[2:-2] != self.size_total_image)

    def get_transformed_image_sample_array(self, image_sample_array):

        return self.transformImages_generator.get_inverse_transformed_image(image_sample_array)

    def compute(self, predict_data):

        if not self.check_shape_predict_data(predict_data.shape):
            message = "wrong shape of input predictions data array..." % (predict_data.shape)
            CatchErrorException(message)

        predict_full_array = np.zeros(self.size_total_image, dtype=FORMATPROBABILITYDATA)

        self.set_calc_reconstructed_image_array(predict_full_array)

        for i in range(self.num_trans_per_sample):
            for index in range(self.num_samples_total):
                self.adding_reconstructed_image_sample_array(index, self.get_processed_image_sample_array(self.get_transformed_image_sample_array(predict_data[i][index])))
            # endfor
        #endfor

        # multiply by factor to account for multiple overlaps of images samples and average over various transformations
        return np.multiply(predict_full_array, self.factor_overlap_images_samples_per_voxel)


class SlidingWindowPlusTransformReconstructorImages2D(SlidingWindowPlusTransformReconstructorImages):

    def __init__(self, size_image_sample, size_total_image, transformImages_generator, num_trans_per_sample, prop_overlap, size_outUnet_sample=None):

        self.slidingWindow_generator = SlidingWindowImages2D(size_image_sample, prop_overlap, size_total=size_total_image)

        num_samples_total = self.slidingWindow_generator.get_num_images()

        if size_outUnet_sample and size_outUnet_sample != size_image_sample:
            filterImages_calculator = FilteringValidUnetOutput2D(size_image_sample, size_outUnet_sample)
        else:
            filterImages_calculator = None

        super(SlidingWindowPlusTransformReconstructorImages2D, self).__init__(size_image_sample, size_total_image, num_samples_total, transformImages_generator, num_trans_per_sample, filterImages_calculator)


    def adding_reconstructed_image_sample_array(self, index, image_sample_array):

        (x_left, x_right, y_down, y_up) = self.slidingWindow_generator.get_limits_image(index)

        # full_array[x_left:x_right, y_down:y_up, ...] += batch_array
        self.reconstructed_image_array[x_left:x_right, y_down:y_up] += image_sample_array


class SlidingWindowPlusTransformReconstructorImages3D(SlidingWindowPlusTransformReconstructorImages):

    def __init__(self, size_image_sample, size_total_image, transformImages_generator, num_trans_per_sample, prop_overlap, size_outUnet_sample=None):

        self.slidingWindow_generator = SlidingWindowImages3D(size_image_sample, prop_overlap, size_total=size_total_image)

        num_samples_total = self.slidingWindow_generator.get_num_images()

        if size_outUnet_sample and size_outUnet_sample != size_image_sample:
            filterImages_calculator = FilteringValidUnetOutput3D(size_image_sample, size_outUnet_sample)
        else:
            filterImages_calculator = None

        super(SlidingWindowPlusTransformReconstructorImages3D, self).__init__(size_image_sample, size_total_image, num_samples_total, transformImages_generator, num_trans_per_sample, filterImages_calculator)


    def adding_reconstructed_image_sample_array(self, index, image_sample_array):

        (z_back, z_front, x_left, x_right, y_down, y_up) = self.slidingWindow_generator.get_limits_image(index)

        # full_array[z_back:z_front, x_left:x_right, y_down:y_up, ...] += batch_array
        self.reconstructed_image_array[z_back:z_front, x_left:x_right, y_down:y_up] += image_sample_array


class SlicingPlusTransformReconstructorImages2D(SlidingWindowPlusTransformReconstructorImages2D):

    def __init__(self, size_image_sample, size_total_image, transformImages_generator, num_trans_per_sample, size_outUnet_sample=None):
        super(SlicingPlusTransformReconstructorImages2D, self).__init__(size_image_sample, size_total_image, transformImages_generator, num_trans_per_sample, (0.0, 0.0), size_outUnet_sample)

class SlicingPlusTransformReconstructorImages3D(SlidingWindowPlusTransformReconstructorImages3D):

    def __init__(self, size_image_sample, size_total_image, transformImages_generator, num_trans_per_sample, size_outUnet_sample=None):
        super(SlicingPlusTransformReconstructorImages3D, self).__init__(size_image_sample, size_total_image, transformImages_generator, num_trans_per_sample, (0.0, 0.0, 0.0), size_outUnet_sample)