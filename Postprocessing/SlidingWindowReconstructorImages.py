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
from Postprocessing.FilteringValidUnetOutput import *
from Postprocessing.BaseImageReconstructor import *


class SlidingWindowReconstructorImages(BaseImageFilteredReconstructor):

    def __init__(self, size_image_sample, size_total_image, num_samples_total, filterImages_calculator=None):

        self.size_image_sample = size_image_sample
        self.size_total_image  = size_total_image
        self.num_samples_total = num_samples_total

        super(SlidingWindowReconstructorImages, self).__init__(size_image_sample, filterImages_calculator)

        self.compute_factor_num_overlap_images_samples_per_voxel()


    def check_shape_predict_data(self, predict_data_shape):

        return (len(predict_data_shape) == len(self.size_total_image) + 2) and \
               (predict_data_shape[0] == self.num_samples_total) and \
               (predict_data_shape[1:-2] != self.size_total_image)

    def adding_reconstructed_image_sample_array(self, index, image_sample_array):
        pass

    def set_calc_reconstructed_image_array(self, input_array):
        self.reconstructed_image_array = input_array


    def compute_factor_num_overlap_images_samples_per_voxel(self):
        # compute how many times a batch image overlaps in same voxel

        num_overlap_images_samples_per_voxels = np.zeros(self.size_total_image, dtype=np.int8)

        self.set_calc_reconstructed_image_array(num_overlap_images_samples_per_voxels)

        sample_count_ones_array = np.ones(self.size_image_sample, dtype=np.int8)

        for index in range(self.num_samples_total):
            self.adding_reconstructed_image_sample_array(index, sample_count_ones_array)
        # endfor

        # get positions with result '0': there's no overlap
        pos_non_overlap = np.argwhere(num_overlap_images_samples_per_voxels == 0)

        self.factor_num_overlap_images_samples_per_voxel = np.divide(np.ones(self.size_total_image, dtype=np.float32),
                                                                     num_overlap_images_samples_per_voxels)
        # remove pos where there was division by zero
        for pos in pos_non_overlap:
            self.factor_num_overlap_images_samples_per_voxel[tuple(pos)] = 0.0


    def get_filtering_map_array(self):

        filtering_map_array = np.zeros(self.size_total_image, dtype=FORMATPROBABILITYDATA)

        self.set_calc_reconstructed_image_array(filtering_map_array)

        sample_filtering_map_array = self.filterImages_calculator.get_prob_outnnet_array()

        for index in range(self.num_samples_total):
            self.adding_reconstructed_image_sample_array(index, sample_filtering_map_array)
        # endfor

        # multiply by factor to account for multiple overlaps of images samples
        return np.multiply(filtering_map_array, self.factor_num_overlap_images_samples_per_voxel)


    def compute(self, predict_data):

        if not self.check_shape_predict_data(predict_data.shape):
            message = "wrong shape of input predictions data array..." % (predict_data.shape)
            CatchErrorException(message)

        predict_full_array = np.zeros(self.size_total_image, dtype=FORMATPROBABILITYDATA)

        self.set_calc_reconstructed_image_array(predict_full_array)

        for index in range(self.num_samples_total):
            self.adding_reconstructed_image_sample_array(index, self.get_processed_image_sample_array(predict_data[index]))
        # endfor

        # multiply by factor to account for multiple overlaps of images samples
        return np.multiply(predict_full_array, self.factor_num_overlap_images_samples_per_voxel)


class SlidingWindowReconstructorImages2D(SlidingWindowReconstructorImages):

    def __init__(self, size_image_sample, size_total_image, prop_overlap, size_outUnet_sample=None):

        self.slidingWindow_generator = SlidingWindowImages2D(size_image_sample, prop_overlap, size_total=size_total_image)

        num_samples_total = self.slidingWindow_generator.get_num_images()

        if size_outUnet_sample and size_outUnet_sample != size_image_sample:
            filterImages_calculator = FilteringValidUnetOutput2D(size_image_sample, size_outUnet_sample)
        else:
            filterImages_calculator = None

        super(SlidingWindowReconstructorImages2D, self).__init__(size_image_sample, size_total_image, num_samples_total, filterImages_calculator)


    def adding_reconstructed_image_sample_array(self, index, image_sample_array):

        (x_left, x_right, y_down, y_up) = self.slidingWindow_generator.get_limits_image(index)

        # full_array[x_left:x_right, y_down:y_up, ...] += batch_array
        self.reconstructed_image_array[x_left:x_right, y_down:y_up] += image_sample_array


class SlidingWindowReconstructorImages3D(SlidingWindowReconstructorImages):

    def __init__(self, size_image_sample, size_total_image, prop_overlap, size_outUnet_sample=None):

        self.slidingWindow_generator = SlidingWindowImages3D(size_image_sample, prop_overlap, size_total=size_total_image)

        num_samples_total = self.slidingWindow_generator.get_num_images()

        if size_outUnet_sample and size_outUnet_sample != size_image_sample:
            filterImages_calculator = FilteringValidUnetOutput3D(size_image_sample, size_outUnet_sample)
        else:
            filterImages_calculator = None

        super(SlidingWindowReconstructorImages3D, self).__init__(size_image_sample, size_total_image, num_samples_total, filterImages_calculator)


    def adding_reconstructed_image_sample_array(self, index, image_sample_array):

        (z_back, z_front, x_left, x_right, y_down, y_up) = self.slidingWindow_generator.get_limits_image(index)

        # full_array[z_back:z_front, x_left:x_right, y_down:y_up, ...] += batch_array
        self.reconstructed_image_array[z_back:z_front, x_left:x_right, y_down:y_up] += image_sample_array


class SlicingReconstructorImages2D(SlidingWindowReconstructorImages2D):

    def __init__(self, size_image_sample, size_total_image, size_outUnet_sample=None):
        super(SlicingReconstructorImages2D, self).__init__(size_image_sample, size_total_image, (0.0, 0.0), size_outUnet_sample)

class SlicingReconstructorImages3D(SlidingWindowReconstructorImages3D):

    def __init__(self, size_image_sample, size_total_image, size_outUnet_sample=None):
        super(SlicingReconstructorImages3D, self).__init__(size_image_sample, size_total_image, (0.0, 0.0, 0.0), size_outUnet_sample)