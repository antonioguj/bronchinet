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

    def __init__(self, size_image_sample, size_total_image, num_samples_total):

        self.size_image_sample = size_image_sample
        self.size_total_image  = size_total_image
        self.num_samples_total = num_samples_total

        super(SlidingWindowReconstructorImages, self).__init__(size_image_sample)

        self.compute_factor_overlap_images_samples_per_voxel()


    def check_shape_predict_data(self, predict_data_shape):

        return (len(predict_data_shape) == len(self.size_total_image) + 2) and \
               (predict_data_shape[0] == self.num_samples_total) and \
               (predict_data_shape[1:-2] != self.size_total_image)

    def adding_reconstructed_image_sample_array(self, index, image_sample_array):
        pass

    def set_calc_reconstructed_image_array(self, image_array):
        self.reconstructed_image_array = image_array

    def compute_factor_overlap_images_samples_per_voxel(self):
        # Compute how many times a batch image overlaps in same voxel

        num_overlap_images_samples_per_voxels = np.zeros(self.size_total_image, dtype=np.int8)

        self.set_calc_reconstructed_image_array(num_overlap_images_samples_per_voxels)

        for index in range(self.num_samples_total):
            self.adding_reconstructed_image_sample_array(index, np.ones(self.size_image_sample, dtype=np.int8))
        # endfor

        # get position where there's no overlap
        pos_non_overlap = np.argwhere(num_overlap_images_samples_per_voxels == 0)

        self.factor_overlap_images_samples_per_voxel = np.divide(np.ones(self.size_total_image, dtype=np.float32),
                                                                 num_overlap_images_samples_per_voxels)

        # remove pos where there was division by zero
        for pos in pos_non_overlap:
            self.factor_overlap_images_samples_per_voxel[tuple(pos)] = 0.0


    def compute(self, predict_data):

        if not self.check_shape_predict_data(predict_data.shape):
            message = "wrong shape of input predictions data array..." % (predict_data.shape)
            CatchErrorException(message)

        predict_full_array = np.zeros(self.size_total_image, dtype=FORMATPREDICTDATA)

        self.set_calc_reconstructed_image_array(predict_full_array)

        for index in range(self.num_samples_total):
            self.adding_reconstructed_image_sample_array(index, self.get_reconstructed_image_sample_array(predict_data[index]))
        # endfor

        # multiply by factor to account for multiple overlaps of images samples
        return np.multiply(predict_full_array, self.factor_overlap_images_samples_per_voxel)


class SlidingWindowReconstructorImages2D(SlidingWindowReconstructorImages):

    def __init__(self, size_image_sample, size_total_image, prop_overlap):

        self.slidingWindow_generator = SlidingWindowImages2D(size_image_sample, prop_overlap, size_total=size_total_image)

        num_samples_total = self.slidingWindow_generator.get_num_images()

        super(SlidingWindowReconstructorImages2D, self).__init__(size_image_sample, size_total_image, num_samples_total)


    def adding_reconstructed_image_sample_array(self, index, image_sample_array):

        (x_left, x_right, y_down, y_up) = self.slidingWindow_generator.get_limits_image(index)

        # full_array[x_left:x_right, y_down:y_up, ...] += batch_array
        self.reconstructed_image_array[x_left:x_right, y_down:y_up] += image_sample_array


class SlidingWindowReconstructorImages3D(SlidingWindowReconstructorImages):

    def __init__(self, size_image_sample, size_total_image, prop_overlap):

        self.slidingWindow_generator = SlidingWindowImages3D(size_image_sample, prop_overlap, size_total=size_total_image)

        num_samples_total = self.slidingWindow_generator.get_num_images()

        super(SlidingWindowReconstructorImages3D, self).__init__(size_image_sample, size_total_image, num_samples_total)


    def adding_reconstructed_image_sample_array(self, index, image_sample_array):

        (z_back, z_front, x_left, x_right, y_down, y_up) = self.slidingWindow_generator.get_limits_image(index)

        # full_array[z_back:z_front, x_left:x_right, y_down:y_up, ...] += batch_array
        self.reconstructed_image_array[z_back:z_front, x_left:x_right, y_down:y_up] += image_sample_array


class SlicingReconstructorImages2D(SlidingWindowReconstructorImages2D):

    def __init__(self, size_image_sample, size_total_image):
        super(SlicingReconstructorImages2D, self).__init__(size_image_sample, size_total_image, (0.0, 0.0))

class SlicingReconstructorImages3D(SlidingWindowReconstructorImages3D):

    def __init__(self, size_image_sample, size_total_image):
        super(SlicingReconstructorImages3D, self).__init__(size_image_sample, size_total_image, (0.0, 0.0, 0.0))