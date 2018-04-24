#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
from CommonUtil.ErrorMessages import *
from Preprocessing.SlidingWindowImages import *
import numpy as np


class ReconstructorImages(object):

    def __init__(self, size_total_image, size_image_batch, num_images_total):

        self.size_total_image = size_total_image
        self.size_image_batch = size_image_batch
        self.num_images_total = num_images_total

        self.compute_factor_overlap_batches_per_voxel()

    def adding_reconstructed_batch(self, index, full_array, array_batch):
        pass


    def compute_factor_overlap_batches_per_voxel(self):
        # Compute how many times a batch image overlaps in same voxel

        num_overlap_batches_voxels = np.zeros(self.size_total_image, dtype=np.int8)

        for index in range(self.num_images_total):
            self.adding_reconstructed_batch(index, num_overlap_batches_voxels, np.ones(self.size_image_batch))
        #endfor

        # get position where there's no overlap
        pos_non_overlap = np.argwhere(num_overlap_batches_voxels == 0)

        self.factor_overlap_batches_per_voxel = np.divide(np.ones(self.size_total_image, dtype=np.float32), num_overlap_batches_voxels)

        # remove pos where there was division by zero
        for pos in pos_non_overlap:
            self.factor_overlap_batches_per_voxel[tuple(pos)] = 0.0

    def compute(self, yPredict):

        if yPredict.shape[0] != self.num_images_total:
            message = "size of \'yPredict\' not equal to num image batches..."
            CatchErrorException(message)

        predictions_array = np.zeros(self.size_total_image, dtype=FORMATPREDICTDATA)

        for index in range(self.num_images_total):
            self.adding_reconstructed_batch(index, predictions_array, yPredict[index])
        #endfor

        # multiply by factor to account for multiple overlaps of batch images
        return np.multiply(predictions_array, self.factor_overlap_batches_per_voxel)


class ReconstructorImages2D(ReconstructorImages):

    def __init__(self, size_total_image, size_image_batch, prop_overlap=(0.0, 0.0)):

        self.batchReconstructor = SlidingWindowImages2D(size_total_image, size_image_batch, prop_overlap)

        super(ReconstructorImages2D, self).__init__(size_total_image, size_image_batch, self.batchReconstructor.get_num_images_total())

    def adding_reconstructed_batch(self, index, full_array, array_batch):

        (x_left, x_right, y_down, y_up) = self.batchReconstructor.get_limits_image(index)

        full_array[..., x_left:x_right, y_down:y_up] += array_batch


class ReconstructorImages3D(ReconstructorImages):

    def __init__(self, size_total_image, size_image_batch, prop_overlap=(0.0, 0.0, 0.0)):

        self.batchReconstructor = SlidingWindowImages3D(size_total_image, size_image_batch, prop_overlap)

        super(ReconstructorImages3D, self).__init__(size_total_image, size_image_batch, self.batchReconstructor.get_num_images_total())

    def adding_reconstructed_batch(self, index, full_array, array_batch):

        (z_back, z_front, x_left, x_right, y_down, y_up) = self.batchReconstructor.get_limits_image(index)

        full_array[..., z_back:z_front, x_left:x_right, y_down:y_up] += array_batch