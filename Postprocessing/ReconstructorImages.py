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


class ReconstructorImage(object):

    def __init__(self, size_total_image, size_image_batch, prop_overlap=(0.0, 0.0, 0.0)):

        self.size_total_image  = size_total_image
        self.size_image_batch  = size_image_batch
        self.prop_overlap      = prop_overlap
        self.batchReconstructor= SlidingWindowImages(size_image_batch, prop_overlap)

        self.compute_factor_overlap_batches_per_voxel()


    def compute_factor_overlap_batches_per_voxel(self):
        # Compute how many times a batch image overlaps in same voxel

        (num_images_z, num_images_x, num_images_y) = self.batchReconstructor.get_num_images_3d(self.size_total_image)
        num_images = num_images_x * num_images_y * num_images_z

        num_overlap_batches_voxels = np.ndarray(self.size_total_image, dtype=np.int8)
        num_overlap_batches_voxels[:,:,:] = 0

        for index in range(num_images):

            (index_z, index_x, index_y) = self.batchReconstructor.get_indexes_3d(index, (num_images_x, num_images_y))

            ((z_back, z_front), (x_left, x_right), (y_down, y_up)) = self.batchReconstructor.get_limits_image_3d((index_z, index_x, index_y))

            num_overlap_batches_voxels[z_back:z_front, x_left:x_right, y_down:y_up] += np.ones(self.size_image_batch, dtype=np.int8)
        #endfor

        # get position where there's no overlap
        pos_non_overlap = np.argwhere(num_overlap_batches_voxels == 0)

        self.factor_overlap_batches_per_voxel = np.divide(np.ones(self.size_total_image, dtype=np.float32), num_overlap_batches_voxels)

        # remove pos where there was division by zero
        for pos in pos_non_overlap:
            self.factor_overlap_batches_per_voxel[tuple(pos)] = 0.0


    def compute(self, yPredict):

        (num_images_z, num_images_x, num_images_y) = self.batchReconstructor.get_num_images_3d(self.size_total_image)
        num_images = num_images_x * num_images_y * num_images_z

        if yPredict.shape[0] != num_images:
            message = "size of \'yPredict\' not equal to num image batches..."
            CatchErrorException(message)

        predictions_array = np.ndarray(self.size_total_image, dtype=FORMATPREDICTDATA)
        predictions_array[:,:,:] = 0

        for index in range(num_images):

            (index_z, index_x, index_y) = self.batchReconstructor.get_indexes_3d(index, (num_images_x, num_images_y))

            ((z_back, z_front), (x_left, x_right), (y_down, y_up)) = self.batchReconstructor.get_limits_image_3d((index_z, index_x, index_y))

            predictions_array[z_back:z_front, x_left:x_right, y_down:y_up] += np.asarray(yPredict[index], dtype=predictions_array.dtype)
        #endfor

        # multiply by factor to account for multiple overlaps of batch images
        return np.multiply(predictions_array, self.factor_overlap_batches_per_voxel)