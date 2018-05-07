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

    def __init__(self, size_image_batch, size_total_image, num_images_total, num_classes_out=1):

        self.size_image_batch = size_image_batch
        self.size_total_image = size_total_image
        self.num_images_total = num_images_total
        self.num_classes_out  = num_classes_out

        self.compute_factor_overlap_batches_per_voxel()


    def run_check_shape_predict_data(self, predict_data_shape):

        return (predict_data_shape[0] == self.num_images_total) and (predict_data_shape[1:-2] != self.size_image_batch) and (len(predict_data_shape) == len(self.size_image_batch) + 2)

    def get_reconstructed_batch_array(self, array_batch):

        if self.num_classes_out == 1:
            return np.squeeze(array_batch, axis=-1)
        else:
            return self.get_reconstructed_batch_multiclass(array_batch)

    def set_calc_reconstructed_array(self, full_array):
        self.calc_reconstructed_array = full_array

    def get_reconstructed_batch_multiclass(self, batch_array):
        pass
    
    def add_reconstructed_batch_array(self, index, full_array, batch_array):
        pass

    def compute_factor_overlap_batches_per_voxel(self):
        # Compute how many times a batch image overlaps in same voxel

        num_overlap_batches_voxels = np.zeros(self.size_total_image, dtype=np.int8)

        self.set_calc_reconstructed_array(num_overlap_batches_voxels)

        for index in range(self.num_images_total):
            self.add_reconstructed_batch_array(index, np.ones(self.size_image_batch, dtype=np.int8))
        #endfor

        # get position where there's no overlap
        pos_non_overlap = np.argwhere(num_overlap_batches_voxels == 0)

        self.factor_overlap_batches_per_voxel = np.divide(np.ones(self.size_total_image, dtype=np.float32), num_overlap_batches_voxels)

        # remove pos where there was division by zero
        for pos in pos_non_overlap:
            self.factor_overlap_batches_per_voxel[tuple(pos)] = 0.0


    def compute(self, predict_data):

        if not self.run_check_shape_predict_data(predict_data.shape):
            message = "wrong shape of input predictions array..." %(predict_data.shape)
            CatchErrorException(message)

        predictMasks_array = np.zeros(self.size_total_image, dtype=FORMATPREDICTDATA)

        self.set_calc_reconstructed_array(predictMasks_array)

        for index in range(self.num_images_total):
            self.add_reconstructed_batch_array(index, self.get_reconstructed_batch_array(predict_data[index]))
        #endfor

        # multiply by factor to account for multiple overlaps of batch images
        return np.multiply(predictMasks_array, self.factor_overlap_batches_per_voxel)


class ReconstructorImages2D(ReconstructorImages):

    def __init__(self, size_image_batch, size_total_image, prop_overlap=(0.0, 0.0), num_classes_out=1):

        self.images_reconstructor = SlidingWindowImages2D(size_image_batch, prop_overlap, size_total=size_total_image)

        num_images_total = self.images_reconstructor.get_num_images()

        super(ReconstructorImages2D, self).__init__(size_image_batch, size_total_image, num_images_total, num_classes_out=num_classes_out)

    def add_reconstructed_batch_array(self, index, batch_array):

        (x_left, x_right, y_down, y_up) = self.images_reconstructor.get_limits_image(index)

        #full_array[x_left:x_right, y_down:y_up, ...] += batch_array
        self.calc_reconstructed_array[x_left:x_right, y_down:y_up] += batch_array

    def get_reconstructed_batch_multiclass(self, batch_array):

        new_batch_array = np.ndarray(self.size_image_batch, dtype=batch_array.dtype)

        for i in range(self.size_image_batch[0]):
            for j in range(self.size_image_batch[1]):
                index_argmax = np.argmax(batch_array[i,j,:])
                new_batch_array[i,j] = index_argmax * batch_array[i,j,index_argmax]
            #endfor
        #endfor
        return new_batch_array


class ReconstructorImages3D(ReconstructorImages):

    def __init__(self, size_image_batch, size_total_image, prop_overlap=(0.0, 0.0, 0.0), num_classes_out=1):

        self.images_reconstructor = SlidingWindowImages3D(size_image_batch, prop_overlap, size_total=size_total_image)

        num_images_total = self.images_reconstructor.get_num_images()

        super(ReconstructorImages3D, self).__init__(size_image_batch, size_total_image, num_images_total, num_classes_out=num_classes_out)

    def add_reconstructed_batch_array(self, index, batch_array):

        (z_back, z_front, x_left, x_right, y_down, y_up) = self.images_reconstructor.get_limits_image(index)

        #full_array[z_back:z_front, x_left:x_right, y_down:y_up, ...] += batch_array
        self.calc_reconstructed_array[z_back:z_front, x_left:x_right, y_down:y_up] += batch_array

    def get_reconstructed_batch_multiclass(self, batch_array):

        new_batch_array = np.ndarray(self.size_image_batch, dtype=batch_array.dtype)

        for i in range(self.size_image_batch[0]):
            for j in range(self.size_image_batch[1]):
                for k in range(self.size_image_batch[2]):
                    index_argmax = np.argmax(batch_array[i,j,k,:])
                    new_batch_array[i,j,k] = index_argmax
                #endfor
            #endfor
        #endfor
        return new_batch_array