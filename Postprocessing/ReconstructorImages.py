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


    def run_check_shape_Ypredict(self, yPredict_shape):

        return (yPredict_shape[0] == self.num_images_total) and (yPredict_shape[1:-2] != self.size_image_batch) and (len(yPredict_shape) == len(self.size_image_batch) + 2)

    def get_predictMasks_batch_array(self, predictMasks_batch):

        if self.num_classes_out == 1:
            return np.squeeze(predictMasks_batch, axis=-1)
        else:
            return self.get_multiclass_predict_array(predictMasks_batch)

    def get_multiclass_predict_array(self, predict_array):
        pass
    
    def add_predictMasks_batch_toFullArray(self, index, predictMasks_array, predictMasks_batch):
        pass


    def compute_factor_overlap_batches_per_voxel(self):
        # Compute how many times a batch image overlaps in same voxel

        num_overlap_batches_voxels = np.zeros(self.size_total_image, dtype=np.int8)

        for index in range(self.num_images_total):
            self.add_predictMasks_batch_toFullArray(index, num_overlap_batches_voxels, np.ones(self.size_image_batch, dtype=np.int8))
        #endfor

        # get position where there's no overlap
        pos_non_overlap = np.argwhere(num_overlap_batches_voxels == 0)

        self.factor_overlap_batches_per_voxel = np.divide(np.ones(self.size_total_image, dtype=np.float32), num_overlap_batches_voxels)

        # remove pos where there was division by zero
        for pos in pos_non_overlap:
            self.factor_overlap_batches_per_voxel[tuple(pos)] = 0.0

    def compute(self, yPredict):

        if not self.run_check_shape_Ypredict(yPredict.shape):
            message = "wrong shape of input predictions array..." %(yPredict.shape)
            CatchErrorException(message)

        self.num_classes_out = yPredict.shape[-1]

        predictMasks_array = np.zeros(self.size_total_image, dtype=FORMATPREDICTDATA)

        for index in range(self.num_images_total):
            self.add_predictMasks_batch_toFullArray(index, predictMasks_array, self.get_predictMasks_batch_array(yPredict[index]))
        #endfor

        # multiply by factor to account for multiple overlaps of batch images
        return np.multiply(predictMasks_array, self.factor_overlap_batches_per_voxel)


class ReconstructorImages2D(ReconstructorImages):

    def __init__(self, size_total_image, size_image_batch, prop_overlap=(0.0, 0.0)):

        self.batchReconstructor = SlidingWindowImages2D(size_total_image, size_image_batch, prop_overlap)

        super(ReconstructorImages2D, self).__init__(size_total_image, size_image_batch, self.batchReconstructor.get_num_images_total())

    def get_multiclass_predict_array(self, predict_array):

        new_predict_array = np.ndarray(self.size_image_batch, dtype=predict_array.dtype)

        for i in range(self.size_image_batch[0]):
            for j in range(self.size_image_batch[1]):
                index_argmax = np.argmax(predict_array[i,j,:])
                new_predict_array[i,j] = index_argmax * predict_array[i,j,index_argmax]
            #endfor
        #endfor
        return new_predict_array

    def add_predictMasks_batch_toFullArray(self, index, predictMasks_array, predictMasks_batch):

        (x_left, x_right, y_down, y_up) = self.batchReconstructor.get_limits_image(index)

        predictMasks_array[..., x_left:x_right, y_down:y_up] += predictMasks_batch


class ReconstructorImages3D(ReconstructorImages):

    def __init__(self, size_total_image, size_image_batch, prop_overlap=(0.0, 0.0, 0.0)):

        self.batchReconstructor = SlidingWindowImages3D(size_total_image, size_image_batch, prop_overlap)

        super(ReconstructorImages3D, self).__init__(size_total_image, size_image_batch, self.batchReconstructor.get_num_images_total())

    def get_multiclass_predict_array(self, predict_array):

        new_predict_array = np.ndarray(self.size_image_batch, dtype=predict_array.dtype)

        for i in range(self.size_image_batch[0]):
            for j in range(self.size_image_batch[1]):
                for k in range(self.size_image_batch[2]):
                    index_argmax = np.argmax(predict_array[i,j,k,:])
                    new_predict_array[i,j,k] = index_argmax
                #endfor
            #endfor
        #endfor
        return new_predict_array

    def add_predictMasks_batch_toFullArray(self, index, predictMasks_array, predictMasks_batch):

        (z_back, z_front, x_left, x_right, y_down, y_up) = self.batchReconstructor.get_limits_image(index)

        predictMasks_array[..., z_back:z_front, x_left:x_right, y_down:y_up] += predictMasks_batch