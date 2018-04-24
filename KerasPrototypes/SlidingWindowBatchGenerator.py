#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.LoadDataManager import OperationsArraysUseInKeras
from Preprocessing.SlidingWindowImages import *
from keras.preprocessing import image
import numpy as np
np.random.seed(2017)


class SlidingWindowBatchGenerator(image.Iterator):

    def __init__(self, list_Xdata, list_Ydata, size_image, prop_overlap, num_classes_out=1, size_outnnet=None, batch_size=1, shuffle=True, seed=None):

        self.list_Xdata = list_Xdata
        self.list_Ydata = list_Ydata
        self.type_Xdata = list_Xdata[0].dtype
        self.type_Ydata = list_Ydata[0].dtype

        self.size_image      = size_image
        self.num_classes_out = num_classes_out
        if size_outnnet and (size_outnnet != size_image):
            self.size_outnnet = size_outnnet
        else:
            self.size_outnnet = size_image

        self.opersArrays = OperationsArraysUseInKeras(size_image, num_classes_out=num_classes_out, size_outnnet=size_outnnet)

        self.num_channels_in = self.opersArrays.get_num_channels_array(self.list_Xdata[0].shape)

        self.list_slidingWindow_images_generator = []
        self.list_indexes_compute_images_batches = []

        for ifile, (Xdata, Ydata) in enumerate(zip(list_Xdata, list_Ydata)):

            self.list_slidingWindow_images_generator.append(SlidingWindowImages3D(Xdata.shape, self.size_image, prop_overlap))

            num_images_total = self.list_slidingWindow_images_generator[ifile].get_num_images_total()

            print('File %s: Generate images by SlidingWindow: size: %s; num batches: %s...' %(ifile, Xdata.shape, num_images_total))

            for index in range(num_images_total):
                # Store indexes to compute images batches: (idx_fullimage, idx_image_batch
                self.list_indexes_compute_images_batches.append((ifile, index))
            #endfor
        #endfor

        num_images_total_all = len(self.list_indexes_compute_images_batches)

        if (shuffle):
            # shuffle indexes to compute images batches
            randomIndexes = np.random.choice(num_images_total_all, size=num_images_total_all, replace=False)

            list_indexes_compute_images_batches_old = self.list_indexes_compute_images_batches
            self.list_indexes_compute_images_batches = []
            for index in randomIndexes:
                self.list_indexes_compute_images_batches.append(list_indexes_compute_images_batches_old[index])
            #endfor

        super(SlidingWindowBatchGenerator, self).__init__(num_images_total_all, batch_size, shuffle, seed)


    def _get_batches_of_transformed_samples(self, list_indexes_array):

        # overwrite function to retrieve images batches
        num_images_batch = len(list_indexes_array)

        xData_shape = self.opersArrays.get_shape_out_array(num_images_batch, num_channels=self.num_channels_in)
        yData_shape = self.opersArrays.get_shape_out_array(num_images_batch, num_channels=self.num_classes_out)
        Xdata_batch = np.ndarray(xData_shape, dtype=self.type_Xdata)
        Ydata_batch = np.ndarray(yData_shape, dtype=self.type_Ydata)

        for i, index in enumerate(list_indexes_array):

            (Xdata_batch_tmp, Ydata_batch_tmp) = self.get_X_Y_data_index_batch(index)

            Xdata_batch[i] = self.opersArrays.get_array_reshaped_Keras(self.opersArrays.get_array_reshaped(Xdata_batch_tmp))

            if self.num_classes_out > 1:
                Ydata_batch[i] = self.opersArrays.get_array_reshaped_Keras(self.opersArrays.get_array_categorical_masks(self.opersArrays.get_array_cropImages_outNnet(Ydata_batch_tmp)))
            else:
                Ydata_batch[i] = self.opersArrays.get_array_reshaped_Keras(self.opersArrays.get_array_reshaped(self.opersArrays.get_array_cropImages_outNnet(Ydata_batch_tmp)))
        #endfor

        return (Xdata_batch, Ydata_batch)

    def get_X_Y_data_index_batch(self, index):

        (idx_fullimage, index_image_batch) = self.list_indexes_compute_images_batches[index]

        return (self.list_slidingWindow_images_generator[idx_fullimage].get_image_array(self.list_Xdata[idx_fullimage], index_image_batch),
                self.list_slidingWindow_images_generator[idx_fullimage].get_image_array(self.list_Ydata[idx_fullimage], index_image_batch))