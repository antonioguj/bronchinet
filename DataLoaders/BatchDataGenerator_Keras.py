#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from DataLoaders.ArrayShapeManager import *
from keras.preprocessing import image
import numpy as np
np.random.seed(2017)


class TrainingBatchDataGenerator(image.Iterator):

    def __init__(self, size_image,
                 list_xData_array,
                 list_yData_array,
                 images_generator,
                 num_classes_out= 1,
                 size_output_Unet= None,
                 batch_size =1,
                 shuffle =True,
                 seed =None):
        self.size_image = size_image
        self.list_xData_array = list_xData_array
        self.type_xData = list_xData_array[0].dtype
        self.list_yData_array = list_yData_array
        self.type_yData = list_yData_array[0].dtype
        self.images_generator = images_generator

        self.array_shape_manager = ArrayShapeManager(size_image,
                                                     is_shaped_Keras= True,
                                                     num_classes_out= num_classes_out,
                                                     size_output_Unet= size_output_Unet)

        self.num_channels_in  = self.array_shape_manager.get_num_channels_array(self.list_xData_array[0].shape)
        self.num_classes_out  = num_classes_out
        num_samples = self.compute_pairIndexes_samples(shuffle, seed)

        super(TrainingBatchDataGenerator, self).__init__(num_samples,
                                                         batch_size,
                                                         shuffle,
                                                         seed)

    def compute_pairIndexes_samples(self, shuffle, seed=None):
        self.list_pairIndexes_samples = []
        for ifile, xData_array in enumerate(self.list_xData_array):
            self.images_generator.complete_init_data(xData_array.shape)
            num_samples_file = self.images_generator.get_num_images()
            #store pair of indexes: (idx_file, idx_batch)
            for index in range(num_samples_file):
                self.list_pairIndexes_samples.append((ifile, index))
            #endfor
        #endfor

        num_samples = len(self.list_pairIndexes_samples)
        if (shuffle):
            if seed:
                np.random.seed(seed)
            randomIndexes = np.random.choice(num_samples, size=num_samples, replace=False)
            self.list_pairIndexes_samples_old = self.list_pairIndexes_samples
            self.list_pairIndexes_samples = []
            for index in randomIndexes:
                self.list_pairIndexes_samples.append(self.list_pairIndexes_samples_old[index])
            #endfor

        return num_samples


    def _get_batches_of_transformed_samples(self, indexes_batch):

        num_samples_batch = len(indexes_batch)
        out_xData_array_shape = self.array_shape_manager.get_shape_out_array(num_samples_batch, self.num_channels_in)
        out_yData_array_shape = self.array_shape_manager.get_shape_out_array(num_samples_batch, self.num_classes_out)
        out_xData_array = np.ndarray(out_xData_array_shape, dtype=self.type_xData)
        out_yData_array = np.ndarray(out_yData_array_shape, dtype=self.type_yData)

        for i, index in enumerate(indexes_batch):
            (index_file, index_sample_file) = self.list_pairIndexes_samples[index]
            self.images_generator.complete_init_data(self.list_xData_array[index_file].shape)
            (xData_elem, yData_elem) = self.images_generator.get_images_array(self.list_xData_array[index_file],
                                                                              index=index_sample_file,
                                                                              masks_array=self.list_yData_array[index_file])
            out_xData_array[i] = self.array_shape_manager.get_xData_array_reshaped(xData_elem)
            out_yData_array[i] = self.array_shape_manager.get_yData_array_reshaped(yData_elem)
        #endfor
        return (out_xData_array, out_yData_array)