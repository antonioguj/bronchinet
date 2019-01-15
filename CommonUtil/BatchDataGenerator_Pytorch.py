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
from CommonUtil.ArrayShapeManager import *
from torch.utils import data
import numpy as np
np.random.seed(2017)


class TrainingBatchDataGenerator(data.DataLoader):

    def __init__(self, size_image, list_xData_array, list_yData_array, images_generator, num_classes_out=1, size_outUnet=None, batch_size=1, shuffle=True, seed=None):

        data_sample_generator = DataSampleGenerator(size_image, list_xData_array, list_yData_array, images_generator, num_classes_out, size_outUnet)

        super(TrainingBatchDataGenerator, self).__init__(data_sample_generator, batch_size=batch_size, shuffle=shuffle)


class DataSampleGenerator(data.Dataset):

    def __init__(self, size_image, list_xData_array, list_yData_array, images_generator, num_classes_out=1, size_outUnet=None):
        self.size_image       = size_image
        self.list_xData_array = list_xData_array
        self.type_xData       = list_xData_array[0].dtype
        self.list_yData_array = list_yData_array
        self.type_yData       = list_yData_array[0].dtype

        self.images_generator = images_generator

        self.array_shape_manager = ArrayShapeManager(size_image, is_shaped_Keras=True, num_classes_out=num_classes_out, size_outUnet=size_outUnet)

        self.num_channels_in  = self.array_shape_manager.get_num_channels_array(self.list_xData_array[0].shape)
        self.num_classes_out  = num_classes_out

        self.num_samples = self.compute_pairIndexes_samples()


    def compute_pairIndexes_samples(self):

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
        return num_samples


    def __getitem__(self, index):

        (index_file, index_sample_file) = self.list_pairIndexes_samples[index]

        self.images_generator.complete_init_data(self.list_xData_array[index_file].shape)

        (xData_elem, yData_elem) = self.images_generator.get_images_array(self.list_xData_array[index_file],
                                                                          index=index_sample_file,
                                                                          masks_array=self.list_yData_array[index_file])

        out_xData_array = self.array_shape_manager.get_xData_array_reshaped(xData_elem)
        out_yData_array = self.array_shape_manager.get_yData_array_reshaped(yData_elem)

        return (out_xData_array, out_yData_array)

    def __len__(self):
        return self.num_samples