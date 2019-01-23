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
import torch
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

    @staticmethod
    def convert_image_torchtensor(in_array):
        return torch.from_numpy(np.expand_dims(in_array, axis=0).copy()).type(torch.FloatTensor)


    def get_full_data(self):

        out_data_shape  = [self.num_samples] + list(self.size_image)
        out_xData_array = np.ndarray(out_data_shape, dtype= self.type_xData)
        out_yData_array = np.ndarray(out_data_shape, dtype= self.type_yData)

        for i in range(self.num_samples):
            (out_xData_array[i], out_yData_array[i]) = self.__getitem__(i)
        #endfor

        return (out_xData_array, out_yData_array)


    def __getitem__(self, index):

        (index_file, index_sample_file) = self.list_pairIndexes_samples[index]

        self.images_generator.complete_init_data(self.list_xData_array[index_file].shape)

        (xData_elem, yData_elem) = self.images_generator.get_images_array(self.list_xData_array[index_file],
                                                                          index=index_sample_file,
                                                                          masks_array=self.list_yData_array[index_file])

        out_xData_array = self.convert_image_torchtensor(xData_elem)
        out_yData_array = self.convert_image_torchtensor(yData_elem)

        return (out_xData_array, out_yData_array)

    def __len__(self):
        return self.num_samples