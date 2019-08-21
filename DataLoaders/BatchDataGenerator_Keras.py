#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Preprocessing.BoundingBoxes import *
from Preprocessing.OperationImages import CropImages
from keras.preprocessing import image
import numpy as np
np.random.seed(2017)


class TrainingBatchDataGenerator(image.Iterator):

    def __init__(self, size_image,
                 list_xData_array,
                 list_yData_array,
                 images_generator,
                 num_channels_in= 1,
                 num_classes_out= 1,
                 isUse_valid_convs= False,
                 size_output_model= None,
                 batch_size= 1,
                 shuffle= True,
                 seed= None,
                 iswrite_datagen_info= True):
        self.size_image = size_image
        self.num_channels_in = num_channels_in

        self.list_xData_array = list_xData_array
        self.type_xData = list_xData_array[0].dtype
        self.list_yData_array = list_yData_array
        self.type_yData = list_yData_array[0].dtype
        self.images_generator = images_generator

        self.num_classes_out = num_classes_out
        self.isUse_valid_convs = isUse_valid_convs
        if isUse_valid_convs and size_output_model:
            self.size_output_model = size_output_model
        else:
            self.size_output_model = size_image
        self.iswrite_datagen_info = iswrite_datagen_info

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

            if self.iswrite_datagen_info:
                size_total_image = xData_array.shape
                num_samples_3dirs = self.images_generator.get_num_images_dirs()
                print("sliding-window gen. from image \'%s\' of size: " "\'%s\': num samples \'%s\', in local dirs: \'%s\'"
                      %(ifile, str(size_total_image), num_samples_file, str(num_samples_3dirs)))

                limits_images_dirs = self.images_generator.get_limits_images_dirs()
                ndirs = len(num_samples_3dirs)
                for i in range(ndirs):
                    print("coords of images in dir \'%s\': \'%s\'..." %(i, limits_images_dirs[i]))
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

    def get_crop_output(self, in_array, size_crop):
        crop_bounding_box = BoundingBoxes.compute_bounding_box_centered_image_3D(size_crop, self.size_image)
        return CropImages.compute3D(in_array, crop_bounding_box)


    def _get_batches_of_transformed_samples(self, indexes_batch):
        num_samples_batch = len(indexes_batch)
        out_xData_array_shape = [num_samples_batch] + list(self.size_image) + [self.num_channels_in]
        out_yData_array_shape = [num_samples_batch] + list(self.size_output_model) + [self.num_classes_out]
        out_xData_array = np.ndarray(out_xData_array_shape, dtype=self.type_xData)
        out_yData_array = np.ndarray(out_yData_array_shape, dtype=self.type_yData)

        for i, index in enumerate(indexes_batch):
            (index_file, index_sample_file) = self.list_pairIndexes_samples[index]
            self.images_generator.complete_init_data(self.list_xData_array[index_file].shape)
            (xData_elem, yData_elem) = self.images_generator.get_images_array(self.list_xData_array[index_file],
                                                                              index=index_sample_file,
                                                                              masks_array=self.list_yData_array[index_file])
            out_xData_array[i] = xData_elem
            if self.isUse_valid_convs:
                out_yData_array[i] = self.get_crop_output(yData_elem, self.size_output_model)
            else:
                out_yData_array[i] = yData_elem
        #endfor
        return (out_xData_array, out_yData_array)