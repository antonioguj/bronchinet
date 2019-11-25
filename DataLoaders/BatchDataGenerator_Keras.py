#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from DataLoaders.BatchDataGenerator import *
from keras.preprocessing import image
import numpy as np
np.random.seed(2017)



class BatchDataGenerator_Keras(BatchDataGenerator, image.Iterator):

    def __init__(self, size_image,
                 list_xData_array,
                 list_yData_array,
                 images_generator,
                 num_channels_in= 1,
                 num_classes_out= 1,
                 isUse_valid_convs= False,
                 size_output_image= None,
                 batch_size= 1,
                 shuffle= True,
                 seed= None,
                 iswrite_datagen_info= True):
        super(BatchDataGenerator_Keras, self).__init__(size_image,
                                                       list_xData_array,
                                                       list_yData_array,
                                                       images_generator,
                                                       num_channels_in=num_channels_in,
                                                       num_classes_out=num_classes_out,
                                                       isUse_valid_convs=isUse_valid_convs,
                                                       size_output_image=size_output_image,
                                                       batch_size=batch_size,
                                                       iswrite_datagen_info=iswrite_datagen_info)

        self.num_images = self.compute_pairIndexes_imagesFile(shuffle, seed=seed)

        super(BatchDataGenerator_Keras, self).__init__(self.num_images, batch_size, shuffle, seed)


    def get_reshaped_output_array(self, in_array):
        return np.expand_dims(in_array, axis=-1)


    def get_formated_output_xData(self, in_array):
        return self.get_reshaped_output_array(in_array)

    def get_formated_output_yData(self, in_array):
        if self.isUse_valid_convs:
            out_array = self.get_cropped_output(in_array)
        else:
            out_array = in_array
        return self.get_formated_output_xData(out_array)


    def _get_batches_of_transformed_samples(self, indexes_batch):
        num_images_batch = len(indexes_batch)
        out_xData_array_shape = [num_images_batch] + list(self.size_image) + [self.num_channels_in]
        out_yData_array_shape = [num_images_batch] + list(self.size_output_model) + [self.num_classes_out]
        out_xData_array = np.ndarray(out_xData_array_shape, dtype=self.type_xData)
        out_yData_array = np.ndarray(out_yData_array_shape, dtype=self.type_yData)

        for i, index in enumerate(indexes_batch):
            (out_xData_array[i], out_yData_array[i]) = self.get_item(index)
        #endfor

        return (out_xData_array, out_yData_array)