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
from torch.utils import data
import torch



class BatchDataGenerator_Pytorch(BatchDataGenerator):

    def __init__(self, size_image,
                 list_xData_array,
                 list_yData_array,
                 images_generator,
                 num_channels_in= 1,
                 num_classes_out= 1,
                 is_outputUnet_validconvs= False,
                 size_output_image= None,
                 batch_size= 1,
                 shuffle= True,
                 seed= None,
                 iswrite_datagen_info= True,
                 is_datagen_in_gpu= True,
                 is_datagen_halfPrec= False):
        super(BatchDataGenerator_Pytorch, self).__init__(size_image,
                                                         list_xData_array,
                                                         list_yData_array,
                                                         images_generator,
                                                         num_channels_in=num_channels_in,
                                                         num_classes_out=num_classes_out,
                                                         is_outputUnet_validconvs=is_outputUnet_validconvs,
                                                         size_output_image=size_output_image,
                                                         batch_size=batch_size,
                                                         shuffle=shuffle,
                                                         seed=seed,
                                                         iswrite_datagen_info= iswrite_datagen_info)
        if is_datagen_in_gpu:
            if is_datagen_halfPrec:
                self.type_data_generated_torch = torch.cuda.HalfTensor
            else:
                self.type_data_generated_torch = torch.cuda.FloatTensor
        else:
            if is_datagen_halfPrec:
                self.type_data_generated_torch = torch.HalfTensor
            else:
                self.type_data_generated_torch = torch.FloatTensor


    def __getitem__(self, index):
        return super(BatchDataGenerator_Pytorch, self).get_item(index)


    def get_image_torchtensor(self, in_array):
        return torch.from_numpy(in_array.copy()).type(self.type_data_generated_torch)

    def get_reshaped_output_array(self, in_array):
        return np.expand_dims(in_array, axis=0)


    def get_formated_output_xData(self, in_array):
        out_array = self.get_reshaped_output_array(in_array)
        return self.get_image_torchtensor(out_array)

    def get_formated_output_yData(self, in_array):
        if self.is_outputUnet_validconvs:
            out_array = self.get_cropped_output(in_array)
        else:
            out_array = in_array
        return self.get_formated_output_xData(out_array)



class WrapperBatchGenerator_Pytorch(data.DataLoader):

    def __init__(self, size_image,
                 list_xData_array,
                 list_yData_array,
                 images_generator,
                 num_channels_in= 1,
                 num_classes_out= 1,
                 is_outputUnet_validconvs= False,
                 size_output_image= None,
                 batch_size= 1,
                 shuffle= True,
                 seed= None,
                 iswrite_datagen_info= True,
                 is_datagen_in_gpu= True,
                 is_datagen_halfPrec= False):
        self.batch_data_generator = BatchDataGenerator_Pytorch(size_image,
                                                               list_xData_array,
                                                               list_yData_array,
                                                               images_generator,
                                                               num_channels_in=num_channels_in,
                                                               num_classes_out=num_classes_out,
                                                               is_outputUnet_validconvs=is_outputUnet_validconvs,
                                                               size_output_image=size_output_image,
                                                               batch_size=batch_size,
                                                               shuffle=shuffle,
                                                               seed=seed,
                                                               iswrite_datagen_info=iswrite_datagen_info,
                                                               is_datagen_in_gpu=is_datagen_in_gpu,
                                                               is_datagen_halfPrec=is_datagen_halfPrec)
        super(WrapperBatchGenerator_Pytorch, self).__init__(self.batch_data_generator,
                                                            batch_size= batch_size,
                                                            shuffle= shuffle)

    def get_full_data(self):
        return self.batch_data_generator.get_full_data()