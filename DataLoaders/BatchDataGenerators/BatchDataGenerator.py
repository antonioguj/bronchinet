#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.ErrorMessages import *
from OperationImages.BoundingBoxes import *
from OperationImages.OperationImages import CropImages
import numpy as np
np.random.seed(2017)



class BatchDataGenerator(object):

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
                 iswrite_datagen_info=True):
        self.size_image = size_image
        self.num_channels_in = num_channels_in

        self.list_xData_array = list_xData_array
        self.type_xData = list_xData_array[0].dtype
        self.list_yData_array = list_yData_array
        self.type_yData = list_yData_array[0].dtype

        if len(list_xData_array) != len(list_yData_array):
            message = 'BatchGenerator: num arrays xData \'%s\' not equal to num arrays yData \'%s\'' %(len(list_xData_array),
                                                                                                       len(list_yData_array))
            CatchErrorException(message)

        self.images_generator = images_generator

        self.num_classes_out = num_classes_out

        self.is_outputUnet_validconvs = is_outputUnet_validconvs
        if is_outputUnet_validconvs and size_output_image and \
            (size_image == size_output_image):
            self.size_output_image = size_output_image
            self.crop_output_bounding_box = BoundingBoxes.compute_bounding_box_centered_image_fit_image(self.size_output_image,
                                                                                                        self.size_image)
            ndims = len(size_image)
            if ndims==2:
                self.fun_crop_images = CropImages.compute2D
            elif ndims==3:
                self.fun_crop_images = CropImages.compute3D
            else:
                raise Exception('Error: self.ndims')
        else:
            self.is_outputUnet_validconvs = False
            self.size_output_image = size_image

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.iswrite_datagen_info = iswrite_datagen_info

        self.num_images = self.compute_list_indexes_imagefile()

        self.on_epoch_end()



    def __len__(self):
        "Denotes the number of batches per epoch"
        return (self.num_images + self.batch_size - 1) // self.batch_size


    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(self.num_images)
        if self.shuffle == True:
            if self.seed is not None:
                np.random.seed(self.seed)

            np.random.shuffle(self.indexes)


    def __getitem__(self, index):
        "Generate one batch of data"
        # generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        num_images_batch = len(indexes)

        out_xData_array_shape = [num_images_batch] + list(self.size_image) + [self.num_channels_in]
        out_yData_array_shape = [num_images_batch] + list(self.size_output_image) + [self.num_classes_out]

        out_xData_array = np.ndarray(out_xData_array_shape, dtype=self.type_xData)
        out_yData_array = np.ndarray(out_yData_array_shape, dtype=self.type_yData)

        for i, index in enumerate(indexes):
            (out_xData_array[i], out_yData_array[i]) = self.get_item(index)
        # endfor

        return (out_xData_array, out_yData_array)


    def compute_list_indexes_imagefile(self):
        self.list_indexes_imagefile = []

        for ifile, xData_array in enumerate(self.list_xData_array):
            self.images_generator.update_image_data(xData_array.shape)

            num_images_file = self.images_generator.get_num_images()

            #store pair of indexes: (idx_file, idx_batch)
            for index in range(num_images_file):
                self.list_indexes_imagefile.append((ifile, index))
            #endfor

            if self.iswrite_datagen_info:
                message = self.images_generator.get_text_description()
                print("Image file: \'%s\'..." %(ifile))
                print(message)
        #endfor

        num_images = len(self.list_indexes_imagefile)
        return num_images


    def get_cropped_output(self, in_array):
        return self.fun_crop_images(in_array, self.crop_output_bounding_box)


    def get_item(self, index):
        (index_file, index_image_file) = self.list_indexes_imagefile[index]
        self.images_generator.update_image_data(self.list_xData_array[index_file].shape)

        return self.images_generator.get_image_2arrays(self.list_xData_array[index_file],
                                                       in2nd_array= self.list_yData_array[index_file],
                                                       index=index_image_file,
                                                       seed=None)

    def get_full_data(self):
        out_xData_shape = [self.num_images] + list(self.size_image)
        out_yData_shape = [self.num_images] + list(self.size_output_image)
        out_xData_array = np.ndarray(out_xData_shape, dtype= self.type_xData)
        out_yData_array = np.ndarray(out_yData_shape, dtype= self.type_yData)

        for i in range(self.num_images):
            (out_xData_array[i], out_yData_array[i]) = self.get_item(i)
        #endfor

        return (out_xData_array, out_yData_array)