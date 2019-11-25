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
import numpy as np
np.random.seed(2017)



class BatchDataGenerator(object):

    def __init__(self, size_image,
                 list_xData_array,
                 list_yData_array,
                 images_generator,
                 num_channels_in= 1,
                 num_classes_out= 1,
                 isUse_valid_convs= False,
                 size_output_image= None,
                 batch_size= 1,
                 iswrite_datagen_info=True):
        self.size_image = size_image
        self.num_channels_in = num_channels_in

        self.list_xData_array = list_xData_array
        self.type_xData = list_xData_array[0].dtype
        self.list_yData_array = list_yData_array
        self.type_yData = list_yData_array[0].dtype

        self.images_generator = images_generator

        self.num_classes_out = num_classes_out

        self.isUse_valid_convs = isUse_valid_convs
        if isUse_valid_convs and size_output_image and \
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
            self.isUse_valid_convs = False

        self.batch_size = batch_size
        self.num_images = None

        self.iswrite_datagen_info = iswrite_datagen_info


    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return self.num_images


    def compute_pairIndexes_imagesFile(self, shuffle, seed=None):
        self.list_pairIndexes_imagesFile = []

        for ifile, xData_array in enumerate(self.list_xData_array):
            self.images_generator.update_image_data(xData_array.shape)

            num_images_file = self.images_generator.get_num_images()

            #store pair of indexes: (idx_file, idx_batch)
            for index in range(num_images_file):
                self.list_pairIndexes_imagesFile.append((ifile, index))
            #endfor

            if self.iswrite_datagen_info:
                message = self.images_generator.get_text_description()
                print("Image file: \'%s\'..." %(ifile))
                print(message)
        #endfor

        if (shuffle):
            if seed is not None:
                np.random.seed(seed)

            num_images = len(self.list_pairIndexes_imagesFile)
            random_indexes = np.random.choice(num_images, size=num_images, replace=False)

            self.list_pairIndexes_imagesFile_prev = self.list_pairIndexes_imagesFile
            self.list_pairIndexes_imagesFile = []
            for index in random_indexes:
                self.list_pairIndexes_imagesFile.append(self.list_pairIndexes_imagesFile_prev[index])
            #endfor

        num_images = len(self.list_pairIndexes_imagesFile)
        return num_images


    def get_cropped_output(self, in_array):
        return self.fun_crop_images(in_array, self.crop_output_bounding_box)


    def get_formated_output_xData(self, in_array):
        return NotImplemented

    def get_formated_output_yData(self, in_array):
        return NotImplemented


    def get_item_noformat(self, index):
        (index_file, index_image_file) = self.list_pairIndexes_imagesFile[index]
        self.images_generator.update_image_data(self.list_xData_array[index_file].shape)

        return self.images_generator.get_image_2arrays(self.list_xData_array[index_file],
                                                       in2nd_array= self.list_yData_array[index_file],
                                                       index=index_image_file,
                                                       seed=None)


    def get_item(self, index):
        (out_xData_elem, out_yData_elem) = self.get_item_noformat(index)
        return (self.get_formated_output_xData(out_xData_elem),
                self.get_formated_output_yData(out_yData_elem))


    def get_full_data(self):
        out_xData_shape = [self.num_images] + list(self.size_image)
        out_yData_shape = [self.num_images] + list(self.size_output_image)
        out_xData_array = np.ndarray(out_xData_shape, dtype= self.type_xData)
        out_yData_array = np.ndarray(out_yData_shape, dtype= self.type_yData)

        for i in range(self.num_images):
            (out_xData_array[i], out_yData_array[i]) = self.get_item(i)
        #endfor

        return (out_xData_array, out_yData_array)