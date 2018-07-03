#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from keras.backend import image_data_format as K_image_data_format
from keras.utils import to_categorical as K_to_categorical
import numpy as np


class ArrayShapeManager(object):

    def __init__(self, size_image, is_shaped_Keras=False, num_classes_out=1, size_outnnet=None):
        self.size_image      = size_image
        self.is_shaped_Keras = is_shaped_Keras
        self.num_classes_out = num_classes_out
        if size_outnnet and (size_outnnet != size_image):
            self.size_outnnet = size_outnnet
        else:
            self.size_outnnet = size_image


    def is_images_array_without_channels(self, in_array_shape):
        return len(in_array_shape) == len(self.size_image)

    def get_num_channels_array(self, in_array_shape):
        if self.is_images_array_without_channels(in_array_shape):
            return None
        else:
            return in_array_shape[-1]

    def get_shape_out_array(self, num_samples, num_channels):
        if self.is_shaped_Keras and not num_channels:
            # arrays in Keras always have one dim reserved for channels
            num_channels = 1
        if num_channels:
            return [num_samples] + list(self.size_image) + [num_channels]
        else:
            return [num_samples] + list(self.size_image)

    def get_array_with_channels(self, in_array, num_channels=None):
        if self.is_shaped_Keras and not num_channels:
            # arrays in Keras always have one dim reserved for channels
            num_channels = 1
        if num_channels == 1:
            return np.expand_dims(in_array, axis=-1)
        elif num_channels > 1:
            return np.reshape(in_array, in_array.shape + [num_channels])
        else:
            return in_array

    def get_array_categorical_masks(self, yData):
        return K_to_categorical(yData, num_classes=self.num_classes_out)


    @staticmethod
    def get_shape_Keras(num_images, size_image, num_channels):
        if K_image_data_format() == 'channels_first':
            return [num_images, num_channels] + list(size_image)
        elif K_image_data_format() == 'channels_last':
            return [num_images] + list(size_image) + [num_channels]
        else:
            return 0

    @staticmethod
    def get_array_reshaped_Keras(array):
        if K_image_data_format() == 'channels_first':
            # need to roll last dimensions, channels, to second dim:
            return np.rollaxis(array, -1, 1)
        elif K_image_data_format() == 'channels_last':
            return array
        else:
            return 0


    @staticmethod
    def get_limits_cropImage(size_image, size_outnnet):
        if (size_image == size_outnnet):
            list_out_aux = [[0] + [s_i] for s_i in size_image]
        else:
            list_out_aux = [[(s_i - s_o) / 2] + [(s_i + s_o) / 2] for (s_i, s_o) in zip(size_image, size_outnnet)]
        # flatten out list of lists and return tuple
        return tuple(reduce(lambda el1, el2: el1 + el2, list_out_aux))

    def get_array_shaped_outNnet(self, yData):
        if (self.size_image==self.size_outnnet):
            return yData
        else:
            if len(self.size_image)==2:
                return self.get_array_shaped_outNnet_2D(yData)
            elif len(self.size_image)==3:
                return self.get_array_shaped_outNnet_3D(yData)

    def get_array_shaped_outNnet_2D(self, yData):
        (x_left, x_right, y_down, y_up) = self.get_limits_cropImage(self.size_image, self.size_outnnet)

        if self.is_images_array_without_channels(yData.shape):
            return yData[..., x_left:x_right, y_down:y_up]
        else:
            return yData[..., x_left:x_right, y_down:y_up, :]

    def get_array_shaped_outNnet_3D(self, yData):
        (z_back, z_front, x_left, x_right, y_down, y_up) = self.get_limits_cropImage(self.size_image, self.size_outnnet)

        if self.is_images_array_without_channels(yData.shape):
            return yData[..., z_back:z_front, x_left:x_right, y_down:y_up]
        else:
            return yData[..., z_back:z_front, x_left:x_right, y_down:y_up, :]


    def get_xData_array_reshaped(self, xData):
        if self.is_shaped_Keras:
            if self.is_images_array_without_channels(xData.shape):
                return self.get_array_reshaped_Keras(self.get_array_with_channels(xData))
            else:
                return self.get_array_reshaped_Keras(xData)
        else:
            if self.is_images_array_without_channels(xData.shape):
                return self.get_array_with_channels(xData)
            else:
                return xData

    def get_yData_array_reshaped(self, yData):
        if self.is_shaped_Keras:
            if self.num_classes_out > 1:
                if (self.size_image==self.size_outnnet):
                    return self.get_array_reshaped_Keras(self.get_array_categorical_masks(yData))
                else:
                    return self.get_array_reshaped_Keras(self.get_array_categorical_masks(self.get_array_shaped_outNnet(yData)))
            else:
                if (self.size_image==self.size_outnnet):
                    if self.is_images_array_without_channels(yData.shape):
                        return self.get_array_reshaped_Keras(self.get_array_with_channels(yData))
                    else:
                        return self.get_array_reshaped_Keras(yData)
                else:
                    if self.is_images_array_without_channels(yData.shape):
                        return self.get_array_reshaped_Keras(self.get_array_with_channels(self.get_array_shaped_outNnet(yData)))
                    else:
                        return self.get_array_reshaped_Keras(self.get_array_shaped_outNnet(yData))
        else:
            if self.num_classes_out > 1:
                if (self.size_image==self.size_outnnet):
                    return self.get_array_categorical_masks(yData)
                else:
                    return self.get_array_categorical_masks(self.get_array_shaped_outNnet(yData))
            else:
                if (self.size_image==self.size_outnnet):
                    if self.is_images_array_without_channels(yData.shape):
                        return self.get_array_with_channels(yData)
                    else:
                        return yData
                else:
                    if self.is_images_array_without_channels(yData.shape):
                        return self.get_array_with_channels(self.get_array_shaped_outNnet(yData))
                    else:
                        return self.get_array_shaped_outNnet(yData)


class ArrayShapeManagerInBatches(ArrayShapeManager):

    def __init__(self, size_image, is_shaped_Keras=False, num_classes_out=1, size_outnnet=None):

        super(ArrayShapeManagerInBatches, self).__init__(size_image, is_shaped_Keras, num_classes_out, size_outnnet)

    def is_images_array_without_channels(self, in_array_shape):
        return len(in_array_shape) == len(self.size_image) + 1