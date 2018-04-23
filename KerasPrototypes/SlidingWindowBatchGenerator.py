#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Preprocessing.SlidingWindowImages import *
from keras.preprocessing import image
from keras.backend import image_data_format as K_image_data_format
from keras.utils import to_categorical as K_to_categorical
import numpy as np
np.random.seed(2017)


class SlidingWindowBatchGenerator(image.Iterator):

    def __init__(self, list_Xdata, list_Ydata, size_image, prop_overlap, num_channels_in=1, num_classes_out=1, size_outnnet=None, batch_size=1, shuffle=True, seed=None):
        self.list_Xdata = list_Xdata
        self.list_Ydata = list_Ydata

        self.size_image      = size_image
        self.num_channels_in = num_channels_in
        self.num_classes_out = num_classes_out
        if size_outnnet and (size_outnnet != size_image):
            self.size_outnnet = size_outnnet
        else:
            self.size_outnnet = size_image

        self.type_Xdata = list_Xdata[0].dtype
        self.type_Ydata = list_Ydata[0].dtype

        self.slidingWindowImages = SlidingWindowImages(size_image, prop_overlap)

        self.list_indexes_compute_images_batches = []

        for i, (Xdata, Ydata) in enumerate(zip(list_Xdata, list_Ydata)):

            (size_fullimage_z, size_fullimage_x, size_fullimage_y) = Xdata.shape

            (num_images_z, num_images_x, num_images_y) = self.slidingWindowImages.get_num_images_3d((size_fullimage_z, size_fullimage_x, size_fullimage_y))
            num_images = num_images_x * num_images_y * num_images_z

            print('Image %s: Generate batches images by Sliding Window: size: %s; num batches: %s, in Z_X_Y: %s...' %(i, Xdata.shape, num_images, (num_images_z, num_images_x, num_images_y)))

            for index in range(num_images):

                (index_z, index_x, index_y) = self.slidingWindowImages.get_indexes_3d(index, (num_images_x, num_images_y))

                # Store indexes to compute images batches: (idx_fullimage, idx_Z_subimage, idx_X_subimage, idx_Y_subimage)
                self.list_indexes_compute_images_batches.append((i, index_z, index_x, index_y))
            #endfor
        #endfor

        num_total_images = len(self.list_indexes_compute_images_batches)

        if (shuffle):
            # shuffle indexes to compute images batches
            randomIndexes = np.random.choice(num_total_images, size=num_total_images, replace=False)

            list_indexes_compute_images_batches_old = self.list_indexes_compute_images_batches
            self.list_indexes_compute_images_batches = []
            for rdm_index in randomIndexes:
                self.list_indexes_compute_images_batches.append( list_indexes_compute_images_batches_old[rdm_index] )
            #endfor

        super(SlidingWindowBatchGenerator, self).__init__(num_total_images, batch_size, shuffle, seed)


    def _get_batches_of_transformed_samples(self, index_array):
        # overwrite function to retrieve images batches
        num_images_batch = len(index_array)

        Xdata_batch = np.ndarray([num_images_batch] + list(self.size_image)  + [self.num_channels_in], dtype=self.type_Xdata)
        Ydata_batch = np.ndarray([num_images_batch] + list(self.size_outnnet)+ [self.num_classes_out], dtype=self.type_Ydata)

        for i, index in enumerate(index_array):

            (idx_fullimage, idx_z_image, idx_x_image, idx_y_image) = self.list_indexes_compute_images_batches[index]

            ((z_back, z_front), (x_left, x_right), (y_down, y_up)) = self.slidingWindowImages.get_limits_image_3d((idx_z_image, idx_x_image, idx_y_image))

            Xdata_batch[i] = self.list_Xdata[idx_fullimage][z_back:z_front, x_left:x_right, y_down:y_up]

            if self.num_classes_out > 1:
                Ydata_batch[i] = self.get_array_categorical_masks(self.get_array_cropImages_outNnet(self.list_Ydata[idx_fullimage][z_back:z_front, x_left:x_right, y_down:y_up]))
            else:
                Ydata_batch[i] = self.get_array_cropImages_outNnet(self.list_Ydata[idx_fullimage][z_back:z_front, x_left:x_right, y_down:y_up])
        #endfor

        return (self.get_array_reshapedKeras(Xdata_batch), self.get_array_reshapedKeras(Ydata_batch))


    def get_array_categorical_masks(self, yData):
        return K_to_categorical(yData, num_classes=self.num_classes_out)


    @staticmethod
    def getArrayShapeKeras(num_batches, size_image, num_channels):
        if K_image_data_format() == 'channels_first':
            return [num_batches, num_channels] + list(size_image)
        else:
            return [num_batches] + list(size_image) + [num_channels]

    def get_array_reshapedKeras(self, array):
        if len(array.shape) == (len(self.size_image) + 1):
            # array_shape: (num_batch, size_batch) #not multichannel
            return array.reshape(self.getArrayShapeKeras(array.shape[0], array.shape[1:], 1))
        else:
            # array_shape: (num_batch, size_batch, num_channel) #multichannel
            return array.reshape(self.getArrayShapeKeras(array.shape[0], array.shape[2:-1], array.shape[-1]))


    @staticmethod
    def get_limits_cropImage(size_image, size_outnnet):
        if (size_image==size_outnnet):
            return size_image
        else:
            return tuple(((s_i- s_o)/2, (s_i + s_o)/2) for (s_i, s_o) in zip(size_image, size_outnnet))

    def get_array_cropImages_outNnet(self, yData):
        if (self.size_image==self.size_outnnet):
            return yData
        else:
            if len(self.size_image)==2:
                ((x_left, x_right), (y_down, y_up)) = self.get_limits_cropImage(self.size_image, self.size_outnnet)
                return yData[..., x_left:x_right, y_down:y_up]
            elif len(self.size_image)==3:
                ((z_back, z_front), (x_left, x_right), (y_down, y_up)) = self.get_limits_cropImage(self.size_image, self.size_outnnet)
                return yData[..., z_back:z_front, x_left:x_right, y_down:y_up]