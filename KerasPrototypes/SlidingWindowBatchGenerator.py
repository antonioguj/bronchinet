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
from keras import backend as K
import numpy as np
np.random.seed(2017)


class SlidingWindowBatchGenerator(image.Iterator):

    def __init__(self, list_Xdata, list_Ydata, size_image, prop_overlap, size_outNnet=None, batch_size=1, shuffle=True, seed=None):

        self.list_Xdata = list_Xdata
        self.list_Ydata = list_Ydata
        self.size_image = size_image
        if size_outNnet and (size_outNnet != size_image):
            self.size_outNnet = size_outNnet
        else:
            self.size_outNnet = size_image

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

        Xdata_batch = np.ndarray([num_images_batch] + list(self.size_image  ), dtype=self.type_Xdata)
        Ydata_batch = np.ndarray([num_images_batch] + list(self.size_outNnet), dtype=self.type_Ydata)

        for i, index in enumerate(index_array):

            (idx_fullimage, idx_z_image, idx_x_image, idx_y_image) = self.list_indexes_compute_images_batches[index]

            ((z_back, z_front), (x_left, x_right), (y_down, y_up)) = self.slidingWindowImages.get_limits_image_3d((idx_z_image, idx_x_image, idx_y_image))

            Xdata_batch[i] = self.list_Xdata[idx_fullimage][z_back:z_front, x_left:x_right, y_down:y_up]
            Ydata_batch[i] = self.get_array_cropImages_outNnet(self.list_Ydata[idx_fullimage][z_back:z_front, x_left:x_right, y_down:y_up])
        #endfor

        return (self.get_array_reshapedKeras(Xdata_batch), self.get_array_reshapedKeras(Ydata_batch))


    @staticmethod
    def getArrayShapeKeras(num_batches, size_image):
        if K.image_data_format() == 'channels_first':
            return [num_batches, 1] + list(size_image)
        else:
            return [num_batches] + list(size_image) + [1]

    @classmethod
    def get_array_reshapedKeras(cls, array):
        return array.reshape(cls.getArrayShapeKeras(array.shape[0], array.shape[1:]))

    @staticmethod
    def get_limits_CropImage(size_image, size_outNnet):
        if (size_image==size_outNnet):
            return size_image
        else:
            return tuple(((s_i- s_o)/2, (s_i + s_o)/2) for (s_i, s_o) in zip(size_image, size_outNnet))

    def get_array_cropImages_outNnet(self, yData):

        if (self.size_image==self.size_outNnet):
            return yData
        else:
            ((z_back, z_front), (x_left, x_right), (y_down, y_up)) = self.get_limits_CropImage(self.size_image, self.size_outNnet)
            print ((z_back, z_front), (x_left, x_right), (y_down, y_up))
            return yData[..., z_back:z_front, x_left:x_right, y_down:y_up]