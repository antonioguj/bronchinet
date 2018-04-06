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
from Preprocessing.SlidingWindowImages import *
from keras.preprocessing import image
from keras import backend as K
import numpy as np
np.random.seed(2017)


class SlidingWindowBatchGenerator(image.Iterator):

    def __init__(self, list_Xdata, list_Ydata, size_image, prop_overlap, batch_size=1, shuffle=True, seed=None):

        self.list_Xdata = list_Xdata
        self.list_Ydata = list_Ydata
        self.size_image = size_image

        self.slidingWindowImages = SlidingWindowImages(size_image, prop_overlap)

        self.list_indexes_compute_images_batches = []

        for i, (Xdata, Ydata) in enumerate(zip(list_Xdata, list_Ydata)):

            (size_fullimage_z, size_fullimage_x, size_fullimage_y) = Xdata.shape

            (num_images_x, num_images_y, num_images_z) = self.slidingWindowImages.get_num_images_3d((size_fullimage_z, size_fullimage_x, size_fullimage_y))
            num_images = num_images_x * num_images_y * num_images_z

            print('Image %s: Generate batches images by Sliding Window: size: %s; num batches: %s, in x_y_z: %s...' %(i, Xdata.shape, num_images, (num_images_x, num_images_y, num_images_z)))

            for index in range(num_images):

                (index_x, index_y, index_z) = self.slidingWindowImages.get_indexes_3d(index, (num_images_x, num_images_y))

                # Store indexes to compute images batches: (idx_fullimage, idx_x_subimage, idx_y_subimage, idx_z_subimage)
                self.list_indexes_compute_images_batches.append((i, index_x, index_y, index_z))
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

        Xdata_batch = np.ndarray([num_images_batch] + list(self.size_image), dtype=FORMATIMAGEDATA)
        Ydata_batch = np.ndarray([num_images_batch] + list(self.size_image), dtype=FORMATMASKDATA)

        for i, index in enumerate(index_array):

            (idx_fullimage, idx_x_image, idx_y_image, idx_z_image) = self.list_indexes_compute_images_batches[index]

            (x_left, x_right, y_down, y_up, z_back, z_front) = self.slidingWindowImages.get_limits_image_3d((idx_x_image, idx_y_image, idx_z_image))

            Xdata_batch[i] = self.list_Xdata[idx_fullimage][z_back:z_front, x_left:x_right, y_down:y_up]
            Ydata_batch[i] = self.list_Ydata[idx_fullimage][z_back:z_front, x_left:x_right, y_down:y_up]
        #endfor

        return (Xdata_batch.reshape(self.getArrayDims(num_images_batch)),
                Ydata_batch.reshape(self.getArrayDims(num_images_batch)))


    def getArrayDims(self, num_batches):
        if K.image_data_format() == 'channels_first':
            return [num_batches, 1] + list(self.size_image)
        else:
            return [num_batches] + list(self.size_image) + [1]



class SlidingWindowBatchGeneratorKerasOld(SlidingWindowBatchGenerator):

    def __init__(self, list_Xdata, list_Ydata, size_image, prop_overlap, batch_size=1, shuffle=True, seed=None):
        super(SlidingWindowBatchGeneratorKerasOld, self).__init__(list_Xdata, list_Ydata, size_image, prop_overlap, batch_size, shuffle, seed)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def next(self):
        # Returns the next batch
        with self.lock:
            index_array, _, _ = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)