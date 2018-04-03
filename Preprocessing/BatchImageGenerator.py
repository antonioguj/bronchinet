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
import numpy as np
np.random.seed(2017)

SHUFFLEIMAGES = True


class ImageBatchGenerator_1Array(object):

    def __init__(self, images_operator, images_array, size_image, size_batch=1, shuffle_images=SHUFFLEIMAGES):
        self.images_operator  = images_operator
        self.images_array     = images_array
        self.size_image       = size_image
        (self.num_images_x, self.num_images_y, self.num_images_z) = images_operator.get_num_images_3d(images_array.shape)
        self.num_total_images = self.num_images_z * self.num_images_x * self.num_images_y
        self.size_batch       = size_batch
        self.num_batches      = (self.num_total_images + self.size_batch - 1)//self.size_batch # round-up

        self.reset(shuffle_images)


    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):

        indexes_batch = next(self.indexes_generator())
        return self.get_batch_images(indexes_batch)

    def indexes_generator(self):

        while self.count_batch < self.num_batches:
            count_index  = self.count_batch * self.size_batch
            self.count_batch += 1
            yield self.list_indexes_total[count_index:count_index + self.size_batch]

    def get_batch_images(self, indexes_batch):

        num_images_batch = len(indexes_batch)

        out_images_array = np.ndarray([num_images_batch] + list(self.size_image), dtype=self.images_array.dtype)

        for i, index in enumerate(indexes_batch):

            (index_x, index_y, index_z) = self.images_operator.get_indexes_3d(index, (self.num_images_x, self.num_images_y))

            (x_left, x_right, y_down, y_up, z_back, z_front) = self.images_operator.get_limits_image_3d((index_x, index_y, index_z))

            out_images_array[i] = np.asarray(self.images_array[z_back:z_front, x_left:x_right, y_down:y_up], dtype=self.images_array.dtype)
        #endfor

        return out_images_array

    def reset(self, shuffle):
        if (shuffle):
            self.list_indexes_total = np.random.choice(self.num_total_images, size=self.num_total_images, replace=False)
        else:
            self.list_indexes_total = range(self.num_total_images)
        self.count_batch = 0


class ImageBatchGenerator_2Arrays(ImageBatchGenerator_1Array):

    def __init__(self, image_operator, images_array, masks_array, size_image, size_batch=1, shuffle_images=SHUFFLEIMAGES):
        self.masks_array = masks_array
        super(ImageBatchGenerator_2Arrays, self).__init__(image_operator, images_array, size_image, size_batch, shuffle_images)

    def get_batch_images(self, indexes_batch):

        num_images_batch = len(indexes_batch)

        out_images_array = np.ndarray([num_images_batch] + list(self.size_image), dtype=self.images_array.dtype)
        out_masks_array  = np.ndarray([num_images_batch] + list(self.size_image), dtype=self.masks_array.dtype)

        for i, index in enumerate(indexes_batch):

            (index_x, index_y, index_z) = self.images_operator.get_indexes_3d(index, (self.num_images_x, self.num_images_y))

            (x_left, x_right, y_down, y_up, z_back, z_front) = self.images_operator.get_limits_image_3d((index_x, index_y, index_z))

            out_images_array[i] = np.asarray(self.images_array[z_back:z_front, x_left:x_right, y_down:y_up], dtype=self.images_array.dtype)
            out_masks_array [i] = np.asarray(self.masks_array [z_back:z_front, x_left:x_right, y_down:y_up], dtype=self.masks_array.dtype)
        #endfor

        return (out_images_array, out_masks_array)