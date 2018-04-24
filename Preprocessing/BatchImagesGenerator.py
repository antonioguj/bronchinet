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


class BatchImagesGenerator_1Array(object):

    def __init__(self, images_array, size_image, images_generator, size_batch=1, shuffle_images=SHUFFLEIMAGES):
        self.images_array     = images_array
        self.size_image       = size_image
        self.images_generator = images_generator
        self.num_images_total = images_generator.get_num_images_total()
        self.size_batch       = size_batch
        self.num_batches      = (self.num_images_total + self.size_batch - 1)//self.size_batch # round-up

        self.reset(shuffle_images)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):

        self.list_indexes_batch = next(self.indexes_generator())
        return self.get_list_images_batch()

    def get_num_channels_array(self, in_array_shape):
        if len(in_array_shape) == len(self.size_image):
            return 1
        else:
            return in_array_shape[-1]

    def get_shape_out_array(self, num_images, num_channels=1):
        if num_channels == 1:
            return [num_images] + list(self.size_image)
        else:
            return [num_images] + list(self.size_image) + [num_channels]

    def reset(self, shuffle):
        if (shuffle):
            self.list_indexes_total = np.random.choice(self.num_images_total, size=self.num_images_total, replace=False)
        else:
            self.list_indexes_total = range(self.num_images_total)
        self.count_batch = 0

    def indexes_generator(self):

        while self.count_batch < self.num_batches:
            count_index  = self.count_batch * self.size_batch
            self.count_batch += 1
            yield self.list_indexes_total[count_index:count_index + self.size_batch]

    def get_list_images_batch(self):

        num_images_out  = len(self.list_indexes_batch)
        out_array_shape = self.get_shape_out_array(num_images_out, num_channels=self.get_num_channels_array(self.images_array.shape))

        out_images_array = np.ndarray(out_array_shape, dtype=self.images_array.dtype)

        for i, index in enumerate(self.list_indexes_batch):
            out_images_array[i] = self.images_generator.get_image_array(self.images_array, index)
        #endfor

        return out_images_array


class BatchImagesGenerator_2Arrays(BatchImagesGenerator_1Array):

    def __init__(self, images_array, masks_array, size_image, images_generator, size_batch=1, shuffle_images=SHUFFLEIMAGES):
        self.masks_array = masks_array
        super(BatchImagesGenerator_2Arrays, self).__init__(images_array, size_image, images_generator, size_batch, shuffle_images)

    def get_list_images_batch(self):

        num_images_out   = len(self.list_indexes_batch)
        out_array1_shape = self.get_shape_out_array(num_images_out, num_channels=self.get_num_channels_array(self.images_array.shape))
        out_array2_shape = self.get_shape_out_array(num_images_out, num_channels=self.get_num_channels_array(self.masks_array.shape))

        out_images_array = np.ndarray(out_array1_shape, dtype=self.images_array.dtype)
        out_masks_array  = np.ndarray(out_array2_shape, dtype=self.masks_array.dtype)

        for i, index in enumerate(self.list_indexes_batch):
            out_images_array[i] = self.images_generator.get_image_array(self.images_array, index)
            out_masks_array [i] = self.images_generator.get_image_array(self.masks_array, index)
        #endfor

        return (out_images_array, out_masks_array)