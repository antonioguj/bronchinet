#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.LoadDataManager import *
from Preprocessing.SlidingWindowImages import *
from keras.preprocessing import image
import numpy as np


class SlidingWindowBatchGenerator(image.Iterator):

    def __init__(self, list_Xdata, list_Ydata,
                      (size_img_z, size_img_x, size_img_y),
                      (prop_overlap_z, prop_overlap_x, prop_overlap_y),
                       batch_size=1, shuffle=False, seed=None):

        self.list_Xdata = list_Xdata
        self.list_Ydata = list_Ydata
        self.size_image = (size_img_z, size_img_x, size_img_y)
        self.prop_overlap = (prop_overlap_z, prop_overlap_x, prop_overlap_y)

        self.slidingWindowImages = SlidingWindowImages(self.size_image, self.prop_overlap)

        self.list_size_data = []
        self.list_size_data_x_y_z = []
        self.list_num_batches = []
        self.list_begin_index_batch = []

        count_batches = 0
        for i, (Xdata, Ydata) in enumerate(zip(list_Xdata, list_Ydata)):

            (sizetotal_z, sizetotal_x, sizetotal_y) = Xdata.shape

            (num_images_x, num_images_y, num_images_z) = self.slidingWindowImages.get_num_images_3d((sizetotal_z, sizetotal_x, sizetotal_y))
            num_images = num_images_x * num_images_y * num_images_z

            print('Image %s: Generate batches images by Sliding Window: size: %s; num batches: %s, in x_y_z: %s...' %(i, Xdata.shape, num_images,
                                                                                                                      (num_images_x, num_images_y, num_images_z)))
            num_batches = (num_images + batch_size - 1)//batch_size # round-up

            self.list_size_data.append( num_images )
            self.list_size_data_x_y_z.append( (num_images_x, num_images_y, num_images_z) )
            self.list_num_batches.append( num_batches )
            self.list_begin_index_batch.append( count_batches )

            count_batches += num_batches
        #endfor

        size_data_total = sum(self.list_size_data)
        super(SlidingWindowBatchGenerator, self).__init__(size_data_total, batch_size, shuffle, seed)

        #reset
        self.on_epoch_end()


    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))

        # retrieve last index lower than 'idx': first index batch in block
        idx_block        = self.get_index_block(idx)
        idx_batch_inblock= self.get_index_batch_inblock(idx, idx_block)

        self._set_index_array(idx_block)

        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)

        current_index = self.batch_size * idx_batch_inblock
        indexes_array = self.index_array[current_index:current_index + self.batch_size]

        self.total_batches_seen += 1

        return self.get_batches_cropped_images(idx_block, indexes_array, self.list_size_data_x_y_z[idx_block])

    def __len__(self):
        return sum(self.list_num_batches)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        with self.lock:
            indexes_array = next(self.index_generator)

        return self.get_batches_cropped_images(self.index_block, indexes_array, self.list_size_data_x_y_z[self.index_block])

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self._reset()
        #while 1:
        while self.batch_index < len(self):
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)

            if (self._is_next_block_batches()):
                self._next_block_batches()

            current_index = self.batch_size * self.batch_index_inblock

            self._next_batch_inblock()

            yield self.index_array[current_index:current_index + self.batch_size]


    def _set_index_array(self, idx_block):
        self.index_array = np.arange(self.list_size_data[idx_block])
        if self.shuffle:
            self.index_array = np.random.permutation(self.list_size_data[idx_block])

    def on_epoch_begin(self):
        self._reset()

    def on_epoch_end(self):
        self._reset()

    def _reset(self):
        self.batch_index = 0
        self.index_block = 0
        self._reset_inblock(0)
        self._set_index_array(0)

    def _reset_inblock(self, idx_block):
        self.batch_index_inblock = 0
        self.last_batch_index_inblock = self.list_num_batches[idx_block]

    def get_index_block(self, idx):
        return np.where(idx >= np.asarray(self.list_begin_index_batch))[0][-1]

    def get_index_batch_inblock(self, idx, idx_block):
        return idx - self.list_begin_index_batch[idx_block]

    def get_indexes_array_batch(self, idx_batch):
        current_index = self.batch_size * idx_batch
        return self.index_array[current_index:current_index + self.batch_size]

    def _is_next_block_batches(self):
        return self.batch_index_inblock == self.last_batch_index_inblock

    def _next_batch_inblock(self):
        self.batch_index += 1
        self.batch_index_inblock += 1
        self.total_batches_seen += 1

    def _next_block_batches(self):
        self.index_block += 1
        self._reset_inblock(self.index_block)
        self._set_index_array(self.index_block)


    def get_batches_cropped_images(self, index_block, indexes_array, (size_data_X, size_data_Y, size_data_Z)):

        (indexes_X_batch,
         indexes_Y_batch,
         indexes_Z_batch) = self.get_indexes_3dirs(indexes_array, size_data_X, size_data_Y)

        if K.image_data_format() == 'channels_first':
            Xdata_batch = np.ndarray([self.batch_size, 1] + list(self.size_image), dtype=FORMATIMAGEDATA)
            Ydata_batch = np.ndarray([self.batch_size, 1] + list(self.size_image), dtype=FORMATMASKDATA)
        else:
            Xdata_batch = np.ndarray([self.batch_size] + list(self.size_image) + [1], dtype=FORMATIMAGEDATA)
            Ydata_batch = np.ndarray([self.batch_size] + list(self.size_image) + [1], dtype=FORMATMASKDATA)

        for i, (index_x_batch, index_y_batch, index_z_batch) in enumerate(zip(indexes_X_batch,
                                                                              indexes_Y_batch,
                                                                              indexes_Z_batch)):

            (x_left, x_right, y_down, y_up, z_back, z_front) = self.compute_batch_boundingBox(index_x_batch, index_y_batch, index_z_batch)

            Xdata_batch[i] = self.list_Xdata[index_block][z_back:z_front, x_left:x_right, y_down:y_up].reshape([self.batch_size] + list(self.size_image) + [1])
            Ydata_batch[i] = self.list_Ydata[index_block][z_back:z_front, x_left:x_right, y_down:y_up].reshape([self.batch_size] + list(self.size_image) + [1])
        #endfor

        return (Xdata_batch, Ydata_batch)


    def get_indexes_3dirs(self, index_batch, size_data_X, size_data_Y):

        return self.slidingWindowImages.get_indexes_3d(index_batch, (size_data_X, size_data_Y))

    def compute_batch_boundingBox(self, index_x, index_y, index_z):

        return self.slidingWindowImages.get_limits_image_3d((index_x, index_y, index_z))