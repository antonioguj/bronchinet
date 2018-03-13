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
from CommonUtil.FileDataManager import *
from keras.preprocessing import image
import numpy as np


class SlidingWindowBatchGenerator(image.Iterator):

    def __init__(self, list_Xdata, list_Ydata, prop_overlap=(0.5, 0.5, 0.5), batch_size=32, shuffle=False, seed=None):

        self.list_Xdata = list_Xdata
        self.list_Ydata = list_Ydata
        self.prop_X = prop_overlap[1]
        self.prop_Y = prop_overlap[2]
        self.prop_Z = prop_overlap[0]

        self.list_size_data = []
        self.list_size_data_XYZ = []
        self.list_num_batches = []
        self.list_begin_index_batch = []

        count_batches = 0
        for i, (Xdata, Ydata) in enumerate(zip(list_Xdata, list_Ydata)):

            (size_img_Z, size_img_X, size_img_Y) = Xdata.shape

            num_cropimgs_X = int(np.floor((size_img_X - self.prop_X*IMAGES_HEIGHT) /(1 - self.prop_X)/IMAGES_HEIGHT))
            num_cropimgs_Y = int(np.floor((size_img_Y - self.prop_Y*IMAGES_WIDTH ) /(1 - self.prop_Y)/IMAGES_WIDTH ))
            num_cropimgs_Z = int(np.floor((size_img_Z - self.prop_Z*IMAGES_DEPTHZ) /(1 - self.prop_Z)/IMAGES_DEPTHZ))
            num_cropimages = num_cropimgs_X * num_cropimgs_Y * num_cropimgs_Z

            num_batches = (num_cropimages + batch_size - 1)//batch_size # round-up

            print('Sliding Window: build num images: %s; in X_Y_Z dirs: (%s,%s,%s)' % (num_cropimages, num_cropimgs_X, num_cropimgs_Y, num_cropimgs_Z))

            self.list_size_data.append( num_cropimages )
            self.list_size_data_XYZ.append( (num_cropimgs_X, num_cropimgs_Y, num_cropimgs_Z) )
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

        (size_data_X, size_data_Y, size_data_Z) = self.list_size_data_XYZ[idx_block]

        current_index = self.batch_size * idx_batch_inblock
        indexes_array = self.index_array[current_index:current_index + self.batch_size]

        self.total_batches_seen += 1

        return self.get_batches_cropped_images(idx_block, indexes_array, size_data_X, size_data_Y, size_data_Z)


    def __len__(self):
        return sum(self.list_num_batches)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        with self.lock:
            indexes_array = next(self.index_generator)

        return self.get_batches_cropped_images(idx_block, indexes_array, size_data_X, size_data_Y, size_data_Z)

    def _flow_index(self):
        self._reset()
        while 1:
            if (self._is_next_block_batches()):
                self._next_block_batches()

            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)

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


    def get_batches_cropped_images(self, idx_block, indexes_array, size_data_X, size_data_Y, size_data_Z):

        (indexes_X_batch,
         indexes_Y_batch,
         indexes_Z_batch) = self.get_index_X_Y_Z(indexes_array, size_data_X, size_data_Y, size_data_Z)

        Xdata_batch = np.ndarray([self.batch_size, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATIMAGEDATA)
        Ydata_batch = np.ndarray([self.batch_size, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATMASKDATA )

        for i, (idx_X_batch, idx_Y_batch, idx_Z_batch) in enumerate(zip(indexes_X_batch,
                                                                        indexes_Y_batch,
                                                                        indexes_Z_batch)):
            # (x_left, x_right, y_down, y_up, z_back, z_front)
            bound_box_batch = self.compute_batch_bound_box(idx_X_batch, idx_Y_batch, idx_Z_batch)

            Xdata_batch[i] = self.list_Xdata[idx_block][bound_box_batch[4]:bound_box_batch[5],
                                                        bound_box_batch[0]:bound_box_batch[1],
                                                        bound_box_batch[2]:bound_box_batch[3]]
            Ydata_batch[i] = self.list_Ydata[idx_block][bound_box_batch[4]:bound_box_batch[5],
                                                        bound_box_batch[0]:bound_box_batch[1],
                                                        bound_box_batch[2]:bound_box_batch[3]]
        #endfor

        if K.image_data_format() == 'channels_first':
            Xdata_batch = Xdata_batch.reshape([self.batch_size, 1, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH])
            Ydata_batch = Ydata_batch.reshape([self.batch_size, 1, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH])
        else:
            Xdata_batch = Xdata_batch.reshape([self.batch_size, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH, 1])
            Ydata_batch = Ydata_batch.reshape([self.batch_size, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH, 1])

        return (Xdata_batch, Ydata_batch)


    def get_index_X_Y_Z(self, idx_batch, size_data_X, size_data_Y, size_data_Z):

        idx_Z_batch = idx_batch // (size_data_X * size_data_Y)
        idx_batch_XY= idx_batch % (size_data_X * size_data_Y)
        idx_Y_batch = idx_batch_XY // size_data_X
        idx_X_batch = idx_batch_XY % size_data_X

        return (idx_X_batch, idx_Y_batch, idx_Z_batch)

    def compute_batch_bound_box(self, idx_X, idx_Y, idx_Z):

        x_left = int(idx_X * (1.0 - self.prop_X) * IMAGES_HEIGHT)
        x_right= x_left + IMAGES_HEIGHT
        y_down = int(idx_Y * (1.0 - self.prop_Y) * IMAGES_WIDTH)
        y_up   = y_down + IMAGES_WIDTH
        z_back = int(idx_Z * (1.0 - self.prop_Z) * IMAGES_DEPTHZ)
        z_front= z_back + IMAGES_DEPTHZ

        return (x_left, x_right, y_down, y_up, z_back, z_front)