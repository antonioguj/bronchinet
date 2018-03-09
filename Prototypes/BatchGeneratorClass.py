#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.FileDataManager import *
from keras.preprocessing import image
import numpy as np


class BatchGenerator(image.Iterator):

    def __init__(self, list_files_X, list_files_Y, batch_size=32, shuffle=False, seed=None):
        self.list_files_X = list_files_X
        self.list_files_Y = list_files_Y

        self.size_data_files = np.ndarray(len(list_files_X), dtype=np.int)
        self.num_batches_files = np.ndarray(len(list_files_X), dtype=np.int)
        self.begin_index_batch_files = np.ndarray(len(list_files_X), dtype=np.int)

        count_batches = 0
        for i, (file_X, file_Y) in enumerate(zip(list_files_X, list_files_Y)):

            (xData, yData) = FileDataManager.loadDataFiles2D(file_X, file_Y)

            size_data = xData.shape[0]
            num_batches = (size_data + batch_size - 1)//batch_size # round-up

            self.size_data_files[i] = size_data
            self.num_batches_files[i] = num_batches
            self.begin_index_batch_files[i] = count_batches

            count_batches += num_batches
        #endfor

        size_data_total = sum(self.size_data_files)
        super(BatchGenerator, self).__init__(size_data_total, batch_size, shuffle, seed)

        #reset
        self.on_epoch_end()


    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if idx == self.last_idx+1:
            return self._getitem_sorted(idx)
        else:
            raise ValueError('ERROR, batch indexes are not sorted. '
                             'Please set \'shuffle==False\' in Keras model.fit_generator()...')
            return self._getitem_notsorted(idx)


    def _getitem_sorted(self, idx):

        # batch indexes are ordered. Files only need to be loaded at certain times. Much faster
        if (idx==0):
            self._reset()

        if (self._is_next_file_batches()):
            self._next_file_batches()

        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)

        current_index = self.batch_size * self.batch_index_infile
        indexes_array = self.index_array[current_index:current_index + self.batch_size]

        self._next_batch_infile()
        self.last_idx = idx

        return (self.this_Xdata[indexes_array],
                self.this_Ydata[indexes_array])


    def _getitem_notsorted(self, idx):

        # retrieve last index lower than 'idx': first index batch in file
        idx_file        = np.where(idx >= self.begin_index_batch_files)[0][-1]
        idx_batch_infile= idx - self.begin_index_batch_files[idx_file]

        self._set_index_array(idx_file)
        self._load_data_file(idx_file)

        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)

        current_index = self.batch_size * idx_batch_infile
        indexes_array = self.index_array[current_index:current_index + self.batch_size]

        self.total_batches_seen += 1
        self.last_idx = idx

        return (self.this_Xdata[indexes_array],
                self.this_Ydata[indexes_array])


    def __len__(self):
        return sum(self.num_batches_files)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        with self.lock:
            indexes_array = next(self.index_generator)

        return (self.this_Xdata[indexes_array],
                self.this_Ydata[indexes_array])


    def _flow_index(self):
        self._reset()
        while 1:
            if (self._is_next_file_batches()):
                self._next_file_batches()

            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)

            current_index = self.batch_size * self.batch_index_infile

            self._next_batch_infile()

            yield self.index_array[current_index:current_index + self.batch_size]


    def _set_index_array(self, idx_file):
        self.index_array = np.arange(self.size_data_files[idx_file])
        if self.shuffle:
            self.index_array = np.random.permutation(self.size_data_files[idx_file])

    def on_epoch_begin(self):
        self._set_index_array(0)
        self._load_data_file(0)

    def on_epoch_end(self):
        self.last_idx = -1
        self._reset()

    def _reset(self):
        self.batch_index = 0
        self.index_file = 0
        self._reset_infile(0)
        self._set_index_array(0)
        self._load_data_file(0)

    def _reset_infile(self, idx_file):
        self.batch_index_infile = 0
        self.last_batch_index_infile = self.num_batches_files[idx_file]

    def _is_next_file_batches(self):
        return self.batch_index_infile == self.last_batch_index_infile

    def _next_batch_infile(self):
        self.batch_index += 1
        self.batch_index_infile += 1
        self.total_batches_seen += 1

    def _next_file_batches(self):
        self.index_file += 1
        self._reset_infile(self.index_file)
        self._set_index_array(self.index_file)
        self._load_data_file(self.index_file)

    def _load_data_file(self, idx_file):
        (self.this_Xdata, self.this_Ydata) = FileDataManager.loadDataFiles2D(self.list_files_X[idx_file],
                                                                             self.list_files_Y[idx_file])