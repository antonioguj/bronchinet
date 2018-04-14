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
from keras.preprocessing import image
import numpy as np
import random


class ManyFilesBatchGenerator(image.Iterator):

    def __init__(self, list_files_X, list_files_Y, batch_size=32, shuffle=False, seed=None):

        self.list_files_X = list_files_X
        self.list_files_Y = list_files_Y

        self.type_Xdata = list_files_X[0].dtype
        self.type_Ydata = list_files_Y[0].dtype

        self.list_size_data = []
        self.list_num_batches = []
        self.list_begin_index_batch = []

        count_batches = 0
        for i, (file_X, file_Y) in enumerate(zip(list_files_X, list_files_Y)):

            (xData, yData) = LoadDataManager.loadData_1FileBatches(file_X, file_Y)

            size_data = xData.shape[0]
            num_batches = (size_data + batch_size - 1)//batch_size # round-up

            self.list_size_data.append( size_data )
            self.list_num_batches.append( num_batches )
            self.list_begin_index_batch.append( count_batches )

            count_batches += num_batches
        #endfor

        size_data_total = sum(self.list_size_data)
        super(ManyFilesBatchGenerator, self).__init__(size_data_total, batch_size, shuffle, seed)

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

        return (self.this_Xdata[indexes_array], self.this_Ydata[indexes_array])


    def _getitem_notsorted(self, idx):

        # retrieve last index lower than 'idx': first index batch in file
        idx_file        = self.get_index_file(idx)
        idx_batch_infile= self.get_index_batch_infile(idx, idx_file)

        self._set_index_array(idx_file)
        self._load_data_file(idx_file)

        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)

        current_index = self.batch_size * idx_batch_infile
        indexes_array = self.index_array[current_index:current_index + self.batch_size]

        self.total_batches_seen += 1
        self.last_idx = idx

        return (self.this_Xdata[indexes_array], self.this_Ydata[indexes_array])


    def __len__(self):
        return sum(self.list_num_batches)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        with self.lock:
            indexes_array = next(self.index_generator)

        return (self.this_Xdata[indexes_array], self.this_Ydata[indexes_array])

    def _flow_index(self):
        self._reset()
        #while 1:
        while self.batch_index < len(self):
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)

            if (self._is_next_file_batches()):
                self._next_file_batches()

            current_index = self.batch_size * self.batch_index_infile

            self._next_batch_infile()

            yield self.index_array[current_index:current_index + self.batch_size]


    def _set_index_array(self, idx_file):
        self.index_array = np.arange(self.list_size_data[idx_file])
        if self.shuffle:
            self.index_array = np.random.permutation(self.list_size_data[idx_file])

    def on_epoch_begin(self):
        self.last_idx = -1
        self._reset()

    def on_epoch_end(self):
        self.last_idx = -1
        self._reset()

    def _reset(self):
        self.batch_index = 0
        self.index_file = 0
        self._shuffle_files_list()
        self._reset_infile(0)
        self._set_index_array(0)
        self._load_data_file(0)

    def _reset_infile(self, idx_file):
        self.batch_index_infile = 0
        self.last_batch_index_infile = self.list_num_batches[idx_file]

    def get_index_file(self, idx):
        return np.where(idx >= np.asarray(self.list_begin_index_batch))[0][-1]

    def get_index_batch_infile(self, idx, idx_file):
        return idx - self.list_begin_index_batch[idx_file]

    def get_indexes_array_batch(self, idx_batch):
        current_index = self.batch_size * idx_batch
        return self.index_array[current_index:current_index + self.batch_size]

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
        (self.this_Xdata, self.this_Ydata) = LoadDataManager.loadData_1FileBatches(self.list_files_X[idx_file],
                                                                                   self.list_files_Y[idx_file],
                                                                                   shuffleImages=True)

    def _shuffle_files_list(self):
        #need to shuffle all list at the same time
        in_list_aux = list(zip(self.list_files_X,
                                          self.list_files_Y,
                                          self.list_size_data,
                                          self.list_num_batches,
                                          self.list_begin_index_batch))
        out_list_aux = random.shuffle(in_list_aux)
        (self.list_files_X,
         self.list_files_Y,
         self.list_size_data,
         self.list_num_batches,
         self.list_begin_index_batch) = zip(*out_list_aux)