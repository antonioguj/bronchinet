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
import numpy as np
np.random.seed(2017)


class BatchDataGenerator(object):

    def __init__(self, size_image, numtot_samples, size_batch=1, shuffle=SHUFFLEIMAGES, seed=None):

        self.size_image     = size_image
        self.numtot_samples = numtot_samples
        self.size_batch     = size_batch
        self.num_batches    = (self.numtot_samples + self.size_batch - 1)//self.size_batch # round-up

        self.reset(shuffle, seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        self.indexes_batch = next(self.indexes_generator())
        return self.get_data_samples_batch(self.indexes_batch)

    def get_data_samples_batch(self, indexes_batch):
        pass

    def reset(self, shuffle, seed=None):
        if (shuffle):
            if seed:
                np.random.seed(seed)
            self.indexes_total = np.random.choice(self.numtot_samples, size=self.numtot_samples, replace=False)
        else:
            self.indexes_total = range(self.numtot_samples)
        self.count_batch = 0

    def indexes_generator(self):

        while self.count_batch < self.num_batches:
            count_index = self.count_batch * self.size_batch
            self.count_batch += 1
            yield self.indexes_total[count_index:count_index + self.size_batch]

    def is_images_array_without_channels(self, in_array_shape):
        return len(in_array_shape) == len(self.size_image)

    def get_num_channels_array(self, in_array_shape):
        if self.is_images_array_without_channels(in_array_shape):
            return 1
        else:
            return in_array_shape[-1]

    def get_shape_out_array(self, in_array_shape, num_samples):
        if not num_samples:
            num_samples = self.images_generator.get_num_images()

        if self.is_images_array_without_channels(in_array_shape):
            return [num_samples] + list(self.size_image)
        else:
            num_channels = self.get_num_channels_array(in_array_shape)
            return [num_samples] + list(self.size_image) + [num_channels]


class BatchDataGenerator_1Array(BatchDataGenerator):

    def __init__(self, size_image, xData_array, images_generator, numtot_samples=None, size_batch=1, shuffle=SHUFFLEIMAGES, seed=None):

        self.xData_array      = xData_array
        self.images_generator = images_generator
        self.images_generator.complete_init_data(xData_array.shape[0:3])

        if not numtot_samples:
            numtot_samples = images_generator.get_num_images()

        super(BatchDataGenerator_1Array, self).__init__(size_image, numtot_samples, size_batch, shuffle, seed)

    def get_data_samples_batch(self, indexes_batch):

        num_samples_batch = len(indexes_batch)
        out_array_shape   = self.get_shape_out_array(self.xData_array.shape, num_samples_batch)

        out_xData_array = np.ndarray(out_array_shape, dtype=self.xData_array.dtype)

        for i, index in enumerate(indexes_batch):
            out_xData_array[i] = self.images_generator.get_image_array(self.xData_array, index)
        #endfor

        return out_xData_array

    def get_images_all(self):

        self.get_data_samples_batch(self.indexes_total)


class BatchDataGenerator_2Arrays(BatchDataGenerator_1Array):

    def __init__(self, size_image, xData_array, yData_array, images_generator, numtot_samples=None, size_batch=1, shuffle=SHUFFLEIMAGES, seed=None):

        self.yData_array = yData_array

        super(BatchDataGenerator_2Arrays, self).__init__(size_image, xData_array, images_generator, numtot_samples, size_batch, shuffle, seed)

    def get_data_samples_batch(self, indexes_batch):

        num_samples_batch     = len(indexes_batch)
        out_xData_array_shape = self.get_shape_out_array(self.xData_array.shape, num_samples_batch)
        out_yData_array_shape = self.get_shape_out_array(self.yData_array.shape, num_samples_batch)

        out_xData_array = np.ndarray(out_xData_array_shape, dtype=self.xData_array.dtype)
        out_yData_array = np.ndarray(out_yData_array_shape, dtype=self.yData_array.dtype)

        for i, index in enumerate(indexes_batch):
            out_xData_array[i] = self.images_generator.get_image_array(self.xData_array, index)
            out_yData_array[i] = self.images_generator.get_image_array(self.yData_array, index)
        #endfor

        return (out_xData_array, out_yData_array)


class BatchDataGenerator_List1Array(BatchDataGenerator):

    def __init__(self, size_image, list_xData_array, images_generator, numtot_samples=None, size_batch=1, shuffle=SHUFFLEIMAGES, seed=None):

        self.list_xData_array = list_xData_array
        self.images_generator = images_generator
        self.images_generator.complete_init_data(self.list_xData_array.shape[0:3])

        self.compute_pairIndex_samples(numtot_samples, shuffle, seed)

        numtot_samples = len(self.list_pairIndexes_samples)

        super(BatchDataGenerator_List1Array, self).__init__(size_image, numtot_samples, size_batch, shuffle, seed)

    def compute_pairIndex_samples(self, numtot_samples, shuffle, seed=None):

        self.list_pairIndex_samples = []
        if numtot_samples:
            # round-up
            numtot_samples = (numtot_samples + len(self.list_xData_array) - 1) // len(self.list_xData_array)

            for ifile in range(len(self.list_xData_array)):
                num_samples_file = numtot_samples // len(self.list_xData_array)

                #store pair of indexes: (idx_file, idx_batch)
                for index in range(num_samples_file):
                    self.list_pairIndex_samples.append((ifile, index))
                #endfor
            #endfor
        else:
            for ifile, xData_array in enumerate(self.list_xData_array):

                self.images_generator.complete_init_data(xData_array.shape[0:3])

                num_samples_file = self.images_generator.get_num_images()

                # store pair of indexes: (idx_file, idx_batch)
                for index in range(num_samples_file):
                    self.list_pairIndex_samples.append((ifile, index))
                # endfor
            # endfor

        numtot_samples = len(self.list_pairIndexes_samples)

        if (shuffle):
            if seed:
                np.random.seed(seed)
            randomIndexes = np.random.choice(numtot_samples, size=numtot_samples, replace=False)

            self.list_pairIndexes_samples_old = self.list_pairIndexes_samples
            self.list_pairIndexes_samples = []
            for index in randomIndexes:
                self.list_pairIndexes_samples.append(self.list_pairIndexes_samples_old[index])
            #endfor

    def get_data_samples_batch(self, indexes_batch):

        num_samples_batch = len(indexes_batch)
        out_array_shape   = self.get_shape_out_array(self.list_xData_array[0].shape, num_samples_batch)

        out_xData_array = np.ndarray(out_array_shape, dtype=self.list_xData_array[0].dtype)

        for i, index in enumerate(indexes_batch):
            (index_file, index_sample_file) = self.list_pairIndexes_samples[index]

            self.images_generator.complete_init_data(self.list_xData_array[index_file].shape[0:3])

            out_xData_array[i] = self.images_generator.get_image_array(self.list_xData_array[index_file], index_sample_file)
        #endfor

        return out_xData_array


class BatchDataGenerator_List2Arrays(BatchDataGenerator_List1Array):

    def __init__(self, size_image, list_xData_array, list_yData_array, images_generator, numtot_samples=None, size_batch=1, shuffle=SHUFFLEIMAGES, seed=None):

        self.list_yData_array = list_yData_array

        super(BatchDataGenerator_List2Arrays, self).__init__(size_image, list_xData_array, images_generator, numtot_samples, size_batch, shuffle, seed)

    def get_data_samples_batch(self, indexes_batch):

        num_samples_batch     = len(indexes_batch)
        out_xData_array_shape = self.get_shape_out_array(self.list_xData_array[0].shape, num_samples_batch)
        out_yData_array_shape = self.get_shape_out_array(self.list_yData_array[0].shape, num_samples_batch)

        out_xData_array = np.ndarray(out_xData_array_shape, dtype=self.list_xData_array[0].dtype)
        out_yData_array = np.ndarray(out_yData_array_shape, dtype=self.list_yData_array[0].dtype)

        for i, index in enumerate(indexes_batch):
            (index_file, index_sample_file) = self.list_pairIndexes_samples[index]

            self.images_generator.complete_init_data(self.list_xData_array[index_file].shape[0:3])

            out_xData_array[i] = self.images_generator.get_image_array(self.list_xData_array[index_file], index_sample_file)
            out_yData_array[i] = self.images_generator.get_image_array(self.list_yData_array[index_file], index_sample_file)
        #endfor

        return (out_xData_array, out_yData_array)