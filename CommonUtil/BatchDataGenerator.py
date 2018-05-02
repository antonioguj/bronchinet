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

    def get_num_channels_array(self, in_array_shape):
        if len(in_array_shape) == len(self.size_image):
            return 1
        else:
            return in_array_shape[-1]

    def get_shape_out_array(self, num_samples, num_channels=1):
        if num_channels == 1:
            return [num_samples] + list(self.size_image)
        else:
            return [num_samples] + list(self.size_image) + [num_channels]


class BatchDataGenerator_1Array(BatchDataGenerator):

    def __init__(self, size_image, xData_array, images_generator, numtot_samples=None, size_batch=1, shuffle=SHUFFLEIMAGES, seed=None):

        self.xData_array     = xData_array
        self.images_generator= images_generator
        self.images_generator.complete_init_data(xData_array.shape[0:3])
        self.num_channels_x  = self.get_num_channels_array(self.xData_array.shape)

        if not numtot_samples:
            numtot_samples = images_generator.get_num_images()

        super(BatchDataGenerator_1Array, self).__init__(size_image, numtot_samples, size_batch, shuffle, seed)

    def get_data_samples_batch(self, indexes_batch):

        num_samples_batch = len(indexes_batch)
        out_array_shape   = self.get_shape_out_array(num_samples_batch, num_channels=self.num_channels_x)

        out_xData_array = np.ndarray(out_array_shape, dtype=self.xData_array.dtype)

        for i, index in enumerate(indexes_batch):
            out_xData_array[i] = self.images_generator.get_image_array(self.xData_array, index)
        #endfor

        return out_xData_array

    def get_images_all(self):

        self.get_data_samples_batch(self.indexes_total)


class BatchDataGenerator_2Arrays(BatchDataGenerator_1Array):

    def __init__(self, size_image, xData_array, yData_array, images_generator, numtot_samples=None, size_batch=1, shuffle=SHUFFLEIMAGES, seed=None):

        self.yData_array   = yData_array
        self.num_channels_y= self.get_num_channels_array(self.yData_array.shape)

        super(BatchDataGenerator_2Arrays, self).__init__(size_image, xData_array, images_generator, numtot_samples, size_batch, shuffle, seed)

    def get_data_samples_batch(self, indexes_batch):

        num_samples_batch     = len(indexes_batch)
        out_xData_array_shape = self.get_shape_out_array(num_samples_batch, num_channels=self.num_channels_x)
        out_yData_array_shape = self.get_shape_out_array(num_samples_batch, num_channels=self.num_channels_y)

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
        self.type_xData       = list_xData_array[0].dtype
        self.images_generator = images_generator
        self.images_generator.complete_init_data(self.list_xData_array[0].shape[0:3])
        self.num_channels_x   = self.get_num_channels_array(self.list_xData_array[0].shape)

        numtot_samples = self.compute_pairIndex_samples(numtot_samples)

        super(BatchDataGenerator_List1Array, self).__init__(size_image, numtot_samples, size_batch, shuffle, seed)

    def compute_pairIndex_samples(self, numtot_samples):

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

                #store pair of indexes: (idx_file, idx_batch)
                for index in range(num_samples_file):
                    self.list_pairIndex_samples.append((ifile, index))
                #endfor
            #endfor
        return len(self.list_pairIndex_samples)

    def reset(self, shuffle, seed=None):
        if (shuffle):
            if seed:
                np.random.seed(seed)
            randomIndexes = np.random.choice(self.numtot_samples, size=self.numtot_samples, replace=False)
            self.indexes_total = []
            for index in randomIndexes:
                self.indexes_total.append(self.list_pairIndex_samples[index])
            #endfor
        else:
            self.indexes_total = self.list_pairIndex_samples

    def get_data_samples_batch(self, indexes_batch):

        num_samples_batch = len(indexes_batch)
        out_array_shape   = self.get_shape_out_array(num_samples_batch, num_channels=self.num_channels_x)

        out_xData_array = np.ndarray(out_array_shape, dtype=self.xData_array.dtype)

        for i, (index_file, index_sample_file) in enumerate(indexes_batch):
            self.images_generator.complete_init_data(self.list_xData_array[index_file].shape[0:3])

            out_xData_array[i] = self.images_generator.get_image_array(self.list_xData_array[index_file], index_sample_file)
        #endfor

        return out_xData_array


class BatchDataGenerator_List2Arrays(BatchDataGenerator_List1Array):

    def __init__(self, size_image, list_xData_array, list_yData_array, images_generator, numtot_samples=None, size_batch=1, shuffle=SHUFFLEIMAGES, seed=None):

        self.list_yData_array= list_yData_array
        self.type_yData      = list_yData_array[0].dtype
        self.num_channels_y  = self.get_num_channels_array(self.list_yData_array[0].shape)

        super(BatchDataGenerator_List2Arrays, self).__init__(size_image, list_xData_array, images_generator, numtot_samples, size_batch, shuffle, seed)

    def get_data_samples_batch(self, indexes_batch):

        num_samples_batch     = len(indexes_batch)
        out_xData_array_shape = self.get_shape_out_array(num_samples_batch, num_channels=self.num_channels_x)
        out_yData_array_shape = self.get_shape_out_array(num_samples_batch, num_channels=self.num_channels_y)

        out_xData_array = np.ndarray(out_xData_array_shape, dtype=self.xData_array.dtype)
        out_yData_array = np.ndarray(out_yData_array_shape, dtype=self.yData_array.dtype)

        for i, (index_file, index_sample_file) in enumerate(indexes_batch):
            self.images_generator.complete_init_data(self.list_xData_array[index_file].shape[0:3])

            out_xData_array[i] = self.images_generator.get_image_array(self.list_xData_array[index_file], index_sample_file)
            out_yData_array[i] = self.images_generator.get_image_array(self.list_yData_array[index_file], index_sample_file)
        #endfor

        return (out_xData_array, out_yData_array)