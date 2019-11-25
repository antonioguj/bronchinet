#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
import numpy as np
np.random.seed(2017)


class BatchDataGenerator(object):

    def __init__(self, numtot_samples, size_batch=1, shuffle=True, seed=None):

        self.numtot_samples= numtot_samples
        self.size_batch    = size_batch
        self.num_batches   = (self.numtot_samples + self.size_batch - 1)//self.size_batch # round-up

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


class BatchDataGenerator_1Array(BatchDataGenerator):

    def __init__(self, size_image, xData_array, images_generator, numtot_samples=None, size_batch=1, shuffle=True, seed=None):

        self.xData_array = xData_array
        self.type_xData  = xData_array.dtype

        self.array_shape_manager = ArrayShapeManager(size_image, is_shaped_Keras=False)

        self.num_channels_x = self.array_shape_manager.get_num_channels_array(xData_array.shape)

        self.images_generator = images_generator
        self.images_generator.update_image_data(xData_array.shape)

        if not numtot_samples:
            numtot_samples = images_generator.get_num_images()

        super(BatchDataGenerator_1Array, self).__init__(numtot_samples, size_batch, shuffle, seed)


    def get_data_samples_batch(self, indexes_batch):

        num_samples_batch = len(indexes_batch)

        out_array_shape = self.array_shape_manager.get_shape_out_array(num_samples_batch, num_channels=self.num_channels_x)
        out_xData_array = np.ndarray(out_array_shape, dtype=self.type_xData)

        for i, index in enumerate(indexes_batch):
            out_xData_array[i] = self.images_generator.get_image(self.xData_array, index=index)
        #endfor

        return out_xData_array

    def get_images_all(self):

        self.get_data_samples_batch(self.indexes_total)


class BatchDataGenerator_2Arrays(BatchDataGenerator_1Array):

    def __init__(self, size_image, xData_array, yData_array, images_generator, numtot_samples=None, size_batch=1, shuffle=True, seed=None):

        self.yData_array = yData_array
        self.type_yData  = yData_array.dtype

        super(BatchDataGenerator_2Arrays, self).__init__(size_image, xData_array, images_generator, numtot_samples, size_batch, shuffle, seed)

        self.num_channels_y = self.array_shape_manager.get_num_channels_array(yData_array.shape)


    def get_data_samples_batch(self, indexes_batch):

        num_samples_batch = len(indexes_batch)

        out_xData_array_shape = self.array_shape_manager.get_shape_out_array(num_samples_batch, num_channels=self.num_channels_x)
        out_yData_array_shape = self.array_shape_manager.get_shape_out_array(num_samples_batch, num_channels=self.num_channels_y)

        out_xData_array = np.ndarray(out_xData_array_shape, dtype=self.type_xData)
        out_yData_array = np.ndarray(out_yData_array_shape, dtype=self.type_yData)

        for i, index in enumerate(indexes_batch):
            (out_xData_array[i], out_yData_array[i]) = self.images_generator.get_image(self.xData_array,
                                                                                       index=index,
                                                                                       in2nd_array=self.yData_array)
        #endfor

        return (out_xData_array, out_yData_array)


class BatchDataGenerator_List1Array(BatchDataGenerator):

    def __init__(self, size_image, list_xData_array, images_generator, numtot_samples=None, size_batch=1, shuffle=True, seed=None):

        self.list_xData_array = list_xData_array
        self.type_xData       = list_xData_array[0].dtype

        self.array_shape_manager = ArrayShapeManager(size_image, is_shaped_Keras=False)

        self.num_channels_x = self.array_shape_manager.get_num_channels_array(list_xData_array[0].shape)

        self.images_generator = images_generator

        numtot_samples = self.compute_pairIndexes_samples(numtot_samples, shuffle, seed)

        super(BatchDataGenerator_List1Array, self).__init__(numtot_samples, size_batch, shuffle, seed)


    def compute_pairIndexes_samples(self, numtot_samples, shuffle, seed=None):

        self.list_pairIndexes_samples = []
        if numtot_samples:
            # round-up
            numtot_samples = (numtot_samples + len(self.list_xData_array) - 1) // len(self.list_xData_array)

            for ifile in range(len(self.list_xData_array)):
                num_samples_file = numtot_samples // len(self.list_xData_array)

                #store pair of indexes: (idx_file, idx_batch)
                for index in range(num_samples_file):
                    self.list_pairIndexes_samples.append((ifile, index))
                #endfor
            #endfor
        else:
            for ifile, xData_array in enumerate(self.list_xData_array):

                self.images_generator.update_image_data(xData_array.shape)

                num_samples_file = self.images_generator.get_num_images()

                # store pair of indexes: (idx_file, idx_batch)
                for index in range(num_samples_file):
                    self.list_pairIndexes_samples.append((ifile, index))
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

        return numtot_samples


    def get_data_samples_batch(self, indexes_batch):

        num_samples_batch = len(indexes_batch)

        out_array_shape = self.array_shape_manager.get_shape_out_array(num_samples_batch, num_channels=self.num_channels_x)
        out_xData_array = np.ndarray(out_array_shape, dtype=self.type_xData)

        for i, index in enumerate(indexes_batch):
            (index_file, index_sample_file) = self.list_pairIndexes_samples[index]

            self.images_generator.update_image_data(self.list_xData_array[index_file].shape)

            out_xData_array[i] = self.images_generator.get_image(self.list_xData_array[index_file],
                                                                 index=index_sample_file)
        #endfor

        return out_xData_array


class BatchDataGenerator_List2Arrays(BatchDataGenerator_List1Array):

    def __init__(self, size_image, list_xData_array, list_yData_array, images_generator, numtot_samples=None, size_batch=1, shuffle=True, seed=None):

        self.list_yData_array = list_yData_array
        self.type_yData       = list_yData_array[0].dtype

        super(BatchDataGenerator_List2Arrays, self).__init__(size_image, list_xData_array, images_generator, numtot_samples, size_batch, shuffle, seed)

        self.num_channels_y = self.array_shape_manager.get_num_channels_array(list_yData_array[0].shape)


    def get_data_samples_batch(self, indexes_batch):

        num_samples_batch = len(indexes_batch)

        out_xData_array_shape = self.array_shape_manager.get_shape_out_array(num_samples_batch, num_channels=self.num_channels_x)
        out_yData_array_shape = self.array_shape_manager.get_shape_out_array(num_samples_batch, num_channels=self.num_channels_y)

        out_xData_array = np.ndarray(out_xData_array_shape, dtype=self.type_xData)
        out_yData_array = np.ndarray(out_yData_array_shape, dtype=self.type_yData)

        for i, index in enumerate(indexes_batch):
            (index_file, index_sample_file) = self.list_pairIndexes_samples[index]

            self.images_generator.update_image_data(self.list_xData_array[index_file].shape)

            (out_xData_array[i], out_yData_array[i]) = self.images_generator.get_image(self.list_xData_array[index_file],
                                                                                       index=index_sample_file,
                                                                                       in2nd_array=self.list_yData_array[index_file])
        #endfor

        return (out_xData_array, out_yData_array)



class ArrayShapeManager(object):

    def __init__(self, size_image,
                 is_shaped_Keras=False,
                 num_classes_out=1,
                 size_output_Unet=None):
        self.size_image = size_image
        self.is_shaped_Keras = is_shaped_Keras
        self.num_classes_out = num_classes_out
        if size_output_Unet and (size_output_Unet != size_image):
            self.size_output_Unet = size_output_Unet
        else:
            self.size_output_Unet = size_image


    def is_images_array_without_channels(self, in_array_shape):
        return len(in_array_shape) == len(self.size_image)

    def get_num_channels_array(self, in_array_shape):
        if self.is_images_array_without_channels(in_array_shape):
            return None
        else:
            return in_array_shape[-1]

    def get_shape_out_array(self, num_samples,
                            num_channels):
        if self.is_shaped_Keras and not num_channels:
            # arrays in Keras always have one dim reserved for channels
            num_channels = 1
        if num_channels:
            return [num_samples] + list(self.size_image) + [num_channels]
        else:
            return [num_samples] + list(self.size_image)

    def get_array_with_channels(self, in_array,
                                num_channels=None):
        if self.is_shaped_Keras and not num_channels:
            # arrays in Keras always have one dim reserved for channels
            num_channels = 1
        if num_channels == 1:
            return np.expand_dims(in_array, axis=-1)
        elif num_channels > 1:
            return np.reshape(in_array, in_array.shape + [num_channels])
        else:
            return in_array

    # def get_array_categorical_masks(self, yData):
    #     return K_to_categorical(yData, num_classes=self.num_classes_out)


    @staticmethod
    def get_shape_Keras(num_images,
                        size_image,
                        num_channels):
        return [num_images] + list(size_image) + [num_channels]
        # if K_image_data_format() == 'channels_first':
        #     return [num_images, num_channels] + list(size_image)
        # elif K_image_data_format() == 'channels_last':
        #     return [num_images] + list(size_image) + [num_channels]
        # else:
        #     return 0

    @staticmethod
    def get_array_reshaped_Keras(array):
        return array
        # if K_image_data_format() == 'channels_first':
        #     # need to roll last dimensions, channels, to second dim:
        #     return np.rollaxis(array, -1, 1)
        # elif K_image_data_format() == 'channels_last':
        #     return array
        # else:
        #     return 0


    @staticmethod
    def get_limits_cropImage(size_image,
                             size_output_Unet):
        if (size_image == size_output_Unet):
            list_out_aux = [[0] + [s_i] for s_i in size_image]
        else:
            list_out_aux = [[(s_i - s_o) / 2] + [(s_i + s_o) / 2] for (s_i, s_o) in zip(size_image, size_output_Unet)]
        # flatten out list of lists and return tuple
        return tuple(reduce(lambda el1, el2: el1 + el2, list_out_aux))

    def get_array_shaped_outNnet(self, yData):
        if (self.size_image==self.size_output_Unet):
            return yData
        else:
            if len(self.size_image)==2:
                return self.get_array_shaped_outNnet_2D(yData)
            elif len(self.size_image)==3:
                return self.get_array_shaped_outNnet_3D(yData)

    def get_array_shaped_outNnet_2D(self, yData):
        (x_left, x_right, y_down, y_up) = self.get_limits_cropImage(self.size_image, self.size_output_Unet)

        if self.is_images_array_without_channels(yData.shape):
            return yData[..., x_left:x_right, y_down:y_up]
        else:
            return yData[..., x_left:x_right, y_down:y_up, :]

    def get_array_shaped_outNnet_3D(self, yData):
        (z_back, z_front, x_left, x_right, y_down, y_up) = self.get_limits_cropImage(self.size_image, self.size_output_Unet)

        if self.is_images_array_without_channels(yData.shape):
            return yData[..., z_back:z_front, x_left:x_right, y_down:y_up]
        else:
            return yData[..., z_back:z_front, x_left:x_right, y_down:y_up, :]


    def get_xData_array_reshaped(self, xData):
        if self.is_shaped_Keras:
            if self.is_images_array_without_channels(xData.shape):
                return self.get_array_reshaped_Keras(self.get_array_with_channels(xData))
            else:
                return self.get_array_reshaped_Keras(xData)
        else:
            if self.is_images_array_without_channels(xData.shape):
                return self.get_array_with_channels(xData)
            else:
                return xData

    def get_yData_array_reshaped(self, yData):
        if self.is_shaped_Keras:
            if self.num_classes_out > 1:
                if (self.size_image==self.size_output_Unet):
                    return self.get_array_reshaped_Keras(self.get_array_categorical_masks(yData))
                else:
                    return self.get_array_reshaped_Keras(self.get_array_categorical_masks(self.get_array_shaped_outNnet(yData)))
            else:
                if (self.size_image==self.size_output_Unet):
                    if self.is_images_array_without_channels(yData.shape):
                        return self.get_array_reshaped_Keras(self.get_array_with_channels(yData))
                    else:
                        return self.get_array_reshaped_Keras(yData)
                else:
                    if self.is_images_array_without_channels(yData.shape):
                        return self.get_array_reshaped_Keras(self.get_array_with_channels(self.get_array_shaped_outNnet(yData)))
                    else:
                        return self.get_array_reshaped_Keras(self.get_array_shaped_outNnet(yData))
        else:
            if self.num_classes_out > 1:
                if (self.size_image==self.size_output_Unet):
                    return self.get_array_categorical_masks(yData)
                else:
                    return self.get_array_categorical_masks(self.get_array_shaped_outNnet(yData))
            else:
                if (self.size_image==self.size_output_Unet):
                    if self.is_images_array_without_channels(yData.shape):
                        return self.get_array_with_channels(yData)
                    else:
                        return yData
                else:
                    if self.is_images_array_without_channels(yData.shape):
                        return self.get_array_with_channels(self.get_array_shaped_outNnet(yData))
                    else:
                        return self.get_array_shaped_outNnet(yData)


class ArrayShapeManagerInBatches(ArrayShapeManager):

    def __init__(self, size_image,
                 is_shaped_Keras=False,
                 num_classes_out=1,
                 size_output_Unet=None):
        super(ArrayShapeManagerInBatches, self).__init__(size_image,
                                                         is_shaped_Keras,
                                                         num_classes_out,
                                                         size_output_Unet)
    # def is_images_array_without_channels(self, in_array_shape):
    #     return len(in_array_shape) == len(self.size_image) + 1