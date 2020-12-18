
from typing import List, Tuple
import numpy as np
np.random.seed(2017)

from common.exceptionmanager import catch_error_exception
from common.functionutil import ImagesUtil
from imageoperators.boundingboxes import BoundingBoxes
from imageoperators.imageoperator import CropImage
from preprocessing.imagegenerator import ImageGenerator


class BatchDataGenerator(object):

    def __init__(self,
                 size_data: int,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None
                 ) -> None:
        self._size_data = size_data
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed

        self._on_epoch_end()

    def __len__(self) -> int:
        "Denotes the number of batches per epoch"
        return (self._size_data + self._batch_size - 1) // self._batch_size

    def _on_epoch_end(self) -> None:
        "Updates indexes after each epoch"
        self._indexes = np.arange(self._size_data)
        if (self._size_data % self._batch_size != 0):
            extra_indexes = np.random.randint(self._size_data, size=self._batch_size)
            self._indexes = np.concatenate([self._indexes, extra_indexes])

        if self._shuffle == True:
            if self._seed is not None:
                np.random.seed(self._seed)
            np.random.shuffle(self._indexes)

    def _get_indexes_batch(self, index: int) -> List[int]:
        return self._indexes[index * self._batch_size: (index + 1) * self._batch_size]

    def __getitem__(self, index: int):
        raise NotImplementedError

    def get_full_data(self) -> np.ndarray:
        raise NotImplementedError


class BatchImageDataGenerator_1Image(BatchDataGenerator):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 list_Xdata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 type_image_format: str = 'channels_last',
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = True
                 ) -> None:
        self._size_image = size_image
        self._list_Xdata = list_Xdata
        self._dtype_Xdata = list_Xdata[0].dtype
        self._images_generator = images_generator
        self._num_channels_in = num_channels_in

        self._num_images = self._compute_list_indexes_images_files(is_print_datagen_info)

        self._is_reshape_channels_first = type_image_format == 'channels_first'

        super(BatchImageDataGenerator_1Image, self).__init__(self._num_images, batch_size, shuffle, seed)

    def _compute_list_indexes_images_files(self, is_print_datagen_info: bool = False) -> int:
        "Store pairs of indexes (index_file, index_batch)"
        self._list_indexes_imagefile = []

        for ifile, i_Xdata in enumerate(self._list_Xdata):
            self._images_generator.update_image_data(i_Xdata.shape)
            num_images_file = self._images_generator.get_num_images()

            for index in range(num_images_file):
                self._list_indexes_imagefile.append((ifile, index))

            if is_print_datagen_info:
                message = self._images_generator.get_text_description()
                print("Image file: \'%s\'..." % (ifile))
                print(message[:-1])   # remove trailing '\n'

        num_images = len(self._list_indexes_imagefile)
        return num_images

    def __getitem__(self, index: int) -> np.ndarray:
        return self._get_data_batch(index)

    def _get_data_batch(self, index: int) -> np.ndarray:
        "Generate one batch of data"
        indexes_batch = self._get_indexes_batch(index)
        num_images_batch = len(indexes_batch)

        out_shape_Xdata = (num_images_batch,) + self._size_image + (self._num_channels_in,)
        out_Xdata = np.ndarray(out_shape_Xdata, dtype=self._dtype_Xdata)

        for i, index in enumerate(indexes_batch):
            out_Xdata[i] = self._get_data_sample(index)

        return self._process_batch_data(out_Xdata)

    def _get_data_sample(self, index: int) -> np.ndarray:
        "Generate one sample of batch of data"
        (index_file, index_image_file) = self._list_indexes_imagefile[index]
        self._images_generator.update_image_data(self._list_Xdata[index_file].shape)

        out_Xdata_elem = self._images_generator.get_image(self._list_Xdata[index_file],
                                                          index=index_image_file, seed=None)
        return self._process_sample_Xdata(out_Xdata_elem)

    def get_full_data(self) -> np.ndarray:
        "Generate full data including all batches"
        out_shape_Xdata = (self._num_images,) + self._size_image + (self._num_channels_in,)
        out_Xdata = np.ndarray(out_shape_Xdata, dtype= self._dtype_Xdata)

        for i in range(self._num_images):
            out_Xdata[i] = self._get_data_sample(i)

        return self._process_batch_data(out_Xdata)

    def _process_sample_Xdata(self, in_image: np.ndarray) -> np.ndarray:
        if ImagesUtil.is_without_channels(self._size_image, in_image.shape):
            in_image = np.expand_dims(in_image, axis=-1)
        return in_image
        
    def _process_batch_data(self, in_batch_data: np.ndarray) -> np.ndarray:
        if self._is_reshape_channels_first:
            return ImagesUtil.reshape_channels_first(in_batch_data)
        else:
            return in_batch_data


class BatchImageDataGenerator_2Images(BatchImageDataGenerator_1Image):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 list_Xdata: List[np.ndarray],
                 list_Ydata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 num_classes_out: int = 1,
                 type_image_format: str = 'channels_last',
                 is_output_nnet_validconvs: bool = False,
                 size_output_image: Tuple[int, ...] = None,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = True
                 ) -> None:
        super(BatchImageDataGenerator_2Images, self).__init__(size_image,
                                                              list_Xdata,
                                                              images_generator,
                                                              num_channels_in=num_channels_in,
                                                              type_image_format=type_image_format,
                                                              batch_size=batch_size,
                                                              shuffle=shuffle,
                                                              seed=seed,
                                                              is_print_datagen_info=is_print_datagen_info)
        self._size_image = size_image
        self._list_Ydata = list_Ydata
        self._dtype_Ydata = list_Ydata[0].dtype
        self._num_classes_out = num_classes_out

        if len(self._list_Xdata) != len(self._list_Ydata):
            message = 'Size of list Xdata \'%s\' not equal to size of list Ydata \'%s\'' %(len(list_Xdata), len(list_Ydata))
            catch_error_exception(message)

        self._is_output_nnet_validconvs = is_output_nnet_validconvs
        if is_output_nnet_validconvs and size_output_image and \
            (size_image != size_output_image):
            self._size_output_image = size_output_image
            self._output_crop_bounding_box = BoundingBoxes.compute_bounding_box_centered_image_fit_image(self._size_output_image,
                                                                                                         self._size_image)
            ndims = len(size_image)
            if ndims==2:
                self._func_crop_images = CropImage._compute2D
            elif ndims==3:
                self._func_crop_images = CropImage._compute3D
            else:
                message = 'BatchImageDataGenerator_2Images:__init__: wrong \'ndims\': %s...' % (ndims)
                catch_error_exception(message)
        else:
            self._is_output_nnet_validconvs = False
            self._size_output_image = size_image

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._get_data_batch(index)

    def _get_data_batch(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        "Generate one batch of data"
        indexes_batch = self._get_indexes_batch(index)
        num_images_batch = len(indexes_batch)

        out_shape_Xdata = (num_images_batch,) + self._size_image + (self._num_channels_in,)
        out_shape_Ydata = (num_images_batch,) + self._size_output_image + (self._num_classes_out,)
        out_Xdata = np.ndarray(out_shape_Xdata, dtype=self._dtype_Xdata)
        out_Ydata = np.ndarray(out_shape_Ydata, dtype=self._dtype_Ydata)

        for i, index in enumerate(indexes_batch):
            (out_Xdata[i], out_Ydata[i]) = self._get_data_sample(index)

        return (self._process_batch_data(out_Xdata),
                self._process_batch_data(out_Ydata))

    def _get_data_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        "Generate one sample of batch of data"
        (index_file, index_image_file) = self._list_indexes_imagefile[index]
        self._images_generator.update_image_data(self._list_Xdata[index_file].shape)

        (out_Xdata_elem, out_Ydata_elem) = self._images_generator.get_2images(self._list_Xdata[index_file],
                                                                              self._list_Ydata[index_file],
                                                                              index=index_image_file, seed=None)
        return (self._process_sample_Xdata(out_Xdata_elem),
                self._process_sample_Ydata(out_Ydata_elem))

    def get_full_data(self) -> Tuple[np.ndarray, np.ndarray]:
        "Generate full data including all batches"
        out_shape_Xdata = (self._num_images,) + self._size_image + (self._num_channels_in,)
        out_shape_Ydata = (self._num_images,) + self._size_output_image + (self._num_classes_out,)
        out_Xdata = np.ndarray(out_shape_Xdata, dtype= self._dtype_Xdata)
        out_Ydata = np.ndarray(out_shape_Ydata, dtype= self._dtype_Ydata)

        for i in range(self._num_images):
            (out_Xdata[i], out_Ydata[i]) = self._get_data_sample(i)

        return (self._process_batch_data(out_Xdata),
                self._process_batch_data(out_Ydata))

    def _get_cropped_sample(self, in_image: np.ndarray) -> np.ndarray:
        return self._func_crop_images(in_image, self._output_crop_bounding_box)

    def _process_sample_Ydata(self, in_image: np.ndarray) -> np.ndarray:
        if self._is_output_nnet_validconvs:
            in_image = self._get_cropped_sample(in_image)
        return self._process_sample_Xdata(in_image)
