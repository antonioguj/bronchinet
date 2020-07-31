
from typing import List, Tuple
import numpy as np
np.random.seed(2017)

from common.exception_manager import catch_error_exception
from imageoperators.boundingboxes import BoundingBoxes
from imageoperators.imageoperator import CropImage
from preprocessing.imagegenerator import ImageGenerator


class BatchDataGenerator(object):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 list_Xdata: List[np.ndarray],
                 list_Ydata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 num_classes_out: int = 1,
                 is_output_nnet_validconvs: bool = False,
                 size_output_image: Tuple[int, ...] = None,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 iswrite_datagen_info: bool = True
                 ) -> None:
        self._size_image = size_image
        self._num_channels_in = num_channels_in

        self._list_Xdata = list_Xdata
        self._list_Ydata = list_Ydata
        self._dtype_Xdata = list_Xdata[0].dtype
        self._dtype_Ydata = list_Ydata[0].dtype

        if len(list_Xdata) != len(list_Ydata):
            message = 'Size of list Xdata \'%s\' not equal to size of list Ydata \'%s\'' %(len(list_Xdata), len(list_Ydata))
            catch_error_exception(message)

        self._images_generator = images_generator

        self._num_classes_out = num_classes_out

        self._is_outputUnet_validconvs = is_output_nnet_validconvs
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
                message = 'BatchDataGenerator:__init__: wrong \'ndims\': %s...' % (self._ndims)
                catch_error_exception(message)
        else:
            self._is_outputUnet_validconvs = False
            self._size_output_image = size_image

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed

        self._iswrite_datagen_info = iswrite_datagen_info

        self._num_images = self._compute_list_indexes_imagefile()

        self._on_epoch_end()

    def __len__(self) -> int:
        "Denotes the number of batches per epoch"
        return (self._num_images + self._batch_size - 1) // self._batch_size

    def _on_epoch_end(self) -> None:
        "Updates indexes after each epoch"
        self._indexes = np.arange(self._num_images)
        if self._shuffle == True:
            if self._seed is not None:
                np.random.seed(self._seed)
            np.random.shuffle(self._indexes)

    def _compute_list_indexes_imagefile(self) -> int:
        "Store pairs of indexes (index_file, index_batch)"
        self._list_indexes_imagefile = []

        for ifile, Xdata in enumerate(self._list_Xdata):
            self._images_generator.update_image_data(Xdata.shape)
            num_images_file = self._images_generator.get_num_images()

            for index in range(num_images_file):
                self._list_indexes_imagefile.append((ifile, index))

            if self._iswrite_datagen_info:
                message = self._images_generator.get_text_description()
                print("Image file: \'%s\'..." %(ifile))
                print(message)

        num_images = len(self._list_indexes_imagefile)
        return num_images

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        "Generate one batch of data"
        indexes = self._indexes[index * self._batch_size: (index + 1) * self._batch_size]  # indexes of this batch
        num_images_batch = len(indexes)

        out_shape_Xdata = [num_images_batch] + list(self._size_image) + [self._num_channels_in]
        out_shape_Ydata = [num_images_batch] + list(self._size_output_image) + [self._num_classes_out]
        out_Xdata = np.ndarray(out_shape_Xdata, dtype=self._dtype_Xdata)
        out_Ydata = np.ndarray(out_shape_Ydata, dtype=self._dtype_Ydata)

        for i, index in enumerate(indexes):
            (out_Xdata[i], out_Ydata[i]) = self._get_item(index)

        return (out_Xdata, out_Ydata)

    def _get_item(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        "Generate one sample of batch of data"
        (index_file, index_image_file) = self._list_indexes_imagefile[index]
        self._images_generator.update_image_data(self._list_Xdata[index_file].shape)

        (out_Xdata_item, out_Ydata_item) = self._images_generator.get_2images(self._list_Xdata[index_file],
                                                                              self._list_Ydata[index_file],
                                                                              index=index_image_file,
                                                                              seed=None)
        return (self._get_formated_output_Xdata(out_Xdata_item),
                self._get_formated_output_Ydata(out_Ydata_item))

    def get_full_data(self) -> Tuple[np.ndarray, np.ndarray]:
        "Generate full data including all batches"
        out_shape_Xdata = [self._num_images] + list(self._size_image) + [self._num_channels_in]
        out_shape_Ydata = [self._num_images] + list(self._size_output_image) + [self._num_classes_out]
        out_Xdata = np.ndarray(out_shape_Xdata, dtype= self._dtype_Xdata)
        out_Ydata = np.ndarray(out_shape_Ydata, dtype= self._dtype_Ydata)

        for i in range(self._num_images):
            (out_Xdata[i], out_Ydata[i]) = self._get_item(i)

        return (out_Xdata, out_Ydata)

    def _get_cropped_output(self, in_image: np.ndarray) -> np.ndarray:
        return self._func_crop_images(in_image, self._output_crop_bounding_box)

    def _get_formated_output_Xdata(self, in_image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _get_formated_output_Ydata(self, in_image: np.ndarray) -> np.ndarray:
        raise NotImplementedError