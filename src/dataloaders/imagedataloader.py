
from typing import List, Tuple, Union
import numpy as np

from common.exceptionmanager import catch_error_exception
from common.functionutil import is_exist_file
from dataloaders.imagefilereader import ImageFileReader


class ImageDataLoader(object):

    @classmethod
    def load_1file(cls, filename: str) -> np.ndarray:
        if not is_exist_file(filename):
            message = 'input file does not exist: \'%s\'' % (filename)
            catch_error_exception(message)

        return ImageFileReader.get_image(filename)

    @classmethod
    def load_2files(cls,
                    filename_1: str,
                    filename_2: str
                    ) -> Tuple[np.ndarray, np.ndarray]:
        if not is_exist_file(filename_1):
            message = 'input file 1 does not exist: \'%s\'' % (filename_1)
            catch_error_exception(message)
        if not is_exist_file(filename_2):
            message = 'input file 1 does not exist: \'%s\'' % (filename_2)
            catch_error_exception(message)

        out_image_1 = ImageFileReader.get_image(filename_1)
        out_image_2 = ImageFileReader.get_image(filename_2)

        if out_image_1.shape != out_image_2.shape:
            message = 'input image 1 and 2 of different size: (\'%s\' != \'%s\')' \
                      % (out_image_1.shape, out_image_2.shape)
            catch_error_exception(message)

        return (out_image_1, out_image_2)

    @classmethod
    def load_1list_files(cls, list_filenames: List[str]) -> List[np.ndarray]:
        out_list_images = []
        for in_file in list_filenames:
            out_image = cls.load_1file(in_file)
            out_list_images.append(out_image)

        return out_list_images

    @classmethod
    def load_2list_files(cls,
                         list_filenames_1: List[str],
                         list_filenames_2: List[str]
                         ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if len(list_filenames_1) != len(list_filenames_2):
            message = 'number files in list1 (%s) and list2 (%s) are not equal' \
                      % (len(list_filenames_1), len(list_filenames_2))
            catch_error_exception(message)

        out_list_images_1 = []
        out_list_images_2 = []
        for in_file_1, in_file_2 in zip(list_filenames_1, list_filenames_2):
            (out_image_1, out_image_2) = cls.load_2files(in_file_1, in_file_2)
            out_list_images_1.append(out_image_1)
            out_list_images_2.append(out_image_2)

        return (out_list_images_1, out_list_images_2)


class ImageDataBatchesLoader(ImageDataLoader):
    _max_load_images_default = None

    def __init__(self, size_image: Union[Tuple[int, int, int], Tuple[int, int]]) -> None:
        self._size_image = size_image

    @staticmethod
    def _shuffle_data(in_imagedata_1: np.ndarray,
                      in_imagedata_2: np.ndarray = None
                      ) -> Tuple[np.ndarray, None]:
        # randomly shuffle the elements in image data
        indexes_shuffled = np.arange(in_imagedata_1.shape[0])
        np.random.shuffle(indexes_shuffled)
        if in_imagedata_2 is not None:
            return (in_imagedata_1[indexes_shuffled[:]], in_imagedata_2[indexes_shuffled[:]])
        else:
            return (in_imagedata_1[indexes_shuffled[:]], None)

    def load_1file(self,
                   filename: str,
                   max_load_images: Union[int, None] = _max_load_images_default,
                   is_shuffle: bool = False
                   ) -> np.ndarray:
        in_stack_images = super(ImageDataBatchesLoader, self).load_1file(filename)
        num_images_stack = in_stack_images.shape[0]

        if in_stack_images[0].shape != self._size_image:
            message = 'image size in input stack is different from image size in class: (\'%s\' != \'%s\'). change ' \
                      'input size in class to be equal to the first' % (in_stack_images[0].shape, self._size_image)
            catch_error_exception(message)

        if max_load_images is not None and (num_images_stack > max_load_images):
            out_batch_images = in_stack_images[0:max_load_images]
        else:
            out_batch_images = in_stack_images

        if is_shuffle:
            (out_batch_images, _) = self._shuffle_data(out_batch_images)

        return out_batch_images

    def load_2files(self,
                    filename_1: str,
                    filename_2: str,
                    max_load_images: Union[int, None] = _max_load_images_default,
                    is_shuffle: bool = False
                    ) -> Tuple[np.ndarray, np.ndarray]:
        (in_stack_images_1, in_stack_images_2) = super(ImageDataBatchesLoader, self).load_2files(filename_1, filename_2)
        num_images_stack = in_stack_images_1.shape[0]

        if in_stack_images_1[0].shape != self._size_image:
            message = 'image size in input stack is different from image size in class: (\'%s\' != \'%s\'). change ' \
                      'input size in class to be equal to the first' % (in_stack_images_1[0].shape, self._size_image)
            catch_error_exception(message)

        if max_load_images is not None and (num_images_stack > max_load_images):
            out_batch_images_1 = in_stack_images_1[0:max_load_images]
            out_batch_images_2 = in_stack_images_2[0:max_load_images]
        else:
            out_batch_images_1 = in_stack_images_1
            out_batch_images_2 = in_stack_images_2

        if is_shuffle:
            (out_batch_images_1, out_batch_images_2) = self._shuffle_data(out_batch_images_1, out_batch_images_2)

        return (out_batch_images_1, out_batch_images_2)

    def load_1list_files(self,
                         list_filenames: List[str],
                         max_load_images: Union[int, None] = _max_load_images_default,
                         is_shuffle: bool = False
                         ) -> List[np.ndarray]:
        out_dtype = super(ImageDataBatchesLoader, self).load_1file(list_filenames[0]).dtype
        out_batch_images = np.array([], dtype=out_dtype).reshape((0,) + self._size_image)
        sumrun_out_images = 0

        for in_file in list_filenames:
            in_stack_images = super(ImageDataBatchesLoader, self).load_1file(in_file)
            num_images_stack = in_stack_images.shape[0]
            sumrun_out_images = sumrun_out_images + num_images_stack

            if max_load_images is not None and (sumrun_out_images > max_load_images):
                num_images_rest_batch = num_images_stack - (sumrun_out_images - max_load_images)
                in_stack_images = in_stack_images[0:num_images_rest_batch]

            out_batch_images = np.concatenate((out_batch_images, in_stack_images), axis=0)

        if is_shuffle:
            (out_batch_images, _) = self._shuffle_data(out_batch_images)

        return out_batch_images

    def load_2list_files(self,
                         list_filenames_1: List[str],
                         list_filenames_2: List[str],
                         max_load_images: Union[int, None] = _max_load_images_default,
                         is_shuffle: bool = False
                         ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if len(list_filenames_1) != len(list_filenames_2):
            message = 'number files in list1 (%s) and list2 (%s) are not equal' \
                      % (len(list_filenames_1), len(list_filenames_2))
            catch_error_exception(message)

        out_dtype_1 = super(ImageDataBatchesLoader, self).load_1file(list_filenames_1[0]).dtype
        out_dtype_2 = super(ImageDataBatchesLoader, self).load_1file(list_filenames_2[0]).dtype
        out_batch_images_1 = np.array([], dtype=out_dtype_1).reshape((0,) + self._size_image)
        out_batch_images_2 = np.array([], dtype=out_dtype_2).reshape((0,) + self._size_image)
        sumrun_out_images = 0

        for in_file1, in_file2 in zip(list_filenames_1, list_filenames_2):
            (in_stack_images_1, in_stack_images_2) = super(ImageDataBatchesLoader, self).load_2files(in_file1, in_file2)
            num_images_stack = in_stack_images_1.shape[0]
            sumrun_out_images = sumrun_out_images + num_images_stack

            if max_load_images is not None and (sumrun_out_images > max_load_images):
                num_images_rest_batch = num_images_stack - (sumrun_out_images - max_load_images)
                in_stack_images_1 = in_stack_images_1[0:num_images_rest_batch]
                in_stack_images_2 = in_stack_images_2[0:num_images_rest_batch]

            out_batch_images_1 = np.concatenate((out_batch_images_1, in_stack_images_1), axis=0)
            out_batch_images_2 = np.concatenate((out_batch_images_2, in_stack_images_2), axis=0)

        if is_shuffle:
            (out_batch_images_1, out_batch_images_2) = self._shuffle_data(out_batch_images_1, out_batch_images_2)

        return (out_batch_images_1, out_batch_images_2)
