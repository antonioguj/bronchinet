
from typing import List, Tuple, Union
import numpy as np

from torch.utils import data as data_torch
import torch

from common.constant import IS_MODEL_GPU, IS_MODEL_HALFPREC
from common.functionutil import ImagesUtil
from dataloaders.batchdatagenerator import BatchImageDataGenerator1Image, BatchImageDataGenerator2Images
from preprocessing.imagegenerator import ImageGenerator

if IS_MODEL_GPU:
    if IS_MODEL_HALFPREC:
        OutputDataType = torch.cuda.HalfTensor
    else:
        OutputDataType = torch.cuda.FloatTensor
else:
    if IS_MODEL_HALFPREC:
        OutputDataType = torch.HalfTensor
    else:
        OutputDataType = torch.FloatTensor


class TrainBatchImageDataGenerator1Image(BatchImageDataGenerator1Image):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 list_xdata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = False
                 ) -> None:
        super(TrainBatchImageDataGenerator1Image, self).__init__(size_image,
                                                                 list_xdata,
                                                                 images_generator,
                                                                 num_channels_in=num_channels_in,
                                                                 type_image_format='channels_first',
                                                                 batch_size=batch_size,
                                                                 shuffle=shuffle,
                                                                 seed=seed,
                                                                 is_print_datagen_info=is_print_datagen_info)

    def __getitem__(self, index: int) -> np.ndarray:
        out_xdata = self._get_data_sample(index)
        out_xdata = ImagesUtil.reshape_channels_first(out_xdata, is_input_sample=True)
        return torch.from_numpy(out_xdata.copy()).type(OutputDataType)


class TrainBatchImageDataGenerator2Images(BatchImageDataGenerator2Images):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 list_xdata: List[np.ndarray],
                 list_ydata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 num_classes_out: int = 1,
                 is_nnet_validconvs: bool = False,
                 size_output_image: Union[Tuple[int, int, int], Tuple[int, int]] = None,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = False
                 ) -> None:
        super(TrainBatchImageDataGenerator2Images, self).__init__(size_image,
                                                                  list_xdata,
                                                                  list_ydata,
                                                                  images_generator,
                                                                  num_channels_in=num_channels_in,
                                                                  num_classes_out=num_classes_out,
                                                                  type_image_format='channels_first',
                                                                  is_nnet_validconvs=is_nnet_validconvs,
                                                                  size_output_image=size_output_image,
                                                                  batch_size=batch_size,
                                                                  shuffle=shuffle,
                                                                  seed=seed,
                                                                  is_print_datagen_info=is_print_datagen_info)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        (out_xdata, out_ydata) = self._get_data_sample(index)
        out_xdata = ImagesUtil.reshape_channels_first(out_xdata, is_input_sample=True)
        out_ydata = ImagesUtil.reshape_channels_first(out_ydata, is_input_sample=True)
        return (torch.from_numpy(out_xdata.copy()).type(OutputDataType),
                torch.from_numpy(out_ydata.copy()).type(OutputDataType))


class WrapperTrainBatchImageDataGenerator1Image(data_torch.DataLoader):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 list_xdata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = False
                 ) -> None:
        self._batchdata_generator = TrainBatchImageDataGenerator1Image(size_image,
                                                                       list_xdata,
                                                                       images_generator,
                                                                       num_channels_in=num_channels_in,
                                                                       batch_size=batch_size,
                                                                       shuffle=shuffle,
                                                                       seed=seed,
                                                                       is_print_datagen_info=is_print_datagen_info)
        super(WrapperTrainBatchImageDataGenerator1Image, self).__init__(self._batchdata_generator,
                                                                        batch_size=batch_size,
                                                                        shuffle=shuffle)

    def get_full_data(self) -> np.ndarray:
        return self._batchdata_generator.get_full_data()


class WrapperTrainBatchImageDataGenerator2Images(data_torch.DataLoader):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 list_xdata: List[np.ndarray],
                 list_ydata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 num_classes_out: int = 1,
                 is_nnet_validconvs: bool = False,
                 size_output_image: Union[Tuple[int, int, int], Tuple[int, int]] = None,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = False
                 ) -> None:
        self._batchdata_generator = TrainBatchImageDataGenerator2Images(size_image,
                                                                        list_xdata,
                                                                        list_ydata,
                                                                        images_generator,
                                                                        num_channels_in=num_channels_in,
                                                                        num_classes_out=num_classes_out,
                                                                        is_nnet_validconvs=is_nnet_validconvs,
                                                                        size_output_image=size_output_image,
                                                                        batch_size=batch_size,
                                                                        shuffle=shuffle,
                                                                        seed=seed,
                                                                        is_print_datagen_info=is_print_datagen_info)
        super(WrapperTrainBatchImageDataGenerator2Images, self).__init__(self._batchdata_generator,
                                                                         batch_size=batch_size,
                                                                         shuffle=shuffle)

    def get_full_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._batchdata_generator.get_full_data()
