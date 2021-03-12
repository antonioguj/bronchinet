
from typing import List, Tuple
import numpy as np

from torch.utils import data as data_torch
import torch

from common.functionutil import ImagesUtil
from dataloaders.batchdatagenerator import BatchImageDataGenerator_1Image, BatchImageDataGenerator_2Images, BatchImageDataGenerator_ManyImagesPerLabel
from preprocessing.imagegenerator import ImageGenerator


class TrainBatchImageDataGenerator_1Image(BatchImageDataGenerator_1Image):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 list_Xdata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = True,
                 is_datagen_gpu: bool = True,
                 is_datagen_halfPrec: bool = False
                 ) -> None:
        super(TrainBatchImageDataGenerator_1Image, self).__init__(size_image,
                                                                  list_Xdata,
                                                                  images_generator,
                                                                  num_channels_in=num_channels_in,
                                                                  type_image_format='channels_first',
                                                                  batch_size=batch_size,
                                                                  shuffle=shuffle,
                                                                  seed=seed,
                                                                  is_print_datagen_info=is_print_datagen_info)
        if is_datagen_gpu:
            if is_datagen_halfPrec:
                self._type_data_generated = torch.cuda.HalfTensor
            else:
                self._type_data_generated = torch.cuda.FloatTensor
        else:
            if is_datagen_halfPrec:
                self._type_data_generated = torch.HalfTensor
            else:
                self._type_data_generated = torch.FloatTensor

    def __getitem__(self, index: int) -> np.ndarray:
        out_Xdata = self._get_data_sample(index)
        out_Xdata = ImagesUtil.reshape_channels_first(out_Xdata, is_input_sample=True)
        return torch.from_numpy(out_Xdata.copy()).type(self._type_data_generated)
        #return torch.from_numpy(out_Xdata).type(self._type_data_generated)


class TrainBatchImageDataGenerator_2Images(BatchImageDataGenerator_2Images):

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
                 is_print_datagen_info: bool = True,
                 is_datagen_gpu: bool = True,
                 is_datagen_halfPrec: bool = False
                 ) -> None:
        super(TrainBatchImageDataGenerator_2Images, self).__init__(size_image,
                                                                   list_Xdata,
                                                                   list_Ydata,
                                                                   images_generator,
                                                                   num_channels_in=num_channels_in,
                                                                   num_classes_out=num_classes_out,
                                                                   type_image_format='channels_first',
                                                                   is_output_nnet_validconvs=is_output_nnet_validconvs,
                                                                   size_output_image=size_output_image,
                                                                   batch_size=batch_size,
                                                                   shuffle=shuffle,
                                                                   seed=seed,
                                                                   is_print_datagen_info=is_print_datagen_info)
        if is_datagen_gpu:
            if is_datagen_halfPrec:
                self._type_data_generated = torch.cuda.HalfTensor
            else:
                self._type_data_generated = torch.cuda.FloatTensor
        else:
            if is_datagen_halfPrec:
                self._type_data_generated = torch.HalfTensor
            else:
                self._type_data_generated = torch.FloatTensor

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        (out_Xdata, out_Ydata) = self._get_data_sample(index)
        out_Xdata = ImagesUtil.reshape_channels_first(out_Xdata, is_input_sample=True)
        out_Ydata = ImagesUtil.reshape_channels_first(out_Ydata, is_input_sample=True)
        return (torch.from_numpy(out_Xdata.copy()).type(self._type_data_generated),
                torch.from_numpy(out_Ydata.copy()).type(self._type_data_generated))
        #return (torch.from_numpy(out_Xdata).type(self._type_data_generated),
        #        torch.from_numpy(out_Ydata).type(self._type_data_generated))


class TrainBatchImageDataGenerator_ManyImagesPerLabel(BatchImageDataGenerator_ManyImagesPerLabel):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 num_images_per_label: int,
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
                 is_print_datagen_info: bool = True,
                 is_datagen_gpu: bool = True,
                 is_datagen_halfPrec: bool = False
                 ) -> None:
        super(TrainBatchImageDataGenerator_ManyImagesPerLabel, self).__init__(size_image,
                                                                              num_images_per_label,
                                                                              list_Xdata,
                                                                              list_Ydata,
                                                                              images_generator,
                                                                              num_channels_in=num_channels_in,
                                                                              num_classes_out=num_classes_out,
                                                                              type_image_format='channels_first',
                                                                              is_output_nnet_validconvs=is_output_nnet_validconvs,
                                                                              size_output_image=size_output_image,
                                                                              batch_size=batch_size,
                                                                              shuffle=shuffle,
                                                                              seed=seed,
                                                                              is_print_datagen_info=is_print_datagen_info)
        if is_datagen_gpu:
            if is_datagen_halfPrec:
                self._type_data_generated = torch.cuda.HalfTensor
            else:
                self._type_data_generated = torch.cuda.FloatTensor
        else:
            if is_datagen_halfPrec:
                self._type_data_generated = torch.HalfTensor
            else:
                self._type_data_generated = torch.FloatTensor

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        (out_Xdata, out_Ydata) = self._get_data_sample(index)
        out_Xdata = ImagesUtil.reshape_channels_first(out_Xdata, is_input_sample=True)
        out_Ydata = ImagesUtil.reshape_channels_first(out_Ydata, is_input_sample=True)
        return (torch.from_numpy(out_Xdata.copy()).type(self._type_data_generated),
                torch.from_numpy(out_Ydata.copy()).type(self._type_data_generated))
        #return (torch.from_numpy(out_Xdata).type(self._type_data_generated),
        #        torch.from_numpy(out_Ydata).type(self._type_data_generated))


class Wrapper_TrainBatchImageDataGenerator_1Image(data_torch.DataLoader):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 list_Xdata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = True,
                 is_datagen_gpu: bool = True,
                 is_datagen_halfPrec: bool = False
                 ) -> None:
        self._batchdata_generator = TrainBatchImageDataGenerator_1Image(size_image,
                                                                        list_Xdata,
                                                                        images_generator,
                                                                        num_channels_in=num_channels_in,
                                                                        batch_size=batch_size,
                                                                        shuffle=shuffle,
                                                                        seed=seed,
                                                                        is_print_datagen_info=is_print_datagen_info,
                                                                        is_datagen_gpu=is_datagen_gpu,
                                                                        is_datagen_halfPrec=is_datagen_halfPrec)
        super(Wrapper_TrainBatchImageDataGenerator_1Image, self).__init__(self._batchdata_generator,
                                                                          batch_size=batch_size,
                                                                          shuffle=shuffle)
    def get_full_data(self) -> np.ndarray:
        return self._batchdata_generator.get_full_data()


class Wrapper_TrainBatchImageDataGenerator_2Images(data_torch.DataLoader):

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
                 is_print_datagen_info: bool = True,
                 is_datagen_gpu: bool = True,
                 is_datagen_halfPrec: bool = False
                 ) -> None:
        self._batchdata_generator = TrainBatchImageDataGenerator_2Images(size_image,
                                                                         list_Xdata,
                                                                         list_Ydata,
                                                                         images_generator,
                                                                         num_channels_in=num_channels_in,
                                                                         num_classes_out=num_classes_out,
                                                                         is_output_nnet_validconvs=is_output_nnet_validconvs,
                                                                         size_output_image=size_output_image,
                                                                         batch_size=batch_size,
                                                                         shuffle=shuffle,
                                                                         seed=seed,
                                                                         is_print_datagen_info=is_print_datagen_info,
                                                                         is_datagen_gpu=is_datagen_gpu,
                                                                         is_datagen_halfPrec=is_datagen_halfPrec)
        super(Wrapper_TrainBatchImageDataGenerator_2Images, self).__init__(self._batchdata_generator,
                                                                           batch_size=batch_size,
                                                                           shuffle=shuffle)
    def get_full_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._batchdata_generator.get_full_data()


class Wrapper_TrainBatchImageDataGenerator_ManyImagesPerLabel(data_torch.DataLoader):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 num_images_per_label: int,
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
                 is_print_datagen_info: bool = True,
                 is_datagen_gpu: bool = True,
                 is_datagen_halfPrec: bool = False
                 ) -> None:
        self._batchdata_generator = TrainBatchImageDataGenerator_ManyImagesPerLabel(size_image,
                                                                                    num_images_per_label,
                                                                                    list_Xdata,
                                                                                    list_Ydata,
                                                                                    images_generator,
                                                                                    num_channels_in=num_channels_in,
                                                                                    num_classes_out=num_classes_out,
                                                                                    is_output_nnet_validconvs=is_output_nnet_validconvs,
                                                                                    size_output_image=size_output_image,
                                                                                    batch_size=batch_size,
                                                                                    shuffle=shuffle,
                                                                                    seed=seed,
                                                                                    is_print_datagen_info=is_print_datagen_info,
                                                                                    is_datagen_gpu=is_datagen_gpu,
                                                                                    is_datagen_halfPrec=is_datagen_halfPrec)
        super(Wrapper_TrainBatchImageDataGenerator_ManyImagesPerLabel, self).__init__(self._batchdata_generator,
                                                                                      batch_size=batch_size,
                                                                                      shuffle=shuffle)
    def get_full_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._batchdata_generator.get_full_data()
