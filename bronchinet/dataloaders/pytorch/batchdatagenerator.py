
from typing import List, Tuple
import numpy as np

from torch.utils import data as data_torch
import torch

from common.functionutil import ImagesUtil
from dataloaders.batchdatagenerator import BatchImageDataGenerator1Image, BatchImageDataGenerator2Images, \
                                           BatchImageDataGeneratorManyImagesPerLabel
from preprocessing.imagegenerator import ImageGenerator


class TrainBatchImageDataGenerator1Image(BatchImageDataGenerator1Image):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 list_xdata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = True,
                 is_datagen_gpu: bool = True,
                 is_datagen_halfprec: bool = False
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
        if is_datagen_gpu:
            if is_datagen_halfprec:
                self._type_data_generated = torch.cuda.HalfTensor
            else:
                self._type_data_generated = torch.cuda.FloatTensor
        else:
            if is_datagen_halfprec:
                self._type_data_generated = torch.HalfTensor
            else:
                self._type_data_generated = torch.FloatTensor

    def __getitem__(self, index: int) -> np.ndarray:
        out_xdata = self._get_data_sample(index)
        out_xdata = ImagesUtil.reshape_channels_first(out_xdata, is_input_sample=True)
        return torch.from_numpy(out_xdata.copy()).type(self._type_data_generated)
        # return torch.from_numpy(out_xdata).type(self._type_data_generated)


class TrainBatchImageDataGenerator2Images(BatchImageDataGenerator2Images):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 list_xdata: List[np.ndarray],
                 list_ydata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 num_classes_out: int = 1,
                 is_nnet_validconvs: bool = False,
                 size_output_image: Tuple[int, ...] = None,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = True,
                 is_datagen_gpu: bool = True,
                 is_datagen_halfprec: bool = False
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
        if is_datagen_gpu:
            if is_datagen_halfprec:
                self._type_data_generated = torch.cuda.HalfTensor
            else:
                self._type_data_generated = torch.cuda.FloatTensor
        else:
            if is_datagen_halfprec:
                self._type_data_generated = torch.HalfTensor
            else:
                self._type_data_generated = torch.FloatTensor

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        (out_xdata, out_ydata) = self._get_data_sample(index)
        out_xdata = ImagesUtil.reshape_channels_first(out_xdata, is_input_sample=True)
        out_ydata = ImagesUtil.reshape_channels_first(out_ydata, is_input_sample=True)
        return (torch.from_numpy(out_xdata.copy()).type(self._type_data_generated),
                torch.from_numpy(out_ydata.copy()).type(self._type_data_generated))
        # return (torch.from_numpy(out_xdata).type(self._type_data_generated),
        #         torch.from_numpy(out_ydata).type(self._type_data_generated))


class TrainBatchImageDataGeneratorManyImagesPerLabel(BatchImageDataGeneratorManyImagesPerLabel):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 num_images_per_label: int,
                 list_xdata: List[np.ndarray],
                 list_ydata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 num_classes_out: int = 1,
                 is_nnet_validconvs: bool = False,
                 size_output_image: Tuple[int, ...] = None,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = True,
                 is_datagen_gpu: bool = True,
                 is_datagen_halfprec: bool = False
                 ) -> None:
        super(TrainBatchImageDataGeneratorManyImagesPerLabel, self).__init__(
            size_image,
            num_images_per_label,
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
        if is_datagen_gpu:
            if is_datagen_halfprec:
                self._type_data_generated = torch.cuda.HalfTensor
            else:
                self._type_data_generated = torch.cuda.FloatTensor
        else:
            if is_datagen_halfprec:
                self._type_data_generated = torch.HalfTensor
            else:
                self._type_data_generated = torch.FloatTensor

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        (out_xdata, out_ydata) = self._get_data_sample(index)
        out_xdata = ImagesUtil.reshape_channels_first(out_xdata, is_input_sample=True)
        out_ydata = ImagesUtil.reshape_channels_first(out_ydata, is_input_sample=True)
        return (torch.from_numpy(out_xdata.copy()).type(self._type_data_generated),
                torch.from_numpy(out_ydata.copy()).type(self._type_data_generated))
        # return (torch.from_numpy(out_xdata).type(self._type_data_generated),
        #         torch.from_numpy(out_ydata).type(self._type_data_generated))


class WrapperTrainBatchImageDataGenerator1Image(data_torch.DataLoader):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 list_xdata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = True,
                 is_datagen_gpu: bool = True,
                 is_datagen_halfprec: bool = False
                 ) -> None:
        self._batchdata_generator = TrainBatchImageDataGenerator1Image(size_image,
                                                                       list_xdata,
                                                                       images_generator,
                                                                       num_channels_in=num_channels_in,
                                                                       batch_size=batch_size,
                                                                       shuffle=shuffle,
                                                                       seed=seed,
                                                                       is_print_datagen_info=is_print_datagen_info,
                                                                       is_datagen_gpu=is_datagen_gpu,
                                                                       is_datagen_halfprec=is_datagen_halfprec)
        super(WrapperTrainBatchImageDataGenerator1Image, self).__init__(self._batchdata_generator,
                                                                        batch_size=batch_size,
                                                                        shuffle=shuffle)

    def get_full_data(self) -> np.ndarray:
        return self._batchdata_generator.get_full_data()


class WrapperTrainBatchImageDataGenerator2Images(data_torch.DataLoader):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 list_xdata: List[np.ndarray],
                 list_ydata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 num_classes_out: int = 1,
                 is_nnet_validconvs: bool = False,
                 size_output_image: Tuple[int, ...] = None,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = True,
                 is_datagen_gpu: bool = True,
                 is_datagen_halfprec: bool = False
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
                                                                        is_print_datagen_info=is_print_datagen_info,
                                                                        is_datagen_gpu=is_datagen_gpu,
                                                                        is_datagen_halfprec=is_datagen_halfprec)
        super(WrapperTrainBatchImageDataGenerator2Images, self).__init__(self._batchdata_generator,
                                                                         batch_size=batch_size,
                                                                         shuffle=shuffle)

    def get_full_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._batchdata_generator.get_full_data()


class WrapperTrainBatchImageDataGeneratorManyImagesPerLabel(data_torch.DataLoader):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 num_images_per_label: int,
                 list_xdata: List[np.ndarray],
                 list_ydata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 num_classes_out: int = 1,
                 is_nnet_validconvs: bool = False,
                 size_output_image: Tuple[int, ...] = None,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = True,
                 is_datagen_gpu: bool = True,
                 is_datagen_halfprec: bool = False
                 ) -> None:
        self._batchdata_generator = TrainBatchImageDataGeneratorManyImagesPerLabel(
            size_image,
            num_images_per_label,
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
            is_print_datagen_info=is_print_datagen_info,
            is_datagen_gpu=is_datagen_gpu,
            is_datagen_halfprec=is_datagen_halfprec)
        super(WrapperTrainBatchImageDataGeneratorManyImagesPerLabel, self).__init__(self._batchdata_generator,
                                                                                    batch_size=batch_size,
                                                                                    shuffle=shuffle)

    def get_full_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._batchdata_generator.get_full_data()
