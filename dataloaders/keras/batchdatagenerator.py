
from typing import List, Tuple
import numpy as np

from tensorflow.keras.utils import Sequence as Sequence_keras

from dataloaders.batchdatagenerator import BatchImageDataGenerator_1Image, BatchImageDataGenerator_2Images
from preprocessing.imagegenerator import ImageGenerator


class TrainBatchImageDataGenerator_1Image(BatchImageDataGenerator_1Image, Sequence_keras):

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
                                                                  type_image_format='channels_last',
                                                                  batch_size=batch_size,
                                                                  shuffle=shuffle,
                                                                  seed=seed,
                                                                  is_print_datagen_info=is_print_datagen_info)
        Sequence_keras.__init__(self)
        self._type_data_generated = np.float32

    def __len__(self) -> int:
        return super(TrainBatchImageDataGenerator_1Image, self).__len__()

    def _on_epoch_end(self) -> None:
        super(TrainBatchImageDataGenerator_1Image, self)._on_epoch_end()

    def __getitem__(self, index: int) -> np.ndarray:
        out_Xdata = super(TrainBatchImageDataGenerator_1Image, self).__getitem__(index)
        return out_Xdata.astype(dtype=self._type_data_generated)


class TrainBatchImageDataGenerator_2Images(BatchImageDataGenerator_2Images, Sequence_keras):

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
                                                                   type_image_format='channels_last',
                                                                   is_output_nnet_validconvs=is_output_nnet_validconvs,
                                                                   size_output_image=size_output_image,
                                                                   batch_size=batch_size,
                                                                   shuffle=shuffle,
                                                                   seed=seed,
                                                                   is_print_datagen_info=is_print_datagen_info)
        Sequence_keras.__init__(self)
        self._type_data_generated = np.float32

    def __len__(self) -> int:
        return super(TrainBatchImageDataGenerator_2Images, self).__len__()

    def _on_epoch_end(self) -> None:
        super(TrainBatchImageDataGenerator_2Images, self)._on_epoch_end()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        (out_Xdata, out_Ydata) = super(TrainBatchImageDataGenerator_2Images, self).__getitem__(index)
        return (out_Xdata.astype(dtype=self._type_data_generated),
                out_Ydata.astype(dtype=self._type_data_generated))