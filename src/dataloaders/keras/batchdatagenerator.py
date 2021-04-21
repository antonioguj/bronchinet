
from typing import List, Tuple, Union
import numpy as np

from tensorflow.keras.utils import Sequence as Sequence_keras

from dataloaders.batchdatagenerator import BatchImageDataGenerator1Image, BatchImageDataGenerator2Images
from preprocessing.imagegenerator import ImageGenerator

OutputDataType = np.float32


class TrainBatchImageDataGenerator1Image(BatchImageDataGenerator1Image, Sequence_keras):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 list_xdata: List[np.ndarray],
                 image_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 is_print_datagen_info: bool = False
                 ) -> None:
        super(TrainBatchImageDataGenerator1Image, self).__init__(size_image,
                                                                 list_xdata,
                                                                 image_generator,
                                                                 num_channels_in=num_channels_in,
                                                                 type_image_format='channels_last',
                                                                 batch_size=batch_size,
                                                                 shuffle=shuffle,
                                                                 seed=seed,
                                                                 is_print_datagen_info=is_print_datagen_info)
        Sequence_keras.__init__(self)

    def __len__(self) -> int:
        return super(TrainBatchImageDataGenerator1Image, self).__len__()

    def _on_epoch_end(self) -> None:
        super(TrainBatchImageDataGenerator1Image, self)._on_epoch_end()

    def __getitem__(self, index: int) -> np.ndarray:
        out_xdata = super(TrainBatchImageDataGenerator1Image, self).__getitem__(index)
        return out_xdata.astype(dtype=OutputDataType)


class TrainBatchImageDataGenerator2Images(BatchImageDataGenerator2Images, Sequence_keras):

    def __init__(self,
                 size_image: Union[Tuple[int, int, int], Tuple[int, int]],
                 list_xdata: List[np.ndarray],
                 list_ydata: List[np.ndarray],
                 image_generator: ImageGenerator,
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
                                                                  image_generator,
                                                                  num_channels_in=num_channels_in,
                                                                  num_classes_out=num_classes_out,
                                                                  type_image_format='channels_last',
                                                                  is_nnet_validconvs=is_nnet_validconvs,
                                                                  size_output_image=size_output_image,
                                                                  batch_size=batch_size,
                                                                  shuffle=shuffle,
                                                                  seed=seed,
                                                                  is_print_datagen_info=is_print_datagen_info)
        Sequence_keras.__init__(self)

    def __len__(self) -> int:
        return super(TrainBatchImageDataGenerator2Images, self).__len__()

    def _on_epoch_end(self) -> None:
        super(TrainBatchImageDataGenerator2Images, self)._on_epoch_end()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        (out_xdata, out_ydata) = super(TrainBatchImageDataGenerator2Images, self).__getitem__(index)
        return (out_xdata.astype(dtype=OutputDataType),
                out_ydata.astype(dtype=OutputDataType))
