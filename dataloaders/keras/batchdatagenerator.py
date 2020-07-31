
from typing import List, Tuple
import numpy as np
from tensorflow.keras.utils import Sequence as Sequence_Keras

from dataloaders.batchdatagenerator import BatchDataGenerator
from preprocessing.imagegenerator import ImageGenerator


class BatchDataGenerator_Keras(BatchDataGenerator, Sequence_Keras):

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
                 iswrite_datagen_info: bool = True,
                 is_datagen_in_gpu: bool = True,
                 is_datagen_halfPrec: bool = False
                 ) -> None:
        super(BatchDataGenerator_Keras, self).__init__(size_image,
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
                                                       iswrite_datagen_info=iswrite_datagen_info)
        self._dtype_Xdata = np.float32
        self._dtype_Ydata = np.float32

    def __len__(self) -> int:
        return super(BatchDataGenerator_Keras, self).__len__()

    def _on_epoch_end(self) -> None:
        super(BatchDataGenerator_Keras, self)._on_epoch_end()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return super(BatchDataGenerator_Keras, self).__getitem__(index)

    def _get_reshaped_output_image(self, in_image: np.ndarray) -> np.ndarray:
        return np.expand_dims(in_image, axis=-1)

    def _get_formated_output_Xdata(self, in_image: np.ndarray) -> np.ndarray:
        return self._get_reshaped_output_image(in_image)

    def _get_formated_output_Ydata(self, in_image: np.ndarray) -> np.ndarray:
        if self._is_outputUnet_validconvs:
            out_image = self._get_cropped_output(in_image).astype(dtype=self._dtype_Ydata)
        else:
            out_image = in_image.astype(dtype=self._dtype_Ydata)
        return self._get_formated_output_Xdata(out_image)