
from typing import List, Tuple
import numpy as np
from torch.utils import data
import torch

from dataloaders.batchdatagenerator import BatchDataGenerator
from preprocessing.imagegenerator import ImageGenerator


class BatchDataGenerator_Pytorch(BatchDataGenerator):

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
        super(BatchDataGenerator_Pytorch, self).__init__(size_image,
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
                                                         iswrite_datagen_info= iswrite_datagen_info)
        if is_datagen_in_gpu:
            if is_datagen_halfPrec:
                self._type_data_generated_torch = torch.cuda.HalfTensor
            else:
                self._type_data_generated_torch = torch.cuda.FloatTensor
        else:
            if is_datagen_halfPrec:
                self._type_data_generated_torch = torch.HalfTensor
            else:
                self._type_data_generated_torch = torch.FloatTensor

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return super(BatchDataGenerator_Pytorch, self)._get_item(index)

    def _get_image_torchtensor(self, in_image: np.ndarray) -> np.ndarray:
        return torch.from_numpy(in_image.copy()).type(self._type_data_generated_torch)

    def _get_reshaped_output_image(self, in_image: np.ndarray) -> np.ndarray:
        return np.expand_dims(in_image, axis=0)

    def _get_formated_output_Xdata(self, in_image: np.ndarray) -> np.ndarray:
        out_image = self._get_reshaped_output_image(in_image)
        return self._get_image_torchtensor(out_image)

    def _get_formated_output_Ydata(self, in_image: np.ndarray) -> np.ndarray:
        if self._is_outputUnet_validconvs:
            out_image = self._get_cropped_output(in_image)
        else:
            out_image = in_image
        return self._get_formated_output_Xdata(out_image)


class WrapperBatchGenerator_Pytorch(data.DataLoader):

    def __init__(self,
                 size_image: Tuple[int, ...],
                 list_Xdata: List[np.ndarray],
                 list_Ydata: List[np.ndarray],
                 images_generator: ImageGenerator,
                 num_channels_in: int = 1,
                 num_classes_out: int = 1,
                 is_outputUnet_validconvs: bool = False,
                 size_output_image: Tuple[int, ...] = None,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: int = None,
                 iswrite_datagen_info: bool = True,
                 is_datagen_in_gpu: bool = True,
                 is_datagen_halfPrec: bool = False
                 ) -> None:
        self._batchdata_generator = BatchDataGenerator_Pytorch(size_image,
                                                               list_Xdata,
                                                               list_Ydata,
                                                               images_generator,
                                                               num_channels_in=num_channels_in,
                                                               num_classes_out=num_classes_out,
                                                               is_output_nnet_validconvs=is_outputUnet_validconvs,
                                                               size_output_image=size_output_image,
                                                               batch_size=batch_size,
                                                               shuffle=shuffle,
                                                               seed=seed,
                                                               iswrite_datagen_info=iswrite_datagen_info,
                                                               is_datagen_in_gpu=is_datagen_in_gpu,
                                                               is_datagen_halfPrec=is_datagen_halfPrec)
        super(WrapperBatchGenerator_Pytorch, self).__init__(self._batchdata_generator,
                                                            batch_size= batch_size,
                                                            shuffle= shuffle)

    def get_full_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._batchdata_generator.get_full_data()