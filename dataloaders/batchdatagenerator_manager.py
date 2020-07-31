
from typing import List, Tuple
import numpy as np

from common.constant import TYPE_DNNLIB_USED
from common.exception_manager import catch_error_exception
if TYPE_DNNLIB_USED == 'Pytorch':
    from dataloaders.pytorch.batchdatagenerator import WrapperBatchGenerator_Pytorch as DNNBatchDataGenerator
elif TYPE_DNNLIB_USED == 'Keras':
    from dataloaders.keras.batchdatagenerator import BatchDataGenerator_Keras as DNNBatchDataGenerator
from preprocessing.imagegenerator import ImageGenerator
from preprocessing.imagegenerator_manager import get_images_generator


def get_batchdata_generator_with_generator(size_images: Tuple[int, ...],
                                           list_Xdata: List[np.ndarray],
                                           list_Ydata: List[np.ndarray],
                                           images_generator: ImageGenerator,
                                           is_output_nnet_validconvs: bool = False,
                                           size_output_images: Tuple[int, ...] = None,
                                           batch_size: int = 1,
                                           shuffle: bool = True,
                                           seed: int = None,
                                           is_datagen_in_gpu: bool = True,
                                           is_datagen_halfPrec: bool = False
                                           ) -> DNNBatchDataGenerator:
    if TYPE_DNNLIB_USED == 'Keras' and not is_datagen_in_gpu:
        message = 'Networks implementation in Keras is always in gpu...'
        catch_error_exception(message)

    if TYPE_DNNLIB_USED == 'Keras' and is_datagen_halfPrec:
        message = 'Networks implementation in Keras not available in Half Precision...'
        catch_error_exception(message)

    batch_data_generator = DNNBatchDataGenerator(size_images,
                                                 list_Xdata,
                                                 list_Ydata,
                                                 images_generator,
                                                 is_output_nnet_validconvs=is_output_nnet_validconvs,
                                                 size_output_image=size_output_images,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 seed=seed,
                                                 is_datagen_in_gpu=is_datagen_in_gpu,
                                                 is_datagen_halfPrec=is_datagen_halfPrec)
    return batch_data_generator


def get_batchdata_generator(size_images: Tuple[int, ...],
                            list_Xdata: List[np.ndarray],
                            list_Ydata: List[np.ndarray],
                            use_sliding_window_images: bool,
                            slide_window_prop_overlap: Tuple[int, ...],
                            use_random_window_images: bool,
                            num_random_patches_epoch: int,
                            use_transform_rigid_images: bool,
                            use_transform_elasticdeform_images: bool,
                            is_output_nnet_validconvs: bool = False,
                            size_output_images: Tuple[int, ...] = None,
                            batch_size: int = 1,
                            shuffle: bool = True,
                            seed: int = None,
                            is_datagen_in_gpu: bool = True,
                            is_datagen_halfPrec: bool = False
                            ) -> DNNBatchDataGenerator:
    if len(list_Xdata)==1:
        size_full_image = list_Xdata[0].shape
    else:
        size_full_image = 0

    images_generator = get_images_generator(size_images,
                                            use_sliding_window_images,
                                            slide_window_prop_overlap,
                                            use_random_window_images,
                                            num_random_patches_epoch,
                                            use_transform_rigid_images,
                                            use_transform_elasticdeform_images,
                                            size_volume_image=size_full_image)

    return get_batchdata_generator_with_generator(size_images,
                                                  list_Xdata,
                                                  list_Ydata,
                                                  images_generator,
                                                  is_output_nnet_validconvs=is_output_nnet_validconvs,
                                                  size_output_images=size_output_images,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  seed=seed,
                                                  is_datagen_in_gpu=is_datagen_in_gpu,
                                                  is_datagen_halfPrec=is_datagen_halfPrec)