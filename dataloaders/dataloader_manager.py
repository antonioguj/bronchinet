
from typing import List, Tuple, Union
import numpy as np

from common.constant import TYPE_DNNLIB_USED, IS_MODEL_IN_GPU, IS_MODEL_HALF_PRECISION
from common.exceptionmanager import catch_error_exception
if TYPE_DNNLIB_USED == 'Pytorch':
    from dataloaders.pytorch.batchdatagenerator import Wrapper_TrainBatchImageDataGenerator_1Image as TrainBatchImageDataGenerator_1Image, \
                                                       Wrapper_TrainBatchImageDataGenerator_2Images as TrainBatchImageDataGenerator_2Images
elif TYPE_DNNLIB_USED == 'Keras':
    from dataloaders.keras.batchdatagenerator import TrainBatchImageDataGenerator_1Image, \
                                                     TrainBatchImageDataGenerator_2Images
from dataloaders.batchdatagenerator import BatchImageDataGenerator_1Image, BatchImageDataGenerator_2Images
from dataloaders.imagedataloader import ImageDataLoader, ImageDataInBatchesLoader
from preprocessing.preprocessing_manager import get_images_generator


def get_imagedataloader_1image(list_filenames_1: List[str],
                               size_images: Tuple[int, ...],
                               use_sliding_window_images: bool,
                               slide_window_prop_overlap: Tuple[int, ...],
                               use_transform_rigid_images: bool,
                               use_transform_elasticdeform_images: bool,
                               use_random_window_images: bool = False,
                               num_random_patches_epoch: int = 0,
                               batch_size: int = 1,
                               shuffle: bool = True,
                               seed: int = None
                               ) -> Union[BatchImageDataGenerator_1Image, List[np.ndarray]]:
    if use_sliding_window_images or \
        use_random_window_images or \
        use_transform_rigid_images or \
        use_transform_elasticdeform_images:
        print("Generate Data Loader with Batch Generator...")

        list_Xdata = ImageDataLoader.load_1list_files(list_filenames_1)

        size_full_image = list_Xdata[0].shape if len(list_Xdata) == 1 else (0, 0, 0)
        num_channels_in = list_Xdata[0].shape[-1]

        images_generator = get_images_generator(size_images,
                                                use_sliding_window_images,
                                                slide_window_prop_overlap,
                                                use_random_window_images,
                                                num_random_patches_epoch,
                                                use_transform_rigid_images,
                                                use_transform_elasticdeform_images,
                                                size_volume_image=size_full_image)
        return BatchImageDataGenerator_1Image(size_images,
                                              list_Xdata,
                                              images_generator,
                                              num_channels_in=num_channels_in,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              seed=seed)
    else:
        print("Load Data directly from stored Batches...")

        batches_Xdata = ImageDataInBatchesLoader(size_images).load_1list_files(list_filenames_1)
        return batches_Xdata


def get_imagedataloader_2images(list_filenames_1: List[str],
                                list_filenames_2: List[str],
                                size_images: Tuple[int, ...],
                                use_sliding_window_images: bool,
                                slide_window_prop_overlap: Tuple[int, ...],
                                use_transform_rigid_images: bool,
                                use_transform_elasticdeform_images: bool,
                                use_random_window_images: bool = False,
                                num_random_patches_epoch: int = 0,
                                is_output_nnet_validconvs: bool = False,
                                size_output_images: Tuple[int, ...] = None,
                                batch_size: int = 1,
                                shuffle: bool = True,
                                seed: int = None
                                ) -> Union[BatchImageDataGenerator_2Images, Tuple[List[np.ndarray], List[np.ndarray]]]:
    if use_sliding_window_images or \
        use_random_window_images or \
        use_transform_rigid_images or \
        use_transform_elasticdeform_images:
        print("Generate Data Loader with Batch Generator...")

        (list_Xdata, list_Ydata) = ImageDataLoader.load_2list_files(list_filenames_1, list_filenames_2)

        size_full_image = list_Xdata[0].shape if len(list_Xdata) == 1 else (0, 0, 0)
        num_channels_in = list_Xdata[0].shape[-1]
        num_classes_out = list_Ydata[0].shape[-1]

        images_generator = get_images_generator(size_images,
                                                use_sliding_window_images,
                                                slide_window_prop_overlap,
                                                use_random_window_images,
                                                num_random_patches_epoch,
                                                use_transform_rigid_images,
                                                use_transform_elasticdeform_images,
                                                size_volume_image=size_full_image)
        return BatchImageDataGenerator_2Images(size_images,
                                               list_Xdata,
                                               list_Ydata,
                                               images_generator,
                                               num_channels_in=num_channels_in,
                                               num_classes_out=num_classes_out,
                                               is_output_nnet_validconvs=is_output_nnet_validconvs,
                                               size_output_image=size_output_images,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               seed=seed)
    else:
        print("Load Data directly from stored Batches...")

        (batches_Xdata, batches_Ydata) = ImageDataInBatchesLoader(size_images).load_2list_files(list_filenames_1, list_filenames_2)
        return (batches_Xdata, batches_Ydata)


def get_train_imagedataloader_1image(list_filenames_1: List[str],
                                     size_images: Tuple[int, ...],
                                     use_sliding_window_images: bool,
                                     slide_window_prop_overlap: Tuple[int, ...],
                                     use_transform_rigid_images: bool,
                                     use_transform_elasticdeform_images: bool,
                                     use_random_window_images: bool = False,
                                     num_random_patches_epoch: int = 0,
                                     batch_size: int = 1,
                                     shuffle: bool = True,
                                     seed: int = None,
                                     ) -> Union[TrainBatchImageDataGenerator_1Image, List[np.ndarray]]:
    if use_sliding_window_images or \
        use_random_window_images or \
        use_transform_rigid_images or \
        use_transform_elasticdeform_images:
        print("Generate Data Loader with Batch Generator...")

        list_Xdata = ImageDataLoader.load_1list_files(list_filenames_1)

        size_full_image = list_Xdata[0].shape if len(list_Xdata) == 1 else (0, 0, 0)
        num_channels_in = 1

        images_generator = get_images_generator(size_images,
                                                use_sliding_window_images,
                                                slide_window_prop_overlap,
                                                use_random_window_images,
                                                num_random_patches_epoch,
                                                use_transform_rigid_images,
                                                use_transform_elasticdeform_images,
                                                size_volume_image=size_full_image)
        return TrainBatchImageDataGenerator_1Image(size_images,
                                                   list_Xdata,
                                                   images_generator,
                                                   num_channels_in=num_channels_in,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   seed=seed,
                                                   is_datagen_gpu=IS_MODEL_IN_GPU,
                                                   is_datagen_halfPrec=IS_MODEL_HALF_PRECISION)
    else:
        print("Load Data directly from stored Batches...")

        batches_Xdata = ImageDataInBatchesLoader(size_images).load_1list_files(list_filenames_1)
        return batches_Xdata


def get_train_imagedataloader_2images(list_filenames_1: List[str],
                                      list_filenames_2: List[str],
                                      size_images: Tuple[int, ...],
                                      use_sliding_window_images: bool,
                                      slide_window_prop_overlap: Tuple[int, ...],
                                      use_transform_rigid_images: bool,
                                      use_transform_elasticdeform_images: bool,
                                      use_random_window_images: bool = False,
                                      num_random_patches_epoch: int = 0,
                                      is_output_nnet_validconvs: bool = False,
                                      size_output_images: Tuple[int, ...] = None,
                                      batch_size: int = 1,
                                      shuffle: bool = True,
                                      seed: int = None,
                                      ) -> Union[TrainBatchImageDataGenerator_2Images, Tuple[List[np.ndarray], List[np.ndarray]]]:
    if use_sliding_window_images or \
        use_random_window_images or \
        use_transform_rigid_images or \
        use_transform_elasticdeform_images:
        print("Generate Data Loader with Batch Generator...")

        (list_Xdata, list_Ydata) = ImageDataLoader.load_2list_files(list_filenames_1, list_filenames_2)

        size_full_image = list_Xdata[0].shape if len(list_Xdata) == 1 else (0, 0, 0)
        num_channels_in = 1
        num_classes_out = 1

        images_generator = get_images_generator(size_images,
                                                use_sliding_window_images,
                                                slide_window_prop_overlap,
                                                use_random_window_images,
                                                num_random_patches_epoch,
                                                use_transform_rigid_images,
                                                use_transform_elasticdeform_images,
                                                size_volume_image=size_full_image)
        return TrainBatchImageDataGenerator_2Images(size_images,
                                                    list_Xdata,
                                                    list_Ydata,
                                                    images_generator,
                                                    num_channels_in=num_channels_in,
                                                    num_classes_out=num_classes_out,
                                                    is_output_nnet_validconvs=is_output_nnet_validconvs,
                                                    size_output_image=size_output_images,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    seed=seed,
                                                    is_datagen_gpu=IS_MODEL_IN_GPU,
                                                    is_datagen_halfPrec=IS_MODEL_HALF_PRECISION)
    else:
        print("Load Data directly from stored Batches...")

        (batches_Xdata, batches_Ydata) = ImageDataInBatchesLoader(size_images).load_2list_files(list_filenames_1, list_filenames_2)
        return (batches_Xdata, batches_Ydata)