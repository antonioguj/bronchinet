
from typing import List, Tuple, Union
import numpy as np

from common.constant import TYPE_DNNLIB_USED, IS_MODEL_IN_GPU, IS_MODEL_HALF_PRECISION
from common.exceptionmanager import catch_error_exception
if TYPE_DNNLIB_USED == 'Pytorch':
    from dataloaders.pytorch.batchdatagenerator import Wrapper_TrainBatchImageDataGenerator_1Image as TrainBatchImageDataGenerator_1Image, \
                                                       Wrapper_TrainBatchImageDataGenerator_2Images as TrainBatchImageDataGenerator_2Images, \
                                                       Wrapper_TrainBatchImageDataGenerator_ManyImagesPerLabel as TrainBatchImageDataGenerator_ManyImagesPerLabel
elif TYPE_DNNLIB_USED == 'Keras':
    from dataloaders.keras.batchdatagenerator import TrainBatchImageDataGenerator_1Image, \
                                                     TrainBatchImageDataGenerator_2Images, \
                                                     TrainBatchImageDataGenerator_ManyImagesPerLabel
from dataloaders.batchdatagenerator import BatchImageDataGenerator_1Image, BatchImageDataGenerator_2Images, BatchImageDataGenerator_ManyImagesPerLabel
from dataloaders.imagedataloader import ImageDataLoader, ImageDataInBatchesLoader
from preprocessing.preprocessing_manager import get_images_generator


def get_imagedataloader_1image(list_filenames_1: List[str],
                               size_in_images: Tuple[int, ...],
                               use_sliding_window_images: bool,
                               prop_overlap_slide_window: Tuple[int, ...],
                               use_transform_rigid_images: bool,
                               use_transform_elasticdeform_images: bool,
                               use_random_window_images: bool = False,
                               num_random_patches_epoch: int = 0,
                               batch_size: int = 1,
                               is_shuffle: bool = True,
                               manual_seed: int = None,
                               is_load_images_from_batches: bool = False
                               ) -> Union[BatchImageDataGenerator_1Image, List[np.ndarray]]:
    print("Generate Data Loader with Batch Generator...")

    if is_load_images_from_batches:
        list_Xdata = ImageDataInBatchesLoader(size_in_images).load_1list_files(list_filenames_1, is_shuffle=is_shuffle)
    else:
        list_Xdata = ImageDataLoader.load_1list_files(list_filenames_1)

    if not (use_sliding_window_images or use_random_window_images) and (len(list_Xdata) == 1):
        size_in_images = list_Xdata[0].shape

    size_full_image = list_Xdata[0].shape if len(list_Xdata) == 1 else (0, 0, 0)
    num_channels_in = 1

    images_generator = get_images_generator(size_in_images,
                                            use_sliding_window_images=use_sliding_window_images,
                                            prop_overlap_slide_window=prop_overlap_slide_window,
                                            use_random_window_images=use_random_window_images,
                                            num_random_patches_epoch=num_random_patches_epoch,
                                            use_transform_rigid_images=use_transform_rigid_images,
                                            use_transform_elasticdeform_images=use_transform_elasticdeform_images,
                                            size_volume_image=size_full_image)
    return BatchImageDataGenerator_1Image(size_in_images,
                                          list_Xdata,
                                          images_generator,
                                          num_channels_in=num_channels_in,
                                          batch_size=batch_size,
                                          shuffle=is_shuffle,
                                          seed=manual_seed)


def get_imagedataloader_2images(list_filenames_1: List[str],
                                list_filenames_2: List[str],
                                size_in_images: Tuple[int, ...],
                                use_sliding_window_images: bool,
                                prop_overlap_slide_window: Tuple[int, ...],
                                use_transform_rigid_images: bool,
                                use_transform_elasticdeform_images: bool,
                                use_random_window_images: bool = False,
                                num_random_patches_epoch: int = 0,
                                is_output_nnet_validconvs: bool = False,
                                size_output_images: Tuple[int, ...] = None,
                                batch_size: int = 1,
                                is_shuffle: bool = True,
                                manual_seed: int = None,
                                is_load_images_from_batches: bool = False
                                ) -> Union[BatchImageDataGenerator_2Images, Tuple[List[np.ndarray], List[np.ndarray]]]:
    print("Generate Data Loader with Batch Generator...")

    if is_load_images_from_batches:
        (list_Xdata, list_Ydata) = ImageDataInBatchesLoader(size_in_images).load_2list_files(list_filenames_1, list_filenames_2,
                                                                                             is_shuffle=is_shuffle)
    else:
        (list_Xdata, list_Ydata) = ImageDataLoader.load_2list_files(list_filenames_1, list_filenames_2)

    if not (use_sliding_window_images or use_random_window_images) and (len(list_Xdata) == 1):
        size_in_images = list_Xdata[0].shape

    size_full_image = list_Xdata[0].shape if len(list_Xdata) == 1 else (0, 0, 0)
    num_channels_in = 1
    num_classes_out = 1

    images_generator = get_images_generator(size_in_images,
                                            use_sliding_window_images=use_sliding_window_images,
                                            prop_overlap_slide_window=prop_overlap_slide_window,
                                            use_random_window_images=use_random_window_images,
                                            num_random_patches_epoch=num_random_patches_epoch,
                                            use_transform_rigid_images=use_transform_rigid_images,
                                            use_transform_elasticdeform_images=use_transform_elasticdeform_images,
                                            size_volume_image=size_full_image)
    return BatchImageDataGenerator_2Images(size_in_images,
                                           list_Xdata,
                                           list_Ydata,
                                           images_generator,
                                           num_channels_in=num_channels_in,
                                           num_classes_out=num_classes_out,
                                           is_output_nnet_validconvs=is_output_nnet_validconvs,
                                           size_output_image=size_output_images,
                                           batch_size=batch_size,
                                           shuffle=is_shuffle,
                                           seed=manual_seed)


def get_train_imagedataloader_1image(list_filenames_1: List[str],
                                     size_in_images: Tuple[int, ...],
                                     use_sliding_window_images: bool,
                                     prop_overlap_slide_window: Tuple[int, ...],
                                     use_transform_rigid_images: bool,
                                     use_transform_elasticdeform_images: bool,
                                     use_random_window_images: bool = False,
                                     num_random_patches_epoch: int = 0,
                                     batch_size: int = 1,
                                     is_shuffle: bool = True,
                                     manual_seed: int = None,
                                     is_load_images_from_batches: bool = False
                                     ) -> Union[TrainBatchImageDataGenerator_1Image, List[np.ndarray]]:
    print("Generate Data Loader with Batch Generator...")

    if is_load_images_from_batches:
        list_Xdata = ImageDataInBatchesLoader(size_in_images).load_1list_files(list_filenames_1, is_shuffle=is_shuffle)
    else:
        list_Xdata = ImageDataLoader.load_1list_files(list_filenames_1)

    size_full_image = list_Xdata[0].shape if len(list_Xdata) == 1 else (0, 0, 0)
    num_channels_in = 1

    images_generator = get_images_generator(size_in_images,
                                            use_sliding_window_images=use_sliding_window_images,
                                            prop_overlap_slide_window=prop_overlap_slide_window,
                                            use_random_window_images=use_random_window_images,
                                            num_random_patches_epoch=num_random_patches_epoch,
                                            use_transform_rigid_images=use_transform_rigid_images,
                                            use_transform_elasticdeform_images=use_transform_elasticdeform_images,
                                            size_volume_image=size_full_image)
    return TrainBatchImageDataGenerator_1Image(size_in_images,
                                               list_Xdata,
                                               images_generator,
                                               num_channels_in=num_channels_in,
                                               batch_size=batch_size,
                                               shuffle=is_shuffle,
                                               seed=manual_seed,
                                               is_datagen_gpu=IS_MODEL_IN_GPU,
                                               is_datagen_halfPrec=IS_MODEL_HALF_PRECISION)


def get_train_imagedataloader_2images(list_filenames_1: List[str],
                                      list_filenames_2: List[str],
                                      size_in_images: Tuple[int, ...],
                                      use_sliding_window_images: bool,
                                      prop_overlap_slide_window: Tuple[int, ...],
                                      use_transform_rigid_images: bool,
                                      use_transform_elasticdeform_images: bool,
                                      use_random_window_images: bool = False,
                                      num_random_patches_epoch: int = 0,
                                      is_output_nnet_validconvs: bool = False,
                                      size_output_images: Tuple[int, ...] = None,
                                      batch_size: int = 1,
                                      is_shuffle: bool = True,
                                      manual_seed: int = None,
                                      is_load_many_images_per_label: bool = False,
                                      num_images_per_label: int = 0,
                                      is_load_images_from_batches: bool = False
                                      ) -> Union[TrainBatchImageDataGenerator_2Images, Tuple[List[np.ndarray], List[np.ndarray]]]:
    print("Generate Data Loader with Batch Generator...")

    if is_load_images_from_batches:
        if is_load_many_images_per_label:
            list_Xdata = ImageDataInBatchesLoader(size_in_images).load_1list_files(list_filenames_1, is_shuffle=is_shuffle)
            list_Ydata = ImageDataInBatchesLoader(size_in_images).load_1list_files(list_filenames_2, is_shuffle=is_shuffle)
        else:
            (list_Xdata, list_Ydata) = ImageDataInBatchesLoader(size_in_images).load_2list_files(list_filenames_1, list_filenames_2,
                                                                                                 is_shuffle=is_shuffle)
    else:
        if is_load_many_images_per_label:
            list_Xdata = ImageDataLoader.load_1list_files(list_filenames_1)
            list_Ydata = ImageDataLoader.load_1list_files(list_filenames_2)
        else:
            (list_Xdata, list_Ydata) = ImageDataLoader.load_2list_files(list_filenames_1, list_filenames_2)

    size_full_image = list_Xdata[0].shape if len(list_Xdata) == 1 else (0, 0, 0)
    num_channels_in = 1
    num_classes_out = 1

    images_generator = get_images_generator(size_in_images,
                                            use_sliding_window_images=use_sliding_window_images,
                                            prop_overlap_slide_window=prop_overlap_slide_window,
                                            use_random_window_images=use_random_window_images,
                                            num_random_patches_epoch=num_random_patches_epoch,
                                            use_transform_rigid_images=use_transform_rigid_images,
                                            use_transform_elasticdeform_images=use_transform_elasticdeform_images,
                                            size_volume_image=size_full_image)
    if is_load_many_images_per_label:
        return TrainBatchImageDataGenerator_ManyImagesPerLabel(size_in_images,
                                                               num_images_per_label,
                                                               list_Xdata,
                                                               list_Ydata,
                                                               images_generator,
                                                               num_channels_in=num_channels_in,
                                                               num_classes_out=num_classes_out,
                                                               is_output_nnet_validconvs=is_output_nnet_validconvs,
                                                               size_output_image=size_output_images,
                                                               batch_size=batch_size,
                                                               shuffle=is_shuffle,
                                                               seed=manual_seed,
                                                               is_datagen_gpu=IS_MODEL_IN_GPU,
                                                               is_datagen_halfPrec=IS_MODEL_HALF_PRECISION)
    else:
        return TrainBatchImageDataGenerator_2Images(size_in_images,
                                                    list_Xdata,
                                                    list_Ydata,
                                                    images_generator,
                                                    num_channels_in=num_channels_in,
                                                    num_classes_out=num_classes_out,
                                                    is_output_nnet_validconvs=is_output_nnet_validconvs,
                                                    size_output_image=size_output_images,
                                                    batch_size=batch_size,
                                                    shuffle=is_shuffle,
                                                    seed=manual_seed,
                                                    is_datagen_gpu=IS_MODEL_IN_GPU,
                                                    is_datagen_halfPrec=IS_MODEL_HALF_PRECISION)
