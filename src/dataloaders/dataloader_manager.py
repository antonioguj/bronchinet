
from typing import List, Tuple, Union
import numpy as np

from common.constant import TYPE_DNNLIB_USED, IS_MODEL_IN_GPU, IS_MODEL_HALF_PRECISION
if TYPE_DNNLIB_USED == 'Pytorch':
    from dataloaders.pytorch.batchdatagenerator import \
        WrapperTrainBatchImageDataGenerator1Image as TrainBatchImageDataGenerator1Image, \
        WrapperTrainBatchImageDataGenerator2Images as TrainBatchImageDataGenerator2Images, \
        WrapperTrainBatchImageDataGeneratorManyImagesPerLabel as TrainBatchImageDataGeneratorManyImagesPerLabel
elif TYPE_DNNLIB_USED == 'Keras':
    from dataloaders.keras.batchdatagenerator import TrainBatchImageDataGenerator1Image, \
        TrainBatchImageDataGenerator2Images, TrainBatchImageDataGeneratorManyImagesPerLabel
from dataloaders.batchdatagenerator import BatchImageDataGenerator1Image, BatchImageDataGenerator2Images
from dataloaders.imagedataloader import ImageDataLoader, ImageDataBatchesLoader
from preprocessing.preprocessing_manager import get_images_generator


def get_imagedataloader_1image(list_filenames_1: List[str],
                               size_in_images: Tuple[int, ...],
                               use_sliding_window_images: bool,
                               prop_overlap_slide_window: Tuple[int, ...],
                               use_transform_rigid_images: bool,
                               use_transform_elastic_images: bool,
                               use_random_window_images: bool = False,
                               num_random_patches_epoch: int = 0,
                               batch_size: int = 1,
                               is_shuffle: bool = True,
                               manual_seed: int = None,
                               is_load_images_from_batches: bool = False
                               ) -> Union[BatchImageDataGenerator1Image, List[np.ndarray]]:
    print("Generate Data Loader with Batch Generator...")

    if is_load_images_from_batches:
        list_xdata = ImageDataBatchesLoader(size_in_images).load_1list_files(list_filenames_1, is_shuffle=is_shuffle)
    else:
        list_xdata = ImageDataLoader.load_1list_files(list_filenames_1)

    if not (use_sliding_window_images or use_random_window_images) and (len(list_xdata) == 1):
        size_in_images = list_xdata[0].shape

    size_full_image = list_xdata[0].shape if len(list_xdata) == 1 else (0, 0, 0)
    num_channels_in = 1

    images_generator = get_images_generator(size_in_images,
                                            use_sliding_window_images=use_sliding_window_images,
                                            prop_overlap_slide_window=prop_overlap_slide_window,
                                            use_random_window_images=use_random_window_images,
                                            num_random_patches_epoch=num_random_patches_epoch,
                                            use_transform_rigid_images=use_transform_rigid_images,
                                            use_transform_elastic_images=use_transform_elastic_images,
                                            size_volume_image=size_full_image)
    return BatchImageDataGenerator1Image(size_in_images,
                                         list_xdata,
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
                                use_transform_elastic_images: bool,
                                use_random_window_images: bool = False,
                                num_random_patches_epoch: int = 0,
                                is_nnet_validconvs: bool = False,
                                size_output_images: Tuple[int, ...] = None,
                                batch_size: int = 1,
                                is_shuffle: bool = True,
                                manual_seed: int = None,
                                is_load_images_from_batches: bool = False
                                ) -> Union[BatchImageDataGenerator2Images, Tuple[List[np.ndarray], List[np.ndarray]]]:
    print("Generate Data Loader with Batch Generator...")

    if is_load_images_from_batches:
        (list_xdata, list_ydata) = ImageDataBatchesLoader(size_in_images).load_2list_files(list_filenames_1,
                                                                                           list_filenames_2,
                                                                                           is_shuffle=is_shuffle)
    else:
        (list_xdata, list_ydata) = ImageDataLoader.load_2list_files(list_filenames_1, list_filenames_2)

    if not (use_sliding_window_images or use_random_window_images) and (len(list_xdata) == 1):
        size_in_images = list_xdata[0].shape

    size_full_image = list_xdata[0].shape if len(list_xdata) == 1 else (0, 0, 0)
    num_channels_in = 1
    num_classes_out = 1

    images_generator = get_images_generator(size_in_images,
                                            use_sliding_window_images=use_sliding_window_images,
                                            prop_overlap_slide_window=prop_overlap_slide_window,
                                            use_random_window_images=use_random_window_images,
                                            num_random_patches_epoch=num_random_patches_epoch,
                                            use_transform_rigid_images=use_transform_rigid_images,
                                            use_transform_elastic_images=use_transform_elastic_images,
                                            size_volume_image=size_full_image)
    return BatchImageDataGenerator2Images(size_in_images,
                                          list_xdata,
                                          list_ydata,
                                          images_generator,
                                          num_channels_in=num_channels_in,
                                          num_classes_out=num_classes_out,
                                          is_nnet_validconvs=is_nnet_validconvs,
                                          size_output_image=size_output_images,
                                          batch_size=batch_size,
                                          shuffle=is_shuffle,
                                          seed=manual_seed)


def get_train_imagedataloader_1image(list_filenames_1: List[str],
                                     size_in_images: Tuple[int, ...],
                                     use_sliding_window_images: bool,
                                     prop_overlap_slide_window: Tuple[int, ...],
                                     use_transform_rigid_images: bool,
                                     use_transform_elastic_images: bool,
                                     use_random_window_images: bool = False,
                                     num_random_patches_epoch: int = 0,
                                     batch_size: int = 1,
                                     is_shuffle: bool = True,
                                     manual_seed: int = None,
                                     is_load_images_from_batches: bool = False
                                     ) -> Union[TrainBatchImageDataGenerator1Image, List[np.ndarray]]:
    print("Generate Data Loader with Batch Generator...")

    if is_load_images_from_batches:
        list_xdata = ImageDataBatchesLoader(size_in_images).load_1list_files(list_filenames_1, is_shuffle=is_shuffle)
    else:
        list_xdata = ImageDataLoader.load_1list_files(list_filenames_1)

    size_full_image = list_xdata[0].shape if len(list_xdata) == 1 else (0, 0, 0)
    num_channels_in = 1

    images_generator = get_images_generator(size_in_images,
                                            use_sliding_window_images=use_sliding_window_images,
                                            prop_overlap_slide_window=prop_overlap_slide_window,
                                            use_random_window_images=use_random_window_images,
                                            num_random_patches_epoch=num_random_patches_epoch,
                                            use_transform_rigid_images=use_transform_rigid_images,
                                            use_transform_elastic_images=use_transform_elastic_images,
                                            size_volume_image=size_full_image)
    return TrainBatchImageDataGenerator1Image(size_in_images,
                                              list_xdata,
                                              images_generator,
                                              num_channels_in=num_channels_in,
                                              batch_size=batch_size,
                                              shuffle=is_shuffle,
                                              seed=manual_seed,
                                              is_datagen_gpu=IS_MODEL_IN_GPU,
                                              is_datagen_halfprec=IS_MODEL_HALF_PRECISION)


def get_train_imagedataloader_2images(list_filenames_1: List[str],
                                      list_filenames_2: List[str],
                                      size_in_images: Tuple[int, ...],
                                      use_sliding_window_images: bool,
                                      prop_overlap_slide_window: Tuple[int, ...],
                                      use_transform_rigid_images: bool,
                                      use_transform_elastic_images: bool,
                                      use_random_window_images: bool = False,
                                      num_random_patches_epoch: int = 0,
                                      is_nnet_validconvs: bool = False,
                                      size_output_images: Tuple[int, ...] = None,
                                      batch_size: int = 1,
                                      is_shuffle: bool = True,
                                      manual_seed: int = None,
                                      is_load_many_images_per_label: bool = False,
                                      num_images_per_label: int = 0,
                                      is_load_images_from_batches: bool = False
                                      ) -> Union[TrainBatchImageDataGenerator2Images,
                                                 Tuple[List[np.ndarray], List[np.ndarray]]]:
    print("Generate Data Loader with Batch Generator...")

    if is_load_images_from_batches:
        if is_load_many_images_per_label:
            list_xdata = ImageDataBatchesLoader(size_in_images).load_1list_files(list_filenames_1,
                                                                                 is_shuffle=is_shuffle)
            list_ydata = ImageDataBatchesLoader(size_in_images).load_1list_files(list_filenames_2,
                                                                                 is_shuffle=is_shuffle)
        else:
            (list_xdata, list_ydata) = ImageDataBatchesLoader(size_in_images).load_2list_files(list_filenames_1,
                                                                                               list_filenames_2,
                                                                                               is_shuffle=is_shuffle)
    else:
        if is_load_many_images_per_label:
            list_xdata = ImageDataLoader.load_1list_files(list_filenames_1)
            list_ydata = ImageDataLoader.load_1list_files(list_filenames_2)
        else:
            (list_xdata, list_ydata) = ImageDataLoader.load_2list_files(list_filenames_1, list_filenames_2)

    size_full_image = list_xdata[0].shape if len(list_xdata) == 1 else (0, 0, 0)
    num_channels_in = 1
    num_classes_out = 1

    images_generator = get_images_generator(size_in_images,
                                            use_sliding_window_images=use_sliding_window_images,
                                            prop_overlap_slide_window=prop_overlap_slide_window,
                                            use_random_window_images=use_random_window_images,
                                            num_random_patches_epoch=num_random_patches_epoch,
                                            use_transform_rigid_images=use_transform_rigid_images,
                                            use_transform_elastic_images=use_transform_elastic_images,
                                            size_volume_image=size_full_image)
    if is_load_many_images_per_label:
        return TrainBatchImageDataGeneratorManyImagesPerLabel(size_in_images,
                                                              num_images_per_label,
                                                              list_xdata,
                                                              list_ydata,
                                                              images_generator,
                                                              num_channels_in=num_channels_in,
                                                              num_classes_out=num_classes_out,
                                                              is_nnet_validconvs=is_nnet_validconvs,
                                                              size_output_image=size_output_images,
                                                              batch_size=batch_size,
                                                              shuffle=is_shuffle,
                                                              seed=manual_seed,
                                                              is_datagen_gpu=IS_MODEL_IN_GPU,
                                                              is_datagen_halfprec=IS_MODEL_HALF_PRECISION)
    else:
        return TrainBatchImageDataGenerator2Images(size_in_images,
                                                   list_xdata,
                                                   list_ydata,
                                                   images_generator,
                                                   num_channels_in=num_channels_in,
                                                   num_classes_out=num_classes_out,
                                                   is_nnet_validconvs=is_nnet_validconvs,
                                                   size_output_image=size_output_images,
                                                   batch_size=batch_size,
                                                   shuffle=is_shuffle,
                                                   seed=manual_seed,
                                                   is_datagen_gpu=IS_MODEL_IN_GPU,
                                                   is_datagen_halfprec=IS_MODEL_HALF_PRECISION)
