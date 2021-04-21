
from typing import List, Dict, Tuple, Union, Any

from common.constant import TYPE_DNNLIB_USED
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
from preprocessing.preprocessing_manager import get_image_generator, fill_missing_trans_rigid_params


def get_imagedataloader_1image(list_filenames_1: List[str],
                               size_images: Union[Tuple[int, int, int], Tuple[int, int]],
                               is_generate_patches: bool,
                               type_generate_patches: str,
                               prop_overlap_slide_images: Union[Tuple[float, float, float], Tuple[float, float]],
                               num_random_images: int,
                               is_transform_images: bool,
                               type_transform_images: str,
                               trans_rigid_params: Union[Dict[str, Any], None],
                               batch_size: int = 1,
                               is_shuffle: bool = True,
                               manual_seed: int = None,
                               is_load_images_from_batches: bool = False
                               ) -> BatchImageDataGenerator1Image:
    print("Generate Data Loader with Batch Generator...")

    if is_load_images_from_batches:
        list_xdata = ImageDataBatchesLoader(size_images).load_1list_files(list_filenames_1, is_shuffle=is_shuffle)
    else:
        list_xdata = ImageDataLoader.load_1list_files(list_filenames_1)

    if not is_generate_patches and (len(list_xdata) == 1):
        size_images = list_xdata[0].shape

    size_volume_images = list_xdata[0].shape if len(list_xdata) == 1 else (0, 0, 0)
    num_channels_in = 1

    trans_rigid_params = fill_missing_trans_rigid_params(trans_rigid_params)

    image_generator = get_image_generator(size_images,
                                          is_generate_patches=is_generate_patches,
                                          type_generate_patches=type_generate_patches,
                                          prop_overlap_slide_images=prop_overlap_slide_images,
                                          num_random_images=num_random_images,
                                          is_transform_images=is_transform_images,
                                          type_transform_images=type_transform_images,
                                          trans_rotation_range=trans_rigid_params['rotation_range'],
                                          trans_shift_range=trans_rigid_params['shift_range'],
                                          trans_flip_dirs=trans_rigid_params['flip_dirs'],
                                          trans_zoom_range=trans_rigid_params['zoom_range'],
                                          trans_fill_mode=trans_rigid_params['fill_mode'],
                                          size_volume_images=size_volume_images)
    print(image_generator.get_text_description())

    return BatchImageDataGenerator1Image(size_images,
                                         list_xdata,
                                         image_generator,
                                         num_channels_in=num_channels_in,
                                         batch_size=batch_size,
                                         shuffle=is_shuffle,
                                         seed=manual_seed)


def get_imagedataloader_2images(list_filenames_1: List[str],
                                list_filenames_2: List[str],
                                size_images: Union[Tuple[int, int, int], Tuple[int, int]],
                                is_generate_patches: bool,
                                type_generate_patches: str,
                                prop_overlap_slide_images: Union[Tuple[float, float, float], Tuple[float, float]],
                                num_random_images: int,
                                is_transform_images: bool,
                                type_transform_images: str,
                                trans_rigid_params: Union[Dict[str, Any], None],
                                is_nnet_validconvs: bool = False,
                                size_output_images: Union[Tuple[int, int, int], Tuple[int, int]] = None,
                                batch_size: int = 1,
                                is_shuffle: bool = True,
                                manual_seed: int = None,
                                is_load_images_from_batches: bool = False
                                ) -> BatchImageDataGenerator2Images:
    print("Generate Data Loader with Batch Generator...")

    if is_load_images_from_batches:
        (list_xdata, list_ydata) = ImageDataBatchesLoader(size_images).load_2list_files(list_filenames_1,
                                                                                        list_filenames_2,
                                                                                        is_shuffle=is_shuffle)
    else:
        (list_xdata, list_ydata) = ImageDataLoader.load_2list_files(list_filenames_1, list_filenames_2)

    if not is_generate_patches and (len(list_xdata) == 1):
        size_images = list_xdata[0].shape

    size_volume_images = list_xdata[0].shape if len(list_xdata) == 1 else (0, 0, 0)
    num_channels_in = 1
    num_classes_out = 1

    trans_rigid_params = fill_missing_trans_rigid_params(trans_rigid_params)

    image_generator = get_image_generator(size_images,
                                          is_generate_patches=is_generate_patches,
                                          type_generate_patches=type_generate_patches,
                                          prop_overlap_slide_images=prop_overlap_slide_images,
                                          num_random_images=num_random_images,
                                          is_transform_images=is_transform_images,
                                          type_transform_images=type_transform_images,
                                          trans_rotation_range=trans_rigid_params['rotation_range'],
                                          trans_shift_range=trans_rigid_params['shift_range'],
                                          trans_flip_dirs=trans_rigid_params['flip_dirs'],
                                          trans_zoom_range=trans_rigid_params['zoom_range'],
                                          trans_fill_mode=trans_rigid_params['fill_mode'],
                                          size_volume_images=size_volume_images)
    print(image_generator.get_text_description())

    return BatchImageDataGenerator2Images(size_images,
                                          list_xdata,
                                          list_ydata,
                                          image_generator,
                                          num_channels_in=num_channels_in,
                                          num_classes_out=num_classes_out,
                                          is_nnet_validconvs=is_nnet_validconvs,
                                          size_output_image=size_output_images,
                                          batch_size=batch_size,
                                          shuffle=is_shuffle,
                                          seed=manual_seed)


def get_train_imagedataloader_1image(list_filenames_1: List[str],
                                     size_images: Union[Tuple[int, int, int], Tuple[int, int]],
                                     is_generate_patches: bool,
                                     type_generate_patches: str,
                                     prop_overlap_slide_images: Union[Tuple[float, float, float], Tuple[float, float]],
                                     num_random_images: int,
                                     is_transform_images: bool,
                                     type_transform_images: str,
                                     trans_rigid_params: Union[Dict[str, Any], None],
                                     batch_size: int = 1,
                                     is_shuffle: bool = True,
                                     manual_seed: int = None,
                                     is_load_images_from_batches: bool = False
                                     ) -> TrainBatchImageDataGenerator1Image:
    print("Generate Data Loader with Batch Generator...")

    if is_load_images_from_batches:
        list_xdata = ImageDataBatchesLoader(size_images).load_1list_files(list_filenames_1, is_shuffle=is_shuffle)
    else:
        list_xdata = ImageDataLoader.load_1list_files(list_filenames_1)

    size_volume_images = list_xdata[0].shape if len(list_xdata) == 1 else (0, 0, 0)
    num_channels_in = 1

    trans_rigid_params = fill_missing_trans_rigid_params(trans_rigid_params)

    image_generator = get_image_generator(size_images,
                                          is_generate_patches=is_generate_patches,
                                          type_generate_patches=type_generate_patches,
                                          prop_overlap_slide_images=prop_overlap_slide_images,
                                          num_random_images=num_random_images,
                                          is_transform_images=is_transform_images,
                                          type_transform_images=type_transform_images,
                                          trans_rotation_range=trans_rigid_params['rotation_range'],
                                          trans_shift_range=trans_rigid_params['shift_range'],
                                          trans_flip_dirs=trans_rigid_params['flip_dirs'],
                                          trans_zoom_range=trans_rigid_params['zoom_range'],
                                          trans_fill_mode=trans_rigid_params['fill_mode'],
                                          size_volume_images=size_volume_images)
    print(image_generator.get_text_description())

    return TrainBatchImageDataGenerator1Image(size_images,
                                              list_xdata,
                                              image_generator,
                                              num_channels_in=num_channels_in,
                                              batch_size=batch_size,
                                              shuffle=is_shuffle,
                                              seed=manual_seed)


def get_train_imagedataloader_2images(list_filenames_1: List[str],
                                      list_filenames_2: List[str],
                                      size_images: Union[Tuple[int, int, int], Tuple[int, int]],
                                      is_generate_patches: bool,
                                      type_generate_patches: str,
                                      prop_overlap_slide_images: Union[Tuple[float, float, float], Tuple[float, float]],
                                      num_random_images: int,
                                      is_transform_images: bool,
                                      type_transform_images: str,
                                      trans_rigid_params: Union[Dict[str, Any], None],
                                      is_nnet_validconvs: bool = False,
                                      size_output_images: Union[Tuple[int, int, int], Tuple[int, int]] = None,
                                      batch_size: int = 1,
                                      is_shuffle: bool = True,
                                      manual_seed: int = None,
                                      is_load_many_images_per_label: bool = False,
                                      num_images_per_label: int = 0,
                                      is_load_images_from_batches: bool = False
                                      ) -> Union[TrainBatchImageDataGenerator2Images,
                                                 TrainBatchImageDataGeneratorManyImagesPerLabel]:
    print("Generate Data Loader with Batch Generator...")

    if is_load_images_from_batches:
        if is_load_many_images_per_label:
            list_xdata = ImageDataBatchesLoader(size_images).load_1list_files(list_filenames_1,
                                                                              is_shuffle=is_shuffle)
            list_ydata = ImageDataBatchesLoader(size_images).load_1list_files(list_filenames_2,
                                                                              is_shuffle=is_shuffle)
        else:
            (list_xdata, list_ydata) = ImageDataBatchesLoader(size_images).load_2list_files(list_filenames_1,
                                                                                            list_filenames_2,
                                                                                            is_shuffle=is_shuffle)
    else:
        if is_load_many_images_per_label:
            list_xdata = ImageDataLoader.load_1list_files(list_filenames_1)
            list_ydata = ImageDataLoader.load_1list_files(list_filenames_2)
        else:
            (list_xdata, list_ydata) = ImageDataLoader.load_2list_files(list_filenames_1, list_filenames_2)

    size_volume_images = list_xdata[0].shape if len(list_xdata) == 1 else (0, 0, 0)
    num_channels_in = 1
    num_classes_out = 1

    trans_rigid_params = fill_missing_trans_rigid_params(trans_rigid_params)

    image_generator = get_image_generator(size_images,
                                          is_generate_patches=is_generate_patches,
                                          type_generate_patches=type_generate_patches,
                                          prop_overlap_slide_images=prop_overlap_slide_images,
                                          num_random_images=num_random_images,
                                          is_transform_images=is_transform_images,
                                          type_transform_images=type_transform_images,
                                          trans_rotation_range=trans_rigid_params['rotation_range'],
                                          trans_shift_range=trans_rigid_params['shift_range'],
                                          trans_flip_dirs=trans_rigid_params['flip_dirs'],
                                          trans_zoom_range=trans_rigid_params['zoom_range'],
                                          trans_fill_mode=trans_rigid_params['fill_mode'],
                                          size_volume_images=size_volume_images)
    print(image_generator.get_text_description())

    if is_load_many_images_per_label:
        return TrainBatchImageDataGeneratorManyImagesPerLabel(size_images,
                                                              num_images_per_label,
                                                              list_xdata,
                                                              list_ydata,
                                                              image_generator,
                                                              num_channels_in=num_channels_in,
                                                              num_classes_out=num_classes_out,
                                                              is_nnet_validconvs=is_nnet_validconvs,
                                                              size_output_image=size_output_images,
                                                              batch_size=batch_size,
                                                              shuffle=is_shuffle,
                                                              seed=manual_seed)
    else:
        return TrainBatchImageDataGenerator2Images(size_images,
                                                   list_xdata,
                                                   list_ydata,
                                                   image_generator,
                                                   num_channels_in=num_channels_in,
                                                   num_classes_out=num_classes_out,
                                                   is_nnet_validconvs=is_nnet_validconvs,
                                                   size_output_image=size_output_images,
                                                   batch_size=batch_size,
                                                   shuffle=is_shuffle,
                                                   seed=manual_seed)
