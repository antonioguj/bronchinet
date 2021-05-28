
from typing import Tuple, Any
import numpy as np
import SimpleITK as sitk
import pydicom
import gzip
import warnings
with warnings.catch_warnings():
    # disable FutureWarning: conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated
    warnings.filterwarnings("ignore", category=FutureWarning)
    import nibabel as nib
    import h5py

from common.exceptionmanager import catch_error_exception
from common.functionutil import fileextension


class ImageFileReader(object):

    @classmethod
    def get_image_position(cls, filename: str) -> Tuple[float, float, float]:
        return cls._get_filereader_class(filename).get_image_position(filename)

    @classmethod
    def get_image_voxelsize(cls, filename: str) -> Tuple[float, float, float]:
        return cls._get_filereader_class(filename).get_image_voxelsize(filename)

    @classmethod
    def get_image_metadata_info(cls, filename: str) -> Any:
        return cls._get_filereader_class(filename).get_image_metadata_info(filename)

    @classmethod
    def update_image_metadata_info(cls, filename: str, **kwargs) -> Any:
        in_metadata = cls.get_image_metadata_info(filename)
        return cls._get_filereader_class(filename).update_image_metadata_info(in_metadata, **kwargs)

    @classmethod
    def get_image_size(cls, filename: str) -> Tuple[int, int, int]:
        return cls.get_image(filename).shape

    @classmethod
    def get_image(cls, filename: str) -> np.ndarray:
        return cls._get_filereader_class(filename).get_image(filename)

    @classmethod
    def write_image(cls, filename: str, in_image: np.ndarray, **kwargs) -> None:
        cls._get_filereader_class(filename).write_image(filename, in_image, **kwargs)

    @staticmethod
    def _get_filereader_class(filename: str) -> 'ImageFileReader':
        extension = fileextension(filename)
        if extension == '.nii' or extension == '.nii.gz':
            return NiftiReader
        elif extension == '.dcm':
            return DicomReader
        elif extension == '.mhd':
            return MHDRawReader
        elif extension == '.npy':
            return NumpyReader
        elif extension == '.npz':
            return NumpyZReader
        elif extension == '.hdf5':
            return Hdf5Reader
        else:
            message = "Not valid file extension: %s..." % (extension)
            catch_error_exception(message)


class NiftiReader(ImageFileReader):

    @classmethod
    def get_image_position(cls, filename: str) -> Tuple[float, float, float]:
        affine = cls._get_image_affine_matrix(filename)
        return tuple(affine[:3, -1])

    @classmethod
    def get_image_voxelsize(cls, filename: str) -> Tuple[float, float, float]:
        affine = cls._get_image_affine_matrix(filename)
        return tuple(np.abs(np.diag(affine)[:3]))

    @classmethod
    def get_image_metadata_info(cls, filename: str) -> Any:
        return cls._get_image_affine_matrix(filename)

    @classmethod
    def update_image_metadata_info(cls, in_metadata: Any, **kwargs) -> Any:
        rescale_factor = kwargs['rescale_factor'] if 'rescale_factor' in kwargs.keys() else None
        translate_factor = kwargs['translate_factor'] if 'translate_factor' in kwargs.keys() else None
        return cls._update_affine_matrix(in_metadata, rescale_factor, translate_factor)

    @classmethod
    def get_image(cls, filename: str) -> np.ndarray:
        out_image = nib.load(filename).get_data()
        return cls._fix_dims_image_read(out_image)

    @classmethod
    def write_image(cls, filename: str, in_image: np.ndarray, **kwargs) -> None:
        affine = kwargs['metadata'] if 'metadata' in kwargs.keys() else None
        in_image = cls._fix_dims_image_write(in_image)
        nib_image = nib.Nifti1Image(in_image, affine)
        nib.save(nib_image, filename)

    @staticmethod
    def _get_image_affine_matrix(filename: str) -> np.ndarray:
        return nib.load(filename).affine

    @staticmethod
    def _compute_affine_matrix(image_voxelsize: Tuple[float, float, float],
                               image_position: Tuple[float, float, float],
                               # image_rotation: : Tuple[float, float, float]
                               ) -> np.ndarray:
        # Consider affine transformations composed of rescaling and translation, for the moment
        affine = np.eye(4)
        if image_voxelsize is not None:
            np.fill_diagonal(affine[:3, :3], image_voxelsize)
        if image_position is not None:
            affine[:3, -1] = image_position
        return affine

    @staticmethod
    def _update_affine_matrix(inout_affine: np.ndarray,
                              rescale_factor: Tuple[float, float, float],
                              translate_factor: Tuple[float, float, float],
                              # rotate_factor: Tuple[float, float, float]
                              ) -> np.ndarray:
        # Consider affine transformations composed of rescaling and translation, for the moment
        if rescale_factor is not None:
            rescale_matrix = np.eye(rescale_factor + (1,))
            inout_affine = np.dot(inout_affine, rescale_matrix)
        if translate_factor is not None:
            inout_affine[:3, -1] += translate_factor
        return inout_affine

    @staticmethod
    def _fix_dims_affine_matrix(inout_affine: np.ndarray) -> np.ndarray:
        # Change dimensions from (dz, dx, dy) to (dx, dy, dz) (nifty format)
        # inout_affine[[0, 1, 2], :] = inout_affine[[1, 2, 0], :]
        # inout_affine[:, [0, 1, 2]] = inout_affine[:, [1, 2, 0]]
        # Change dimensions from (dz, dy, dz) to (dx, dy, dz) (nifty format)
        inout_affine[[0, 2], :] = inout_affine[[2, 0], :]
        inout_affine[:, [0, 2]] = inout_affine[:, [2, 0]]
        return inout_affine

    @staticmethod
    def _fix_dims_image_read(in_image: np.ndarray) -> np.ndarray:
        # Roll image dimensions from (dz, dx, dy) to (dx, dy, dz) (nifty format)
        # return np.rollaxis(in_image, 2, 0)
        # Roll image dimensions from (dz, dy, dx) to (dx, dy, dz) (nifty format)
        return np.swapaxes(in_image, 0, 2)

    @staticmethod
    def _fix_dims_image_write(in_image: np.ndarray) -> np.ndarray:
        # Roll image dimensions from (dz, dx, dy) to (dx, dy, dz) (nifty format)
        # return np.rollaxis(in_image, 0, 3)
        # Roll image dimensions from (dz, dy, dx) to (dx, dy, dz) (nifty format)
        return np.swapaxes(in_image, 0, 2)

    @staticmethod
    def fix_dims_image_from_dicom2niix(in_image: np.ndarray) -> np.ndarray:
        return np.flip(in_image, 1)

    @staticmethod
    def fix_dims_image_affine_matrix_from_dicom2niix(inout_affine: np.ndarray) -> np.ndarray:
        inout_affine[1, 1] = - inout_affine[1, 1]
        inout_affine[1, -1] = - inout_affine[1, -1]
        return inout_affine


class DicomReader(ImageFileReader):

    @classmethod
    def get_image_position(cls, filename: str) -> Tuple[float, float, float]:
        ds = pydicom.read_file(filename)
        image_position_str = ds[0x0020, 0x0032].value   # Elem 'Image Position (Patient)'
        return (float(image_position_str[0]),
                float(image_position_str[1]),
                float(image_position_str[2]))

    @classmethod
    def get_image_voxelsize(cls, filename: str) -> Tuple[float, float, float]:
        ds = pydicom.read_file(filename)
        return (float(ds.SpacingBetweenSlices),
                float(ds.PixelSpacing[0]),
                float(ds.PixelSpacing[1]))

    @staticmethod
    def get_dicom_header(filename: str, is_return_tags_description: bool = False) -> Any:
        header_read = pydicom.read_file(filename)
        if is_return_tags_description:
            return {key: (header_read[key].repval, header_read[key].name) for key in header_read.keys()}
        else:
            return {key: header_read[key].repval for key in header_read.keys()}

    @classmethod
    def get_image_metadata_info(cls, filename: str) -> Any:
        image_read = sitk.ReadImage(filename)
        metadata_keys = image_read.GetMetaDataKeys()
        return {key: image_read.GetMetaData(key) for key in metadata_keys}

    @classmethod
    def update_image_metadata_info(cls, in_metadata: Any, **kwargs) -> Any:
        if 'target_metadata' in kwargs.keys():
            target_metadata = kwargs['target_metadata']
            return cls._update_headertags_physical_info(in_metadata, target_metadata)
        else:
            return None

    @classmethod
    def get_image(cls, filename: str) -> np.ndarray:
        image_read = sitk.ReadImage(filename)
        return sitk.GetArrayFromImage(image_read)

    @classmethod
    def write_image(cls, filename: str, in_image: np.ndarray, **kwargs) -> None:
        if in_image.dtype != np.uint16:
            in_image = in_image.astype(np.uint16)
        image_write = sitk.GetImageFromArray(in_image)
        if 'metadata' in kwargs.keys():
            dict_metadata = kwargs['metadata']
            for (key, val) in dict_metadata.items():
                image_write.SetMetaData(key, val)
        sitk.WriteImage(image_write, filename)

    @classmethod
    def write_image_old(cls, filename: str, in_image: np.ndarray, dsref_image) -> None:
        if dsref_image.file_meta.TransferSyntaxUID.is_compressed:
            dsref_image.decompress()
        if in_image.dtype != np.uint16:
            in_image = in_image.astype(np.uint16)
        dsref_image.PixelData = in_image.tostring()
        pydicom.write_file(filename, dsref_image)

    @staticmethod
    def _convert_image_stored_dtype_uint16(in_image: np.ndarray) -> np.ndarray:
        max_val_uint16 = np.iinfo(np.uint16).max
        ind_pos_0 = np.argwhere(in_image == 0)
        in_image = in_image.astype(np.int32) - max_val_uint16 - 1
        in_image[ind_pos_0] = 0
        return in_image

    @staticmethod
    def _update_headertags_physical_info(inout_metadata: Any, target_metadata: Any) -> Any:
        # Dicom header tags for info of i) world coordinates and ii) voxel size
        tag_image_position = '0020|0032'
        tag_image_orientation = '0020|0037'
        tag_spacing_slices = '0018|0088'
        tag_pixel_spacing = '0028|0030'
        list_tags_update = [tag_image_position, tag_image_orientation, tag_spacing_slices, tag_pixel_spacing]
        for itag in list_tags_update:
            inout_metadata[itag] = target_metadata[itag]
        return inout_metadata


class MHDRawReader(ImageFileReader):

    @classmethod
    def get_image(cls, filename: str) -> np.ndarray:
        image_read = sitk.ReadImage(filename)
        return sitk.GetArrayFromImage(image_read)

    @classmethod
    def write_image(cls, filename: str, in_image: np.ndarray, **kwargs) -> None:
        image_write = sitk.GetImageFromArray(in_image)
        sitk.WriteImage(image_write, filename)


class NumpyReader(ImageFileReader):

    @classmethod
    def get_image(cls, filename: str) -> np.ndarray:
        return np.load(filename)

    @classmethod
    def write_image(cls, filename: str, in_image: np.ndarray, **kwargs) -> None:
        np.save(filename, in_image)


class NumpyZReader(ImageFileReader):

    @classmethod
    def get_image(cls, filename: str) -> np.ndarray:
        return np.load(filename)['arr_0']

    @classmethod
    def write_image(cls, filename: str, in_image: np.ndarray, **kwargs) -> None:
        np.savez_compressed(filename, in_image)


class Hdf5Reader(ImageFileReader):

    @classmethod
    def get_image(cls, filename: str) -> np.ndarray:
        data_file = h5py.File(filename, 'r')
        return data_file['data'][:]

    @classmethod
    def write_image(cls, filename: str, in_image: np.ndarray, **kwargs) -> None:
        data_file = h5py.File(filename, 'w')
        data_file.create_dataset('data', data=in_image)
        data_file.close()


class GzipManager(object):

    @staticmethod
    def get_read_file(filename: str) -> Any:
        return gzip.GzipFile(filename, 'r')

    @staticmethod
    def get_write_file(filename: str) -> Any:
        return gzip.GzipFile(filename, 'w')

    @staticmethod
    def close_file(fileobj: Any) -> None:
        fileobj.close()


# all available file readers
DICT_AVAIL_FILE_READER = {'nifti': NiftiReader,
                          'dicom': DicomReader,
                          'numpy': NumpyReader}
