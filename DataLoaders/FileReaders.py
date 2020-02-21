#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.FunctionsUtil import *
import SimpleITK as sitk
import pydicom
from pydicom.dataset import Dataset, FileDataset
import nibabel as nib
import numpy as np
import h5py
import datetime, time
import gzip



class GZIPmanager(object):
    @staticmethod
    def getReadFile(filename):
        return gzip.GzipFile(filename, 'r')

    @staticmethod
    def getWriteFile(filename):
        return gzip.GzipFile(filename, 'w')

    @staticmethod
    def closeFile(fileobj):
        fileobj.close()



class FileReader(object):
    @classmethod
    def getImageMetadataInfo(cls, filename):
        return cls.getFileReaderClass(filename).getImageMetadataInfo(filename)

    @classmethod
    def getImageSize(cls, filename):
        return cls.getImageArray(filename).shape

    @classmethod
    def getImageArray(cls, filename):
        return cls.getFileReaderClass(filename).getImageArray(filename)

    @classmethod
    def writeImageArray(cls, filename, images_array, **kwargs):
        cls.getFileReaderClass(filename).writeImageArray(filename, images_array, **kwargs)

    @staticmethod
    def getFileReaderClass(filename):
        extension = filenameextension(filename)
        if (extension == '.nii' or extension == '.nii.gz'):
            return NIFTIreader
        elif (extension == '.npy'):
            return NUMPYreader
        elif (extension == '.npz'):
            return NUMPYZreader
        elif (extension == '.hdf5'):
            return HDF5reader
        elif (extension == '.mhd'):
            return MHDRAWreader
        elif (extension == '.dcm'):
            return DICOMreader
        else:
            message = "Not valid file extension: %s..." %(extension)
            CatchErrorException(message)

    @staticmethod
    def get2ImageArraysAndCheck(filename1, filename2):
        image1_array = FileReader.getImageArray(filename1)
        image2_array = FileReader.getImageArray(filename2)
        if (image1_array.shape != image2_array.shape):
            message = 'size of image 1 \'%s\' ans image 2 \'%s\' do not match' %(image1_array.shape, image2_array.shape)
            CatchWarningException(message)
            return (False, False, False)
        return (image1_array, image2_array, True)



class NIFTIreader(FileReader):
    @staticmethod
    def getImageAffineMatrix(filename):
        return nib.load(filename).affine

    @staticmethod
    def computeAffineMatrix(img_voxelsize, img_position, img_rotation):
        affine = np.eye(4)
        if img_voxelsize != None:
            np.fill_diagonal(affine[:3,:3], img_voxelsize)
        if img_position != None:
            affine[:3,-1] = img_position
        return affine

    @staticmethod
    def changeDimsAffineMatrix(affine):
        # Change dimensions from (dz, dx, dy) to (dx, dy, dz) (nifty format)
        # affine[[0, 1, 2], :] = affine[[1, 2, 0], :]
        # affine[:, [0, 1, 2]] = affine[:, [1, 2, 0]]
        # Change dimensions from (dz, dy, dz) to (dx, dy, dz) (nifty format)
        affine[[0, 2], :] = affine[[2, 0], :]
        affine[:, [0, 2]] = affine[:, [2, 0]]
        return affine

    @staticmethod
    def fixDimsImageArray_fromDicom2niix(images_array):
        return np.flip(images_array, 1)

    @staticmethod
    def fixDimsImageAffineMatrix_fromDicom2niix(affine):
        affine[1, 1] = - affine[1, 1]
        affine[1,-1] = - affine[1,-1]
        return affine

    @staticmethod
    def changeDimsImageArray_read(images_array):
        # Roll array axis to change dimensions from (dz, dx, dy) to (dx, dy, dz) (nifty format)
        # return np.rollaxis(images_array, 2, 0)
        # Roll array axis to change dimensions from (dz, dy, dx) to (dx, dy, dz) (nifty format)
        return np.swapaxes(images_array, 0, 2)

    @staticmethod
    def changeDimsImageArray_write(images_array):
        # Roll array axis to change dimensions from (dz, dx, dy) to (dx, dy, dz) (nifty format)
        # return np.rollaxis(images_array, 0, 3)
        # Roll array axis to change dimensions from (dz, dy, dx) to (dx, dy, dz) (nifty format)
        return np.swapaxes(images_array, 0, 2)

    @classmethod
    def getImagePosition(cls, filename):
        affine = cls.getImageAffineMatrix(filename)
        return tuple(affine[:3,-1])

    @classmethod
    def getImageVoxelSize(cls, filename):
        affine = cls.getImageAffineMatrix(filename)
        return tuple(np.abs(np.diag(affine)[:3]))

    @classmethod
    def getImageMetadataInfo(cls, filename):
        return cls.getImageAffineMatrix(filename)

    @classmethod
    def getImageArray(cls, filename, isFix_from_dicom2niix=False):
        nib_img = nib.load(filename)
        return cls.changeDimsImageArray_read(nib_img.get_data())

    @classmethod
    def writeImageArray(cls, filename, images_array, **kwargs):
        if 'metadata' in kwargs.keys():
            affine = kwargs['metadata']
        else:
            affine = None
        nib_img = nib.Nifti1Image(cls.changeDimsImageArray_write(images_array), affine)
        nib.save(nib_img, filename)



class NUMPYreader(FileReader):
    @classmethod
    def getImageArray(cls, filename):
        return np.load(filename)

    @classmethod
    def writeImageArray(cls, filename, images_array, **kwargs):
        np.save(filename, images_array)


class NUMPYZreader(FileReader):
    @classmethod
    def getImageArray(cls, filename):
        return np.load(filename)['arr_0']

    @classmethod
    def writeImageArray(cls, filename, images_array, **kwargs):
        np.savez_compressed(filename, images_array)


class HDF5reader(FileReader):
    @classmethod
    def getImageArray(cls, filename):
        data_file = h5py.File(filename, 'r')
        return data_file['data'][:]

    @classmethod
    def writeImageArray(cls, filename, images_array, **kwargs):
        data_file = h5py.File(filename, 'w')
        data_file.create_dataset('data', data=images_array)
        data_file.close()


class MHDRAWreader(FileReader):
    @classmethod
    def getImageArray(cls, filename):
        img_read = sitk.ReadImage(filename)
        return sitk.GetArrayFromImage(img_read)

    @classmethod
    def writeImageArray(cls, filename, images_array, **kwargs):
        img_write = sitk.GetImageFromArray(images_array)
        sitk.WriteImage(img_write, filename)



class DICOMreader(FileReader):
    @staticmethod
    def getImageHeader(filename):
        return pydicom.read_file(filename)

    @classmethod
    def getImagePosition(cls, filename):
        ds = pydicom.read_file(filename)
        img_position_str = ds[0x0020, 0x0032].value   # Elem 'Image Position (Patient)'
        return (float(img_position_str[0]),
                float(img_position_str[1]),
                float(img_position_str[2]))

    @classmethod
    def getImageVoxelSize(cls, filename):
        ds = pydicom.read_file(filename)
        return (float(ds.SpacingBetweenSlices),
                float(ds.PixelSpacing[0]),
                float(ds.PixelSpacing[1]))

    @classmethod
    def getImageMetadataInfo(cls, filename):
        img_read = sitk.ReadImage(filename)
        metadata_keys = img_read.GetMetaDataKeys()
        return {key: img_read.GetMetaData(key) for key in metadata_keys}

    @staticmethod
    def convertImageArrayStoredDtypeUint16(images_array):
        max_val_uint16 = np.iinfo(np.uint16).max
        ind_pos_0 = np.argwhere(images_array == 0)
        images_array = images_array.astype(np.int32) - max_val_uint16 - 1
        images_array[ind_pos_0] = 0
        return images_array

    @classmethod
    def getImageArray(cls, filename):
        img_read = sitk.ReadImage(filename)
        return sitk.GetArrayFromImage(img_read)

    @classmethod
    def writeImageArray(cls, filename, images_array, **kwargs):
        if images_array.dtype != np.uint16:
            images_array = images_array.astype(np.uint16)
        img_write = sitk.GetImageFromArray(images_array)
        if 'metadata' in kwargs.keys():
            dict_metadata = kwargs['metadata']
            for (key, val) in dict_metadata.iteritems():
                img_write.SetMetaData(key, val)
        return sitk.WriteImage(img_write, filename)

    @classmethod
    def writeImageArray_OLD(cls, filename, images_array, ds_refimg):
        if ds_refimg.file_meta.TransferSyntaxUID.is_compressed:
            ds_refimg.decompress()
        if images_array.dtype != np.uint16:
            images_array = images_array.astype(np.uint16)
        ds_refimg.PixelData = images_array.tostring()
        pydicom.write_file(filename, ds_refimg)



# all available file readers
DICTAVAILFILEREADERS = {'nifti': NIFTIreader ,
                        'numpy': NUMPYreader,
                        'dicom': DICOMreader}
