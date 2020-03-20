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
    def getImagePosition(cls, filename):
        return cls.getFileReaderClass(filename).getImagePosition(filename)

    @classmethod
    def getImageVoxelSize(cls, filename):
        return cls.getFileReaderClass(filename).getImageVoxelSize(filename)

    @classmethod
    def getImageMetadataInfo(cls, filename):
        return cls.getFileReaderClass(filename).getImageMetadataInfo(filename)

    @classmethod
    def updateImageMetadataInfo(cls, filename, **kwargs):
        in_metadata = cls.getImageMetadataInfo(filename)
        return cls.getFileReaderClass(filename).updateImageMetadataInfo(in_metadata, **kwargs)

    @classmethod
    def getImageSize(cls, filename):
        return cls.getImageArray(filename).shape

    @classmethod
    def getImageArray(cls, filename):
        return cls.getFileReaderClass(filename).getImageArray(filename)

    @classmethod
    def writeImageArray(cls, filename, img_array, **kwargs):
        cls.getFileReaderClass(filename).writeImageArray(filename, img_array, **kwargs)

    @staticmethod
    def getFileReaderClass(filename):
        extension = fileextension(filename)
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
    def computeAffineMatrix(img_voxelsize, img_position): #, img_rotation):
        # Only consider affine transformations composed on rescaling and translation, for the moment
        affine = np.eye(4)
        if img_voxelsize != None:
            np.fill_diagonal(affine[:3,:3], img_voxelsize)
        if img_position != None:
            affine[:3,-1] = img_position
        return affine

    @staticmethod
    def updateAffineMatrix(in_affine, img_rescalefactor, img_translatefactor): #, img_rotatefactor):
        # Only consider affine transformations composed on rescaling and translation, for the moment
        out_affine = in_affine
        if img_rescalefactor != None:
            rescale_matrix = np.eye(img_rescalefactor + (1,))
            out_affine = np.dot(out_affine, rescale_matrix)
        if img_translatefactor != None:
            out_affine[:3,-1] += img_translatefactor
        return out_affine

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
    def fixDimsImageArray_fromDicom2niix(img_array):
        return np.flip(img_array, 1)

    @staticmethod
    def fixDimsImageAffineMatrix_fromDicom2niix(affine):
        affine[1, 1] = - affine[1, 1]
        affine[1,-1] = - affine[1,-1]
        return affine

    @staticmethod
    def changeDimsImageArray_read(img_array):
        # Roll array axis to change dimensions from (dz, dx, dy) to (dx, dy, dz) (nifty format)
        # return np.rollaxis(img_array, 2, 0)
        # Roll array axis to change dimensions from (dz, dy, dx) to (dx, dy, dz) (nifty format)
        return np.swapaxes(img_array, 0, 2)

    @staticmethod
    def changeDimsImageArray_write(img_array):
        # Roll array axis to change dimensions from (dz, dx, dy) to (dx, dy, dz) (nifty format)
        # return np.rollaxis(img_array, 0, 3)
        # Roll array axis to change dimensions from (dz, dy, dx) to (dx, dy, dz) (nifty format)
        return np.swapaxes(img_array, 0, 2)

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
    def updateImageMetadataInfo(cls, in_metadata, **kwargs):
        if 'img_rescalefactor' in kwargs.keys():
            img_rescalefactor = kwargs['img_rescalefactor']
        else:
            img_rescalefactor = None
        if 'img_translatefactor' in kwargs.keys():
            img_translatefactor = kwargs['img_translatefactor']
        else:
            img_translatefactor = None
        return cls.updateAffineMatrix(in_metadata, img_rescalefactor, img_translatefactor)

    @classmethod
    def getImageArray(cls, filename, isFix_from_dicom2niix=False):
        nib_img = nib.load(filename)
        return cls.changeDimsImageArray_read(nib_img.get_data())

    @classmethod
    def writeImageArray(cls, filename, img_array, **kwargs):
        if 'metadata' in kwargs.keys():
            affine = kwargs['metadata']
        else:
            affine = None
        nib_img = nib.Nifti1Image(cls.changeDimsImageArray_write(img_array), affine)
        nib.save(nib_img, filename)



class NUMPYreader(FileReader):
    @classmethod
    def getImageArray(cls, filename):
        return np.load(filename)

    @classmethod
    def writeImageArray(cls, filename, img_array, **kwargs):
        np.save(filename, img_array)


class NUMPYZreader(FileReader):
    @classmethod
    def getImageArray(cls, filename):
        return np.load(filename)['arr_0']

    @classmethod
    def writeImageArray(cls, filename, img_array, **kwargs):
        np.savez_compressed(filename, img_array)


class HDF5reader(FileReader):
    @classmethod
    def getImageArray(cls, filename):
        data_file = h5py.File(filename, 'r')
        return data_file['data'][:]

    @classmethod
    def writeImageArray(cls, filename, img_array, **kwargs):
        data_file = h5py.File(filename, 'w')
        data_file.create_dataset('data', data=img_array)
        data_file.close()


class MHDRAWreader(FileReader):
    @classmethod
    def getImageArray(cls, filename):
        img_read = sitk.ReadImage(filename)
        return sitk.GetArrayFromImage(img_read)

    @classmethod
    def writeImageArray(cls, filename, img_array, **kwargs):
        img_write = sitk.GetImageFromArray(img_array)
        sitk.WriteImage(img_write, filename)



class DICOMreader(FileReader):
    @staticmethod
    def convertImageArrayStoredDtypeUint16(img_array):
        max_val_uint16 = np.iinfo(np.uint16).max
        ind_pos_0 = np.argwhere(img_array == 0)
        img_array = img_array.astype(np.int32) - max_val_uint16 - 1
        img_array[ind_pos_0] = 0
        return img_array

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

    @classmethod
    def updateImageMetadataInfo(cls, in_metadata, **kwargs):
        print('\'updateImageMetadataInfo\' not implemented for DICOMreader...')
        return NotImplemented

    @classmethod
    def getImageArray(cls, filename):
        img_read = sitk.ReadImage(filename)
        return sitk.GetArrayFromImage(img_read)

    @classmethod
    def writeImageArray(cls, filename, img_array, **kwargs):
        if img_array.dtype != np.uint16:
            img_array = img_array.astype(np.uint16)
        img_write = sitk.GetImageFromArray(img_array)
        if 'metadata' in kwargs.keys():
            dict_metadata = kwargs['metadata']
            for (key, val) in dict_metadata.iteritems():
                img_write.SetMetaData(key, val)
        sitk.WriteImage(img_write, filename)

    @classmethod
    def writeImageArray_OLD(cls, filename, img_array, ds_refimg):
        if ds_refimg.file_meta.TransferSyntaxUID.is_compressed:
            ds_refimg.decompress()
        if img_array.dtype != np.uint16:
            img_array = img_array.astype(np.uint16)
        ds_refimg.PixelData = img_array.tostring()
        pydicom.write_file(filename, ds_refimg)



# all available file readers
DICTAVAILFILEREADERS = {'nifti': NIFTIreader ,
                        'numpy': NUMPYreader,
                        'dicom': DICOMreader}
