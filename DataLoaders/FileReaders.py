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
    def getImageHeaderInfo(cls, filename):
        return cls.getFileReaderClass(filename).getImageHeaderInfo(filename)

    @classmethod
    def getImageSize(cls, filename):
        return cls.getImageArray(filename).shape

    @classmethod
    def getImageArray(cls, filename):
        return cls.getFileReaderClass(filename).getImageArray(filename)

    @classmethod
    def writeImageArray(cls, filename, images_array, img_header_info=None):
        cls.getFileReaderClass(filename).writeImageArray(filename, images_array, img_header_info)

    @staticmethod
    def getFileReaderClass(filename):
        basename, extension = ospath_splitext_recurse(filename)
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

    @classmethod
    def getImagePosition(cls, filename):
        affine = cls.getImageAffineMatrix(filename)
        return tuple(affine[:3,-1])

    @classmethod
    def getImageVoxelSize(cls, filename):
        affine = cls.getImageAffineMatrix(filename)
        return tuple(np.abs(np.diag(affine)[:3]))

    @classmethod
    def getImageHeaderInfo(cls, filename):
        return cls.getImageAffineMatrix(filename)

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
    def getImageArray(cls, filename, isFix_from_dicom2niix=False):
        nib_img = nib.load(filename)
        return cls.changeDimsImageArray_read(nib_img.get_data())

    @classmethod
    def writeImageArray(cls, filename, images_array, img_header_affine=None):
        nib_img = nib.Nifti1Image(cls.changeDimsImageArray_write(images_array), img_header_affine)
        nib.save(nib_img, filename)



class NUMPYreader(FileReader):
    @classmethod
    def getImageArray(cls, filename):
        return np.load(filename)

    @classmethod
    def writeImageArray(cls, filename, images_array, img_header_info=None):
        np.save(filename, images_array)


class NUMPYZreader(FileReader):
    @classmethod
    def getImageArray(cls, filename):
        return np.load(filename)['arr_0']

    @classmethod
    def writeImageArray(cls, filename, images_array, img_header_info=None):
        np.savez_compressed(filename, images_array)


class HDF5reader(FileReader):
    @classmethod
    def getImageArray(cls, filename):
        data_file = h5py.File(filename, 'r')
        return data_file['data'][:]

    @classmethod
    def writeImageArray(cls, filename, images_array, img_header_info=None):
        data_file = h5py.File(filename, 'w')
        data_file.create_dataset('data', data=images_array)
        data_file.close()


class MHDRAWreader(FileReader):
    @classmethod
    def getImageArray(cls, filename):
        ds = sitk.ReadImage(filename)
        return sitk.GetArrayFromImage(ds)

    @classmethod
    def writeImageArray(cls, filename, images_array, img_header_info=None):
        ds = sitk.GetImageFromArray(images_array)
        sitk.WriteImage(ds, filename)



class DICOMreader(FileReader):
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
    def getImageHeaderInfo(cls, filename):
        return {'position': cls.getImagePosition(filename),
                'voxelsize': cls.getImageVoxelSize(filename)}

    @classmethod
    def getImageArray(cls, filename):
        ds = sitk.ReadImage(filename)
        return sitk.GetArrayFromImage(ds)

    @classmethod
    def writeImageArray(cls, filename, images_array, img_header_info=None):
        ds = sitk.GetImageFromArray(images_array)
        sitk.WriteImage(ds, filename)

    @staticmethod
    def writeDICOMimage(filename, images_array):
        ## This code block was taken from the output of a MATLAB secondary
        ## capture.  I do not know what the long dotted UIDs mean, but
        ## this code works.
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
        file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
        file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble="\0" * 128)
        ds.Modality = 'WSD'
        ds.ContentDate = str(datetime.date.today()).replace('-', '')
        ds.ContentTime = str(time.time())  # milliseconds since the epoch
        ds.StudyInstanceUID = '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
        ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
        ds.SOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
        ds.SOPClassUID = 'Secondary Capture Image Storage'
        ds.SecondaryCaptureDeviceManufctur = 'Python 2.7.3'
        ## These are the necessary imaging components of the FileDataset object.
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SmallestImagePixelValue = '\\x00\\x00'
        ds.LargestImagePixelValue = '\\xff\\xff'
        ds.Rows = images_array.shape[0]
        ds.Columns = images_array.shape[1]
        if images_array.dtype != np.uint16:
            images_array = images_array.astype(np.uint16)
        ds.PixelData = images_array.tostring()
        ds.save_as(filename)

    # get dcm header info:
    @staticmethod
    def loadPatientInformation(filename):
        ds = pydicom.read_file(filename)
        information = {}
        information['PatientID'] = ds.PatientID
        information['PatientName'] = ds.PatientName
        information['PatientBirthDate'] = ds.PatientBirthDate
        information['PatientSex'] = ds.PatientSex
        information['StudyID'] = ds.StudyID
        # information['StudyTime'] = ds.Studytime
        information['InstitutionName'] = ds.InstitutionName
        information['Manufacturer'] = ds.Manufacturer
        information['NumberOfFrames'] = ds.NumberOfFrames
        return information

    # copy PixelData info and save image
    @staticmethod
    def copyPixelDataAndSaveImage(origfilename, newfilename):
        orig_ds = pydicom.read_file(origfilename)
        new_ds  = pydicom.read_file(newfilename)
        orig_ds.PixelData = new_ds.PixelData
        orig_ds.save_as(origfilename)



# all available file readers
DICTAVAILFILEREADERS = {'nifti': NIFTIreader ,
                        'numpy': NUMPYreader,
                        'dicom': DICOMreader}
