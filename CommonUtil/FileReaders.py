#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.FunctionsUtil import *
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

    @staticmethod
    def getImageSize(filename):

        basename, extension = ospath_splitext_recurse(filename)

        if (extension == '.dcm'):
            return DICOMreader.getImageSize(filename)
        if (extension == '.dcm.gz'):
            print("Not implemented for extension '.dcm.gz'...")
            return False
            # fileobj = GZIPmanager.getReadFile(filename)
            # out_arrsize = DICOMreader.getImageSize(fileobj)
            # GZIPmanager.closeFile(fileobj)
            # return out_arrsize
        elif (extension == '.nii'):
            return NIFTIreader.getImageSize(filename)
        elif (extension == '.nii.gz'):
            return NIFTIreader.getImageSize(filename)
        elif (extension == '.npy'):
            return NUMPYreader.getImageSize(filename)
        elif (extension == '.npz'):
            return NUMPYZreader.getImageSize(filename)
        elif (extension == '.npy.gz'):
            fileobj = GZIPmanager.getReadFile(filename)
            out_arrsize = NUMPYreader.getImageSize(fileobj)
            GZIPmanager.closeFile(fileobj)
            return out_arrsize
        elif (extension == '.hdf5'):
            return HDF5reader.getImageSize(filename)
        else:
            message = "No valid file extension: %s..." %(extension)
            CatchErrorException(message)

    @staticmethod
    def getImageArray(filename):

        basename, extension = ospath_splitext_recurse(filename)

        if (extension == '.dcm'):
            return DICOMreader.getImageArray(filename)
        if (extension == '.dcm.gz'):
            print("Not implemented for extension '.dcm.gz'...")
            return False
            # fileobj = GZIPmanager.getReadFile(filename)
            # out_array = DICOMreader.getImageArray(fileobj)
            # GZIPmanager.closeFile(fileobj)
            # return out_array
        elif (extension == '.nii'):
            return NIFTIreader.getImageArray(filename)
        elif (extension == '.nii.gz'):
            return NIFTIreader.getImageArray(filename)
        elif (extension == '.npy'):
            return NUMPYreader.getImageArray(filename)
        elif (extension == '.npz'):
            return NUMPYZreader.getImageArray(filename)
        elif (extension == '.npy.gz'):
            fileobj = GZIPmanager.getReadFile(filename)
            out_array = NUMPYreader.getImageArray(fileobj)
            GZIPmanager.closeFile(fileobj)
            return out_array
        elif (extension == '.hdf5'):
            return HDF5reader.getImageArray(filename)
        else:
            message = "No valid file extension: %s..." %(extension)
            CatchErrorException(message)

    @staticmethod
    def writeImageArray(filename, images_array):

        basename, extension = ospath_splitext_recurse(filename)

        if (extension == '.dcm'):
            DICOMreader.writeImageArray(filename, images_array)
        if (extension == '.dcm.gz'):
            print("Not implemented for extension '.dcm.gz'...")
            return False
            # fileobj = GZIPmanager.getWriteFile(filename)
            # DICOMreader.writeImageArray(fileobj, images_array)
            # GZIPmanager.closeFile(fileobj)
        elif (extension == '.nii'):
            NIFTIreader.writeImageArray(filename, images_array)
        elif (extension == '.nii.gz'):
            NIFTIreader.writeImageArray(filename, images_array)
        elif (extension == '.npy'):
            NUMPYreader.writeImageArray(filename, images_array)
        elif (extension == '.npz'):
            NUMPYZreader.writeImageArray(filename, images_array)
        elif (extension == '.npy.gz'):
            fileobj = GZIPmanager.getWriteFile(filename)
            NUMPYreader.writeImageArray(fileobj, images_array)
            GZIPmanager.closeFile(fileobj)
        elif (extension == '.hdf5'):
            HDF5reader.writeImageArray(filename, images_array)
        else:
            message = "No valid file extension: %s..." %(extension)
            CatchErrorException(message)


class HDF5reader(FileReader):

    # get h5py image size:
    @staticmethod
    def getImageSize(filename):
        data_file = h5py.File(filename, 'r')
        return data_file['data'].shape

    # get h5py image array:
    @staticmethod
    def getImageArray(filename):
        data_file = h5py.File(filename, 'r')
        return data_file['data'][:]

    # write h5py file array:
    @staticmethod
    def writeImageArray(filename, images_array):
        data_file = h5py.File(filename, 'w')
        data_file.create_dataset('data', data=images_array)
        data_file.close()


class NUMPYreader(FileReader):

    # get numpy image size:
    @staticmethod
    def getImageSize(filename):
        return np.load(filename).shape

    # get numpy image array:
    @staticmethod
    def getImageArray(filename):
        return np.load(filename)

    # write numpy file array:
    @staticmethod
    def writeImageArray(filename, images_array):
        np.save(filename, images_array)


class NUMPYZreader(FileReader):

    # get numpy image size:
    @staticmethod
    def getImageSize(filename):
        return np.load(filename)['arr_0'].shape

    # get numpy image array:
    @staticmethod
    def getImageArray(filename):
        return np.load(filename)['arr_0']

    # write numpy file array:
    @staticmethod
    def writeImageArray(filename, images_array):
        np.savez_compressed(filename, images_array)


class NIFTIreader(FileReader):
    # In nifty format, the axes are reversed.
    # Need to swap axis and set depth_Z first dim

    # get nifti image size:
    @staticmethod
    def getImageSize(filename):
        nib_im = nib.load(filename)
        return nib_im.get_data().shape[::-1]

    # get nifti image array:
    @staticmethod
    def getImageArray(filename):
        nib_im = nib.load(filename)
        return np.swapaxes(nib_im.get_data(), 0, 2)

    # write nifti file array:
    @staticmethod
    def writeImageArray(filename, images_array):
        nib_im = nib.Nifti1Image(np.swapaxes(images_array, 0, 2), np.eye(4))
        nib.save(nib_im, filename)


class DICOMreader(FileReader):

    # get dcm image dims:
    @staticmethod
    def getImageSize(filename):
        ds = sitk.ReadImage(filename)
        #np.swapaxes(ds.GetSize(), 0, 2)
        return sitk.GetArrayFromImage(ds).shape

    # get dcm voxel size:
    @staticmethod
    def getVoxelSize(filename):
        ds = pydicom.read_file(filename)
        voxel_size = (float(ds.SpacingBetweenSlices),
                      float(ds.PixelSpacing[0]),
                      float(ds.PixelSpacing[1]))
        return voxel_size

    # load dcm file array:
    @staticmethod
    def getImageArray(filename):
        ds = sitk.ReadImage(filename)
        return sitk.GetArrayFromImage(ds)

    # write dcm file array:
    @staticmethod
    def writeImageArray(filename, images_array):
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
DICTAVAILFILEREADERS = {"numpy": NUMPYreader,
                        "dicom": DICOMreader,
                        "nifti": NIFTIreader }
