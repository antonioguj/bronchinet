#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

import SimpleITK as sitk
import nibabel as nib
import dicom
from dicom.dataset import Dataset, FileDataset
import datetime, time
import numpy as np


class FileReader(object):

    @staticmethod
    def getImageArray(filename):
        pass

    @staticmethod
    def writeImageArray(filename, img_array):
        pass


class NUMPYreader(FileReader):

    # get numpy image array:
    @staticmethod
    def getImageArray(filename):

        return np.load(filename)

    # write numpy file array:
    @staticmethod
    def writeImageArray(filename, img_array):

        np.save(filename, img_array)


class NIFTIreader(FileReader):

    # get nifti image array:
    @staticmethod
    def getImageArray(filename):

        nib_im = nib.load(filename)
        return nib_im.get_data()

    # write nifti file array:
    @staticmethod
    def writeImageArray(filename, img_array):

        nib_im = nib.Nifti1Image(img_array, np.eye(4))
        nib.save(nib_im, filename)


class DICOMreader(FileReader):

    # get dcm image dims:
    @staticmethod
    def getImageSize(filename):

        ds = sitk.ReadImage(filename)
        return ds.GetSize()

    # load dcm file array:
    @staticmethod
    def getImageArray(filename):

        ds = sitk.ReadImage(filename)
        return sitk.GetArrayFromImage(ds)

    # write dcm file array:
    @staticmethod
    def writeImageArray(filename, img_array):

        ds = sitk.GetImageFromArray(img_array)
        sitk.WriteImage(ds, filename)

    @staticmethod
    def writeDICOMimage(filename, img_array):

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
        ds.Rows = img_array.shape[0]
        ds.Columns = img_array.shape[1]
        if img_array.dtype != np.uint16:
            img_array = img_array.astype(np.uint16)
        ds.PixelData = img_array.tostring()

        ds.save_as(filename)

    # get dcm header info:
    @staticmethod
    def loadPatientInformation(filename):

        ds = dicom.read_file(filename)

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

        orig_ds = dicom.read_file(origfilename)
        new_ds  = dicom.read_file(newfilename)
        orig_ds.PixelData = new_ds.PixelData
        orig_ds.save_as(origfilename)


# All Available File Readers
DICTAVAILFILEREADERS = {"numpy": NUMPYreader,
                        "dicom": DICOMreader,
                        "nifti": NIFTIreader }
