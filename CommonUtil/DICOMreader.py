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
import dicom


class DICOMreader(object):

    # load dcm file:
    @staticmethod
    def loadImage(filename):

        ds = sitk.ReadImage(filename)
        return sitk.GetArrayFromImage(ds)

    # get image dims:
    @staticmethod
    def getImageSize(filename):

        ds = sitk.ReadImage(filename)
        return ds.GetSize()

    # load dcm file information:
    @staticmethod
    def loadImageInformation(filename):

        information = {}
        ds = dicom.read_file(filename)
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