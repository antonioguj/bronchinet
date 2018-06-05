#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.Constants import *
from CommonUtil.ErrorMessages import *
from CommonUtil.FileReaders import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.WorkDirsManager import *
from Preprocessing.OperationsImages import *
import argparse


def main(args):

    workDirsManager  = WorkDirsManager(args.basedir)
    BaseDataPath     = workDirsManager.getNameBaseDataPath()
    InputImagesPath  = workDirsManager.getNameExistPath(BaseDataPath, 'RawImages')
    InputMasksPath   = workDirsManager.getNameExistPath(BaseDataPath, 'RawMasks')
    OutputImagesPath = workDirsManager.getNameNewPath(BaseDataPath, 'RawImages')
    OutputMasksPath  = workDirsManager.getNameNewPath(BaseDataPath, 'RawMasks')

    # Get the file list:
    nameImagesFiles = '*.dcm'
    nameMasksFiles  = '*.dcm'

    listImagesFiles = findFilesDir(InputImagesPath, nameImagesFiles)
    listMasksFiles  = findFilesDir(InputMasksPath,  nameMasksFiles)

    nbImagesFiles   = len(listImagesFiles)
    nbMasksFiles    = len(listMasksFiles)

    outFilesExtension = '.nii'


    # Run checkers
    if (nbImagesFiles == 0):
        message = "num CTs Images found in dir \'%s\'" %(OrigImagesPath)
        CatchErrorException(message)
    if (nbImagesFiles != nbMasksFiles):
        message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbMasksFiles)
        CatchErrorException(message)


    if isExistdir(joinpathnames(BaseDataPath, 'RawAddMasks')):
        isExistsAddMasks = True

        InputAddMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, 'RawAddMasks')
        OutputAddMasksPath = workDirsManager.getNameNewPath(BaseDataPath, 'RawAddMasks')

        nameAddMasksFiles = '*.dcm'
        listAddMasksFiles = findFilesDir(InputAddMasksPath, nameAddMasksFiles)
        nbAddMasksFiles   = len(listAddMasksFiles)

        if (nbImagesFiles != nbAddMasksFiles):
            message = "num CTs Images %i not equal to num additional Masks %i" % (nbImagesFiles, nbAddMasksFiles)
            CatchErrorException(message)
    else:
        isExistsAddMasks = False



    for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

        print('\'%s\'...' %(imagesFile))

        images_array = FileReader.getImageArray(imagesFile)
        masks_array  = FileReader.getImageArray(masksFile)

        if isExistsAddMasks:
            add_masksFile   = listAddMasksFiles[i]
            add_masks_array = FileReader.getImageArray(add_masksFile)


        print("Saving images in nifty '.nii' format of final dimensions: %s..." %(str(images_array.shape)))

        out_imagesFilename = joinpathnames(OutputImagesPath, filenamenoextension(imagesFile) + outFilesExtension)
        out_masksFilename  = joinpathnames(OutputMasksPath,  filenamenoextension(masksFile)  + outFilesExtension)

        FileReader.writeImageArray(out_imagesFilename, images_array)
        FileReader.writeImageArray(out_masksFilename,  masks_array )

        if isExistsAddMasks:
            out_masksFilename = joinpathnames(OutputAddMasksPath, filenamenoextension(add_masksFile) + outFilesExtension)

            FileReader.writeImageArray(out_masksFilename, add_masks_array)
    #endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)