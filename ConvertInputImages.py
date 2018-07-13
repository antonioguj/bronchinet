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
    OutputImagesPath = workDirsManager.getNameNewPath(BaseDataPath, 'ProcImages')
    OutputMasksPath  = workDirsManager.getNameNewPath(BaseDataPath, 'ProcMasks')

    # Get the file list:
    nameImagesFiles = '*.dcm'
    nameMasksFiles  = '*.dcm'

    listImagesFiles = findFilesDir(InputImagesPath, nameImagesFiles)
    listMasksFiles  = findFilesDir(InputMasksPath,  nameMasksFiles)

    nbImagesFiles   = len(listImagesFiles)
    nbMasksFiles    = len(listMasksFiles)


    # Run checkers
    if (nbImagesFiles == 0):
        message = "num CTs Images found in dir \'%s\'" %(InputImagesPath)
        CatchErrorException(message)
    if (nbImagesFiles != nbMasksFiles):
        message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbMasksFiles)
        CatchErrorException(message)


    if isExistdir(joinpathnames(BaseDataPath, 'RawAddMasks')):
        isExistsAddMasks = True

        InputAddMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, 'RawAddMasks')
        OutputAddMasksPath = workDirsManager.getNameNewPath(BaseDataPath, 'ProcAddMasks')

        nameAddMasksFiles = '*.dcm'
        listAddMasksFiles = findFilesDir(InputAddMasksPath, nameAddMasksFiles)
        nbAddMasksFiles   = len(listAddMasksFiles)

        if (nbImagesFiles != nbAddMasksFiles):
            message = "num CTs Images %i not equal to num additional Masks %i" % (nbImagesFiles, nbAddMasksFiles)
            CatchErrorException(message)
    else:
        isExistsAddMasks = False



    for i, (images_file, masks_file) in enumerate(zip(listImagesFiles, listMasksFiles)):

        print('\'%s\'...' %(images_file))

        images_array = FileReader.getImageArray(images_file)
        masks_array  = FileReader.getImageArray(masks_file)

        if (args.invertImageAxial):
            images_array = FlippingImages.compute(images_array, axis=0)
            masks_array  = FlippingImages.compute(masks_array,  axis=0)


        if isExistsAddMasks:
            add_masks_file   = listAddMasksFiles[i]
            add_masks_array = FileReader.getImageArray(add_masks_file)

            if (args.invertImageAxial):
                add_masks_array = FlippingImages.compute(add_masks_array, axis=0)


        print("Saving images in nifty '.nii' format of final dimensions: %s..." %(str(images_array.shape)))

        out_images_filename = joinpathnames(OutputImagesPath, filenamenoextension(images_file) + '.nii.gz')
        out_masks_filename  = joinpathnames(OutputMasksPath,  filenamenoextension(masks_file)  + '.nii.gz')

        FileReader.writeImageArray(out_images_filename, images_array.astype(FORMATIMAGEDATA))
        FileReader.writeImageArray(out_masks_filename,  masks_array .astype(FORMATIMAGEDATA))

        if isExistsAddMasks:
            out_masks_filename = joinpathnames(OutputAddMasksPath, filenamenoextension(add_masks_file) + '.nii.gz')

            FileReader.writeImageArray(out_masks_filename, add_masks_array.astype(FORMATIMAGEDATA))
    #endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--invertImageAxial', type=str2bool, default=INVERTIMAGEAXIAL)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)