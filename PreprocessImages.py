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

    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath    = workDirsManager.getNameDataPath(args.typedata)
    RawImagesPath   = workDirsManager.getNameExistPath(BaseDataPath, 'RawImages')
    RawMasksPath    = workDirsManager.getNameExistPath(BaseDataPath, 'RawMasks')
    ProcImagesPath  = workDirsManager.getNameNewPath(BaseDataPath, 'ProcImages')
    ProcMasksPath   = workDirsManager.getNameNewPath(BaseDataPath, 'ProcMasks')

    # Get the file list:
    nameImagesFiles = '*.dcm'
    nameMasksFiles  = '*.dcm'

    listImagesFiles = findFilesDir(RawImagesPath, nameImagesFiles)
    listMasksFiles  = findFilesDir(RawMasksPath,  nameMasksFiles)

    nbImagesFiles   = len(listImagesFiles)
    nbMasksFiles    = len(listMasksFiles)

    # Run checkers
    if (nbImagesFiles == 0):
        message = "num CTs Images found in dir \'%s\'" %(RawImagesPath)
        CatchErrorException(message)
    if (nbImagesFiles != nbMasksFiles):
        message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbMasksFiles)
        CatchErrorException(message)


    if isExistdir(joinpathnames(BaseDataPath, 'RawAddMasks')):
        isExistsAddMasks = True

        RawAddMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, 'RawAddMasks')
        ProcAddMasksPath = workDirsManager.getNameNewPath(BaseDataPath, 'ProcAddMasks')

        nameAddMasksFiles = '*.dcm'
        listAddMasksFiles = findFilesDir(RawAddMasksPath, nameAddMasksFiles)
        nbAddMasksFiles   = len(listAddMasksFiles)

        if (nbImagesFiles != nbAddMasksFiles):
            message = "num CTs Images %i not equal to num additional Masks %i" % (nbImagesFiles, nbAddMasksFiles)
            CatchErrorException(message)


    if (args.cropImages):
        namefile_dict = joinpathnames(BaseDataPath, "boundBoxesMasks.npy")
        dict_masks_boundingBoxes = readDictionary(namefile_dict)


    for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

        print('\'%s\'...' %(imagesFile))

        images_array = FileReader.getImageArray(imagesFile)
        masks_array  = FileReader.getImageArray(masksFile)

        if isExistsAddMasks:
            add_masks_array = FileReader.getImageArray(listAddMasksFiles[i])


        if (args.reduceSizeImages):

            if isSmallerTuple(images_array.shape[-2:], args.sizeReducedImages):
                message = "New reduced size: %s, smaller than size of original images: %s..." %(images_array.shape[-2:], args.sizeReducedImages)
                CatchErrorException(message)

            images_array = ResizeImages.compute2D(images_array, args.sizeReducedImages)
            masks_array  = ResizeImages.compute2D(masks_array,  args.sizeReducedImages, isMasks=True)

            if isExistsAddMasks:
                add_masks_array = ResizeImages.compute2D(add_masks_array, args.sizeReducedImages, isMasks=True)

            print("Reduce resolution of images to size: %s. Final dimensions: %s..." %(args.sizeReducedImages, images_array.shape))


        if (args.cropImages):

            crop_boundingBox = dict_masks_boundingBoxes[filenamenoextension(imagesFile)]

            images_array = CropImages.compute3D(images_array, crop_boundingBox)
            masks_array  = CropImages.compute3D(masks_array,  crop_boundingBox)

            if isExistsAddMasks:
                add_masks_array = CropImages.compute3D(add_masks_array, crop_boundingBox)

            print("Crop images to bounding-box: %s. Final dimensions: %s..." %(crop_boundingBox, images_array.shape))


        print("Saving processed images of final dimensions: %s..." %(str(images_array.shape)))

        out_imagesFilename = joinpathnames(ProcImagesPath, filenamenoextension(imagesFile) +'.nii')
        out_masksFilename  = joinpathnames(ProcMasksPath,  filenamenoextension(masksFile)  +'.nii')

        FileReader.writeImageArray(out_imagesFilename, images_array)
        FileReader.writeImageArray(out_masksFilename,  masks_array )

        if isExistsAddMasks:
            out_masksFilename = joinpathnames(ProcAddMasksPath, filenamenoextension(masksFile) + '.nii')

            FileReader.writeImageArray(out_masksFilename, add_masks_array)
    #endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--typedata', default=TYPEDATA)
    parser.add_argument('--reduceSizeImages', type=str2bool, default=REDUCESIZEIMAGES)
    parser.add_argument('--sizeReducedImages', type=str2bool, default=SIZEREDUCEDIMAGES)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--cropSizeBoundingBox', type=str2tupleint, default=CROPSIZEBOUNDINGBOX)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)