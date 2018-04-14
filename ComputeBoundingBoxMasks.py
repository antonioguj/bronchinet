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
from Preprocessing.BoundingBoxMasks import *
from Preprocessing.OperationsImages import *
import argparse



def main(args):

    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath    = workDirsManager.getNameDataPath(args.typedata)
    RawImagesPath   = workDirsManager.getNameNewPath(BaseDataPath, 'RawImages')
    RawMasksPath    = workDirsManager.getNameNewPath(BaseDataPath, 'RawAddMasks')

    dict_masks_boundingBoxes = {}

    maxSize_boundingBox = (0, 0, 0)

    # Get the file list:
    nameImagesFiles = '*.dcm'
    nameMasksFiles  = '*.dcm'

    listImagesFiles = findFilesDir(RawImagesPath, nameImagesFiles)
    listMasksFiles  = findFilesDir(RawMasksPath,  nameMasksFiles )

    nbImagesFiles   = len(listImagesFiles)
    nbMasksFiles    = len(listMasksFiles)

    # Run checkers
    if (nbImagesFiles == 0):
        message = "num CTs Images found in dir \'%s\'" %(RawImagesPath)
        CatchErrorException(message)
    if (nbImagesFiles != nbMasksFiles):
        message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbMasksFiles)
        CatchErrorException(message)


    for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

        print('\'%s\'...' % (imagesFile))

        images_array = FileReader.getImageArray(imagesFile)
        masks_array  = FileReader.getImageArray(masksFile)


        boundingBox = BoundingBoxMasks.compute_with_border_effects(masks_array)

        size_boundingBox = BoundingBoxMasks.computeSizeBoundingBox(boundingBox)

        maxSize_boundingBox = BoundingBoxMasks.computeMaxSizeBoundingBox(size_boundingBox, maxSize_boundingBox)


        # Compute new bounding-box that fits all input images processed
        processed_boundingBox = BoundingBoxMasks.computeCenteredBoundingBox(boundingBox, args.cropSizeBoundingBox, images_array.shape)

        size_processed_boundingBox = BoundingBoxMasks.computeSizeBoundingBox(processed_boundingBox)

        dict_masks_boundingBoxes[filenamenoextension(imagesFile)] = processed_boundingBox

        print("computed bounding-box: %s; processed bounding-box: %s; of size: %s" %(boundingBox, processed_boundingBox, size_processed_boundingBox))


        if (size_processed_boundingBox[1:3] != args.cropSizeBoundingBox):
            message = "size processed bounding-box not correct: %s != %s" % (size_processed_boundingBox, args.cropSizeBoundingBox)
            CatchErrorException(message)

        if BoundingBoxMasks.isBoundingBoxContained(boundingBox, processed_boundingBox):
            message = "Processed bounding-box: %s smaller than original one: %s..." % (processed_boundingBox, boundingBox)
            CatchWarningException(message)
    #endfor


    print("max size bounding-box found: %s; set size bounding-box %s" %(maxSize_boundingBox, args.cropSizeBoundingBox))

    # Save dictionary in csv file
    nameoutfile = joinpathnames(BaseDataPath, "boundBoxesMasks.npy")
    saveDictionary(nameoutfile, dict_masks_boundingBoxes)
    nameoutfile = joinpathnames(BaseDataPath, "boundBoxesMasks.csv")
    saveDictionary_csv(nameoutfile, dict_masks_boundingBoxes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--typedata', default=TYPEDATA)
    parser.add_argument('--cropSizeBoundingBox', type=str2tuplefloat, default=CROPSIZEBOUNDINGBOX)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)