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
from CommonUtil.FileReaders import *
from CommonUtil.FunctionsUtil import *
from CommonUtil.WorkDirsManager import WorkDirsManager
from Preprocessing.BoundingBoxMasks import *
from Preprocessing.OperationsImages import *

BORDER_EFFECTS  = (0, 0, 0)


def main():

    workDirsManager = WorkDirsManager(BASEDIR)
    BaseDataPath    = workDirsManager.getNameDataPath(TYPEDATA)
    ImagesPath      = workDirsManager.getNameNewPath(BaseDataPath, 'RawImages')
    MasksPath       = workDirsManager.getNameNewPath(BaseDataPath, 'RawMasks')
    LungsMasksPath  = workDirsManager.getNameNewPath(BaseDataPath, 'RawLungs')
    CropImagesPath  = workDirsManager.getNameNewPath(BaseDataPath, 'CroppedImages')

    dict_masks_boundingBoxes = {}

    maxSize_boundingBox = (0, 0, 0)


    # Get the file list:
    listImagesFiles     = findFilesDir(ImagesPath     + '/*.dcm')
    listLungsMasksFiles = findFilesDir(LungsMasksPath + '/*.dcm')

    for imagesFile, masksFile in zip(listImagesFiles, listLungsMasksFiles):

        print('\'%s\'...' % (imagesFile))

        images_array = FileReader.getImageArray(imagesFile)
        masks_array  = FileReader.getImageArray(masksFile)


        boundingBox = BoundingBoxMasks.compute_with_border_effects(masks_array, BORDER_EFFECTS)

        size_boundingBox = BoundingBoxMasks.computeSizeBoundingBox(boundingBox)

        maxSize_boundingBox = BoundingBoxMasks.computeMaxSizeBoundingBox(size_boundingBox, maxSize_boundingBox)


        # Compute new bounding-box that fits all input images processed
        processed_boundingBox = BoundingBoxMasks.computeCenteredBoundingBox(boundingBox, CROPSIZEBOUNDINGBOX, images_array.shape)

        size_processed_boundingBox = BoundingBoxMasks.computeSizeBoundingBox(processed_boundingBox)

        cropped_images_array = CropImages.compute3D(images_array, processed_boundingBox)

        dict_masks_boundingBoxes[basename(imagesFile)] = processed_boundingBox


        print("computed bounding-box: %s; processed bounding-box: %s; of size: %s" %(boundingBox, processed_boundingBox, size_processed_boundingBox))

        if (size_processed_boundingBox[1:3] != CROPSIZEBOUNDINGBOX):
            message = "size processed bounding-box not correct: %s != %s" % (size_processed_boundingBox, CROPSIZEBOUNDINGBOX)
            CatchErrorException(message)

        if BoundingBoxMasks.isBoundingBoxContained(boundingBox, processed_boundingBox):
            message = "Processed bounding-box: %s smaller than original one: %s..." % (processed_boundingBox, boundingBox)
            CatchWarningException(message)


        nameoutfile = joinpathnames(CropImagesPath, basename(imagesFile).replace('.dcm','.nii'))
        FileReader.writeImageArray(nameoutfile, cropped_images_array)
    #endfor


    print("max size bounding-box found: %s; set size bounding-box %s" %(maxSize_boundingBox, CROPSIZEBOUNDINGBOX))

    # Save dictionary in csv file
    nameoutfile = joinpathnames(BaseDataPath, "boundBoxesMasks.npy")
    saveDictionary(nameoutfile, dict_masks_boundingBoxes)
    nameoutfile = joinpathnames(BaseDataPath, "boundBoxesMasks.csv")
    saveDictionary_csv(nameoutfile, dict_masks_boundingBoxes)


if __name__ == "__main__":
    main()