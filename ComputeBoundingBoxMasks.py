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

    # ---------- SETTINGS ----------
    dict_masks_boundingBoxes = {}

    max_size_boundingBox = (0, 0, 0)
    min_size_boundingBox = (1.0e+03, 1.0e+03, 1.0e+03)

    nameRawImagesRelPath = 'RawImages'
    nameRawMasksRelPath  = 'ProcAllMasks'

    # Get the file list:
    nameImagesFiles = '*.dcm'
    nameMasksFiles  = '*.nii.gz'

    nameBoundingBoxesMasksNPY = 'boundBoxesMasks.npy'
    nameBoundingBoxesMasksCSV = 'boundBoxesMasks.csv'
    # ---------- SETTINGS ----------


    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath    = workDirsManager.getNameBaseDataPath()
    RawImagesPath   = workDirsManager.getNameExistPath(BaseDataPath, nameRawImagesRelPath)
    RawMasksPath    = workDirsManager.getNameExistPath(BaseDataPath, nameRawMasksRelPath )

    listImagesFiles = findFilesDir(RawImagesPath, nameImagesFiles)
    listMasksFiles  = findFilesDir(RawMasksPath,  nameMasksFiles )

    nbImagesFiles   = len(listImagesFiles)
    nbMasksFiles    = len(listMasksFiles)

    # Run checkers
    if (nbImagesFiles == 0):
        message = "0 Images found in dir \'%s\'" %(RawImagesPath)
        CatchErrorException(message)
    if (nbImagesFiles != nbMasksFiles):
        message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbMasksFiles)
        CatchErrorException(message)



    for images_file, masks_file in zip(listImagesFiles, listMasksFiles):

        print('\'%s\'...' % (images_file))

        images_array = FileReader.getImageArray(images_file)
        masks_array  = FileReader.getImageArray(masks_file)

        if (args.invertImageAxial):
            images_array = FlippingImages.compute(images_array, axis=0)
            masks_array  = FlippingImages.compute(masks_array,  axis=0)

        if (images_array.shape != masks_array.shape):
            message = "size of Images and Masks not equal: %s != %s" % (images_array.shape, masks_array.shape)
            CatchErrorException(message)
        print("Original image of size: %s..." % (str(images_array.shape)))


        boundingBox = BoundingBoxMasks.compute_with_border_effects(masks_array, voxels_buffer_border=VOXELSBUFFERBORDER)

        size_boundingBox = BoundingBoxMasks.compute_size_bounding_box(boundingBox)

        max_size_boundingBox = BoundingBoxMasks.compute_max_size_bounding_box(size_boundingBox, max_size_boundingBox)

        print("computed bounding-box: %s, of size: %s" %(boundingBox, size_boundingBox))

        # if (size_boundingBox != images_array.shape):
        #     message = "booundary-box size: %s, not equal to size of images: %s..." %(size_boundingBox, images_array.shape)
        #     CatchErrorException(message)
        # print("Boundary-box size: %s..." % (str(size_boundingBox)))
        #
        # max_size_boundingBox = BoundingBoxMasks.compute_max_size_bounding_box(size_boundingBox, max_size_boundingBox)
        # min_size_boundingBox = BoundingBoxMasks.compute_min_size_bounding_box(size_boundingBox, min_size_boundingBox)
        #
        # print("computed bounding-box: %s..." %(str(boundingBox)))


        # Compute new bounding-box that fits all input images processed
        processed_boundingBox = BoundingBoxMasks.compute_centered_bounding_box(boundingBox, args.cropSizeBoundingBox, images_array.shape)

        size_processed_boundingBox = BoundingBoxMasks.compute_size_bounding_box(processed_boundingBox)

        dict_masks_boundingBoxes[filenamenoextension(images_file)] = processed_boundingBox

        print("processed bounding-box: %s, of size: %s" %(processed_boundingBox, size_processed_boundingBox))


        if (size_processed_boundingBox[1:3] != args.cropSizeBoundingBox):
            message = "size processed bounding-box not correct: %s != %s" % (size_processed_boundingBox, args.cropSizeBoundingBox)
            CatchErrorException(message)

        # if (size_processed_boundingBox != images_array.shape):
        #     message = "size processed bounding-box not correct: %s != %s" % (size_processed_boundingBox, images_array.shape)
        #     CatchErrorException(message)
        #
        if BoundingBoxMasks.is_bounding_box_contained(boundingBox, processed_boundingBox):
            message = "Processed bounding-box: %s smaller than original one: %s..." % (processed_boundingBox, boundingBox)
            CatchWarningException(message)
    #endfor


    print("max size bounding-box found: %s; set size bounding-box %s" %(max_size_boundingBox, args.cropSizeBoundingBox))


    # Save dictionary in csv file
    nameoutfile = joinpathnames(BaseDataPath, nameBoundingBoxesMasksNPY)
    saveDictionary(nameoutfile, dict_masks_boundingBoxes)
    nameoutfile = joinpathnames(BaseDataPath, nameBoundingBoxesMasksCSV)
    saveDictionary_csv(nameoutfile, dict_masks_boundingBoxes)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--invertImageAxial', type=str2bool, default=INVERTIMAGEAXIAL)
    parser.add_argument('--voxelsBufferBorder', type=str2tuplefloat, default=VOXELSBUFFERBORDER)
    parser.add_argument('--cropSizeBoundingBox', type=str2tuplefloat, default=CROPSIZEBOUNDINGBOX)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)