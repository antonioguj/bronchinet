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
from Preprocessing.BalanceClassesCTs import *
from Preprocessing.OperationsImages import *
from Preprocessing.SlidingWindowImages import *
import argparse


def main(args):

    workDirsManager = WorkDirsManager(args.basedir)
    BaseDataPath    = workDirsManager.getNameDataPath(args.typedata)
    RawImagesPath   = workDirsManager.getNameExistPath(BaseDataPath, 'ProcImages')
    RawMasksPath    = workDirsManager.getNameExistPath(BaseDataPath, 'ProcMasks')
    ProcessDataPath = workDirsManager.getNameNewPath(BaseDataPath, 'ProcInputData')

    # Get the file list:
    nameImagesFiles = '*.nii'
    nameMasksFiles  = '*.nii'

    listImagesFiles = findFilesDir(RawImagesPath, nameImagesFiles)
    listMasksFiles  = findFilesDir(RawMasksPath,  nameMasksFiles)

    nbImagesFiles   = len(listImagesFiles)
    nbMasksFiles    = len(listMasksFiles)

    tempNameProcImagesFiles = 'images-%0.2i_dim'
    tempNameProcMasksFiles  = 'masks-%0.2i_dim'

    # Run checkers
    if (nbImagesFiles == 0):
        message = "num CTs Images found in dir \'%s\'" %(RawImagesPath)
        CatchErrorException(message)
    if (nbImagesFiles != nbMasksFiles):
        message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbMasksFiles)
        CatchErrorException(message)


    if (args.confineMasksToLungs):

        AddMasksPath = workDirsManager.getNameExistPath(BaseDataPath, 'ProcAddMasks')

        nameAddMasksFiles = '*.nii'
        listAddMasksFiles = findFilesDir(AddMasksPath, nameAddMasksFiles)
        nbAddMasksFiles   = len(listAddMasksFiles)

        if (nbImagesFiles != nbAddMasksFiles):
            message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbAddMasksFiles)
            CatchErrorException(message)



    for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

        print('\'%s\'...' %(imagesFile))

        images_array = FileReader.getImageArray(imagesFile)
        masks_array  = FileReader.getImageArray(masksFile)

        if (images_array.shape != masks_array.shape):
            message = "size of images: %s, not equal to size of masks: %s..." %(images_array.shape, masks_array.shape)
            CatchErrorException(message)
        print("Original image of size: %s..." %(str(images_array.shape)))


        if (args.multiClassCase):
            # Check the correct multilabels in "masks_array"
            if not checkCorrectNumClassesInMasks(masks_array, args.numClassesMasks):
                message = "In multiclass case, found wrong values in masks array: %s..." %(np.unique(masks_array))
                CatchErrorException(message)
        else:
            # Turn to binary masks (0, 1)
            masks_array = processBinaryMasks(masks_array)


        if (args.confineMasksToLungs):
            print("Confine masks to exclude the area outside the lungs...")

            exclude_masks_array = FileReader.getImageArray(listAddMasksFiles[i])

            masks_array = ExclusionMasks.compute(masks_array, exclude_masks_array)


        if (args.checkBalanceClasses):
            if (args.confineMasksToLungs):

                (num_pos_class, num_neg_class) = BalanceClassesCTs.compute_excludeAreas(masks_array)
            else:
                (num_pos_class, num_neg_class) = BalanceClassesCTs.compute(masks_array)

            print("Balance classes negative / positive: %s..." %(num_neg_class/num_pos_class))


        if (args.createImagesBatches):

            if (args.slidingWindowImages):

                slidingWindowImagesGenerator = SlidingWindowImages3D(images_array.shape, IMAGES_DIMS_Z_X_Y, args.prop_overlap_Z_X_Y)

                images_array = slidingWindowImagesGenerator.compute_images_array_all(images_array)
                masks_array  = slidingWindowImagesGenerator.compute_images_array_all(masks_array)

                print("Generate batches images by Sliding-window: size: %s; Overlap: %s. Final dimensions: %s..." %(IMAGES_DIMS_Z_X_Y, args.prop_overlap_Z_X_Y, images_array.shape))
            else:

                slicingImagesGenerator = SlicingImages3D(images_array.shape, IMAGES_DIMS_Z_X_Y)

                images_array = slicingImagesGenerator.compute_images_array_all(images_array)
                masks_array  = slicingImagesGenerator.compute_images_array_all(masks_array)

                print("Generate batches images by Slicing volumes: size: %s. Final dimensions: %s..." %(IMAGES_DIMS_Z_X_Y, images_array.shape))


        # Save processed data for training networks
        print("Saving processed data, with dims: %s..." %(tuple2str(images_array.shape)))

        out_imagesFilename = joinpathnames(ProcessDataPath, tempNameProcImagesFiles%(i) + tuple2str(images_array.shape) + getFileExtension(FORMATINOUTDATA))
        out_masksFilename  = joinpathnames(ProcessDataPath, tempNameProcMasksFiles%(i)  + tuple2str(masks_array.shape)  + getFileExtension(FORMATINOUTDATA))

        FileReader.writeImageArray(out_imagesFilename, images_array)
        FileReader.writeImageArray(out_masksFilename,  masks_array )


        if (args.saveVisualProcessData):
            if (args.createImagesBatches):
                print("Saving processed data in image format for visualization...")

                for j, (batch_images_array, batch_masks_array) in enumerate(zip(images_array, masks_array)):

                    out_imagesFilename = joinpathnames(ProcessDataPath, tempNameProcImagesFiles%(i) + tuple2str(images_array.shape[1:]) + '_batch%i'%(j) +'.nii')
                    out_masksFilename  = joinpathnames(ProcessDataPath, tempNameProcMasksFiles%(i) +  tuple2str(masks_array.shape[1:])  + '_batch%i'%(j) +'.nii')

                    FileReader.writeImageArray(out_imagesFilename, batch_images_array)
                    FileReader.writeImageArray(out_masksFilename,  batch_masks_array )
                #endfor
            else:
                out_imagesFilename = joinpathnames(ProcessDataPath, tempNameProcImagesFiles%(i) + tuple2str(images_array.shape) +'.nii')
                out_masksFilename  = joinpathnames(ProcessDataPath, tempNameProcMasksFiles%(i)  + tuple2str(masks_array.shape)  +'.nii')

                FileReader.writeImageArray(out_imagesFilename, images_array)
                FileReader.writeImageArray(out_masksFilename, masks_array)
    #endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--typedata', default=TYPEDATA)
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--confineMasksToLungs', default=CONFINEMASKSTOLUNGS)
    parser.add_argument('--checkBalanceClasses', type=str2bool, default=CHECKBALANCECLASSES)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--createImagesBatches', type=str2bool, default=CREATEIMAGESBATCHES)
    parser.add_argument('--saveVisualProcessData', type=str2bool, default=SAVEVISUALPROCESSDATA)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)