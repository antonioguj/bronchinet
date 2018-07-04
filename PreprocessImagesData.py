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
from Preprocessing.SlidingWindowPlusTransformImages import *
from Preprocessing.TransformationImages import *
import argparse


def main(args):

    workDirsManager     = WorkDirsManager(args.basedir)
    BaseDataPath        = workDirsManager.getNameBaseDataPath()
    OriginImagesPath    = workDirsManager.getNameExistPath(BaseDataPath, 'RawImages')
    OriginMasksPath     = workDirsManager.getNameExistPath(BaseDataPath, 'RawMasks')
    ProcessedImagesPath = workDirsManager.getNameNewPath(BaseDataPath, 'ProcImagesData')
    ProcessedMasksPath  = workDirsManager.getNameNewPath(BaseDataPath, 'ProcMasksData')


    # Get the file list:
    nameImagesFiles = '*.dcm'
    nameMasksFiles  = '*.dcm'

    listImagesFiles = findFilesDir(OriginImagesPath, nameImagesFiles)
    listMasksFiles  = findFilesDir(OriginMasksPath,  nameMasksFiles)

    nbImagesFiles   = len(listImagesFiles)
    nbMasksFiles    = len(listMasksFiles)

    tempNameProcImagesFiles = 'images-%0.2i_dim'
    tempNameProcMasksFiles  = 'masks-%0.2i_dim'

    outFilesExtension = getFileExtension(FORMATINOUTDATA)


    # Run checkers
    if (nbImagesFiles == 0):
        message = "num CTs Images found in dir \'%s\'" %(OriginImagesPath)
        CatchErrorException(message)
    if (nbImagesFiles != nbMasksFiles):
        message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbMasksFiles)
        CatchErrorException(message)


    if (args.confineMasksToLungs):

        OriginAddMasksPath = workDirsManager.getNameExistPath(BaseDataPath, 'RawAddMasks')

        nameAddMasksFiles = '*.dcm'
        listAddMasksFiles = findFilesDir(OriginAddMasksPath, nameAddMasksFiles)
        nbAddMasksFiles   = len(listAddMasksFiles)

        if (nbImagesFiles != nbAddMasksFiles):
            message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbAddMasksFiles)
            CatchErrorException(message)

    if (args.cropImages):
        namefile_dict = joinpathnames(BaseDataPath, "boundBoxesMasks.npy")
        dict_masks_boundingBoxes = readDictionary(namefile_dict)



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

            exclude_masksFile   = listAddMasksFiles[i]
            exclude_masks_array = FileReader.getImageArray(exclude_masksFile)

            masks_array = ExclusionMasks.compute(masks_array, exclude_masks_array)


        if (args.reduceSizeImages):
            print("Reduce resolution of images to size...")

            if isSmallerTuple(images_array.shape[-2:], args.sizeReducedImages):
                message = "New reduced size: %s, smaller than size of original images: %s..." %(images_array.shape[-2:], args.sizeReducedImages)
                CatchErrorException(message)

            images_array = ResizeImages.compute2D(images_array, args.sizeReducedImages)
            masks_array  = ResizeImages.compute2D(masks_array,  args.sizeReducedImages, isMasks=True)

            print("Final dimensions: %s..." %(args.sizeReducedImages, images_array.shape))


        if (args.cropImages):
            print("Crop images to bounding-box...")

            crop_boundingBox = dict_masks_boundingBoxes[filenamenoextension(imagesFile)]

            images_array = CropImages.compute3D(images_array, crop_boundingBox)
            masks_array  = CropImages.compute3D(masks_array,  crop_boundingBox)

            print("Final dimensions: %s..." %(crop_boundingBox, images_array.shape))


        if (args.checkBalanceClasses):
            if (args.confineMasksToLungs):

                (num_pos_class, num_neg_class) = BalanceClassesCTs.compute_excludeAreas(masks_array)
            else:
                (num_pos_class, num_neg_class) = BalanceClassesCTs.compute(masks_array)

            print("Balance classes negative / positive: %s..." %(num_neg_class/num_pos_class))


        if (args.createImagesBatches):
            if (args.slidingWindowImages):
                print("Generate batches images by Sliding-window: size: %s; Overlap: %s..." %(IMAGES_DIMS_Z_X_Y, args.prop_overlap_Z_X_Y))

                slidingWindowImagesGenerator = SlidingWindowImages3D(IMAGES_DIMS_Z_X_Y, args.prop_overlap_Z_X_Y, images_array.shape)

                (images_array, masks_array) = slidingWindowImagesGenerator.compute_images_array_all(images_array, masks_array=masks_array)

                print("Final dimensions: %s..." %(images_array.shape))
            else:
                print("Generate batches images by Slicing volumes: size: %s..." %(IMAGES_DIMS_Z_X_Y))

                slicingImagesGenerator = SlicingImages3D(IMAGES_DIMS_Z_X_Y, images_array.shape)

                (images_array, masks_array) = slicingImagesGenerator.compute_images_array_all(images_array, masks_array=masks_array)

                print("Final dimensions: %s..." %(images_array.shape))


            if (args.elasticDeformationImages):
                print("Generate transformed images by elastic deformations, of type '%s': size: %s..." %(TYPEELASTICDEFORMATION, IMAGES_DIMS_Z_X_Y))

                if (TYPEELASTICDEFORMATION=='pixelwise'):
                    elasticDeformationImagesGenerator = ElasticDeformationPixelwiseImages3D(IMAGES_DIMS_Z_X_Y)
                else: #TYPEELASTICDEFORMATION=='gridwise':
                    elasticDeformationImagesGenerator = ElasticDeformationGridwiseImages3D(IMAGES_DIMS_Z_X_Y)

                (images_array, masks_array) = elasticDeformationImagesGenerator.compute_images_array_all(images_array, masks_array=masks_array)

                print("Final dimensions: %s..." %(images_array.shape))

            elif (args.transformationImages):
                print("Generate random 3D transformations of images: size: %s..." %(IMAGES_DIMS_Z_X_Y))

                transformImagesGenerator = TransformationImages3D(IMAGES_DIMS_Z_X_Y,
                                                                  rotation_XY_range=ROTATION_XY_RANGE,
                                                                  rotation_XZ_range=ROTATION_XZ_RANGE,
                                                                  rotation_YZ_range=ROTATION_YZ_RANGE,
                                                                  height_shift_range=HEIGHT_SHIFT_RANGE,
                                                                  width_shift_range=WIDTH_SHIFT_RANGE,
                                                                  depth_shift_range=DEPTH_SHIFT_RANGE,
                                                                  horizontal_flip=HORIZONTAL_FLIP,
                                                                  vertical_flip=VERTICAL_FLIP,
                                                                  depthZ_flip=DEPTHZ_FLIP)

                (images_array, masks_array) = transformImagesGenerator.compute_images_array_all(images_array, masks_array=masks_array)

                print("Final dimensions: %s..." %(images_array.shape))


        # Save processed data for training networks
        print("Saving processed data, with dims: %s..." %(tuple2str(images_array.shape)))

        out_imagesFilename = joinpathnames(ProcessedImagesPath, tempNameProcImagesFiles%(i) + tuple2str(images_array.shape) + outFilesExtension)
        out_masksFilename  = joinpathnames(ProcessedMasksPath,  tempNameProcMasksFiles%(i)  + tuple2str(masks_array.shape)  + outFilesExtension)

        FileReader.writeImageArray(out_imagesFilename, images_array)
        FileReader.writeImageArray(out_masksFilename,  masks_array )


        if (args.saveVisualProcessData):
            if (args.createImagesBatches):
                print("Saving processed data in image format for visualization...")

                for j, (batch_images_array, batch_masks_array) in enumerate(zip(images_array, masks_array)):

                    out_imagesFilename = joinpathnames(ProcessedImagesPath, tempNameProcImagesFiles%(i) + tuple2str(images_array.shape[1:]) + '_batch%i'%(j) +'.nii.gz')
                    out_masksFilename  = joinpathnames(ProcessedMasksPath,  tempNameProcMasksFiles%(i) +  tuple2str(masks_array.shape[1:])  + '_batch%i'%(j) +'.nii.gz')

                    FileReader.writeImageArray(out_imagesFilename, batch_images_array)
                    FileReader.writeImageArray(out_masksFilename,  batch_masks_array )
                #endfor
            else:
                out_imagesFilename = joinpathnames(ProcessedImagesPath, tempNameProcImagesFiles%(i) + tuple2str(images_array.shape) +'.nii.gz')
                out_masksFilename  = joinpathnames(ProcessedMasksPath,  tempNameProcMasksFiles%(i)  + tuple2str(masks_array.shape)  +'.nii.gz')

                FileReader.writeImageArray(out_imagesFilename, images_array)
                FileReader.writeImageArray(out_masksFilename, masks_array)
    #endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--confineMasksToLungs', default=CONFINEMASKSTOLUNGS)
    parser.add_argument('--reduceSizeImages', type=str2bool, default=REDUCESIZEIMAGES)
    parser.add_argument('--sizeReducedImages', type=str2bool, default=SIZEREDUCEDIMAGES)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--cropSizeBoundingBox', type=str2tupleint, default=CROPSIZEBOUNDINGBOX)
    parser.add_argument('--checkBalanceClasses', type=str2bool, default=CHECKBALANCECLASSES)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=TRANSFORMATIONIMAGES)
    parser.add_argument('--elasticDeformationImages', type=str2bool, default=ELASTICDEFORMATIONIMAGES)
    parser.add_argument('--createImagesBatches', type=str2bool, default=CREATEIMAGESBATCHES)
    parser.add_argument('--saveVisualProcessData', type=str2bool, default=SAVEVISUALPROCESSDATA)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)