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
from Preprocessing.BalanceClassesMasks import *
from Preprocessing.OperationsImages import *
from Preprocessing.SlidingWindowImages import *
from Preprocessing.SlidingWindowPlusTransformImages import *
from Preprocessing.TransformationImages import *
import argparse



def main(args):

    # ---------- SETTINGS ----------
    nameOriginImagesRelPath    = 'ProcImages'
    nameOriginMasksRelPath     = 'ProcMasks'
    nameProcessedImagesRelPath = 'ProcImagesExperData'
    nameProcessedMasksRelPath  = 'ProcMasksExperData'
    nameExcludeMasksRelPath    = 'ProcAddMasks'

    # Get the file list:
    nameImagesFiles       = '*.nii.gz'
    nameMasksFiles        = '*.nii.gz'
    nameExcludeMasksFiles = '*lungs.nii.gz'

    nameBoundingBoxesMasks = 'boundBoxesMasks.npy'

    tempNameProcImagesFiles = 'images-%0.2i_dim%s' + getFileExtension(FORMATINOUTDATA)
    tempNameProcMasksFiles  = 'masks-%0.2i_dim%s'  + getFileExtension(FORMATINOUTDATA)
    # ---------- SETTINGS ----------


    workDirsManager     = WorkDirsManager(args.basedir)
    BaseDataPath        = workDirsManager.getNameBaseDataPath()
    OriginImagesPath    = workDirsManager.getNameExistPath(BaseDataPath, nameOriginImagesRelPath)
    OriginMasksPath     = workDirsManager.getNameExistPath(BaseDataPath, nameOriginMasksRelPath )
    ProcessedImagesPath = workDirsManager.getNameNewPath(BaseDataPath, nameProcessedImagesRelPath)
    ProcessedMasksPath  = workDirsManager.getNameNewPath(BaseDataPath, nameProcessedMasksRelPath )

    listImagesFiles = findFilesDir(OriginImagesPath, nameImagesFiles)
    listMasksFiles  = findFilesDir(OriginMasksPath,  nameMasksFiles)

    nbImagesFiles   = len(listImagesFiles)
    nbMasksFiles    = len(listMasksFiles)

    # Run checkers
    if (nbImagesFiles == 0):
        message = "num CTs Images found in dir \'%s\'" %(OriginImagesPath)
        CatchErrorException(message)
    if (nbImagesFiles != nbMasksFiles):
        message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbMasksFiles)
        CatchErrorException(message)

    if (args.confineMasksToLungs):

        ExcludeMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameExcludeMasksRelPath)

        listExcludeMasksFiles = findFilesDir(ExcludeMasksPath, nameExcludeMasksFiles)
        nbExcludeMasksFiles   = len(listExcludeMasksFiles)

        if (nbImagesFiles != nbExcludeMasksFiles):
            message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbExcludeMasksFiles)
            CatchErrorException(message)

    if (args.cropImages or args.extendSizeImages):

        namefile_dict = joinpathnames(BaseDataPath, nameBoundingBoxesMasks)
        dict_masks_boundingBoxes = readDictionary(namefile_dict)



    for i, (images_file, masks_file) in enumerate(zip(listImagesFiles, listMasksFiles)):

        print('\'%s\'...' %(images_file))

        images_array = FileReader.getImageArray(images_file)
        masks_array  = FileReader.getImageArray(masks_file)

        if (args.invertImageAxial):
            images_array = FlippingImages.compute(images_array, axis=0)
            masks_array  = FlippingImages.compute(masks_array,  axis=0)

        if (images_array.shape != masks_array.shape):
            message = "size of images: %s, not equal to size of masks: %s..." %(images_array.shape, masks_array.shape)
            CatchErrorException(message)
        print("Original image of size: %s..." %(str(images_array.shape)))


        if (args.multiClassCase):
            operationsMasks = OperationsMultiClassMasks(args.numClassesMasks)

            # Check the correct labels in "masks_array"
            if not operationsMasks.check_masks(masks_array):
                message = "found wrong labels in masks array: %s..." %(np.unique(masks_array))
                CatchErrorException(message)

            message = "MULTICLASS CASE STILL IN IMPLEMENTATION...EXIT"
            CatchErrorException(message)
        else:
            # Turn to binary masks (0, 1)
            masks_array = OperationsBinaryMasks.process_masks(masks_array)


        if (args.confineMasksToLungs):
            print("Confine masks to exclude the area outside the lungs...")

            exclude_masks_file  = listExcludeMasksFiles[i]
            exclude_masks_array = FileReader.getImageArray(exclude_masks_file)

            if (args.invertImageAxial):
                exclude_masks_array = FlippingImages.compute(exclude_masks_array, axis=0)

            masks_array = OperationsBinaryMasks.apply_mask_exclude_voxels(masks_array, exclude_masks_array)


        if (args.reduceSizeImages):
            print("Reduce resolution of images to size: %s..." %(args.sizeReducedImages))

            if isSmallerTuple(images_array.shape[-2:], args.sizeReducedImages):
                message = "New reduced size: %s, smaller than size of original images: %s..." %(images_array.shape[-2:], args.sizeReducedImages)
                CatchErrorException(message)

            images_array = ResizeImages.compute2D(images_array, args.sizeReducedImages)
            masks_array  = ResizeImages.compute2D(masks_array,  args.sizeReducedImages, isMasks=True)

            print("Final dimensions: %s..." %(images_array.shape))


        if (args.cropImages):
            crop_boundingBox = dict_masks_boundingBoxes[filenamenoextension(images_file)]

            print("Crop images to bounding-box: %s..." %(str(crop_boundingBox)))

            images_array = CropImages.compute3D(images_array, crop_boundingBox)
            masks_array  = CropImages.compute3D(masks_array,  crop_boundingBox)

            print("Final dimensions: %s..." %(str(images_array.shape)))


        if (args.extendSizeImages):
            print("Extend size images to constant size %s:..." %(str(CROPSIZEBOUNDINGBOX)))

            size_new_image = (images_array.shape[0], CROPSIZEBOUNDINGBOX[0], CROPSIZEBOUNDINGBOX[1])

            backgr_val_images_array = -1000
            backgr_val_masks_array  = -1 if args.confineMasksToLungs else 0

            crop_boundingBox = dict_masks_boundingBoxes[filenamenoextension(images_file)]

            sub_images_array = images_array
            sub_masks_array  = masks_array

            images_array = np.ndarray(size_new_image, dtype=sub_images_array.dtype)
            masks_array  = np.ndarray(size_new_image, dtype=sub_masks_array. dtype)

            images_array[:,:,:] = backgr_val_images_array
            masks_array [:,:,:] = backgr_val_masks_array

            images_array[crop_boundingBox[0][0]:crop_boundingBox[0][1],
                         crop_boundingBox[1][0]:crop_boundingBox[1][1],
                         crop_boundingBox[2][0]:crop_boundingBox[2][1]] = sub_images_array
            masks_array [crop_boundingBox[0][0]:crop_boundingBox[0][1],
                         crop_boundingBox[1][0]:crop_boundingBox[1][1],
                         crop_boundingBox[2][0]:crop_boundingBox[2][1]] = sub_masks_array


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

        out_images_filename = joinpathnames(ProcessedImagesPath, tempNameProcImagesFiles%(i+1, tuple2str(images_array.shape)))
        out_masks_filename  = joinpathnames(ProcessedMasksPath,  tempNameProcMasksFiles% (i+1, tuple2str(masks_array.shape )))

        FileReader.writeImageArray(out_images_filename, images_array)
        FileReader.writeImageArray(out_masks_filename,  masks_array )
    #endfor



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--invertImageAxial', type=str2bool, default=INVERTIMAGEAXIAL)
    parser.add_argument('--confineMasksToLungs', type=str2bool, default=CONFINEMASKSTOLUNGS)
    parser.add_argument('--reduceSizeImages', type=str2bool, default=REDUCESIZEIMAGES)
    parser.add_argument('--sizeReducedImages', type=str2bool, default=SIZEREDUCEDIMAGES)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--extendSizeImages', type=str2bool, default=EXTENDSIZEIMAGES)
    parser.add_argument('--checkBalanceClasses', type=str2bool, default=CHECKBALANCECLASSES)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=TRANSFORMATIONIMAGES)
    parser.add_argument('--elasticDeformationImages', type=str2bool, default=ELASTICDEFORMATIONIMAGES)
    parser.add_argument('--createImagesBatches', type=str2bool, default=CREATEIMAGESBATCHES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)