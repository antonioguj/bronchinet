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
from Preprocessing.OperationsMasks import *
from Preprocessing.SlidingWindowImages import *
from Preprocessing.SlidingWindowPlusTransformImages import *
from Preprocessing.TransformationImages import *
import argparse



def main(args):

    # ---------- SETTINGS ----------
    nameInputImagesRelPath  = 'ProcImages'
    nameInputMasksRelPath   = 'ProcDistTrans'
    nameLungsMasksRelPath   = 'ProcAllMasks'
    nameOutputImagesRelPath = 'ProcImagesExperData'
    nameOutputMasksRelPath  = 'ProcDistTransExperData'

    # Get the file list:
    nameInputImagesFiles = '*.nii.gz'
    nameInputMasksFiles  = '*.nii.gz'
    nameLungsMasksFiles  = '*_lungs.nii.gz'

    nameBoundingBoxesMasks = 'boundBoxesMasks.npy'

    if (args.constructInputDataDLCST):
        tempNameProcImagesFiles  = 'images-%0.2i_img1_dim%s' + getFileExtension(FORMATINOUTDATA)
        tempNameProcMasksFiles   = 'masks-%0.2i_img1_dim%s'  + getFileExtension(FORMATINOUTDATA)
        tempNameProcImagesFiles_2= 'images-%0.2i_img2_dim%s' + getFileExtension(FORMATINOUTDATA)
        tempNameProcMasksFiles_2 = 'masks-%0.2i_img2_dim%s'  + getFileExtension(FORMATINOUTDATA)
    else:
        tempNameProcImagesFiles  = 'images-%0.2i_dim%s' + getFileExtension(FORMATINOUTDATA)
        tempNameProcMasksFiles   = 'masks-%0.2i_dim%s'  + getFileExtension(FORMATINOUTDATA)

    if (args.saveVisualizeProcData):
        nameVisualOutputRelPath = 'VisualizeProcExperData'

        tempNameVisualProcFiles = lambda filename: basename(filename).replace('.npz','.nii.gz')
    # ---------- SETTINGS ----------



    workDirsManager  = WorkDirsManager(args.basedir)
    BaseDataPath     = workDirsManager.getNameBaseDataPath()
    InputImagesPath  = workDirsManager.getNameExistPath(BaseDataPath, nameInputImagesRelPath)
    InputMasksPath   = workDirsManager.getNameExistPath(BaseDataPath, nameInputMasksRelPath )
    OutputImagesPath = workDirsManager.getNameNewPath  (BaseDataPath, nameOutputImagesRelPath)
    OutputMasksPath  = workDirsManager.getNameNewPath  (BaseDataPath, nameOutputMasksRelPath )

    if (args.saveVisualizeProcData):
        VisualOutputPath = workDirsManager.getNameNewPath(BaseDataPath, nameVisualOutputRelPath)

    listImagesFiles = findFilesDir(InputImagesPath, nameInputImagesFiles)
    listMasksFiles  = findFilesDir(InputMasksPath,  nameInputMasksFiles)

    nbImagesFiles   = len(listImagesFiles)
    nbMasksFiles    = len(listMasksFiles)

    # Run checkers
    if (nbImagesFiles == 0):
        message = "num CTs Images found in dir \'%s\'" %(InputImagesPath)
        CatchErrorException(message)
    if (nbImagesFiles != nbMasksFiles):
        message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbMasksFiles)
        CatchErrorException(message)


    if (args.masksToRegionInterest):

        LungsMasksPath = workDirsManager.getNameExistPath(BaseDataPath, nameLungsMasksRelPath)

        listLungsMasksFiles = findFilesDir(LungsMasksPath, nameLungsMasksFiles)
        nbLungsMasksFiles   = len(listLungsMasksFiles)

        if (nbImagesFiles != nbLungsMasksFiles):
            message = "num CTs Images %i not equal to num Lungs Masks %i" %(nbImagesFiles, nbLungsMasksFiles)
            CatchErrorException(message)

    if (args.cropImages or args.extendSizeImages):

        namefile_dict = joinpathnames(BaseDataPath, nameBoundingBoxesMasks)
        dict_masks_boundingBoxes = readDictionary(namefile_dict)



    # START ANALYSIS
    # ------------------------------
    print("-" * 30)
    print("Preprocessing...")
    print("-" * 30)

    for i, (images_file, masks_file) in enumerate(zip(listImagesFiles, listMasksFiles)):

        print('\'%s\'...' %(images_file))
        print('\'%s\'...' %(masks_file))

        images_array = FileReader.getImageArray(images_file)
        masks_array  = FileReader.getImageArray(masks_file)

        if (args.invertImageAxial):
            images_array = FlippingImages.compute(images_array, axis=0)
            masks_array  = FlippingImages.compute(masks_array,  axis=0)

        if (images_array.shape != masks_array.shape):
            message = "size of Images and Masks not equal: %s != %s" % (images_array.shape, masks_array.shape)
            CatchErrorException(message)
        print("Original image of size: %s..." %(str(images_array.shape)))



        if (args.isClassificationCase):
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



        if (args.masksToRegionInterest):
            print("Mask ground-truth to Region of Interest: exclude voxels outside ROI: lungs...")

            lungs_masks_file  = listLungsMasksFiles[i]
            lungs_masks_array = FileReader.getImageArray(lungs_masks_file)

            if (args.invertImageAxial):
                lungs_masks_array = FlippingImages.compute(lungs_masks_array, axis=0)

            masks_array = OperationsBinaryMasks.apply_mask_exclude_voxels(masks_array, lungs_masks_array)



        if (args.constructInputDataDLCST):
            print("Construct data for DLCST: crop images and set one batch per image...")

            bounding_box = dict_masks_boundingBoxes[filenamenoextension(images_file)]

            print("Crop images to bounding-box: %s..." % (str(bounding_box)))

            (bounding_box_left, bounding_box_right) = BoundingBoxMasks.compute_split_bounding_boxes(bounding_box, axis=2)

            images_array_2 = CropImages.compute3D(images_array, bounding_box_right)
            images_array   = CropImages.compute3D(images_array, bounding_box_left)

            masks_array_2 = CropImages.compute3D(masks_array, bounding_box_right)
            masks_array   = CropImages.compute3D(masks_array, bounding_box_left)

            print("Final dimensions: %s..." % (str(images_array.shape)))

        else:
            if (args.reduceSizeImages):
                print("Reduce resolution of images to size: %s..." %(args.sizeReducedImages))

                if isSmallerTuple(images_array.shape[-2:], args.sizeReducedImages):
                    message = "New reduced size: %s, smaller than size of original images: %s..." %(images_array.shape[-2:], args.sizeReducedImages)
                    CatchErrorException(message)

                images_array = ResizeImages.compute2D(images_array, args.sizeReducedImages)
                masks_array  = ResizeImages.compute2D(masks_array,  args.sizeReducedImages, isMasks=True)

                print("Final dimensions: %s..." %(images_array.shape))


            if (args.cropImages):
                bounding_box = dict_masks_boundingBoxes[filenamenoextension(images_file)]

                print("Crop images to bounding-box: %s..." % (str(bounding_box)))

                images_array = CropImages.compute3D(images_array, bounding_box)
                masks_array  = CropImages.compute3D(masks_array,  bounding_box)

                print("Final dimensions: %s..." %(str(images_array.shape)))


            if (args.extendSizeImages):
                print("Extend size images to constant size %s:..." %(str(CROPSIZEBOUNDINGBOX)))

                size_new_image = (images_array.shape[0], CROPSIZEBOUNDINGBOX[0], CROPSIZEBOUNDINGBOX[1])

                backgr_val_images = -1000
                backgr_val_masks  = -1 if args.masksToRegionInterest else 0

                bounding_box = dict_masks_boundingBoxes[filenamenoextension(images_file)]

                images_array = ExtendImages.compute3D(images_array, bounding_box, size_new_image, background_value=backgr_val_images)
                masks_array  = ExtendImages.compute3D(masks_array,  bounding_box, size_new_image, background_value=backgr_val_masks)

                print("Final dimensions: %s..." % (str(images_array.shape)))



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

        out_images_filename = joinpathnames(OutputImagesPath, tempNameProcImagesFiles%(i+1, tuple2str(images_array.shape)))
        out_masks_filename  = joinpathnames(OutputMasksPath,  tempNameProcMasksFiles% (i+1, tuple2str(masks_array.shape )))

        FileReader.writeImageArray(out_images_filename, images_array)
        FileReader.writeImageArray(out_masks_filename,  masks_array )

        if (args.constructInputDataDLCST):
            out_images_filename = joinpathnames(OutputImagesPath, tempNameProcImagesFiles_2%(i+1, tuple2str(images_array.shape)))
            out_masks_filename  = joinpathnames(OutputMasksPath,  tempNameProcMasksFiles_2 %(i+1, tuple2str(masks_array.shape)))

            FileReader.writeImageArray(out_images_filename, images_array_2)
            FileReader.writeImageArray(out_masks_filename,  masks_array_2)

        if (args.saveVisualizeProcData):
            print("Saving processed data to visualize...")

            out_images_filename = joinpathnames(VisualOutputPath, tempNameVisualProcFiles(out_images_filename))
            out_masks_filename  = joinpathnames(VisualOutputPath, tempNameVisualProcFiles(out_masks_filename))

            FileReader.writeImageArray(out_images_filename, images_array)
            FileReader.writeImageArray(out_masks_filename,  masks_array )
    #endfor



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--isClassificationCase', type=str, default=ISCLASSIFICATIONCASE)
    parser.add_argument('--multiClassCase', type=str2bool, default=MULTICLASSCASE)
    parser.add_argument('--numClassesMasks', type=int, default=NUMCLASSESMASKS)
    parser.add_argument('--invertImageAxial', type=str2bool, default=INVERTIMAGEAXIAL)
    parser.add_argument('--masksToRegionInterest', type=str2bool, default=MASKTOREGIONINTEREST)
    parser.add_argument('--reduceSizeImages', type=str2bool, default=REDUCESIZEIMAGES)
    parser.add_argument('--sizeReducedImages', type=str2bool, default=SIZEREDUCEDIMAGES)
    parser.add_argument('--cropImages', type=str2bool, default=CROPIMAGES)
    parser.add_argument('--extendSizeImages', type=str2bool, default=EXTENDSIZEIMAGES)
    parser.add_argument('--constructInputDataDLCST', type=str2bool, default=CONSTRUCTINPUTDATADLCST)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=TRANSFORMATIONIMAGES)
    parser.add_argument('--elasticDeformationImages', type=str2bool, default=ELASTICDEFORMATIONIMAGES)
    parser.add_argument('--createImagesBatches', type=str2bool, default=CREATEIMAGESBATCHES)
    parser.add_argument('--saveVisualizeProcData', type=str2bool, default=SAVEVISUALIZEPROCDATA)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)
