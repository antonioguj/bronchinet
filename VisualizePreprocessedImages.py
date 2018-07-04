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
from Preprocessing.ElasticDeformationImages import *
from Preprocessing.SlidingWindowImages import *
from Preprocessing.SlidingWindowPlusTransformImages import *
from Preprocessing.TransformationImages import *
import argparse


def main(args):

    workDirsManager  = WorkDirsManager(args.basedir)
    BaseDataPath     = workDirsManager.getNameBaseDataPath()
    OriginImagesPath = workDirsManager.getNameExistPath(BaseDataPath, 'ProcImagesData')
    OriginMasksPath  = workDirsManager.getNameExistPath(BaseDataPath, 'ProcMasksData')
    VisualImagesPath = workDirsManager.getNameNewPath  (BaseDataPath, 'VisualTransformImages')

    # Get the file list:
    nameImagesFiles  = 'images*'+ getFileExtension(FORMATINOUTDATA)
    nameMasksFiles   = 'masks*' + getFileExtension(FORMATINOUTDATA)

    nameOutImagesFiles      = lambda in_name: joinpathnames(VisualImagesPath, filenamenoextension(in_name) + '_transform.nii.gz')
    nameOutImagesBatchFiles = lambda in_name, index: joinpathnames(VisualImagesPath, filenamenoextension(in_name) + '_batch' + str(index) + '_transform.nii.gz')

    listImagesFiles = findFilesDir(OriginImagesPath, nameImagesFiles)
    listMasksFiles  = findFilesDir(OriginMasksPath,  nameMasksFiles)

    listImagesFiles = listImagesFiles
    listMasksFiles  = listMasksFiles

    nbImagesFiles = len(listImagesFiles)
    nbMasksFiles  = len(listMasksFiles)


    # Run checkers
    if (nbImagesFiles == 0):
        message = "num CTs Images found in dir \'%s\'" %(OriginImagesPath)
        CatchErrorException(message)
    if (nbImagesFiles != nbMasksFiles):
        message = "num CTs Images %i not equal to num Masks %i" %(nbImagesFiles, nbMasksFiles)
        CatchErrorException(message)


    # tests Transformation parameters
    rotation_XY_range = 0
    rotation_XZ_range = 0
    rotation_YZ_range = 0
    width_shift_range = 0
    height_shift_range = 0
    depth_shift_range = 0
    shear_XY_range = 0
    shear_XZ_range = 0
    shear_YZ_range = 0
    horizontal_flip = False
    vertical_flip = False
    depthZ_flip = False

    type_elastic_deformation = 'gridwise'



    for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

        print('\'%s\'...' % (imagesFile))

        images_array = FileReader.getImageArray(imagesFile)
        masks_array  = FileReader.getImageArray(masksFile)
        masks_array  = processBinaryMasks_Type2(masks_array).astype(dtype=FORMATIMAGEDATA)

        transform_images_array = images_array
        transform_masks_array  = masks_array


        if (args.slidingWindowImages):
            print("Generate batches images by Sliding-window: size: %s; Overlap: %s..." % (IMAGES_DIMS_Z_X_Y, args.prop_overlap_Z_X_Y))

            slidingWindowImagesGenerator = SlidingWindowImages3D(IMAGES_DIMS_Z_X_Y, args.prop_overlap_Z_X_Y, images_array.shape)

            (transform_images_array, transform_masks_array) = slidingWindowImagesGenerator.compute_images_array_all(transform_images_array, masks_array=transform_masks_array)

            #print("Final dimensions: %s..." % (transform_images_array.shape))


        if (args.elasticDeformationImages):
            print("Generate transformed images by elastic deformations, of type '%s': size: %s..." % (TYPEELASTICDEFORMATION, IMAGES_DIMS_Z_X_Y))

            if (type_elastic_deformation == 'pixelwise'):
                elasticDeformationImagesGenerator = ElasticDeformationPixelwiseImages3D(IMAGES_DIMS_Z_X_Y)
            else:  # type_elastic_deformation=='gridwise':
                elasticDeformationImagesGenerator = ElasticDeformationGridwiseImages3D(IMAGES_DIMS_Z_X_Y)

            (transform_images_array, transform_masks_array) = elasticDeformationImagesGenerator.compute_images_array_all(transform_images_array, masks_array=transform_masks_array)

            #print("Final dimensions: %s..." % (transform_images_array.shape))

        elif (args.transformationImages):
            print("Generate random 3D transformations of images: size: %s..." % (IMAGES_DIMS_Z_X_Y))

            transformImagesGenerator = TransformationImages3D(IMAGES_DIMS_Z_X_Y,
                                                              rotation_XY_range=rotation_XY_range,
                                                              rotation_XZ_range=rotation_XZ_range,
                                                              rotation_YZ_range=rotation_YZ_range,
                                                              width_shift_range=width_shift_range,
                                                              height_shift_range=height_shift_range,
                                                              depth_shift_range=depth_shift_range,
                                                              shear_XY_range=shear_XY_range,
                                                              shear_XZ_range=shear_XZ_range,
                                                              shear_YZ_range=shear_YZ_range,
                                                              horizontal_flip=horizontal_flip,
                                                              vertical_flip=vertical_flip,
                                                              depthZ_flip=depthZ_flip)

            (transform_images_array, transform_masks_array) = transformImagesGenerator.compute_images_array_all(transform_images_array, masks_array=transform_masks_array)

            #print("Final dimensions: %s..." % (transform_images_array.shape))


        if (args.slidingWindowImages):
            for j, (batch_images_array, batch_masks_array) in enumerate(zip(transform_images_array, transform_masks_array)):

                FileReader.writeImageArray(nameOutImagesBatchFiles(imagesFile,j), batch_images_array)
                FileReader.writeImageArray(nameOutImagesBatchFiles(masksFile, j), batch_masks_array)
            # endfor
        else:
            FileReader.writeImageArray(nameOutImagesFiles(imagesFile), transform_images_array)
            FileReader.writeImageArray(nameOutImagesFiles(masksFile),  transform_masks_array )
    # endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--slidingWindowImages', type=str2bool, default=SLIDINGWINDOWIMAGES)
    parser.add_argument('--prop_overlap_Z_X_Y', type=str2tuplefloat, default=PROP_OVERLAP_Z_X_Y)
    parser.add_argument('--transformationImages', type=str2bool, default=TRANSFORMATIONIMAGES)
    parser.add_argument('--elasticDeformationImages', type=str2bool, default=ELASTICDEFORMATIONIMAGES)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)