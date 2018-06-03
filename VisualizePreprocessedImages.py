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
    OrigImagesPath   = workDirsManager.getNameExistPath(BaseDataPath, 'ProcImages')
    OrigMasksPath    = workDirsManager.getNameExistPath(BaseDataPath, 'ProcMasks')
    VisualImagesPath = workDirsManager.getNameNewPath  (BaseDataPath, 'VisualTransformImages')

    # Get the file list:
    nameImagesFiles    = '*.nii'
    nameMasksFiles     = '*.nii'
    nameOutImagesFiles = lambda in_name: joinpathnames(VisualImagesPath, filenamenoextension(in_name) + '_transform.nii')

    listImagesFiles = findFilesDir(OrigImagesPath, nameImagesFiles)
    listMasksFiles  = findFilesDir(OrigMasksPath,  nameMasksFiles)

    nbImagesFiles = len(listImagesFiles)
    nbMasksFiles  = len(listMasksFiles)

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

    elastic_deformation = True
    type_elastic_deformation = 'gridwise'


    for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

        print('\'%s\'...' % (imagesFile))

        images_array = FileReader.getImageArray(imagesFile)
        masks_array  = FileReader.getImageArray(masksFile)

        transform_images_array = images_array
        transform_masks_array  = masks_array


        if (elastic_deformation):
            print("compute elastic deformation of images...")

            if (type_elastic_deformation == 'pixelwise'):
                elasticDeformationImages = ElasticDeformationPixelwiseImages3D(IMAGES_DIMS_Z_X_Y)
            else: #type_elastic_deformation =='gridwise'
                elasticDeformationImages = ElasticDeformationGridwiseImages3D(IMAGES_DIMS_Z_X_Y)

            # Important! set seed to apply same random transformation to images and masks
            (transform_images_array, transform_masks_array) = elasticDeformationImages.get_images_array(transform_images_array, masks_array=transform_masks_array)


        print("compute random transformation of images...")
        transformationImages = TransformationImages3D(IMAGES_DIMS_Z_X_Y,
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

        # Important! set seed to apply same random transformation to images and masks
        (transform_images_array, transform_masks_array) = transformationImages.get_images_array(transform_images_array, masks_array=transform_masks_array)


        FileReader.writeImageArray(nameOutImagesFiles(imagesFile), transform_images_array)
        FileReader.writeImageArray(nameOutImagesFiles(masksFile),  transform_masks_array )
    # endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)