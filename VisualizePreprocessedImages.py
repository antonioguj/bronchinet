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
    rotation_XY_range = 20
    rotation_XZ_range = 10
    rotation_YZ_range = 10
    width_shift_range = 50
    height_shift_range = 70
    depth_shift_range = 15
    shear_XY_range = 10
    shear_XZ_range = 5
    shear_YZ_range = 5
    horizontal_flip = True
    vertical_flip = True
    depthZ_flip = True


    for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

        print('\'%s\'...' % (imagesFile))

        images_array = FileReader.getImageArray(imagesFile)
        masks_array  = FileReader.getImageArray(masksFile)


        if (args.transformationImages):
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
            transform_images_array = transformationImages.get_image_array(images_array, seed=i)
            transform_masks_array  = transformationImages.get_image_array(masks_array,  seed=i)
        else:
            transform_images_array = images_array
            transform_masks_array  = masks_array


        FileReader.writeImageArray(nameOutImagesFiles(imagesFile), transform_images_array)
        FileReader.writeImageArray(nameOutImagesFiles(masksFile),  transform_masks_array )
    # endfor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', default=BASEDIR)
    parser.add_argument('--transformationImages', type=str2bool, default=True)
    args = parser.parse_args()

    print("Print input arguments...")
    for key, value in vars(args).iteritems():
        print("\'%s\' = %s" %(key, value))

    main(args)