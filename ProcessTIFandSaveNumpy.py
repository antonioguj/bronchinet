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
from CommonUtil.WorkDirsManager import WorkDirsManager
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
import os


TYPEDATA = 'testing'

MAXIMAGESNPYFILE = 10e+06

SHUFFLEIMAGES = False


def preprocess(image_array):
    image_array_proc = resize(image_array, (IMAGES_HEIGHT, IMAGES_WIDTH), preserve_range=True)
    return image_array_proc


#MAIN
workDirsManager= WorkDirsManager(BASEDIR)
DataPath       = workDirsManager.getNameDataPath(TYPEDATA)
ImagesPath     = workDirsManager.getNameRawImagesDataPath(TYPEDATA)
MasksPath      = workDirsManager.getNameRawMasksDataPath(TYPEDATA)


# Get the file list:
listImagesFiles = os.listdir(ImagesPath)
listImagesFiles = [os.path.join(ImagesPath, file) for file in listImagesFiles]
listMasksFiles  = os.listdir(MasksPath)
listMasksFiles  = [os.path.join(MasksPath, file) for file in listMasksFiles]

num_images = len(listImagesFiles)
num_masks  = len(listMasksFiles)

if (num_images == 0):
    message = "no files found in \"%s\"" %(DataPath)
    CatchErrorException(message)


if( TYPEDATA=="training" ):

    if (num_images != num_masks):
        message = "num_images %i not equal to num_masks %i" %(num_images,num_masks)
        CatchErrorException(message)

    print('-' * 30)
    print('Loading files (%s in total)...' %(num_images))
    print('-' * 30)

    fullImageData = np.ndarray([num_images, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATIMAGEDATA)
    fullMaskData  = np.ndarray([num_masks,  IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATMASKDATA )

    for i, (imageFile, maskFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

        image_array = imread(imageFile, as_grey=True)
        mask_array  = imread(maskFile,  as_grey=True)

        # Turn the mask images to binary images:
        mask_array = np.where(mask_array != 0, 1, 0)

        # Resize images
        image_array = resize(image_array, (IMAGES_HEIGHT, IMAGES_WIDTH), preserve_range=True)
        mask_array  = resize(mask_array,  (IMAGES_HEIGHT, IMAGES_WIDTH), preserve_range=True)

        fullImageData[i] = np.asarray(image_array, dtype=FORMATIMAGEDATA)
        fullMaskData [i] = np.asarray(mask_array,  dtype=FORMATMASKDATA)

        if i % 1000 == 0:
            print('Done: {0}/{1} images'.format(i, num_images))
    #endfor

elif( TYPEDATA=="testing" ):

    print('-' * 30)
    print('Loading files (%s in total)...' %(num_images))
    print('-' * 30)

    fullImageData = np.ndarray([num_images, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATIMAGEDATA)
    fullMaskData  = np.ndarray([num_images], dtype=np.int32)

    for i, imageFile in enumerate(listImagesFiles):

        image_array = imread(imageFile, as_grey=True)

        # Retrieve image ID
        image_ID = int((imageFile.split('/')[-1]).split('.')[0])

        # Resize images
        image_array = resize(image_array, (IMAGES_HEIGHT, IMAGES_WIDTH), preserve_range=True)

        fullImageData[i] = np.asarray(image_array, dtype=FORMATIMAGEDATA)
        fullMaskData [i] = image_ID

        if i % 1000 == 0:
            print('Done: {0}/{1} images'.format(i, num_images))
    #endfor


print('-' * 30)
print('Saving files...')
print('-' * 30)

nameOutImageFile = DataPath + 'images.npy'
nameOutMaskFile  = DataPath + 'masks.npy'

if SHUFFLEIMAGES:
    # Generate random indexes in range (0,num_files) to shuffle input data of DNN
    randIndexes = np.random.choice(num_images, size=num_images, replace=False)

    np.save(nameOutImageFile, fullImageData[randIndexes[:]])
    np.save(nameOutMaskFile,  fullMaskData [randIndexes[:]])
else:
    # Do not shuffle data
    np.save(nameOutImageFile, fullImageData)
    np.save(nameOutMaskFile,  fullMaskData)