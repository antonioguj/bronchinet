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
from CommonUtil.DICOMreader import DICOMreader
from CommonUtil.ErrorMessages import *
from CommonUtil.OthersUtil import splitListInChunks
from CommonUtil.WorkDirsManager import WorkDirsManager
from glob import glob
import numpy as np
import os

import matplotlib.pyplot as plt

TYPEDATA = 'testing'

MAXVOLIMAGEINOUTFILE = 6

CROPPINGIMAGES = True

SHUFFLEIMAGES = True


#MAIN
workDirsManager = WorkDirsManager(BASEDIR)

DataPath        = workDirsManager.getNameDataPath(TYPEDATA)
ImagesPath      = workDirsManager.getNameRawImagesDataPath(TYPEDATA)
GroundTruthPath = workDirsManager.getNameRawGroundTruthDataPath(TYPEDATA)

if (not os.path.isdir(ImagesPath)):
    message = "directory \'%s\' does not exist" % (ImagesPath)
    CatchErrorException(message)

if (not os.path.isdir(GroundTruthPath)):
    message = "directory \'%s\' does not exist" % (GroundTruthPath)
    CatchErrorException(message)


# Get the file list:
listImageFiles      = sorted(glob(ImagesPath + '*.dcm'))
listGroundTruthFiles= sorted(glob(GroundTruthPath  + '*.dcm'))

nbImageFiles       = len(listImageFiles);
nbGroundTruthFiles = len(listGroundTruthFiles);

if (nbImageFiles != nbGroundTruthFiles):
    message = "nbImageFiles %i not equal to nbGroundTruthFiles %i" %(nbImageFiles,nbGroundTruthFiles)
    CatchErrorException(message)

# Write list of files involved
filelist = open(DataPath + "file.txt", 'w')
filelist.write('list of files\n')
filelist.write('%s\n' % (nbImageFiles))
for file in listImageFiles:
    filelist.write('%s\n' % (file))
filelist.close()


# Split the full count of files in "NBOUTPUTFILES" chunks
# if images are very big, memory usage may blow up. Preferible to create several output files
listOfListsImageFiles       = splitListInChunks(listImageFiles,       MAXVOLIMAGEINOUTFILE)
listOfListsGroundTruthFiles = splitListInChunks(listGroundTruthFiles, MAXVOLIMAGEINOUTFILE)

nbSplitBatches = len(listOfListsImageFiles)

print('-' * 30)
print('Nb Total Files (%s). Max Files per Batch (%s). Split Total Files in (%s) Batches...' %(nbImageFiles, MAXVOLIMAGEINOUTFILE, nbSplitBatches))


if (CROPPINGIMAGES):
    sizeBoundBox = (BOUNDBOX_CROPPING[0][1] - BOUNDBOX_CROPPING[0][0],
                    BOUNDBOX_CROPPING[1][1] - BOUNDBOX_CROPPING[1][0])
    print('Cropping Images to Bounding Box of size (%s,%s)' %(sizeBoundBox[0], sizeBoundBox[1]))

    if (sizeBoundBox[0] != IMAGES_HEIGHT or
        sizeBoundBox[1] != IMAGES_WIDTH):
        message = 'size of Input Images not equal to Bounding Box size (%s,%s)'
        CatchErrorException(message)


count = 0

for listImageFiles, listGroundTruthFiles in zip(listOfListsImageFiles,listOfListsGroundTruthFiles):
    # increase counter output files
    count += 1

    nbFiles = len(listImageFiles)

    # First loop: calculate size total all images
    print('-' * 30)
    print('Calculate Dimensions of Batch, with total nb Files (%s)...' %(nbFiles))

    total_numSlices = 0

    for imageFile, groundTruthFile in zip(listImageFiles, listGroundTruthFiles):

        sizeImage       = DICOMreader.getImageSize(imageFile)
        sizeGroundTruth = DICOMreader.getImageSize(groundTruthFile)

        if (sizeImage != sizeGroundTruth):
            message = "size of Images not equal to size of GroundTruth"
            CatchErrorException(message)

        if((sizeImage[1] != IMAGES_HEIGHT or
            sizeImage[2] != IMAGES_WIDTH ) and
            CROPPINGIMAGES == False):
            message = 'size of Images not equal to Input Size (%s,%s)' %(sizeImage[1], sizeImage[2])
            CatchErrorException(message)

        if (CROPPINGIMAGES):
            # Find the number of slices with segmentation masks
            groundTruth_array = DICOMreader.loadImage(groundTruthFile)

            for i, slice_groundTruth in enumerate(groundTruth_array):

                if( len(np.argwhere(slice_groundTruth)) ):
                    total_numSlices += 1
        else:
            total_numSlices += sizeImage[2]

    #endfor

    print('Found total nb Images (%s), of size (%s,%s)' %(total_numSlices, IMAGES_HEIGHT, IMAGES_WIDTH))
    print('-' * 30)


    # Second loop, loading Images and stacking altogether

    print('-' * 30)
    print('Loading files (%s in total)...' %(nbFiles))
    print('-' * 30)

    # Writing out images and masks as 1 channel arrays for input into network
    fullImageData       = np.ndarray([total_numSlices, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATIMAGEDATA)
    fullGroundTruthData = np.ndarray([total_numSlices, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATGROUNDTRUTHDATA)

    count_img = 0

    for imageFile, groundTruthFile in zip(listImageFiles, listGroundTruthFiles):

        print('\'%s\'...' %(imageFile))

        image_array       = DICOMreader.loadImage(imageFile)
        groundTruth_array = DICOMreader.loadImage(groundTruthFile)

        # Save images and ground-truth slices in a global list:
        if (CROPPINGIMAGES):
            for i, (slice_image, slice_groundTruth) in enumerate(zip(image_array, groundTruth_array)):

                if (len(np.argwhere(slice_groundTruth))):

                    # Turn the mask images to binary images:
                    slice_groundTruth = np.where(slice_groundTruth != 0, 1, 0)

                    slice_image = slice_image[BOUNDBOX_CROPPING[0][0]:BOUNDBOX_CROPPING[0][1],
                                              BOUNDBOX_CROPPING[1][0]:BOUNDBOX_CROPPING[1][1]]
                    slice_groundTruth = slice_groundTruth[BOUNDBOX_CROPPING[0][0]:BOUNDBOX_CROPPING[0][1],
                                                          BOUNDBOX_CROPPING[1][0]:BOUNDBOX_CROPPING[1][1]]

                    fullImageData      [count_img] = np.asarray(slice_image,       dtype=FORMATIMAGEDATA)
                    fullGroundTruthData[count_img] = np.asarray(slice_groundTruth, dtype=FORMATGROUNDTRUTHDATA)

                    count_img += 1
            #endfor
        else:
            for i, (slice_image, slice_groundTruth) in enumerate(zip(image_array, groundTruth_array)):

                # Turn the mask images to binary images:
                slice_groundTruth = np.where(slice_groundTruth != 0, 1, 0)

                fullImageData      [count_img + i] = np.asarray(slice_image,       dtype=FORMATIMAGEDATA)
                fullGroundTruthData[count_img + i] = np.asarray(slice_groundTruth, dtype=FORMATGROUNDTRUTHDATA)

            #endfor

            count_img += image_array.shape[0]
    # endfor


    print('-' * 30)
    print('Saving files...')
    print('-' * 30)

    nameOutImageFile      = DataPath + 'imagesCropped-0%i.npy' %(count)
    nameOutGroundTruthFile= DataPath + 'groundTruthCropped-0%i.npy' %(count)

    if SHUFFLEIMAGES:
        # Generate random indexes in range (0,num_files) to shuffle input data of DNN
        randIndexImages = np.random.choice(range(total_numSlices), size=total_numSlices, replace=False)

        np.save(nameOutImageFile,       fullImageData      [randIndexImages[:]])
        np.save(nameOutGroundTruthFile, fullGroundTruthData[randIndexImages[:]])
    else:
        # Do not shuffle data
        np.save(nameOutImageFile,       fullImageData      )
        np.save(nameOutGroundTruthFile, fullGroundTruthData)

#endfor