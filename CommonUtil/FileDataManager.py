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
from keras import backend as K
from glob import glob
import numpy as np
np.random.seed(2017)
import os



class FileDataManager(object):

    SHUFFLEIMAGES = True

    @classmethod
    def loadDataFile2D(cls, pathFile, maxImagesToLoad=MAXIMAGESTOLOAD):

        listImagesFiles      = sorted( glob(pathFile + 'images-*.npy'     ))
        listGroundTruthFiles = sorted( glob(pathFile + 'groundTruth-*.npy'))

        if K.image_data_format() == 'channels_first':
            xData = np.ndarray([maxImagesToLoad, 1, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATIMAGEDATA      )
            yData = np.ndarray([maxImagesToLoad, 1, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATGROUNDTRUTHDATA)
        else:
            xData = np.ndarray([maxImagesToLoad, IMAGES_HEIGHT, IMAGES_WIDTH, 1], dtype=FORMATIMAGEDATA      )
            yData = np.ndarray([maxImagesToLoad, IMAGES_HEIGHT, IMAGES_WIDTH, 1], dtype=FORMATGROUNDTRUTHDATA)

        count_img = 0

        for imageFile, groundTruthFile in zip(listImagesFiles, listGroundTruthFiles):

            if (count_img >= maxImagesToLoad):  # reached the limit of images to load
                break

            xData_part = np.load(imageFile).astype(FORMATIMAGEDATA)
            yData_part = np.load(groundTruthFile).astype(FORMATGROUNDTRUTHDATA)

            numSlices = xData_part.shape[0]

            if K.image_data_format() == 'channels_first':
                xData_part = xData_part.reshape(numSlices, 1, IMAGES_HEIGHT, IMAGES_WIDTH)
                yData_part = yData_part.reshape(numSlices, 1, IMAGES_HEIGHT, IMAGES_WIDTH)
            else:
                xData_part = xData_part.reshape(numSlices, IMAGES_HEIGHT, IMAGES_WIDTH, 1)
                yData_part = yData_part.reshape(numSlices, IMAGES_HEIGHT, IMAGES_WIDTH, 1)

            numSlices = min(numSlices, maxImagesToLoad - count_img)

            for i in range(numSlices):
                xData[count_img + i] = xData_part[i]
                yData[count_img + i] = yData_part[i]

            count_img += numSlices

        # endfor

        if( cls.SHUFFLEIMAGES ):
            # Generate random indexes to shuffle input data
            total_numImages = xData.shape[0]
            randIndexImages = np.random.choice(range(total_numImages), size=total_numImages, replace=False)

            return (xData[randIndexImages[:]], yData[randIndexImages[:]])

        else:
            return (xData, yData)