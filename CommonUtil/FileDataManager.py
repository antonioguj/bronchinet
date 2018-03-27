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
from keras import backend as K
import numpy as np
np.random.seed(2017)
import os

SHUFFLEIMAGES = False
NORMALIZEDATA = False


class FileDataManager(object):

    @classmethod
    def normalizeData(cls):
        if( NORMALIZEDATA ):
            xData = xData.astype('float32')
            mean  = np.mean(xData)
            std   = np.std(xData)
            xData = (xData - mean)/std

    @classmethod
    def shuffleImages(cls, xData, yData):
        # Generate random indexes to shuffle data
        randomIndexes = np.random.choice(range(xData.shape[0]), size=xData.shape[0], replace=False)
        xData = xData[randomIndexes[:]]
        yData = yData[randomIndexes[:]]

        return (xData, yData)


    @classmethod
    def loadDataFiles2D(cls, imagesFile, masksFile, maxSlicesToLoad=1000000, shuffleImages=SHUFFLEIMAGES):

        if( not os.path.exists(imagesFile) or
            not os.path.exists(masksFile) ):
            message = "Images file or Masks file does not exist (\'%s\',\'%s\')" % (imagesFile, masksFile)
            CatchErrorException(message)

        xData = np.load(imagesFile).astype(dtype=FORMATIMAGEDATA)
        yData = np.load(masksFile) .astype(dtype=FORMATMASKDATA )

        if( xData.shape != yData.shape ):
            message = "Images array of different size to Masks array (\'%s\',\'%s\')" % (xData.shape, yData.shape)
            CatchErrorException(message)

        totalSlices = min(xData.shape[0], maxSlicesToLoad)

        if K.image_data_format() == 'channels_first':
            xData = np.asarray(xData[0:totalSlices]).reshape([totalSlices, 1, IMAGES_HEIGHT, IMAGES_WIDTH])
            yData = np.asarray(yData[0:totalSlices]).reshape([totalSlices, 1, IMAGES_HEIGHT, IMAGES_WIDTH])
        else:
            xData = np.asarray(xData[0:totalSlices]).reshape([totalSlices, IMAGES_HEIGHT, IMAGES_WIDTH, 1])
            yData = np.asarray(yData[0:totalSlices]).reshape([totalSlices, IMAGES_HEIGHT, IMAGES_WIDTH, 1])

        if (shuffleImages):
            return cls.shuffleImages(xData, yData)
        else:
            return (xData, yData)


    @classmethod
    def loadDataListFiles2D(cls, listImagesFiles, listMasksFiles, maxSlicesToLoad=1000000, shuffleImages=SHUFFLEIMAGES):

        if( not listImagesFiles or
            not listMasksFiles ):
            message = "No Images files found"
            CatchErrorException(message)

        #First, loop over files to run cheks and compute size output array
        totalSlices = 0
        for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

            if (not os.path.exists(imagesFile) or
                not os.path.exists(masksFile)):
                message = "Images file or Masks file does not exist (\'%s\',\'%s\')" % (imagesFile, masksFile)
                CatchErrorException(message)

            xData_part = np.load(imagesFile).astype(dtype=FORMATIMAGEDATA)
            yData_part = np.load(masksFile) .astype(dtype=FORMATMASKDATA )

            if (xData_part.shape != yData_part.shape):
                message = "Images array of different size to Masks array (\'%s\',\'%s\')" % (xData_part.shape, yData_part.shape)
                CatchErrorException(message)

            totalSlices += xData_part.shape[0]

            if( totalSlices>=maxSlicesToLoad ):
                #reached the max size for output array
                totalSlices = maxSlicesToLoad
                #limit images files to load form
                listImagesFiles = [listImagesFiles[j] for j in range(i)]
                listMasksFiles  = [listMasksFiles [j] for j in range(i)]
                break
        #endfor

        if K.image_data_format() == 'channels_first':
            xData = np.ndarray([totalSlices, 1, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATIMAGEDATA)
            yData = np.ndarray([totalSlices, 1, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATMASKDATA)
        else:
            xData = np.ndarray([totalSlices, IMAGES_HEIGHT, IMAGES_WIDTH, 1], dtype=FORMATIMAGEDATA)
            yData = np.ndarray([totalSlices, IMAGES_HEIGHT, IMAGES_WIDTH, 1], dtype=FORMATMASKDATA)

        countSlice = 0
        for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

            xData_part = np.load(imagesFile).astype(dtype=FORMATIMAGEDATA)
            yData_part = np.load(masksFile) .astype(dtype=FORMATMASKDATA )

            numSlices_part = min(xData_part.shape[0], xData.shape[0] - countSlice)

            if K.image_data_format() == 'channels_first':
                xData[countSlice:countSlice + numSlices_part] = np.asarray(xData_part[0:numSlices_part]).reshape(numSlices_part, 1, IMAGES_HEIGHT, IMAGES_WIDTH)
                yData[countSlice:countSlice + numSlices_part] = np.asarray(yData_part[0:numSlices_part]).reshape(numSlices_part, 1, IMAGES_HEIGHT, IMAGES_WIDTH)
            else:
                xData[countSlice:countSlice + numSlices_part] = np.asarray(xData_part[0:numSlices_part]).reshape(numSlices_part, IMAGES_HEIGHT, IMAGES_WIDTH, 1)
                yData[countSlice:countSlice + numSlices_part] = np.asarray(yData_part[0:numSlices_part]).reshape(numSlices_part, IMAGES_HEIGHT, IMAGES_WIDTH, 1)

            countSlice += numSlices_part
        # endfor

        if (shuffleImages):
            return cls.shuffleImages(xData, yData)
        else:
            return (xData, yData)


    @classmethod
    def loadDataFiles3D(cls, imagesFile, masksFile, maxVolsToLoad=10000, shuffleImages=SHUFFLEIMAGES):

        if( not os.path.exists(imagesFile) or
            not os.path.exists(masksFile) ):
            message = "Images file or Masks file does not exist (\'%s\',\'%s\')" % (imagesFile, masksFile)
            CatchErrorException(message)

        xData = np.load(imagesFile).astype(dtype=FORMATIMAGEDATA)
        yData = np.load(masksFile) .astype(dtype=FORMATMASKDATA )

        if( xData.shape != yData.shape ):
            message = "Images array of different size to Masks array (\'%s\',\'%s\')" % (xData.shape, yData.shape)
            CatchErrorException(message)

        totalVols = min(xData.shape[0], maxVolsToLoad)

        if K.image_data_format() == 'channels_first':
            xData = np.asarray(xData[0:totalVols]).reshape([totalVols, 1, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH])
            yData = np.asarray(yData[0:totalVols]).reshape([totalVols, 1, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH])
        else:
            xData = np.asarray(xData[0:totalVols]).reshape([totalVols, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH, 1])
            yData = np.asarray(yData[0:totalVols]).reshape([totalVols, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH, 1])

        if (shuffleImages):
            return cls.shuffleImages(xData, yData)
        else:
            return (xData, yData)


    @classmethod
    def loadDataListFiles3D(cls, listImagesFiles, listMasksFiles, maxVolsToLoad=10000, shuffleImages=SHUFFLEIMAGES):

        if( not listImagesFiles or
            not listMasksFiles ):
            message = "No Images files found"
            CatchErrorException(message)

        #First, loop over files to run cheks and compute size output array
        totalVols = 0
        for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

            if (not os.path.exists(imagesFile) or
                not os.path.exists(masksFile)):
                message = "Images file or Masks file does not exist (\'%s\',\'%s\')" % (imagesFile, masksFile)
                CatchErrorException(message)

            xData_part = np.load(imagesFile).astype(dtype=FORMATIMAGEDATA)
            yData_part = np.load(masksFile) .astype(dtype=FORMATMASKDATA )

            if (xData_part.shape != yData_part.shape):
                message = "Images array of different size to Masks array (\'%s\',\'%s\')" % (xData_part.shape, yData_part.shape)
                CatchErrorException(message)

            totalVols += xData_part.shape[0]

            if( totalVols>=maxVolsToLoad ):
                #reached the max size for output array
                totalVols = maxVolsToLoad
                #limit images files to load form
                listImagesFiles = [listImagesFiles[j] for j in range(i)]
                listMasksFiles  = [listMasksFiles [j] for j in range(i)]
                break
        #endfor

        if K.image_data_format() == 'channels_first':
            xData = np.ndarray([totalVols, 1, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATIMAGEDATA)
            yData = np.ndarray([totalVols, 1, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH], dtype=FORMATMASKDATA)
        else:
            xData = np.ndarray([totalVols, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH, 1], dtype=FORMATIMAGEDATA)
            yData = np.ndarray([totalVols, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH, 1], dtype=FORMATMASKDATA)

        countVol = 0
        for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

            xData_part = np.load(imagesFile).astype(dtype=FORMATIMAGEDATA)
            yData_part = np.load(masksFile) .astype(dtype=FORMATMASKDATA )

            numVols_part = min(xData_part.shape[0], xData.shape[0] - countVol)

            if K.image_data_format() == 'channels_first':
                xData[countVol:countVol + numVols_part] = np.asarray(xData_part[0:numVols_part]).reshape(numVols_part, 1, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH)
                yData[countVol:countVol + numVols_part] = np.asarray(yData_part[0:numVols_part]).reshape(numVols_part, 1, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH)
            else:
                xData[countVol:countVol + numVols_part] = np.asarray(xData_part[0:numVols_part]).reshape(numVols_part, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH, 1)
                yData[countVol:countVol + numVols_part] = np.asarray(yData_part[0:numVols_part]).reshape(numVols_part, IMAGES_DEPTHZ, IMAGES_HEIGHT, IMAGES_WIDTH, 1)

            countVol += numVols_part
        # endfor

        if (shuffleImages):
            return cls.shuffleImages(xData, yData)
        else:
            return (xData, yData)


    @classmethod
    def loadDataFiles3D_noprocess(cls, imagesFile, masksFile):

        if( not os.path.exists(imagesFile) or
            not os.path.exists(masksFile) ):
            message = "Images file or Masks file does not exist (\'%s\',\'%s\')" % (imagesFile, masksFile)
            CatchErrorException(message)

        xData = np.load(imagesFile).astype(dtype=FORMATIMAGEDATA)
        yData = np.load(masksFile) .astype(dtype=FORMATMASKDATA )

        if( xData.shape != yData.shape ):
            message = "Images array of different size to Masks array (\'%s\',\'%s\')" % (xData.shape, yData.shape)
            CatchErrorException(message)

        return (xData, yData)

    @classmethod
    def loadDataListFiles3D_noprocess(cls, listImagesFiles, listMasksFiles):

        if( not listImagesFiles or
            not listMasksFiles ):
            message = "No Images files found"
            CatchErrorException(message)

        #First, loop over files to run cheks and compute size output array
        xData = []
        yData = []
        for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

            if (not os.path.exists(imagesFile) or
                not os.path.exists(masksFile)):
                message = "Images file or Masks file does not exist (\'%s\',\'%s\')" % (imagesFile, masksFile)
                CatchErrorException(message)

            xData_part = np.load(imagesFile).astype(dtype=FORMATIMAGEDATA)
            yData_part = np.load(masksFile) .astype(dtype=FORMATMASKDATA )

            if (xData_part.shape != yData_part.shape):
                message = "Images array of different size to Masks array (\'%s\',\'%s\')" % (xData_part.shape, yData_part.shape)
                CatchErrorException(message)

            xData.append(xData_part)
            yData.append(yData_part)
        #endfor

        return (xData, yData)