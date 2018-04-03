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
from CommonUtil.FunctionsUtil import *
from Preprocessing.BatchImageGenerator import *
from keras import backend as K
import numpy as np
np.random.seed(2017)

SHUFFLEIMAGES = True
NORMALIZEDATA = False


class LoadDataManager(object):

    def __init__(self, size_image):
        self.size_image = size_image

    @staticmethod
    def runChecker1(imagesFile, masksFile):
        if (not isExistfile(imagesFile) or isExistfile(masksFile)):
            message = "Images or masks file does not exist (\'%s\',\'%s\')" % (imagesFile, masksFile)
            CatchErrorException(message)

    @staticmethod
    def runChecker2(array_shape, size_image):
        if (array_shape[1:] != size_image):
            message = "Images or masks batches of wrong size (\'%s\',\'%s\')" % (array_shape[1:], size_image)
            CatchErrorException(message)

    @staticmethod
    def runChecker3(images_shape, masks_shape):
        if (images_shape != masks_shape):
            message = "Images and masks array of different size (\'%s\',\'%s\')" % (images_shape, masks_shape)
            CatchErrorException(message)

    @classmethod
    def normalizeData(cls, xData):
        mean = np.mean(xData)
        std  = np.std(xData)
        return (xData - mean)/std

    @classmethod
    def shuffleImages(cls, xData, yData):
        # Generate random indexes to shuffle data
        randomIndexes = np.random.choice(range(xData.shape[0]), size=xData.shape[0], replace=False)
        return (xData[randomIndexes[:]],
                yData[randomIndexes[:]])

    def getArrayDims(self, num_batches):
        if K.image_data_format() == 'channels_first':
            return [num_batches, 1] + list(self.size_image)
        else:
            return [num_batches] + list(self.size_image) + [1]


    def loadData_1FileBatches(self, imagesFile, masksFile, max_num_batches=10000, shuffleImages=SHUFFLEIMAGES):

        xData = np.load(imagesFile).astype(dtype=FORMATIMAGEDATA)
        yData = np.load(masksFile) .astype(dtype=FORMATMASKDATA )

        num_batches = min(xData.shape[0], max_num_batches)

        xData = xData[0:num_batches].reshape(self.getArrayDims(num_batches))
        yData = yData[0:num_batches].reshape(self.getArrayDims(num_batches))

        if (shuffleImages):
            return self.shuffleImages(xData, yData)
        else:
            return (xData, yData)


    def loadData_ListFilesBatches(self, listImagesFiles, listMasksFiles, max_num_batches=10000, shuffleImages=SHUFFLEIMAGES):

        #loop over files to compute output array size
        num_batches = 0
        for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

            xData_part   = np.load(imagesFile)
            num_batches += xData_part.shape[0]

            if( num_batches>=max_num_batches ):
                #reached the max size for output array
                num_batches = max_num_batches
                #limit images files to load form
                listImagesFiles = [listImagesFiles[j] for j in range(i+1)]
                listMasksFiles  = [listMasksFiles [j] for j in range(i+1)]
                break
        #endfor

        xData = np.ndarray(self.getArrayDims(num_batches), dtype=FORMATIMAGEDATA)
        yData = np.ndarray(self.getArrayDims(num_batches), dtype=FORMATMASKDATA)

        count_batch = 0
        for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

            xData_part = np.load(imagesFile).astype(dtype=FORMATIMAGEDATA)
            yData_part = np.load(masksFile) .astype(dtype=FORMATMASKDATA )

            num_batches_part = min(xData_part.shape[0], xData.shape[0] - count_batch)

            xData[count_batch:count_batch + num_batches_part] = xData_part[0:num_batches_part].reshape(self.getArrayDims(num_batches_part))
            yData[count_batch:count_batch + num_batches_part] = yData_part[0:num_batches_part].reshape(self.getArrayDims(num_batches_part))

            count_batch += num_batches_part
        # endfor

        if (shuffleImages):
            return self.shuffleImages(xData, yData)
        else:
            return (xData, yData)


    def loadData_1File_BatchGenerator(self, ImagesOperator, imagesFile, masksFile, max_num_batches=10000, shuffleImages=SHUFFLEIMAGES):

        xData = np.load(imagesFile).astype(dtype=FORMATIMAGEDATA)
        yData = np.load(masksFile) .astype(dtype=FORMATMASKDATA )

        imagesBatchGenerator = ImageBatchGenerator_2Arrays(ImagesOperator, xData, yData, self.size_image, size_batch=1, shuffle_images=shuffleImages)

        num_batches = min(len(imagesBatchGenerator), max_num_batches)

        xData = np.ndarray(self.getArrayDims(num_batches), dtype=FORMATIMAGEDATA)
        yData = np.ndarray(self.getArrayDims(num_batches), dtype=FORMATMASKDATA)

        for index in range(num_batches):
            # Retrieve (xData, yData) directly from imagesDataGenerator
            (xData_batch, yData_batch) = next(imagesBatchGenerator)

            xData[index] = xData_batch.reshape(self.getArrayDims(1))
            yData[index] = yData_batch.reshape(self.getArrayDims(1))
        #endfor

        return (xData, yData)


    def loadData_ListFiles_BatchGenerator(self, ImagesOperator, listImagesFiles, listMasksFiles, max_num_batches=10000, shuffleImages=SHUFFLEIMAGES):

        #loop over files to compute output array size
        num_batches = 0
        for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

            xData = np.load(imagesFile)

            imagesBatchGenerator = ImageBatchGenerator_1Array(ImagesOperator, xData, self.size_image, size_batch=1, shuffle_images=shuffleImages)

            num_batches += len(imagesBatchGenerator)

            if( num_batches>=max_num_batches ):
                #reached the max size for output array
                num_batches = max_num_batches
                #limit images files to load form
                listImagesFiles = [listImagesFiles[j] for j in range(i+1)]
                listMasksFiles  = [listMasksFiles [j] for j in range(i+1)]
                break

        xData = np.ndarray(self.getArrayDims(num_batches), dtype=FORMATIMAGEDATA)
        yData = np.ndarray(self.getArrayDims(num_batches), dtype=FORMATMASKDATA)

        count_batch = 0
        for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

            xData_part = np.load(imagesFile).astype(dtype=FORMATIMAGEDATA)
            yData_part = np.load(masksFile) .astype(dtype=FORMATMASKDATA )

            imagesBatchGenerator = ImageBatchGenerator_2Arrays(ImagesOperator, xData_part, yData_part, self.size_image, size_batch=1, shuffle_images=shuffleImages)

            num_batches_part = min(len(imagesBatchGenerator), xData.shape[0] - count_batch)

            for index in range(num_batches_part):
                # Retrieve (xData, yData) directly from imagesDataGenerator
                (xData_batch, yData_batch) = next(imagesBatchGenerator)

                xData[count_batch] = xData_batch.reshape(self.getArrayDims(1))
                yData[count_batch] = yData_batch.reshape(self.getArrayDims(1))

                count_batch += 1
            #endfor
        #endfor

        return (xData, yData)


    def loadData_1File(self, imagesFile, masksFile):

        return (np.load(imagesFile).astype(dtype=FORMATIMAGEDATA),
                np.load(masksFile) .astype(dtype=FORMATMASKDATA))


    def loadData_ListFiles(self, listImagesFiles, listMasksFiles):

        xData = []
        yData = []
        for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):
            xData.append(np.load(imagesFile).astype(dtype=FORMATIMAGEDATA))
            yData.append(np.load(masksFile) .astype(dtype=FORMATMASKDATA ))

        return (xData, yData)