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
from Preprocessing.BatchImageGenerator import *
from keras import backend as K


class LoadDataManager(object):

    def __init__(self, size_image, size_outnnet=None):
        self.size_image = size_image
        if size_outnnet and (size_outnnet != size_image):
            self.size_outnnet = size_outnnet
        else:
            self.size_outnnet = size_image

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
    def normalize_data(cls, xData):
        mean = np.mean(xData)
        std  = np.std(xData)
        return (xData - mean)/std

    @classmethod
    def shuffle_images(cls, xData, yData):
        # Generate random indexes to shuffle data
        randomIndexes = np.random.choice(range(xData.shape[0]), size=xData.shape[0], replace=False)
        return (xData[randomIndexes[:]],
                yData[randomIndexes[:]])


    @staticmethod
    def getArrayShapeKeras(num_batches, size_image):
        if K.image_data_format() == 'channels_first':
            return [num_batches, 1] + list(size_image)
        else:
            return [num_batches] + list(size_image) + [1]

    @classmethod
    def get_array_reshapedKeras(cls, array):
        return array.reshape(cls.getArrayShapeKeras(array.shape[0], array.shape[1:]))

    @staticmethod
    def get_limits_CropImage(size_image, size_outnnet):
        if (size_image==size_outnnet):
            return size_image
        else:
            return tuple(((s_i- s_o)/2, (s_i + s_o)/2) for (s_i, s_o) in zip(size_image, size_outnnet))

    def get_array_cropImages_outNnet(self, yData):

        if (self.size_image==self.size_outnnet):
            return yData
        else:
            ((z_back, z_front), (x_left, x_right), (y_down, y_up)) = self.get_limits_CropImage(self.size_image, self.size_outnnet)
            print ((z_back, z_front), (x_left, x_right), (y_down, y_up))
            return yData[..., z_back:z_front, x_left:x_right, y_down:y_up]


    def loadData_1FileBatches(self, imagesFile, masksFile, max_num_batches=10000, shuffle_images=SHUFFLEIMAGES):

        xData = FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA)
        yData = FileReader.getImageArray(masksFile) .astype(dtype=FORMATMASKDATA )

        num_batches = min(xData.shape[0], max_num_batches)

        xData = self.get_array_reshapedKeras(xData[0:num_batches])
        yData = self.get_array_reshapedKeras(self.get_cropped_images_outNnet(yData[0:num_batches]))

        if (shuffle_images):
            return self.shuffle_images(xData, yData)
        else:
            return (xData, yData)


    def loadData_ListFilesBatches(self, listImagesFiles, listMasksFiles, max_num_batches=10000, shuffle_images=SHUFFLEIMAGES):

        #loop over files to compute output array size
        num_batches = 0
        for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

            xData_part_shape = FileReader.getImageSize(imagesFile)
            num_batches += xData_part_shape[0]

            if( num_batches>=max_num_batches ):
                #reached the max size for output array
                num_batches = max_num_batches
                #limit images files to load form
                listImagesFiles = [listImagesFiles[j] for j in range(i+1)]
                listMasksFiles  = [listMasksFiles [j] for j in range(i+1)]
                break
        #endfor

        xData = np.ndarray(self.getArrayShapeKeras(num_batches, self.size_image  ), dtype=FORMATIMAGEDATA)
        yData = np.ndarray(self.getArrayShapeKeras(num_batches, self.size_outnnet), dtype=FORMATMASKDATA)

        count_batch = 0
        for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

            xData_part = FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA)
            yData_part = FileReader.getImageArray(masksFile) .astype(dtype=FORMATMASKDATA )

            num_batches_part = min(xData_part.shape[0], xData.shape[0] - count_batch)

            xData[count_batch:count_batch + num_batches_part] = self.get_array_reshapedKeras(xData_part[0:num_batches_part])
            yData[count_batch:count_batch + num_batches_part] = self.get_array_reshapedKeras(self.get_array_cropImages_outNnet(yData_part[0:num_batches_part]))

            count_batch += num_batches_part
        # endfor

        if (shuffle_images):
            return self.shuffle_images(xData, yData)
        else:
            return (xData, yData)


    def loadData_1File_BatchGenerator(self, ImagesOperator, imagesFile, masksFile, max_num_batches=10000, shuffle_images=SHUFFLEIMAGES):

        xData = FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA)
        yData = FileReader.getImageArray(masksFile) .astype(dtype=FORMATMASKDATA )

        imagesBatchGenerator = ImageBatchGenerator_2Arrays(ImagesOperator, xData, yData, self.size_image, size_batch=1, shuffle_images=shuffle_images)

        num_batches = min(len(imagesBatchGenerator), max_num_batches)

        xData = np.ndarray(self.getArrayShapeKeras(num_batches, self.size_image  ), dtype=FORMATIMAGEDATA)
        yData = np.ndarray(self.getArrayShapeKeras(num_batches, self.size_outnnet), dtype=FORMATMASKDATA)

        for index in range(num_batches):
            # Retrieve (xData, yData) directly from imagesDataGenerator
            (xData_batch, yData_batch) = next(imagesBatchGenerator)

            xData[index] = self.get_array_reshapedKeras(xData_batch)
            yData[index] = self.get_array_reshapedKeras(self.get_array_cropImages_outNnet(yData_batch))
        #endfor

        return (xData, yData)


    def loadData_ListFiles_BatchGenerator(self, ImagesOperator, listImagesFiles, listMasksFiles, max_num_batches=10000, shuffle_images=SHUFFLEIMAGES):

        #loop over files to compute output array size
        num_batches = 0
        for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

            xData = FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA)

            imagesBatchGenerator = ImageBatchGenerator_1Array(ImagesOperator, xData, self.size_image, size_batch=1, shuffle_images=shuffle_images)

            num_batches += len(imagesBatchGenerator)

            if( num_batches>=max_num_batches ):
                #reached the max size for output array
                num_batches = max_num_batches
                #limit images files to load form
                listImagesFiles = [listImagesFiles[j] for j in range(i+1)]
                listMasksFiles  = [listMasksFiles [j] for j in range(i+1)]
                break

        xData = np.ndarray(self.getArrayShapeKeras(num_batches, self.size_image  ), dtype=FORMATIMAGEDATA)
        yData = np.ndarray(self.getArrayShapeKeras(num_batches, self.size_outnnet), dtype=FORMATMASKDATA)

        count_batch = 0
        for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

            xData_part = FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA)
            yData_part = FileReader.getImageArray(masksFile) .astype(dtype=FORMATMASKDATA )

            imagesBatchGenerator = ImageBatchGenerator_2Arrays(ImagesOperator, xData_part, yData_part, self.size_image, size_batch=1, shuffle_images=shuffle_images)

            num_batches_part = min(len(imagesBatchGenerator), xData.shape[0] - count_batch)

            for index in range(num_batches_part):
                # Retrieve (xData, yData) directly from imagesDataGenerator
                (xData_batch, yData_batch) = next(imagesBatchGenerator)

                xData[count_batch] = self.get_array_reshapedKeras(xData_batch)
                yData[count_batch] = self.get_array_reshapedKeras(self.get_array_cropImages_outNnet(yData_batch))

                count_batch += 1
            #endfor
        #endfor

        return (xData, yData)


    def loadData_1File(self, imagesFile, masksFile):

        return (FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA),
                FileReader.getImageArray(masksFile ).astype(dtype=FORMATMASKDATA ))


    def loadData_ListFiles(self, listImagesFiles, listMasksFiles):

        xData = []
        yData = []
        for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):
            xData.append(FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA))
            yData.append(FileReader.getImageArray(masksFile) .astype(dtype=FORMATMASKDATA ))

        return (xData, yData)