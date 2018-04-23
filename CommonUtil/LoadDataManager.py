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
from keras.backend import image_data_format as K_image_data_format
from keras.utils import to_categorical as K_to_categorical


class LoadDataManager(object):

    def __init__(self, size_image, num_channels_in=1, num_classes_out=1, size_outnnet=None):
        self.size_image      = size_image
        self.num_channels_in = num_channels_in
        self.num_classes_out = num_classes_out
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
    def getArrayShapeKeras(num_batches, size_image, num_channels):
        if K_image_data_format() == 'channels_first':
            return [num_batches, num_channels] + list(size_image)
        else:
            return [num_batches] + list(size_image) + [num_channels]

    def get_array_reshapedKeras(self, array):
        if len(array.shape) == (len(self.size_image) + 1):
            # array_shape: (num_batch, size_batch) #not multichannel
            return array.reshape(self.getArrayShapeKeras(array.shape[0], array.shape[1:], 1))
        else:
            # array_shape: (num_batch, size_batch, num_channel) #multichannel
            return array.reshape(self.getArrayShapeKeras(array.shape[0], array.shape[2:-1], array.shape[-1]))


    @staticmethod
    def get_limits_cropImage(size_image, size_outnnet):
        if (size_image==size_outnnet):
            return size_image
        else:
            return tuple(((s_i- s_o)/2, (s_i + s_o)/2) for (s_i, s_o) in zip(size_image, size_outnnet))

    def get_array_cropImages_outNnet(self, yData):
        if (self.size_image==self.size_outnnet):
            return yData
        else:
            if len(self.size_image)==2:
                ((x_left, x_right), (y_down, y_up)) = self.get_limits_cropImage(self.size_image, self.size_outnnet)
                return yData[..., x_left:x_right, y_down:y_up]
            elif len(self.size_image)==3:
                ((z_back, z_front), (x_left, x_right), (y_down, y_up)) = self.get_limits_cropImage(self.size_image, self.size_outnnet)
                return yData[..., z_back:z_front, x_left:x_right, y_down:y_up]


    def get_array_categorical_masks(self, yData):
        return K_to_categorical(yData, num_classes=self.num_classes_out)


    def loadData_1FileBatches(self, imagesFile, masksFile, max_num_batches=10000, shuffle_images=SHUFFLEIMAGES):

        xData = FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA)
        yData = FileReader.getImageArray(masksFile) .astype(dtype=FORMATMASKDATA )

        num_batches = min(xData.shape[0], max_num_batches)

        xData = self.get_array_reshapedKeras(xData[0:num_batches])

        if self.num_classes_out > 1:
            yData = self.get_array_reshapedKeras(self.get_array_categorical_masks(self.get_array_cropImages_outNnet(yData[0:num_batches])))
        else:
            yData = self.get_array_reshapedKeras(self.get_array_cropImages_outNnet(yData[0:num_batches]))

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

        xData = np.ndarray(self.getArrayShapeKeras(num_batches, self.size_image  , self.num_channels_in), dtype=FORMATIMAGEDATA)
        yData = np.ndarray(self.getArrayShapeKeras(num_batches, self.size_outnnet, self.num_classes_out), dtype=FORMATMASKDATA)

        count_batch = 0
        for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

            xData_part = FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA)
            yData_part = FileReader.getImageArray(masksFile) .astype(dtype=FORMATMASKDATA )

            num_batches_part = min(xData_part.shape[0], xData.shape[0] - count_batch)

            xData[count_batch:count_batch + num_batches_part] = self.get_array_reshapedKeras(xData_part[0:num_batches_part])

            if self.num_classes_out > 1:
                yData[count_batch:count_batch + num_batches_part] = self.get_array_reshapedKeras(self.get_array_categorical_masks(self.get_array_cropImages_outNnet(yData_part[0:num_batches_part])))
            else:
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

        xData = np.ndarray(self.getArrayShapeKeras(num_batches, self.size_image  , self.num_channels_in), dtype=FORMATIMAGEDATA)
        yData = np.ndarray(self.getArrayShapeKeras(num_batches, self.size_outnnet, self.num_classes_out), dtype=FORMATMASKDATA)

        for index in range(num_batches):
            # Retrieve (xData, yData) directly from imagesDataGenerator
            (xData_batch, yData_batch) = next(imagesBatchGenerator)

            xData[index] = self.get_array_reshapedKeras(xData_batch)

            if self.num_classes_out > 1:
                yData[index] = self.get_array_reshapedKeras(self.get_array_categorical_masks(self.get_array_cropImages_outNnet(yData_batch)))
            else:
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

        xData = np.ndarray(self.getArrayShapeKeras(num_batches, self.size_image  , self.num_channels_in), dtype=FORMATIMAGEDATA)
        yData = np.ndarray(self.getArrayShapeKeras(num_batches, self.size_outnnet, self.num_classes_out), dtype=FORMATMASKDATA)

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

                if self.num_classes_out > 1:
                    yData[count_batch] = self.get_array_reshapedKeras(self.get_array_categorical_masks(self.get_array_cropImages_outNnet(yData_batch)))
                else:
                    yData[count_batch] = self.get_array_reshapedKeras(self.get_array_cropImages_outNnet(yData_batch))

                count_batch += 1
            #endfor
        #endfor

        return (xData, yData)


    def loadData_1File(self, imagesFile, masksFile, isCategoricalOut=False):

        isCategoricalOut = isCategoricalOut and (self.num_classes_out > 1)

        if isCategoricalOut:
            return (FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA),
                    self.get_array_categorical_masks(FileReader.getImageArray(masksFile).astype(dtype=FORMATMASKDATA)))
        else:
            return (FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA),
                    FileReader.getImageArray(masksFile ).astype(dtype=FORMATMASKDATA ))


    def loadData_ListFiles(self, listImagesFiles, listMasksFiles, isCategoricalOut=False):

        isCategoricalOut = isCategoricalOut and (self.num_classes_out > 1)

        if isCategoricalOut:
            return ([FileReader.getImageArray(file).astype(dtype=FORMATIMAGEDATA) for file in listImagesFiles],
                    [self.get_array_categorical_masks(FileReader.getImageArray(file).astype(dtype=FORMATMASKDATA)) for file in listMasksFiles])
        else:
            return ([FileReader.getImageArray(file).astype(dtype=FORMATIMAGEDATA) for file in listImagesFiles],
                    [FileReader.getImageArray(file).astype(dtype=FORMATMASKDATA)  for file in listMasksFiles])