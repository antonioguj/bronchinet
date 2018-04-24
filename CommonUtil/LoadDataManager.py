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
from Preprocessing.BatchImagesGenerator import *
from Preprocessing.SlidingWindowImages import *
from keras.backend import image_data_format as K_image_data_format
from keras.utils import to_categorical as K_to_categorical


class LoadDataManager(object):

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
    def get_array_categorical_masks_cls(self, yData, num_classes_out):
        return K_to_categorical(yData, num_classes=num_classes_out)


    @classmethod
    def loadData_1File(cls, imagesFile, masksFile, isCategoricalOut=False):

        if isCategoricalOut:
            return (FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA),
                    cls.get_array_categorical_masks_cls(FileReader.getImageArray(masksFile).astype(dtype=FORMATMASKDATA)))
        else:
            return (FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA),
                    FileReader.getImageArray(masksFile ).astype(dtype=FORMATMASKDATA ))

    @classmethod
    def loadData_ListFiles(cls, listImagesFiles, listMasksFiles, isCategoricalOut=False):

        if isCategoricalOut:
            return ([FileReader.getImageArray(file).astype(dtype=FORMATIMAGEDATA) for file in listImagesFiles],
                    [cls.get_array_categorical_masks_cls(FileReader.getImageArray(file).astype(dtype=FORMATMASKDATA)) for file in listMasksFiles])
        else:
            return ([FileReader.getImageArray(file).astype(dtype=FORMATIMAGEDATA) for file in listImagesFiles],
                    [FileReader.getImageArray(file).astype(dtype=FORMATMASKDATA)  for file in listMasksFiles])


class OperationsArraysUseInKeras(object):

    def __init__(self, size_image, num_classes_out=1, size_outnnet=None):
        self.size_image      = size_image
        self.num_classes_out = num_classes_out
        if size_outnnet and (size_outnnet != size_image):
            self.size_outnnet = size_outnnet
        else:
            self.size_outnnet = size_image

    @staticmethod
    def get_shape_Keras(num_images, size_image, num_channels):
        if K_image_data_format() == 'channels_first':
            return [num_images, num_channels] + list(size_image)
        elif K_image_data_format() == 'channels_last':
            return [num_images] + list(size_image) + [num_channels]
        else:
            return 0

    @staticmethod
    def get_array_reshaped_Keras(array):
        if K_image_data_format() == 'channels_first':
            # need to roll last dimensions, channels, to second dim:
            return np.rollaxis(array, -1, 1)
        elif K_image_data_format() == 'channels_last':
            return array
        else:
            return 0

    def get_num_channels_array(self, in_array_shape):
        if len(in_array_shape) == len(self.size_image):
            return 1
        else:
            return in_array_shape[-1]

    def get_shape_out_array(self, num_images, num_channels=1):
        if num_channels == 1:
            return [num_images] + list(self.size_image) + [1]
        else:
            return [num_images] + list(self.size_image) + [num_channels]

    def get_array_reshaped(self, in_array, num_channels=None):
        if not num_channels:
            num_channels = self.get_num_channels_array(in_array.shape)
        if num_channels == 1:
            return np.expand_dims(in_array, axis=-1)
        else:
            return in_array

    def get_array_categorical_masks(self, yData):
        return K_to_categorical(yData, num_classes=self.num_classes_out)

    @staticmethod
    def get_limits_cropImage(size_image, size_outnnet):

        if (size_image==size_outnnet):
            return size_image
        else:
            return tuple([(s_i- s_o)/2, (s_i + s_o)/2] for (s_i, s_o) in zip(size_image, size_outnnet))

    def get_array_cropImages_outNnet(self, yData):

        if (self.size_image==self.size_outnnet):
            return yData
        else:
            if len(self.size_image)==2:
                (x_left, x_right, y_down, y_up) = self.get_limits_cropImage(self.size_image, self.size_outnnet)
                return yData[..., x_left:x_right, y_down:y_up]
            elif len(self.size_image)==3:
                (z_back, z_front, x_left, x_right, y_down, y_up) = self.get_limits_cropImage(self.size_image, self.size_outnnet)
                return yData[..., z_back:z_front, x_left:x_right, y_down:y_up]


class LoadDataManagerInBatches(LoadDataManager, OperationsArraysUseInKeras):

    def __init__(self, size_image, num_classes_out=1, size_outnnet=None):

        super(LoadDataManagerInBatches, self).__init__(size_image, num_classes_out=num_classes_out, size_outnnet=size_outnnet)

    def get_num_channels_array(self, in_array_shape):
        if len(in_array_shape) == len(self.size_image) + 1:
            return 1
        else:
            return in_array_shape[-1]


    def loadData_1File(self, imagesFile, masksFile, max_num_images=10000, shuffle_images=SHUFFLEIMAGES):

        xData = FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA)
        yData = FileReader.getImageArray(masksFile) .astype(dtype=FORMATMASKDATA )

        num_images = min(xData.shape[0], max_num_images)

        xData = self.get_array_reshaped_Keras(self.get_array_reshaped(xData[0:num_images]))

        if self.num_classes_out > 1:
            yData = self.get_array_reshaped_Keras(self.get_array_categorical_masks(self.get_array_cropImages_outNnet(yData[0:num_images])))
        else:
            yData = self.get_array_reshaped_Keras(self.get_array_reshaped(self.get_array_cropImages_outNnet(yData[0:num_images])))

        if (shuffle_images):
            return self.shuffle_images(xData, yData)
        else:
            return (xData, yData)


    def loadData_ListFiles(self, listImagesFiles, listMasksFiles, max_num_images=10000, shuffle_images=SHUFFLEIMAGES):

        #loop over files to compute output array size
        num_images = 0
        for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

            xData_part_shape = FileReader.getImageSize(imagesFile)
            num_images += xData_part_shape[0]

            if( num_images>=max_num_images ):
                #reached the max size for output array
                num_images = max_num_images
                #limit images files to load form
                listImagesFiles = [listImagesFiles[j] for j in range(i+1)]
                listMasksFiles  = [listMasksFiles [j] for j in range(i+1)]
                break
        #endfor

        xData_shape = self.get_shape_out_array(num_images, num_channels=self.get_num_channels_array(xData_part_shape))
        yData_shape = self.get_shape_out_array(num_images, num_channels=self.num_classes_out)
        xData = np.ndarray(xData_shape, dtype=FORMATIMAGEDATA)
        yData = np.ndarray(yData_shape, dtype=FORMATMASKDATA)

        count_images = 0
        for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

            xData_part = FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA)
            yData_part = FileReader.getImageArray(masksFile) .astype(dtype=FORMATMASKDATA )

            num_images_part = min(xData_part.shape[0], xData.shape[0] - count_images)

            xData[count_images:count_images + num_images_part] = self.get_array_reshaped_Keras(self.get_array_reshaped(xData_part[0:num_images_part]))

            if self.num_classes_out > 1:
                yData[count_images:count_images + num_images_part] = self.get_array_reshaped_Keras(self.get_array_categorical_masks(self.get_array_cropImages_outNnet(yData_part[0:num_images_part])))
            else:
                yData[count_images:count_images + num_images_part] = self.get_array_reshaped_Keras(self.get_array_reshaped(self.get_array_cropImages_outNnet(yData_part[0:num_images_part])))

            count_images += num_images_part
        # endfor

        if (shuffle_images):
            return self.shuffle_images(xData, yData)
        else:
            return (xData, yData)


class LoadDataManagerInBatches_BatchGenerator(LoadDataManager, OperationsArraysUseInKeras):

    def __init__(self, size_image, prop_overlap, type_generator='SlidingWindow', num_classes_out=1, size_outnnet=None):
        self.prop_overlap   = prop_overlap
        self.type_generator = type_generator

        super(LoadDataManagerInBatches_BatchGenerator, self).__init__(size_image, num_classes_out=num_classes_out, size_outnnet=size_outnnet)


    def loadData_1File(self, imagesFile, masksFile, max_num_images=10000, shuffle_images=SHUFFLEIMAGES):

        xData = FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA)
        yData = FileReader.getImageArray(masksFile) .astype(dtype=FORMATMASKDATA )

        if (self.type_generator=='SlidingWindow'):
            imagesGenerator      = SlidingWindowImages3D(xData.shape, self.size_image, self.prop_overlap)
            batchImagesGenerator = BatchImagesGenerator_2Arrays(xData, yData, self.size_image, imagesGenerator, size_batch=1, shuffle_images=shuffle_images)

        num_images = min(len(batchImagesGenerator), max_num_images)

        xData_shape = self.get_shape_out_array(num_images, num_channels=self.get_num_channels_array(xData))
        yData_shape = self.get_shape_out_array(num_images, num_channels=self.num_classes_out)
        xData = np.ndarray(xData_shape, dtype=FORMATIMAGEDATA)
        yData = np.ndarray(yData_shape, dtype=FORMATMASKDATA)

        for i in range(num_images):
            # Retrieve (xData, yData) directly from imagesDataGenerator
            (xData_batch, yData_batch) = next(batchImagesGenerator)

            xData[i] = self.get_array_reshaped_Keras(self.get_array_reshaped(xData_batch[0]))

            if self.num_classes_out > 1:
                yData[i] = self.get_array_reshaped_Keras(self.get_array_categorical_masks(self.get_array_cropImages_outNnet(yData_batch[0])))
            else:
                yData[i] = self.get_array_reshaped_Keras(self.get_array_reshaped(self.get_array_cropImages_outNnet(yData_batch[0])))
        #endfor

        return (xData, yData)


    def loadData_ListFiles(self, listImagesFiles, listMasksFiles, max_num_images=10000, shuffle_images=SHUFFLEIMAGES):

        #loop over files to compute output array size
        num_images = 0
        for i, (imagesFile, masksFile) in enumerate(zip(listImagesFiles, listMasksFiles)):

            xData_part = FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA)

            if (self.type_generator == 'SlidingWindow'):
                imagesGenerator      = SlidingWindowImages3D(xData_part.shape, self.size_image, self.prop_overlap)
                batchImagesGenerator = BatchImagesGenerator_1Array(xData_part, self.size_image, imagesGenerator, size_batch=1, shuffle_images=shuffle_images)

            num_images += len(batchImagesGenerator)

            if( num_images>=max_num_images ):
                #reached the max size for output array
                num_images = max_num_images
                #limit images files to load form
                listImagesFiles = [listImagesFiles[j] for j in range(i+1)]
                listMasksFiles  = [listMasksFiles [j] for j in range(i+1)]
                break

        xData_shape = self.get_shape_out_array(num_images, num_channels=self.get_num_channels_array(xData_part.shape))
        yData_shape = self.get_shape_out_array(num_images, num_channels=self.num_classes_out)
        xData = np.ndarray(xData_shape, dtype=FORMATIMAGEDATA)
        yData = np.ndarray(yData_shape, dtype=FORMATMASKDATA)

        count_images = 0
        for imagesFile, masksFile in zip(listImagesFiles, listMasksFiles):

            xData_part = FileReader.getImageArray(imagesFile).astype(dtype=FORMATIMAGEDATA)
            yData_part = FileReader.getImageArray(masksFile) .astype(dtype=FORMATMASKDATA )

            if (self.type_generator == 'SlidingWindow'):
                imagesGenerator      = SlidingWindowImages3D(xData_part.shape, self.size_image, self.prop_overlap)
                batchImagesGenerator = BatchImagesGenerator_2Arrays(xData_part, yData_part, self.size_image, imagesGenerator, size_batch=1, shuffle_images=shuffle_images)

            num_images_part = min(len(batchImagesGenerator), xData.shape[0] - count_images)

            for i in range(num_images_part):
                # Retrieve (xData, yData) directly from imagesDataGenerator
                (xData_batch, yData_batch) = next(batchImagesGenerator)

                xData[count_images] = self.get_array_reshaped_Keras(self.get_array_reshaped(xData_batch[0]))

                if self.num_classes_out > 1:
                    yData[count_images] = self.get_array_reshaped_Keras(self.get_array_categorical_masks(self.get_array_cropImages_outNnet(yData_batch[0])))
                else:
                    yData[count_images] = self.get_array_reshaped_Keras(self.get_array_reshaped(self.get_array_cropImages_outNnet(yData_batch[0])))

                count_images += 1
            #endfor
        #endfor

        return (xData, yData)