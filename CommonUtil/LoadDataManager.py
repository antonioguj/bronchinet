#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.BatchDataGenerator import *
from CommonUtil.FileReaders import *
from CommonUtil.FunctionsUtil import *


class LoadDataManager(object):

    @staticmethod
    def runChecker1(images_file, masks_file):
        if (not isExistfile(images_file) or isExistfile(masks_file)):
            message = "Images or masks file does not exist (\'%s\',\'%s\')" % (images_file, masks_file)
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
        # generate random indexes to shuffle data
        randomIndexes = np.random.choice(range(xData.shape[0]), size=xData.shape[0], replace=False)
        return (xData[randomIndexes[:]],
                yData[randomIndexes[:]])

    @staticmethod
    def loadData_1File(images_file, masks_file):

        return (FileReader.getImageArray(images_file).astype(dtype=FORMATXDATA),
                FileReader.getImageArray(masks_file ).astype(dtype=FORMATYDATA))

    @staticmethod
    def loadData_ListFiles(listImagesFiles, listMasksFiles):

        return ([FileReader.getImageArray(file).astype(dtype=FORMATXDATA) for file in listImagesFiles],
                [FileReader.getImageArray(file).astype(dtype=FORMATYDATA)  for file in listMasksFiles])


class LoadDataManagerInBatches(LoadDataManager):

    def __init__(self, size_image, num_classes_out=1, size_outUnet=None):

        self.num_classes_out     = num_classes_out
        self.array_shape_manager = ArrayShapeManagerInBatches(size_image, is_shaped_Keras=True, num_classes_out=num_classes_out, size_outUnet=size_outUnet)


    def loadData_1File(self, images_file, masks_file, max_num_images=10000, shuffle_images=SHUFFLEIMAGES):

        xData = FileReader.getImageArray(images_file).astype(dtype=FORMATXDATA)
        yData = FileReader.getImageArray(masks_file) .astype(dtype=FORMATYDATA)

        num_images = min(xData.shape[0], max_num_images)

        xData = self.array_shape_manager.get_xData_array_reshaped(xData[0:num_images])
        yData = self.array_shape_manager.get_yData_array_reshaped(yData[0:num_images])

        if (shuffle_images):
            return self.shuffle_images(xData, yData)
        else:
            return (xData, yData)


    def loadData_ListFiles(self, listImagesFiles, listMasksFiles, max_num_images=10000, shuffle_images=SHUFFLEIMAGES):

        #loop over files to compute output array size
        num_images = 0
        for i, (images_file, masks_file) in enumerate(zip(listImagesFiles, listMasksFiles)):

            xData_part_shape = FileReader.getImageSize(images_file)
            num_images += xData_part_shape[0]

            if( num_images>=max_num_images ):
                #reached the max size for output array
                num_images = max_num_images
                #limit images files to load form
                listImagesFiles = [listImagesFiles[j] for j in range(i+1)]
                listMasksFiles  = [listMasksFiles [j] for j in range(i+1)]
                break
        #endfor

        xData_shape = self.array_shape_manager.get_shape_out_array(num_images, self.array_shape_manager.get_num_channels_array(xData_part_shape))
        yData_shape = self.array_shape_manager.get_shape_out_array(num_images, self.num_classes_out)

        xData = np.ndarray(xData_shape, dtype=FORMATXDATA)
        yData = np.ndarray(yData_shape, dtype=FORMATYDATA)

        count_images = 0
        for images_file, masks_file in zip(listImagesFiles, listMasksFiles):

            xData_part = FileReader.getImageArray(images_file).astype(dtype=FORMATXDATA)
            yData_part = FileReader.getImageArray(masks_file) .astype(dtype=FORMATYDATA)

            num_images_part = min(xData_part.shape[0], xData.shape[0] - count_images)

            xData[count_images:count_images + num_images_part] = self.array_shape_manager.get_xData_array_reshaped(xData_part[0:num_images_part])
            yData[count_images:count_images + num_images_part] = self.array_shape_manager.get_yData_array_reshaped(yData_part[0:num_images_part])

            count_images += num_images_part
        #endfor

        if (shuffle_images):
            return self.shuffle_images(xData, yData)
        else:
            return (xData, yData)


class LoadDataManagerInBatches_DataGenerator(LoadDataManager):

    def __init__(self, size_image, images_generator, num_classes_out=1, size_outUnet=None):

        self.size_image       = size_image
        self.images_generator = images_generator

        self.num_classes_out     = num_classes_out
        self.array_shape_manager = ArrayShapeManager(size_image, is_shaped_Keras=True, num_classes_out=num_classes_out, size_outUnet=size_outUnet)


    def loadData_1File(self, images_file, masks_file, max_num_images=10000, shuffle_images=SHUFFLEIMAGES):

        xData = FileReader.getImageArray(images_file).astype(dtype=FORMATXDATA)
        yData = FileReader.getImageArray(masks_file) .astype(dtype=FORMATYDATA)

        batch_data_generator = BatchDataGenerator_2Arrays(self.size_image, xData, yData, self.images_generator, size_batch=1, shuffle=shuffle_images)

        num_images = min(len(batch_data_generator), max_num_images)

        xData_shape = self.array_shape_manager.get_shape_out_array(num_images, self.array_shape_manager.get_num_channels_array(xData.shape))
        yData_shape = self.array_shape_manager.get_shape_out_array(num_images, self.num_classes_out)

        xData = np.ndarray(xData_shape, dtype=FORMATXDATA)
        yData = np.ndarray(yData_shape, dtype=FORMATYDATA)

        for i in range(num_images):
            # Retrieve (xData, yData) directly from imagesDataGenerator
            (xData_batch, yData_batch) = next(batch_data_generator)

            xData[i] = self.array_shape_manager.get_xData_array_reshaped(xData_batch[0])
            yData[i] = self.array_shape_manager.get_yData_array_reshaped(yData_batch[0])
        #endfor

        return (xData, yData)


    def loadData_ListFiles(self, listImagesFiles, listMasksFiles, max_num_images=10000, shuffle_images=SHUFFLEIMAGES):

        #loop over files to compute output array size
        num_images = 0
        for i, (images_file, masks_file) in enumerate(zip(listImagesFiles, listMasksFiles)):

            xData_part = FileReader.getImageArray(images_file).astype(dtype=FORMATXDATA)

            batch_data_generator = BatchDataGenerator_1Array(self.size_image, xData_part, self.images_generator, size_batch=1, shuffle=shuffle_images)

            num_images += len(batch_data_generator)

            if( num_images>=max_num_images ):
                #reached the max size for output array
                num_images = max_num_images
                #limit images files to load form
                listImagesFiles = [listImagesFiles[j] for j in range(i+1)]
                listMasksFiles  = [listMasksFiles [j] for j in range(i+1)]
                break

        xData_shape = self.array_shape_manager.get_shape_out_array(num_images, self.array_shape_manager.get_num_channels_array(xData_part.shape))
        yData_shape = self.array_shape_manager.get_shape_out_array(num_images, self.num_classes_out)

        xData = np.ndarray(xData_shape, dtype=FORMATXDATA)
        yData = np.ndarray(yData_shape, dtype=FORMATYDATA)

        count_images = 0
        for images_file, masks_file in zip(listImagesFiles, listMasksFiles):

            xData_part = FileReader.getImageArray(images_file).astype(dtype=FORMATXDATA)
            yData_part = FileReader.getImageArray(masks_file) .astype(dtype=FORMATYDATA)

            batch_data_generator = BatchDataGenerator_2Arrays(self.size_image, xData_part, yData_part, self.images_generator, size_batch=1, shuffle=shuffle_images)

            num_images_part = min(len(batch_data_generator), xData.shape[0] - count_images)

            for i in range(num_images_part):
                # Retrieve (xData, yData) directly from imagesDataGenerator
                (xData_batch, yData_batch) = next(batch_data_generator)

                xData[count_images] = self.array_shape_manager.get_xData_array_reshaped(xData_batch[0])
                yData[count_images] = self.array_shape_manager.get_yData_array_reshaped(yData_batch[0])

                count_images += 1
            #endfor
        #endfor

        return (xData, yData)