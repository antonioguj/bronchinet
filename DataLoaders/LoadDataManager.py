#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from Common.Constants import *
from Common.FunctionsUtil import *
from DataLoaders.FileReaders import *



class LoadDataManager(object):

    @classmethod
    def load_1file(cls, in_image_file, in_label_file=None):
        if not isExistfile(in_image_file):
            message = 'input image file does not exist: \'%s\'' % (in_image_file)
            CatchErrorException(message)

        if in_label_file:
            if not isExistfile(in_label_file):
                message = 'input label file does not exist: \'%s\'' % (in_label_file)
                CatchErrorException(message)

            out_image_array = FileReader.get_image_array(in_image_file)
            out_label_array = FileReader.get_image_array(in_label_file)

            if out_image_array.shape != out_label_array.shape:
                message = 'input image and label array of different size: (\'%s\' != \'%s\')' % (out_image_array.shape,
                                                                                                 out_label_array.shape)
                CatchErrorException(message)

            return (out_image_array.astype(dtype=FORMATXDATA),
                    out_label_array.astype(dtype=FORMATYDATA))
        else:
            out_image_array = FileReader.get_image_array(in_image_file)

            return out_image_array.astype(dtype=FORMATXDATA)


    @classmethod
    def load_listFiles(cls, inlist_image_files, inlist_label_files=None):
        if inlist_label_files:
            if len(inlist_image_files) != len(inlist_label_files):
                message = 'number files images (\'%s\') and labels (\'%s\') are not equal' % (len(inlist_image_files),
                                                                                              len(inlist_label_files))
                CatchErrorException(message)
                
            outlist_image_array = []
            outlist_label_array = []
            for in_image_file, in_label_file in zip(inlist_image_files, inlist_label_files):
                (out_image_array, out_label_array) = cls.load_1file(in_image_file, in_label_file)
                outlist_image_array.append(out_image_array)
                outlist_label_array.append(out_label_array)
            # endfor

            return (outlist_image_array, outlist_label_array)
    
        else:
            outlist_image_array = []
            for in_image_file in inlist_image_files:
                outlist_image_array.append(cls.load_1file(in_image_file))
            # endfor

            return outlist_image_array



class LoadDataManagerInBatches(LoadDataManager):
    max_num_images_default = None

    def __init__(self, size_image):
        self.size_image = size_image

    @staticmethod
    def shuffle_images(in_image_data, in_label_data=None):
        # generate random indexes to shuffle data
        random_indexes = np.random.choice(range(in_image_data.shape[0]), size=in_image_data.shape[0], replace=False)
        if in_label_data is not None:
            return (in_image_data[random_indexes[:]], in_label_data[random_indexes[:]])
        else:
            return in_image_data[random_indexes[:]]


    def load_1file(self, in_image_file, in_label_file=None, max_num_images=max_num_images_default, shuffle_images=True):
        if in_label_file:
            (outbatch_image_array, outbatch_label_array) = super(LoadDataManagerInBatches, self).load_1file(in_image_file,
                                                                                                            in_label_file)
        else:
            outbatch_image_array = super(LoadDataManagerInBatches, self).load_1file(in_image_file)

        if outbatch_image_array[0].shape != self.size_image:
            message = 'size of images contained in input batch is different from fixed image size in class: ' \
                      '(\'%s\' != \'%s\'). change the image size in class equal to the first' % (outbatch_image_array[0].shape,
                                                                                                 self.size_image)
            CatchErrorException(message)

        num_images_batch = outbatch_image_array.shape[0]

        if max_num_images and (num_images_batch > max_num_images):
            outbatch_image_array = outbatch_image_array[0:max_num_images]
            if in_label_file:
                outbatch_label_array = outbatch_label_array[0:max_num_images]

        if in_label_file:
            if (shuffle_images):
                return self.shuffle_images(outbatch_image_array, outbatch_label_array)
            else:
                return (outbatch_image_array, outbatch_label_array)
        else:
            if (shuffle_images):
                return self.shuffle_images(outbatch_image_array)
            else:
                return outbatch_image_array


    def load_listFiles(self, inlist_image_files, inlist_label_files=None, max_num_images=max_num_images_default, shuffle_images=True):
        if inlist_label_files:
            if len(inlist_image_files) != len(inlist_label_files):
                message = 'number files images (\'%s\') and labels (\'%s\') are not equal' % (len(inlist_image_files),
                                                                                              len(inlist_label_files))
                CatchErrorException(message)

            outotal_image_array = np.array([], dtype=FORMATXDATA).reshape((0,) + self.size_image)
            outotal_label_array = np.array([], dtype=FORMATYDATA).reshape((0,) + self.size_image)
            sumrun_images_out = 0

            for in_image_file, in_label_file in zip(inlist_image_files, inlist_label_files):
                (outbatch_image_array, outbatch_label_array) = self.load_1file(in_image_file, in_label_file,
                                                                               max_num_images=None, shuffle_images=False)
                num_images_batch  = outbatch_image_array.shape[0]
                sumrun_images_out = sumrun_images_out + num_images_batch

                if max_num_images and (sumrun_images_out > max_num_images):
                    num_images_rest_batch = num_images_batch - (sumrun_images_out - max_num_images)
                    outbatch_image_array  = outbatch_image_array[0:num_images_rest_batch]
                    outbatch_label_array  = outbatch_label_array[0:num_images_rest_batch]

                outotal_image_array = np.concatenate((outotal_image_array, outbatch_image_array), axis=0)
                outotal_label_array = np.concatenate((outotal_label_array, outbatch_label_array), axis=0)
            # endfor

        else:
            outotal_image_array = np.array([], dtype=FORMATXDATA).reshape((0,) + self.size_image)
            sumrun_images_out = 0

            for in_image_file in inlist_image_files:
                outbatch_image_array = self.load_1file(in_image_file, max_num_images=None, shuffle_images=False)

                num_images_batch = outbatch_image_array.shape[0]
                sumrun_images_out = sumrun_images_out + num_images_batch

                if max_num_images and (sumrun_images_out > max_num_images):
                    num_images_rest_batch = num_images_batch - (sumrun_images_out - max_num_images)
                    outbatch_image_array  = outbatch_image_array[0:num_images_rest_batch]

                outotal_image_array = np.concatenate((outotal_image_array, outbatch_image_array), axis=0)
            # endfor

        if inlist_label_files:
            if (shuffle_images):
                return self.shuffle_images(outotal_image_array, outotal_label_array)
            else:
                return (outotal_image_array, outotal_label_array)
        else:
            if (shuffle_images):
                return self.shuffle_images(outotal_image_array)
            else:
                return outotal_image_array