#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from keras.layers import Input, merge, concatenate, Dropout, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D, Conv2DTranspose
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, Cropping3D, Conv3DTranspose
from keras.models import Model, load_model


class NeuralNetwork(object):

    @classmethod
    def getModel(cls):
        pass
    @classmethod
    def getModelAndCompile(cls, optimizer, lossfunction, metrics):
        return cls.getModel().compile(optimizer=optimizer,
                                      loss=lossfunction,
                                      metrics=metrics )
    @staticmethod
    def getLoadSavedModel(model_saved_path, custom_objects=None):
        return load_model(model_saved_path, custom_objects=custom_objects)


class Unet2D(NeuralNetwork):

    nbfilters   = 32
    size_filter = (3, 3)

    @classmethod
    def getModel(cls, image_nx, image_ny, type_padding='same'):

        inputs = Input((image_nx, image_ny, 1))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution2D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(inputs)
        hidlayer_dwl1_2 = Convolution2D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl1_1)
        hidlayer_dwl2_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution2D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_1)
        hidlayer_dwl2_3 = Convolution2D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_2)
        hidlayer_dwl3_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution2D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_1)
        hidlayer_dwl3_3 = Convolution2D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_2)
        hidlayer_dwl4_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl3_3)

        nbfilters_dwl4  = 2*nbfilters_dwl3
        hidlayer_dwl4_2 = Convolution2D(nbfilters_dwl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl4_1)
        hidlayer_dwl4_3 = Convolution2D(nbfilters_dwl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl4_2)
        hidlayer_dwl5_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl4_3)

        nbfilters_dwl5  = 2*nbfilters_dwl4
        hidlayer_dwl5_2 = Convolution2D(nbfilters_dwl5, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl5_1)
        hidlayer_dwl5_3 = Convolution2D(nbfilters_dwl5, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl5_2)

        hidlayer_upl4_1 = UpSampling2D(size=(2, 2))(hidlayer_dwl5_3)
        hidlayer_upl4_1 = merge([hidlayer_upl4_1, hidlayer_dwl4_3], mode='concat', concat_axis=-1)

        nbfilters_upl4  = nbfilters_dwl4
        hidlayer_upl4_2 = Convolution2D(nbfilters_upl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl4_1)
        hidlayer_upl4_3 = Convolution2D(nbfilters_upl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl4_2)

        hidlayer_upl3_1 = UpSampling2D(size=(2, 2))(hidlayer_upl4_3)
        hidlayer_upl3_1 = merge([hidlayer_upl3_1, hidlayer_dwl3_3], mode='concat', concat_axis=-1)

        nbfilters_upl3  = nbfilters_dwl3
        hidlayer_upl3_2 = Convolution2D(nbfilters_upl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl3_1)
        hidlayer_upl3_3 = Convolution2D(nbfilters_upl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl3_2)

        hidlayer_upl2_1 = UpSampling2D(size=(2, 2))(hidlayer_upl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution2D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_1)
        hidlayer_upl2_3 = Convolution2D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_2)

        hidlayer_upl1_1 = UpSampling2D(size=(2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution2D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_1)
        hidlayer_upl1_3 = Convolution2D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_2)

        outputs = Convolution2D(1, (1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet2D_Shallow(NeuralNetwork):

    nbfilters   = 32
    size_filter = (3, 3)

    @classmethod
    def getModel(cls, image_nx, image_ny, type_padding='same'):

        inputs = Input((image_nx, image_ny, 1))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution2D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(inputs)
        hidlayer_dwl1_2 = Convolution2D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl1_1)
        hidlayer_dwl2_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution2D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_1)
        hidlayer_dwl2_3 = Convolution2D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_2)
        hidlayer_dwl3_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution2D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_1)
        hidlayer_dwl3_3 = Convolution2D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_2)

        hidlayer_upl2_1 = UpSampling2D(size=(2, 2))(hidlayer_dwl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution2D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_1)
        hidlayer_upl2_3 = Convolution2D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_2)

        hidlayer_upl1_1 = UpSampling2D(size=(2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_1], mode='concat', concat_axis=-1)

        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution2D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_1)
        hidlayer_upl1_3 = Convolution2D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_2)

        outputs = Convolution2D(1, (1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet2D_Dropout(NeuralNetwork):

    nbfilters   = 32
    size_filter = (3, 3)
    dropoutrate = 0.2

    @classmethod
    def getModel(cls, image_nx, image_ny, type_padding='same'):

        inputs = Input((image_nx, image_ny, 1))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution2D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(inputs)
        hidlayer_dwl1_1 = Dropout(cls.dropoutrate)(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = Convolution2D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = Dropout(cls.dropoutrate)(hidlayer_dwl1_2)
        hidlayer_dwl2_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution2D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_1)
        hidlayer_dwl2_2 = Dropout(cls.dropoutrate)( hidlayer_dwl2_2)
        hidlayer_dwl2_3 = Convolution2D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = Dropout(cls.dropoutrate)( hidlayer_dwl2_3)
        hidlayer_dwl3_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution2D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_1)
        hidlayer_dwl3_2 = Dropout(cls.dropoutrate)(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = Convolution2D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = Dropout(cls.dropoutrate)(hidlayer_dwl3_3)
        hidlayer_dwl4_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl3_3)

        nbfilters_dwl4  = 2*nbfilters_dwl3
        hidlayer_dwl4_2 = Convolution2D(nbfilters_dwl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl4_1)
        hidlayer_dwl4_2 = Dropout(cls.dropoutrate)(hidlayer_dwl4_2)
        hidlayer_dwl4_3 = Convolution2D(nbfilters_dwl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl4_2)
        hidlayer_dwl4_3 = Dropout(cls.dropoutrate)(hidlayer_dwl4_3)
        hidlayer_dwl5_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl4_3)

        nbfilters_dwl5  = 2*nbfilters_dwl4
        hidlayer_dwl5_2 = Convolution2D(nbfilters_dwl5, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl5_1)
        hidlayer_dwl5_2 = Dropout(cls.dropoutrate)(hidlayer_dwl5_2)
        hidlayer_dwl5_3 = Convolution2D(nbfilters_dwl5, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl5_2)
        hidlayer_dwl5_3 = Dropout(cls.dropoutrate)(hidlayer_dwl5_3)

        hidlayer_upl4_1 = UpSampling2D(size=(2, 2))(hidlayer_dwl5_3)
        hidlayer_upl4_1 = merge([hidlayer_upl4_1, hidlayer_dwl4_3], mode='concat', concat_axis=-1)

        nbfilters_upl4  = nbfilters_dwl4
        hidlayer_upl4_2 = Convolution2D( nbfilters_upl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl4_1)
        hidlayer_upl4_2 = Dropout(cls.dropoutrate)(hidlayer_upl4_2)
        hidlayer_upl4_3 = Convolution2D( nbfilters_upl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl4_2)
        hidlayer_upl4_3 = Dropout(cls.dropoutrate)(hidlayer_upl4_3)

        hidlayer_upl3_1 = UpSampling2D(size=(2, 2))(hidlayer_upl4_3)
        hidlayer_upl3_1 = merge([hidlayer_upl3_1, hidlayer_dwl3_3], mode='concat', concat_axis=-1)

        nbfilters_upl3  = nbfilters_dwl3
        hidlayer_upl3_2 = Convolution2D( nbfilters_upl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl3_1)
        hidlayer_upl3_2 = Dropout(cls.dropoutrate)(hidlayer_upl3_2)
        hidlayer_upl3_3 = Convolution2D( nbfilters_upl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl3_2)
        hidlayer_upl3_3 = Dropout(cls.dropoutrate)(hidlayer_upl3_3)

        hidlayer_upl2_1 = UpSampling2D(size=(2, 2))(hidlayer_upl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution2D( nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_1)
        hidlayer_upl2_2 = Dropout(cls.dropoutrate)(hidlayer_upl2_2)
        hidlayer_upl2_3 = Convolution2D( nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_2)
        hidlayer_upl2_3 = Dropout(cls.dropoutrate)(hidlayer_upl2_3)

        hidlayer_upl1_1 = UpSampling2D(size=(2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution2D( nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_1)
        hidlayer_upl1_2 = Dropout(cls.dropoutrate)(hidlayer_upl1_2)
        hidlayer_upl1_3 = Convolution2D( nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_2)
        hidlayer_upl1_3 = Dropout(cls.dropoutrate)(hidlayer_upl1_3)

        outputs = Convolution2D(1, (1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet2D_Batchnorm(NeuralNetwork):

    nbfilters   = 32
    size_filter = (3, 3)

    @classmethod
    def getModel(cls, image_nx, image_ny, type_padding='same'):

        inputs = Input((image_nx, image_ny, 1))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution2D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(inputs)
        hidlayer_dwl1_1 = BatchNormalization()(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = Convolution2D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = BatchNormalization()(hidlayer_dwl1_2)
        hidlayer_dwl2_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution2D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_1)
        hidlayer_dwl2_2 = BatchNormalization()(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = Convolution2D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = BatchNormalization()(hidlayer_dwl2_3)
        hidlayer_dwl3_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution2D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_1)
        hidlayer_dwl3_2 = BatchNormalization()(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = Convolution2D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = BatchNormalization()(hidlayer_dwl3_3)
        hidlayer_dwl4_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl3_3)

        nbfilters_dwl4  = 2*nbfilters_dwl3
        hidlayer_dwl4_2 = Convolution2D(nbfilters_dwl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl4_1)
        hidlayer_dwl4_2 = BatchNormalization()(hidlayer_dwl4_2)
        hidlayer_dwl4_3 = Convolution2D(nbfilters_dwl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl4_2)
        hidlayer_dwl4_3 = BatchNormalization()(hidlayer_dwl4_3)
        hidlayer_dwl5_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl4_3)

        nbfilters_dwl5  = 2*nbfilters_dwl4
        hidlayer_dwl5_2 = Convolution2D(nbfilters_dwl5, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl5_1)
        hidlayer_dwl5_2 = BatchNormalization()(hidlayer_dwl5_2)
        hidlayer_dwl5_3 = Convolution2D(nbfilters_dwl5, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl5_2)
        hidlayer_dwl5_3 = BatchNormalization()(hidlayer_dwl5_3)

        hidlayer_upl4_1 = UpSampling2D(size=(2, 2))(hidlayer_dwl5_3)
        hidlayer_upl4_1 = merge([hidlayer_upl4_1, hidlayer_dwl4_3], mode='concat', concat_axis=-1)

        nbfilters_upl4  = nbfilters_dwl4
        hidlayer_upl4_2 = Convolution2D( nbfilters_upl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl4_1)
        hidlayer_upl4_2 = BatchNormalization()(hidlayer_upl4_2)
        hidlayer_upl4_3 = Convolution2D( nbfilters_upl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl4_2)
        hidlayer_upl4_3 = BatchNormalization()(hidlayer_upl4_3)

        hidlayer_upl3_1 = UpSampling2D(size=(2, 2))(hidlayer_upl4_3)
        hidlayer_upl3_1 = merge([hidlayer_upl3_1, hidlayer_dwl3_3], mode='concat', concat_axis=-1)

        nbfilters_upl3  = nbfilters_dwl3
        hidlayer_upl3_2 = Convolution2D( nbfilters_upl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl3_1)
        hidlayer_upl3_2 = BatchNormalization()(hidlayer_upl3_2)
        hidlayer_upl3_3 = Convolution2D( nbfilters_upl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl3_2)
        hidlayer_upl3_3 = BatchNormalization()(hidlayer_upl3_3)

        hidlayer_upl2_1 = UpSampling2D(size=(2, 2))(hidlayer_upl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution2D( nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_1)
        hidlayer_upl2_2 = BatchNormalization()(hidlayer_upl2_2)
        hidlayer_upl2_3 = Convolution2D( nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_2)
        hidlayer_upl2_3 = BatchNormalization()(hidlayer_upl2_3)

        hidlayer_upl1_1 = UpSampling2D(size=(2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution2D( nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_1)
        hidlayer_upl1_2 = BatchNormalization()(hidlayer_upl1_2)
        hidlayer_upl1_3 = Convolution2D( nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_2)
        hidlayer_upl1_3 = BatchNormalization()(hidlayer_upl1_3)

        outputs = Convolution2D(1, (1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet2D_Shallow_Dropout(NeuralNetwork):

    nbfilters   = 32
    size_filter = (3, 3)
    dropoutrate = 0.2

    @classmethod
    def getModel(cls, image_nx, image_ny, type_padding='same'):

        inputs = Input((image_nx, image_ny, 1))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution2D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(inputs)
        hidlayer_dwl1_1 = Dropout(cls.dropoutrate)(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = Convolution2D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = Dropout(cls.dropoutrate)(hidlayer_dwl1_2)
        hidlayer_dwl2_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution2D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_1)
        hidlayer_dwl2_2 = Dropout(cls.dropoutrate)(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = Convolution2D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = Dropout(cls.dropoutrate)(hidlayer_dwl2_3)
        hidlayer_dwl3_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution2D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_1)
        hidlayer_dwl3_2 = Dropout(cls.dropoutrate)(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = Convolution2D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = Dropout(cls.dropoutrate)(hidlayer_dwl3_3)

        hidlayer_upl2_1 = UpSampling2D(size=(2, 2))(hidlayer_dwl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution2D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_1)
        hidlayer_upl2_2 = Dropout(cls.dropoutrate)(hidlayer_upl2_2)
        hidlayer_upl2_3 = Convolution2D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_2)
        hidlayer_upl2_3 = Dropout(cls.dropoutrate)(hidlayer_upl2_3)

        hidlayer_upl1_1 = UpSampling2D(size=(2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_1], mode='concat', concat_axis=-1)

        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution2D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_1)
        hidlayer_upl1_2 = Dropout(cls.dropoutrate)(hidlayer_upl1_2)
        hidlayer_upl1_3 = Convolution2D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_2)
        hidlayer_upl1_3 = Dropout(cls.dropoutrate)(hidlayer_upl1_3)

        outputs = Convolution2D(1, (1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet2D_Shallow_Batchnorm(NeuralNetwork):

    nbfilters   = 32
    size_filter = (3, 3)

    @classmethod
    def getModel(cls, image_nx, image_ny, type_padding='same'):

        inputs = Input((image_nx, image_ny, 1))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution2D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(inputs)
        hidlayer_dwl1_1 = BatchNormalization()(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = Convolution2D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = BatchNormalization()(hidlayer_dwl1_2)
        hidlayer_dwl2_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution2D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_1)
        hidlayer_dwl2_2 = BatchNormalization()(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = Convolution2D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = BatchNormalization()(hidlayer_dwl2_3)
        hidlayer_dwl3_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution2D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_1)
        hidlayer_dwl3_2 = BatchNormalization()(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = Convolution2D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = BatchNormalization()(hidlayer_dwl3_3)

        hidlayer_upl2_1 = UpSampling2D(size=(2, 2))(hidlayer_dwl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution2D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_1)
        hidlayer_upl2_2 = BatchNormalization()(hidlayer_upl2_2)
        hidlayer_upl2_3 = Convolution2D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_2)
        hidlayer_upl2_3 = BatchNormalization()(hidlayer_upl2_3)

        hidlayer_upl1_1 = UpSampling2D(size=(2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_1], mode='concat', concat_axis=-1)

        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution2D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_1)
        hidlayer_upl1_2 = BatchNormalization()(hidlayer_upl1_2)
        hidlayer_upl1_3 = Convolution2D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_2)
        hidlayer_upl1_3 = BatchNormalization()(hidlayer_upl1_3)

        outputs = Convolution2D(1, (1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet3D(NeuralNetwork):

    nbfilters   = 32
    size_filter = (3, 3, 3)

    @classmethod
    def getModel(cls, (image_nz, image_nx, image_ny), type_padding='same'):

        inputs = Input((image_nz, image_nx, image_ny, 1))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution3D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(inputs)
        hidlayer_dwl1_2 = Convolution3D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl1_1)
        hidlayer_dwl2_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution3D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_1)
        hidlayer_dwl2_3 = Convolution3D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_2)
        hidlayer_dwl3_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution3D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_1)
        hidlayer_dwl3_3 = Convolution3D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_2)
        hidlayer_dwl4_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl3_3)

        nbfilters_dwl4  = 2*nbfilters_dwl3
        hidlayer_dwl4_2 = Convolution3D(nbfilters_dwl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl4_1)
        hidlayer_dwl4_3 = Convolution3D(nbfilters_dwl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl4_2)
        hidlayer_dwl5_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl4_3)

        nbfilters_dwl5  = 2*nbfilters_dwl4
        hidlayer_dwl5_2 = Convolution3D(nbfilters_dwl5, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl5_1)
        hidlayer_dwl5_3 = Convolution3D(nbfilters_dwl5, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl5_2)

        hidlayer_upl4_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_dwl5_3)
        hidlayer_upl4_1 = merge([hidlayer_upl4_1, hidlayer_dwl4_3], mode='concat', concat_axis=-1)

        nbfilters_upl4  = nbfilters_dwl4
        hidlayer_upl4_2 = Convolution3D(nbfilters_upl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl4_1)
        hidlayer_upl4_3 = Convolution3D(nbfilters_upl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl4_2)

        hidlayer_upl3_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_upl4_3)
        hidlayer_upl3_1 = merge([hidlayer_upl3_1, hidlayer_dwl3_3], mode='concat', concat_axis=-1)

        nbfilters_upl3  = nbfilters_dwl3
        hidlayer_upl3_2 = Convolution3D(nbfilters_upl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl3_1)
        hidlayer_upl3_3 = Convolution3D(nbfilters_upl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl3_2)

        hidlayer_upl2_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_upl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution3D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_1)
        hidlayer_upl2_3 = Convolution3D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_2)

        hidlayer_upl1_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution3D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_1)
        hidlayer_upl1_3 = Convolution3D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_2)

        outputs = Convolution3D(1, (1, 1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet3D_Dropout(NeuralNetwork):

    nbfilters   = 26
    size_filter = (3, 3, 3)
    dropoutrate = 0.2

    @classmethod
    def getModel(cls, (image_nz, image_nx, image_ny), type_padding='same'):

        inputs = Input((image_nz, image_nx, image_ny, 1))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution3D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(inputs)
        hidlayer_dwl1_1 = Dropout(cls.dropoutrate)(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = Convolution3D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = Dropout(cls.dropoutrate)(hidlayer_dwl1_2)
        hidlayer_dwl2_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution3D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_1)
        hidlayer_dwl2_2 = Dropout(cls.dropoutrate)(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = Convolution3D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = Dropout(cls.dropoutrate)(hidlayer_dwl2_3)
        hidlayer_dwl3_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution3D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_1)
        hidlayer_dwl3_2 = Dropout(cls.dropoutrate)(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = Convolution3D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = Dropout(cls.dropoutrate)(hidlayer_dwl3_3)
        hidlayer_dwl4_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl3_3)

        nbfilters_dwl4  = 2*nbfilters_dwl3
        hidlayer_dwl4_2 = Convolution3D(nbfilters_dwl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl4_1)
        hidlayer_dwl4_2 = Dropout(cls.dropoutrate)(hidlayer_dwl4_2)
        hidlayer_dwl4_3 = Convolution3D(nbfilters_dwl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl4_2)
        hidlayer_dwl4_3 = Dropout(cls.dropoutrate)(hidlayer_dwl4_3)
        hidlayer_dwl5_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl4_3)

        nbfilters_dwl5  = 2*nbfilters_dwl4
        hidlayer_dwl5_2 = Convolution3D(nbfilters_dwl5, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl5_1)
        hidlayer_dwl5_2 = Dropout(cls.dropoutrate)(hidlayer_dwl5_2)
        hidlayer_dwl5_3 = Convolution3D(nbfilters_dwl5, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl5_2)
        hidlayer_dwl5_3 = Dropout(cls.dropoutrate)(hidlayer_dwl5_3)

        hidlayer_upl4_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_dwl5_3)
        hidlayer_upl4_1 = merge([hidlayer_upl4_1, hidlayer_dwl4_3], mode='concat', concat_axis=-1)

        nbfilters_upl4  = nbfilters_dwl4
        hidlayer_upl4_2 = Convolution3D(nbfilters_upl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl4_1)
        hidlayer_upl4_2 = Dropout(cls.dropoutrate)(hidlayer_upl4_2)
        hidlayer_upl4_3 = Convolution3D(nbfilters_upl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl4_2)
        hidlayer_upl4_3 = Dropout(cls.dropoutrate)(hidlayer_upl4_3)

        hidlayer_upl3_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_upl4_3)
        hidlayer_upl3_1 = merge([hidlayer_upl3_1, hidlayer_dwl3_3], mode='concat', concat_axis=-1)

        nbfilters_upl3  = nbfilters_dwl3
        hidlayer_upl3_2 = Convolution3D(nbfilters_upl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl3_1)
        hidlayer_upl3_2 = Dropout(cls.dropoutrate)(hidlayer_upl3_2)
        hidlayer_upl3_3 = Convolution3D(nbfilters_upl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl3_2)
        hidlayer_upl3_3 = Dropout(cls.dropoutrate)(hidlayer_upl3_3)

        hidlayer_upl2_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_upl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution3D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_1)
        hidlayer_upl2_2 = Dropout(cls.dropoutrate)(hidlayer_upl2_2)
        hidlayer_upl2_3 = Convolution3D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_2)
        hidlayer_upl2_3 = Dropout(cls.dropoutrate)(hidlayer_upl2_3)

        hidlayer_upl1_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution3D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_1)
        hidlayer_upl1_2 = Dropout(cls.dropoutrate)(hidlayer_upl1_2)
        hidlayer_upl1_3 = Convolution3D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_2)
        hidlayer_upl1_3 = Dropout(cls.dropoutrate)(hidlayer_upl1_3)

        outputs = Convolution3D(1, (1, 1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet3D_Batchnorm(NeuralNetwork):

    nbfilters   = 28
    size_filter = (3, 3, 3)

    @classmethod
    def getModel(cls, (image_nz, image_nx, image_ny), type_padding='same'):

        inputs = Input((image_nz, image_nx, image_ny, 1))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution3D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(inputs)
        hidlayer_dwl1_1 = BatchNormalization()(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = Convolution3D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = BatchNormalization()(hidlayer_dwl1_2)
        hidlayer_dwl2_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution3D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_1)
        hidlayer_dwl2_2 = BatchNormalization()(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = Convolution3D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = BatchNormalization()(hidlayer_dwl2_3)
        hidlayer_dwl3_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution3D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_1)
        hidlayer_dwl3_2 = BatchNormalization()(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = Convolution3D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = BatchNormalization()(hidlayer_dwl3_3)
        hidlayer_dwl4_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl3_3)

        nbfilters_dwl4  = 2*nbfilters_dwl3
        hidlayer_dwl4_2 = Convolution3D(nbfilters_dwl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl4_1)
        hidlayer_dwl4_2 = BatchNormalization()(hidlayer_dwl4_2)
        hidlayer_dwl4_3 = Convolution3D(nbfilters_dwl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl4_2)
        hidlayer_dwl4_3 = BatchNormalization()(hidlayer_dwl4_3)
        hidlayer_dwl5_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl4_3)

        nbfilters_dwl5  = 2*nbfilters_dwl4
        hidlayer_dwl5_2 = Convolution3D(nbfilters_dwl5, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl5_1)
        hidlayer_dwl5_2 = BatchNormalization()(hidlayer_dwl5_2)
        hidlayer_dwl5_3 = Convolution3D(nbfilters_dwl5, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl5_2)
        hidlayer_dwl5_3 = BatchNormalization()(hidlayer_dwl5_3)

        hidlayer_upl4_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_dwl5_3)
        hidlayer_upl4_1 = merge([hidlayer_upl4_1, hidlayer_dwl4_3], mode='concat', concat_axis=-1)

        nbfilters_upl4  = nbfilters_dwl4
        hidlayer_upl4_2 = Convolution3D(nbfilters_upl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl4_1)
        hidlayer_upl4_2 = BatchNormalization()(hidlayer_upl4_2)
        hidlayer_upl4_3 = Convolution3D(nbfilters_upl4, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl4_2)
        hidlayer_upl4_3 = BatchNormalization()(hidlayer_upl4_3)

        hidlayer_upl3_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_upl4_3)
        hidlayer_upl3_1 = merge([hidlayer_upl3_1, hidlayer_dwl3_3], mode='concat', concat_axis=-1)

        nbfilters_upl3  = nbfilters_dwl3
        hidlayer_upl3_2 = Convolution3D(nbfilters_upl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl3_1)
        hidlayer_upl3_2 = BatchNormalization()(hidlayer_upl3_2)
        hidlayer_upl3_3 = Convolution3D(nbfilters_upl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl3_2)
        hidlayer_upl3_3 = BatchNormalization()(hidlayer_upl3_3)

        hidlayer_upl2_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_upl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution3D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_1)
        hidlayer_upl2_2 = BatchNormalization()(hidlayer_upl2_2)
        hidlayer_upl2_3 = Convolution3D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_2)
        hidlayer_upl2_3 = BatchNormalization()(hidlayer_upl2_3)

        hidlayer_upl1_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution3D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_1)
        hidlayer_upl1_2 = BatchNormalization()(hidlayer_upl1_2)
        hidlayer_upl1_3 = Convolution3D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_2)
        hidlayer_upl1_3 = BatchNormalization()(hidlayer_upl1_3)

        outputs = Convolution3D(1, (1, 1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet3D_Shallow(NeuralNetwork):

    nbfilters   = 32
    size_filter = (3, 3, 3)

    @classmethod
    def getModel(cls, (image_nz, image_nx, image_ny), type_padding='same'):

        inputs = Input((image_nz, image_nx, image_ny, 1))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution3D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(inputs)
        hidlayer_dwl1_2 = Convolution3D(nbfilters_dwl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl1_1)
        hidlayer_dwl2_1 = MaxPooling3D (pool_size=(2, 2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution3D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_1)
        hidlayer_dwl2_3 = Convolution3D(nbfilters_dwl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl2_2)
        hidlayer_dwl3_1 = MaxPooling3D (pool_size=(2, 2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution3D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_1)
        hidlayer_dwl3_3 = Convolution3D(nbfilters_dwl3, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_dwl3_2)

        hidlayer_upl2_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_dwl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution3D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_1)
        hidlayer_upl2_3 = Convolution3D(nbfilters_upl2, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl2_2)

        hidlayer_upl1_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_1], mode='concat', concat_axis=-1)

        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution3D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_1)
        hidlayer_upl1_3 = Convolution3D(nbfilters_upl1, cls.size_filter, activation='relu', padding=type_padding)(hidlayer_upl1_2)

        outputs = Convolution3D(1, (1, 1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


# All Available Networks
DICTAVAILNETWORKS2D = {"Unet2D":                  Unet2D,
                       "Unet2D_Dropout":          Unet2D_Dropout,
                       "Unet2D_Batchnorm":        Unet2D_Batchnorm,
                       "Unet2D_Shallow":          Unet2D_Shallow,
                       "Unet2D_Shallow_Dropout":  Unet2D_Shallow_Dropout,
                       "Unet2D_Shallow_Batchnorm":Unet2D_Shallow_Batchnorm }
DICTAVAILNETWORKS3D = {"Unet3D":                  Unet3D,
                       "Unet3D_Dropout":          Unet3D_Dropout,
                       "Unet3D_Batchnorm":        Unet3D_Batchnorm,
                       "Unet3D_Shallow":          Unet3D_Shallow }