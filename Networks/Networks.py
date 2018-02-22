#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D
from keras.models import Model


class NeuralNetwork(object):

    @classmethod
    def getModel(cls):
        pass
    @classmethod
    def getModelAndCompile(cls, optimizer, lossfunction, metrics):
        return cls.getModel().compile(optimizer=optimizer,
                                      loss=lossfunction,
                                      metrics=metrics )


class Unet2D(NeuralNetwork):

    nbfilters  = 32
    filter_size= (3, 3)

    @classmethod
    def getModel(cls, image_nx, image_ny):

        inputs = Input((image_nx, image_ny, 1))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution2D(nbfilters_dwl1, cls.filter_size, activation='relu', border_mode='same')(inputs)
        hidlayer_dwl1_2 = Convolution2D(nbfilters_dwl1, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl1_1)
        hidlayer_dwl2_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution2D(nbfilters_dwl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl2_1)
        hidlayer_dwl2_3 = Convolution2D(nbfilters_dwl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl2_2)
        hidlayer_dwl3_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution2D(nbfilters_dwl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl3_1)
        hidlayer_dwl3_3 = Convolution2D(nbfilters_dwl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl3_2)
        hidlayer_dwl4_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl3_3)

        nbfilters_dwl4  = 2*nbfilters_dwl3
        hidlayer_dwl4_2 = Convolution2D(nbfilters_dwl4, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl4_1)
        hidlayer_dwl4_3 = Convolution2D(nbfilters_dwl4, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl4_2)
        hidlayer_dwl5_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl4_3)

        nbfilters_dwl5  = 2*nbfilters_dwl4
        hidlayer_dwl5_2 = Convolution2D(nbfilters_dwl5, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl5_1)
        hidlayer_dwl5_3 = Convolution2D(nbfilters_dwl5, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl5_2)
        
        hidlayer_upl4_1 = UpSampling2D(size=(2, 2))(hidlayer_dwl5_3)
        hidlayer_upl4_1 = merge([hidlayer_upl4_1, hidlayer_dwl4_3], mode='concat', concat_axis=-1)

        nbfilters_upl4  = nbfilters_dwl4
        hidlayer_upl4_2 = Convolution2D(nbfilters_upl4, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl4_1)
        hidlayer_upl4_3 = Convolution2D(nbfilters_upl4, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl4_2)
        
        hidlayer_upl3_1 = UpSampling2D(size=(2, 2))(hidlayer_upl4_3)
        hidlayer_upl3_1 = merge([hidlayer_upl3_1, hidlayer_dwl3_3], mode='concat', concat_axis=-1)
        
        nbfilters_upl3  = nbfilters_dwl3
        hidlayer_upl3_2 = Convolution2D(nbfilters_upl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl3_1)
        hidlayer_upl3_3 = Convolution2D(nbfilters_upl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl3_2)
        
        hidlayer_upl2_1 = UpSampling2D(size=(2, 2))(hidlayer_upl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)
        
        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution2D(nbfilters_upl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl2_1)
        hidlayer_upl2_3 = Convolution2D(nbfilters_upl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl2_2)
        
        hidlayer_upl1_1 = UpSampling2D(size=(2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)
        
        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution2D(nbfilters_upl1, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl1_1)
        hidlayer_upl1_3 = Convolution2D(nbfilters_upl1, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl1_2)

        outputs         = Convolution2D(1, (1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet2D_Shallow(NeuralNetwork):

    nbfilters  = 32
    filter_size= (3, 3)

    @classmethod
    def getModel(cls, image_nx, image_ny):

        inputs = Input((image_nx, image_ny, 1))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution2D(nbfilters_dwl1, cls.filter_size, activation='relu', border_mode='same')(inputs)
        hidlayer_dwl1_2 = Convolution2D(nbfilters_dwl1, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl1_1)
        hidlayer_dwl2_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution2D(nbfilters_dwl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl2_1)
        hidlayer_dwl2_3 = Convolution2D(nbfilters_dwl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl2_2)
        hidlayer_dwl3_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution2D(nbfilters_dwl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl3_1)
        hidlayer_dwl3_3 = Convolution2D(nbfilters_dwl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl3_2)

        hidlayer_upl2_1 = UpSampling2D(size=(2, 2))(hidlayer_dwl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution2D(nbfilters_upl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl2_1)
        hidlayer_upl2_3 = Convolution2D(nbfilters_upl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl2_2)

        hidlayer_upl1_1 = UpSampling2D(size=(2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_1], mode='concat', concat_axis=-1)

        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution2D(nbfilters_upl1, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl1_1)
        hidlayer_upl1_3 = Convolution2D(nbfilters_upl1, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl1_2)

        outputs         = Convolution2D(1, (1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet2D_Dropout(NeuralNetwork):

    nbfilters   = 32
    filter_size = (3, 3)
    dropoutrate = 0.2

    @classmethod
    def getModel(cls, image_nx, image_ny):

        inputs = Input((image_nx, image_ny, 1))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution2D(nbfilters_dwl1, cls.filter_size, activation='relu', border_mode='same')(inputs)
        hidlayer_dwl1_1 = Dropout(cls.dropoutrate)(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = Convolution2D(nbfilters_dwl1, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = Dropout(cls.dropoutrate)(hidlayer_dwl1_2)
        hidlayer_dwl2_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution2D(nbfilters_dwl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl2_1)
        hidlayer_dwl2_2 = Dropout(cls.dropoutrate)( hidlayer_dwl2_2)
        hidlayer_dwl2_3 = Convolution2D(nbfilters_dwl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = Dropout(cls.dropoutrate)( hidlayer_dwl2_3)
        hidlayer_dwl3_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution2D(nbfilters_dwl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl3_1)
        hidlayer_dwl3_2 = Dropout(cls.dropoutrate)(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = Convolution2D(nbfilters_dwl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = Dropout(cls.dropoutrate)(hidlayer_dwl3_3)
        hidlayer_dwl4_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl3_3)

        nbfilters_dwl4  = 2*nbfilters_dwl3
        hidlayer_dwl4_2 = Convolution2D(nbfilters_dwl4, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl4_1)
        hidlayer_dwl4_2 = Dropout(cls.dropoutrate)(hidlayer_dwl4_2)
        hidlayer_dwl4_3 = Convolution2D(nbfilters_dwl4, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl4_2)
        hidlayer_dwl4_3 = Dropout(cls.dropoutrate)(hidlayer_dwl4_3)
        hidlayer_dwl5_1 = MaxPooling2D (pool_size=(2, 2))(hidlayer_dwl4_3)

        nbfilters_dwl5  = 2*nbfilters_dwl4
        hidlayer_dwl5_2 = Convolution2D(nbfilters_dwl5, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl5_1)
        hidlayer_dwl5_2 = Dropout(cls.dropoutrate)(hidlayer_dwl5_2)
        hidlayer_dwl5_3 = Convolution2D(nbfilters_dwl5, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl5_2)
        hidlayer_dwl5_3 = Dropout(cls.dropoutrate)(hidlayer_dwl5_3)

        hidlayer_upl4_1 = UpSampling2D(size=(2, 2))(hidlayer_dwl5_3)
        hidlayer_upl4_1 = merge([hidlayer_upl4_1, hidlayer_dwl4_3], mode='concat', concat_axis=-1)

        nbfilters_upl4  = nbfilters_dwl4
        hidlayer_upl4_2 = Convolution2D( nbfilters_upl4, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl4_1)
        hidlayer_upl4_2 = Dropout(cls.dropoutrate)(hidlayer_upl4_2)
        hidlayer_upl4_3 = Convolution2D( nbfilters_upl4, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl4_2)
        hidlayer_upl4_3 = Dropout(cls.dropoutrate)(hidlayer_upl4_3)

        hidlayer_upl3_1 = UpSampling2D(size=(2, 2))(hidlayer_upl4_3)
        hidlayer_upl3_1 = merge([hidlayer_upl3_1, hidlayer_dwl3_3], mode='concat', concat_axis=-1)

        nbfilters_upl3  = nbfilters_dwl3
        hidlayer_upl3_2 = Convolution2D( nbfilters_upl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl3_1)
        hidlayer_upl3_2 = Dropout(cls.dropoutrate)(hidlayer_upl3_2)
        hidlayer_upl3_3 = Convolution2D( nbfilters_upl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl3_2)
        hidlayer_upl3_3 = Dropout(cls.dropoutrate)(hidlayer_upl3_3)

        hidlayer_upl2_1 = UpSampling2D(size=(2, 2))(hidlayer_upl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution2D( nbfilters_upl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl2_1)
        hidlayer_upl2_2 = Dropout(cls.dropoutrate)(hidlayer_upl2_2)
        hidlayer_upl2_3 = Convolution2D( nbfilters_upl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl2_2)
        hidlayer_upl2_3 = Dropout(cls.dropoutrate)(hidlayer_upl2_3)

        hidlayer_upl1_1 = UpSampling2D(size=(2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution2D( nbfilters_upl1, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl1_1)
        hidlayer_upl1_2 = Dropout(cls.dropoutrate)(hidlayer_upl1_2)
        hidlayer_upl1_3 = Convolution2D( nbfilters_upl1, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl1_2)
        hidlayer_upl1_3 = Dropout(cls.dropoutrate)(hidlayer_upl1_3)

        outputs         = Convolution2D(1, (1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet3D(NeuralNetwork):

    nbfilters  = 32
    filter_size= (3, 3, 3)

    @classmethod
    def getModel(cls, image_nx, image_ny, image_nz):

        inputs = Input((image_nx, image_ny, image_nz))

        nbfilters_dwl1  = cls.nbfilters
        hidlayer_dwl1_1 = Convolution3D(nbfilters_dwl1, cls.filter_size, activation='relu', border_mode='same')(inputs)
        hidlayer_dwl1_2 = Convolution3D(nbfilters_dwl1, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl1_1)
        hidlayer_dwl2_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl1_2)

        nbfilters_dwl2  = 2*nbfilters_dwl1
        hidlayer_dwl2_2 = Convolution3D(nbfilters_dwl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl2_1)
        hidlayer_dwl2_3 = Convolution3D(nbfilters_dwl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl2_2)
        hidlayer_dwl3_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl2_3)

        nbfilters_dwl3  = 2*nbfilters_dwl2
        hidlayer_dwl3_2 = Convolution3D(nbfilters_dwl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl3_1)
        hidlayer_dwl3_3 = Convolution3D(nbfilters_dwl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl3_2)
        hidlayer_dwl4_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl3_3)

        nbfilters_dwl4  = 2*nbfilters_dwl3
        hidlayer_dwl4_2 = Convolution3D(nbfilters_dwl4, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl4_1)
        hidlayer_dwl4_3 = Convolution3D(nbfilters_dwl4, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl4_2)
        hidlayer_dwl5_1 = MaxPooling3D(pool_size=(2, 2, 2))(hidlayer_dwl4_3)

        nbfilters_dwl5  = 2*nbfilters_dwl4
        hidlayer_dwl5_2 = Convolution3D(nbfilters_dwl5, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl5_1)
        hidlayer_dwl5_3 = Convolution3D(nbfilters_dwl5, cls.filter_size, activation='relu', border_mode='same')(hidlayer_dwl5_2)

        hidlayer_upl4_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_dwl5_3)
        hidlayer_upl4_1 = merge([hidlayer_upl4_1, hidlayer_dwl4_3], mode='concat', concat_axis=-1)

        nbfilters_upl4  = nbfilters_dwl4
        hidlayer_upl4_2 = Convolution3D(nbfilters_upl4, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl4_1)
        hidlayer_upl4_3 = Convolution3D(nbfilters_upl4, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl4_2)

        hidlayer_upl3_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_upl4_3)
        hidlayer_upl3_1 = merge([hidlayer_upl3_1, hidlayer_dwl3_3], mode='concat', concat_axis=-1)

        nbfilters_upl3  = nbfilters_dwl3
        hidlayer_upl3_2 = Convolution3D(nbfilters_upl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl3_1)
        hidlayer_upl3_3 = Convolution3D(nbfilters_upl3, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl3_2)

        hidlayer_upl2_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_upl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfilters_upl2  = nbfilters_dwl2
        hidlayer_upl2_2 = Convolution3D(nbfilters_upl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl2_1)
        hidlayer_upl2_3 = Convolution3D(nbfilters_upl2, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl2_2)

        hidlayer_upl1_1 = UpSampling3D(size=(2, 2, 2))(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfilters_upl1  = nbfilters_dwl1
        hidlayer_upl1_2 = Convolution3D(nbfilters_upl1, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl1_1)
        hidlayer_upl1_3 = Convolution3D(nbfilters_upl1, cls.filter_size, activation='relu', border_mode='same')(hidlayer_upl1_2)

        outputs = Convolution3D(1, (1, 1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model