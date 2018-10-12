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

    def getModel(self):
        pass

    def getModelAndCompile(self, optimizer, lossfunction, metrics):
        return self.getModel().compile(optimizer=optimizer,
                                      loss=lossfunction,
                                      metrics=metrics )

    @staticmethod
    def getLoadSavedModel(model_saved_path, custom_objects=None):
        return load_model(model_saved_path, custom_objects=custom_objects)


class Unet3D(NeuralNetwork):

    nbfeamaps_first_layer = 16
    size_filter  = (3, 3, 3)
    size_pooling = (2, 2, 2)
    dropout_rate = 0.2

    def __init__(self, size_image, is_dropout=False):
        self.size_image = size_image
        self.is_dropout = is_dropout

    def getModel(self):

        inputs = Input((self.size_image[0], self.size_image[1], self.size_image[2], 1))

        nbfeamaps_dwl1  = self.nbfeamaps_first_layer
        hidlayer_dwl1_1 = Convolution3D(nbfeamaps_dwl1, self.size_filter, activation='relu', padding='same')(inputs)
        hidlayer_dwl1_2 = Convolution3D(nbfeamaps_dwl1, self.size_filter, activation='relu', padding='same')(hidlayer_dwl1_1)
        if self.is_dropout:
            hidlayer_dwl1_2 = Dropout(self.dropout_rate)(hidlayer_dwl1_2)
        hidlayer_dwl2_1 = MaxPooling3D(pool_size=self.size_pooling)(hidlayer_dwl1_2)

        nbfeamaps_dwl2  = 2*nbfeamaps_dwl1
        hidlayer_dwl2_2 = Convolution3D(nbfeamaps_dwl2, self.size_filter, activation='relu', padding='same')(hidlayer_dwl2_1)
        hidlayer_dwl2_3 = Convolution3D(nbfeamaps_dwl2, self.size_filter, activation='relu', padding='same')(hidlayer_dwl2_2)
        if self.is_dropout:
            hidlayer_dwl2_3 = Dropout(self.dropout_rate)(hidlayer_dwl2_3)
        hidlayer_dwl3_1 = MaxPooling3D(pool_size=self.size_pooling)(hidlayer_dwl2_3)

        nbfeamaps_dwl3  = 2*nbfeamaps_dwl2
        hidlayer_dwl3_2 = Convolution3D(nbfeamaps_dwl3, self.size_filter, activation='relu', padding='same')(hidlayer_dwl3_1)
        hidlayer_dwl3_3 = Convolution3D(nbfeamaps_dwl3, self.size_filter, activation='relu', padding='same')(hidlayer_dwl3_2)
        if self.is_dropout:
            hidlayer_dwl3_3 = Dropout(self.dropout_rate)(hidlayer_dwl3_3)
        hidlayer_dwl4_1 = MaxPooling3D(pool_size=self.size_pooling)(hidlayer_dwl3_3)

        nbfeamaps_dwl4  = 2*nbfeamaps_dwl3
        hidlayer_dwl4_2 = Convolution3D(nbfeamaps_dwl4, self.size_filter, activation='relu', padding='same')(hidlayer_dwl4_1)
        hidlayer_dwl4_3 = Convolution3D(nbfeamaps_dwl4, self.size_filter, activation='relu', padding='same')(hidlayer_dwl4_2)
        if self.is_dropout:
            hidlayer_dwl4_3 = Dropout(self.dropout_rate)(hidlayer_dwl4_3)
        hidlayer_dwl5_1 = MaxPooling3D(pool_size=self.size_pooling)(hidlayer_dwl4_3)

        nbfeamaps_dwl5  = 2*nbfeamaps_dwl4
        hidlayer_dwl5_2 = Convolution3D(nbfeamaps_dwl5, self.size_filter, activation='relu', padding='same')(hidlayer_dwl5_1)
        hidlayer_dwl5_3 = Convolution3D(nbfeamaps_dwl5, self.size_filter, activation='relu', padding='same')(hidlayer_dwl5_2)
        if self.is_dropout:
            hidlayer_dwl5_3 = Dropout(self.dropout_rate)(hidlayer_dwl5_3)

        hidlayer_upl4_1 = UpSampling3D(size=self.size_pooling)(hidlayer_dwl5_3)
        hidlayer_upl4_1 = merge([hidlayer_upl4_1, hidlayer_dwl4_3], mode='concat', concat_axis=-1)

        nbfeamaps_upl4  = nbfeamaps_dwl4
        hidlayer_upl4_2 = Convolution3D(nbfeamaps_upl4, self.size_filter, activation='relu', padding='same')(hidlayer_upl4_1)
        hidlayer_upl4_3 = Convolution3D(nbfeamaps_upl4, self.size_filter, activation='relu', padding='same')(hidlayer_upl4_2)
        if self.is_dropout:
            hidlayer_upl4_3 = Dropout(self.dropout_rate)(hidlayer_upl4_3)

        hidlayer_upl3_1 = UpSampling3D(size=self.size_pooling)(hidlayer_upl4_3)
        hidlayer_upl3_1 = merge([hidlayer_upl3_1, hidlayer_dwl3_3], mode='concat', concat_axis=-1)

        nbfeamaps_upl3  = nbfeamaps_dwl3
        hidlayer_upl3_2 = Convolution3D(nbfeamaps_upl3, self.size_filter, activation='relu', padding='same')(hidlayer_upl3_1)
        hidlayer_upl3_3 = Convolution3D(nbfeamaps_upl3, self.size_filter, activation='relu', padding='same')(hidlayer_upl3_2)
        if self.is_dropout:
            hidlayer_upl3_3 = Dropout(self.dropout_rate)(hidlayer_upl3_3)

        hidlayer_upl2_1 = UpSampling3D(size=self.size_pooling)(hidlayer_upl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfeamaps_upl2  = nbfeamaps_dwl2
        hidlayer_upl2_2 = Convolution3D(nbfeamaps_upl2, self.size_filter, activation='relu', padding='same')(hidlayer_upl2_1)
        hidlayer_upl2_3 = Convolution3D(nbfeamaps_upl2, self.size_filter, activation='relu', padding='same')(hidlayer_upl2_2)
        if self.is_dropout:
            hidlayer_upl2_3 = Dropout(self.dropout_rate)(hidlayer_upl2_3)

        hidlayer_upl1_1 = UpSampling3D(size=self.size_pooling)(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfeamaps_upl1  = nbfeamaps_dwl1
        hidlayer_upl1_2 = Convolution3D(nbfeamaps_upl1, self.size_filter, activation='relu', padding='same')(hidlayer_upl1_1)
        hidlayer_upl1_3 = Convolution3D(nbfeamaps_upl1, self.size_filter, activation='relu', padding='same')(hidlayer_upl1_2)
        if self.is_dropout:
            hidlayer_upl1_3 = Dropout(self.dropout_rate)(hidlayer_upl1_3)

        outputs = Convolution3D(1, (1, 1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet3D_Shallow(NeuralNetwork):

    nbfeamaps_first_layer = 16
    size_filter  = (3, 3, 3)
    size_pooling = (2, 2, 2)
    dropout_rate = 0.2

    def __init__(self, size_image, is_dropout=False):
        self.size_image = size_image
        self.is_dropout = is_dropout

    def getModel(self):

        inputs = Input((self.size_image[0], self.size_image[1], self.size_image[2], 1))

        nbfeamaps_dwl1  = self.nbfeamaps_first_layer
        hidlayer_dwl1_1 = Convolution3D(nbfeamaps_dwl1, self.size_filter, activation='relu', padding='same')(inputs)
        hidlayer_dwl1_2 = Convolution3D(nbfeamaps_dwl1, self.size_filter, activation='relu', padding='same')(hidlayer_dwl1_1)
        if self.is_dropout:
            hidlayer_dwl1_2 = Dropout(self.dropout_rate)(hidlayer_dwl1_2)
        hidlayer_dwl2_1 = MaxPooling3D (pool_size=self.size_pooling)(hidlayer_dwl1_2)

        nbfeamaps_dwl2  = 2*nbfeamaps_dwl1
        hidlayer_dwl2_2 = Convolution3D(nbfeamaps_dwl2, self.size_filter, activation='relu', padding='same')(hidlayer_dwl2_1)
        hidlayer_dwl2_3 = Convolution3D(nbfeamaps_dwl2, self.size_filter, activation='relu', padding='same')(hidlayer_dwl2_2)
        if self.is_dropout:
            hidlayer_dwl2_3 = Dropout(self.dropout_rate)(hidlayer_dwl2_3)
        hidlayer_dwl3_1 = MaxPooling3D (pool_size=self.size_pooling)(hidlayer_dwl2_3)

        nbfeamaps_dwl3  = 2*nbfeamaps_dwl2
        hidlayer_dwl3_2 = Convolution3D(nbfeamaps_dwl3, self.size_filter, activation='relu', padding='same')(hidlayer_dwl3_1)
        hidlayer_dwl3_3 = Convolution3D(nbfeamaps_dwl3, self.size_filter, activation='relu', padding='same')(hidlayer_dwl3_2)
        if self.is_dropout:
            hidlayer_dwl3_3 = Dropout(self.dropout_rate)(hidlayer_dwl3_3)

        hidlayer_upl2_1 = UpSampling3D(size=self.size_pooling)(hidlayer_dwl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfeamaps_upl2  = nbfeamaps_dwl2
        hidlayer_upl2_2 = Convolution3D(nbfeamaps_upl2, self.size_filter, activation='relu', padding='same')(hidlayer_upl2_1)
        hidlayer_upl2_3 = Convolution3D(nbfeamaps_upl2, self.size_filter, activation='relu', padding='same')(hidlayer_upl2_2)
        if self.is_dropout:
            hidlayer_upl2_3 = Dropout(self.dropout_rate)(hidlayer_upl2_3)

        hidlayer_upl1_1 = UpSampling3D(size=self.size_pooling)(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfeamaps_upl1  = nbfeamaps_dwl1
        hidlayer_upl1_2 = Convolution3D(nbfeamaps_upl1, self.size_filter, activation='relu', padding='same')(hidlayer_upl1_1)
        hidlayer_upl1_3 = Convolution3D(nbfeamaps_upl1, self.size_filter, activation='relu', padding='same')(hidlayer_upl1_2)
        if self.is_dropout:
            hidlayer_upl1_3 = Dropout(self.dropout_rate)(hidlayer_upl1_3)

        outputs = Convolution3D(1, (1, 1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet3D_Tailored(NeuralNetwork):

    num_featmaps_firstlay_default = 16

    size_filter_dwlys = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (1, 3, 3)]
    size_filter_uplys = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (1, 3, 3)]
    size_pooling_lys  = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)]
    size_cropping_lys = [(0, 90, 90), (0, 41, 41), (0, 16, 16), (0, 4, 4)]

    dropout_rate = 0.2


    def __init__(self, size_image,
                 num_classes_out,
                 num_featmaps_firstlay=num_featmaps_firstlay_default,
                 is_dropout=False,
                 is_batchnormalization=False):
        self.size_image           = size_image
        self.num_classes_out      = num_classes_out
        self.num_featmaps_firstlay= num_featmaps_firstlay
        self.is_dropout           = is_dropout
        self.is_batchnormalization= is_batchnormalization

    def getModel(self):

        inputs = Input((self.size_image[0], self.size_image[1], self.size_image[2], 1))

        nbfeamaps_dwl1  = self.num_featmaps_firstlay
        hidlayer_dwl1_1 = Convolution3D(nbfeamaps_dwl1, self.size_filter_dwlys[0], activation='relu', padding='same')(inputs)
        hidlayer_dwl1_2 = Convolution3D(nbfeamaps_dwl1, self.size_filter_dwlys[0], activation='relu', padding='same')(hidlayer_dwl1_1)
        # if self.is_dropout:
        #     hidlayer_dwl1_2 = Dropout(self.dropout_rate)(hidlayer_dwl1_2)
        hidlayer_dwl2_1 = MaxPooling3D(pool_size=self.size_pooling_lys[0])(hidlayer_dwl1_2)

        nbfeamaps_dwl2  = 2*nbfeamaps_dwl1
        hidlayer_dwl2_2 = Convolution3D(nbfeamaps_dwl2, self.size_filter_dwlys[1], activation='relu', padding='same')(hidlayer_dwl2_1)
        hidlayer_dwl2_3 = Convolution3D(nbfeamaps_dwl2, self.size_filter_dwlys[1], activation='relu', padding='same')(hidlayer_dwl2_2)
        # if self.is_dropout:
        #     hidlayer_dwl2_3 = Dropout(self.dropout_rate)(hidlayer_dwl2_3)
        hidlayer_dwl3_1 = MaxPooling3D(pool_size=self.size_pooling_lys[1])(hidlayer_dwl2_3)

        nbfeamaps_dwl3  = 2*nbfeamaps_dwl2
        hidlayer_dwl3_2 = Convolution3D(nbfeamaps_dwl3, self.size_filter_dwlys[2], activation='relu', padding='same')(hidlayer_dwl3_1)
        hidlayer_dwl3_3 = Convolution3D(nbfeamaps_dwl3, self.size_filter_dwlys[2], activation='relu', padding='same')(hidlayer_dwl3_2)
        # if self.is_dropout:
        #     hidlayer_dwl3_3 = Dropout(self.dropout_rate)(hidlayer_dwl3_3)
        hidlayer_dwl4_1 = MaxPooling3D(pool_size=self.size_pooling_lys[2])(hidlayer_dwl3_3)

        nbfeamaps_dwl4  = 2*nbfeamaps_dwl3
        hidlayer_dwl4_2 = Convolution3D(nbfeamaps_dwl4, self.size_filter_dwlys[3], activation='relu', padding='same')(hidlayer_dwl4_1)
        hidlayer_dwl4_3 = Convolution3D(nbfeamaps_dwl4, self.size_filter_dwlys[3], activation='relu', padding='same')(hidlayer_dwl4_2)
        # if self.is_dropout:
        #     hidlayer_dwl4_3 = Dropout(self.dropout_rate)(hidlayer_dwl4_3)
        hidlayer_dwl5_1 = MaxPooling3D(pool_size=self.size_pooling_lys[3])(hidlayer_dwl4_3)

        nbfeamaps_dwl5  = 2*nbfeamaps_dwl4
        hidlayer_dwl5_2 = Convolution3D(nbfeamaps_dwl5, self.size_filter_dwlys[4], activation='relu', padding='same')(hidlayer_dwl5_1)
        hidlayer_dwl5_3 = Convolution3D(nbfeamaps_dwl5, self.size_filter_dwlys[4], activation='relu', padding='same')(hidlayer_dwl5_2)
        #if self.is_dropout:
        #    hidlayer_dwl5_3 = Dropout(self.dropout_rate)(hidlayer_dwl5_3)
        #if self.is_batchnormalization:
        #    hidlayer_dwl5_3 = BatchNormalization()(hidlayer_dwl5_3)

        hidlayer_upl4_1 = UpSampling3D(size=self.size_pooling_lys[3])(hidlayer_dwl5_3)
        hidlayer_upl4_1 = merge([hidlayer_upl4_1, hidlayer_dwl4_3], mode='concat', concat_axis=-1)

        nbfeamaps_upl4  = nbfeamaps_dwl4
        hidlayer_upl4_3 = Convolution3D(nbfeamaps_upl4, self.size_filter_uplys[3], activation='relu', padding='same')(hidlayer_upl4_1)
        #hidlayer_upl4_3 = Convolution3D(nbfeamaps_upl4, self.size_filter_uplys[3], activation='relu', padding='same')(hidlayer_upl4_2)
        if self.is_dropout:
            hidlayer_upl4_3 = Dropout(self.dropout_rate)(hidlayer_upl4_3)
        #if self.is_batchnormalization:
        #    hidlayer_upl4_3 = BatchNormalization()(hidlayer_upl4_3)

        hidlayer_upl3_1 = UpSampling3D(size=self.size_pooling_lys[2])(hidlayer_upl4_3)
        hidlayer_upl3_1 = merge([hidlayer_upl3_1, hidlayer_dwl3_3], mode='concat', concat_axis=-1)

        nbfeamaps_upl3  = nbfeamaps_dwl3
        hidlayer_upl3_3 = Convolution3D(nbfeamaps_upl3, self.size_filter_uplys[2], activation='relu', padding='same')(hidlayer_upl3_1)
        #hidlayer_upl3_3 = Convolution3D(nbfeamaps_upl3, self.size_filter_uplys[2], activation='relu', padding='same')(hidlayer_upl3_2)
        if self.is_dropout:
            hidlayer_upl3_3 = Dropout(self.dropout_rate)(hidlayer_upl3_3)
        #if self.is_batchnormalization:
        #    hidlayer_upl3_3 = BatchNormalization()(hidlayer_upl3_3)

        hidlayer_upl2_1 = UpSampling3D(size=self.size_pooling_lys[1])(hidlayer_upl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfeamaps_upl2  = nbfeamaps_dwl2
        hidlayer_upl2_3 = Convolution3D(nbfeamaps_upl2, self.size_filter_uplys[1], activation='relu', padding='same')(hidlayer_upl2_1)
        #hidlayer_upl2_3 = Convolution3D(nbfeamaps_upl2, self.size_filter_uplys[1], activation='relu', padding='same')(hidlayer_upl2_2)
        if self.is_dropout:
            hidlayer_upl2_3 = Dropout(self.dropout_rate)(hidlayer_upl2_3)
        if self.is_batchnormalization:
            hidlayer_upl2_3 = BatchNormalization()(hidlayer_upl2_3)

        hidlayer_upl1_1 = UpSampling3D(size=self.size_pooling_lys[0])(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfeamaps_upl1  = nbfeamaps_dwl1
        hidlayer_upl1_3 = Convolution3D(nbfeamaps_upl1, self.size_filter_uplys[0], activation='relu', padding='same')(hidlayer_upl1_1)
        #hidlayer_upl1_3 = Convolution3D(nbfeamaps_upl1, self.size_filter_uplys[0], activation='relu', padding='same')(hidlayer_upl1_2)
        if self.is_dropout:
            hidlayer_upl1_3 = Dropout(self.dropout_rate)(hidlayer_upl1_3)
        if self.is_batchnormalization:
            hidlayer_upl1_3 = BatchNormalization()(hidlayer_upl1_3)

        outputs = Convolution3D(self.num_classes_out, (1, 1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet3D_Shallow_Tailored(NeuralNetwork):

    num_featmaps_firstlay_default = 16

    size_filter_dwlys = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
    size_filter_uplys = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
    size_pooling_lys  = [(2, 2, 2), (2, 2, 2)]
    size_cropping_lys = [(0, 16, 16), (0, 4, 4)]

    dropout_rate = 0.2


    def __init__(self, size_image,
                 num_classes_out,
                 num_featmaps_firstlay=num_featmaps_firstlay_default,
                 is_dropout=False,
                 is_batchnormalization=False):
        self.size_image           = size_image
        self.num_classes_out      = num_classes_out
        self.num_featmaps_firstlay= num_featmaps_firstlay
        self.is_dropout           = is_dropout
        self.is_batchnormalization= is_batchnormalization

    def getModel(self):

        inputs = Input((self.size_image[0], self.size_image[1], self.size_image[2], 1))

        nbfeamaps_dwl1  = self.num_featmaps_firstlay
        hidlayer_dwl1_1 = Convolution3D(nbfeamaps_dwl1, self.size_filter_dwlys[0], activation='relu', padding='same')(inputs)
        hidlayer_dwl1_2 = Convolution3D(nbfeamaps_dwl1, self.size_filter_dwlys[0], activation='relu', padding='same')(hidlayer_dwl1_1)
        #if self.is_dropout:
        #    hidlayer_dwl1_2 = Dropout(self.dropout_rate)(hidlayer_dwl1_2)
        hidlayer_dwl2_1 = MaxPooling3D(pool_size=self.size_pooling_lys[0])(hidlayer_dwl1_2)

        nbfeamaps_dwl2  = 2*nbfeamaps_dwl1
        hidlayer_dwl2_2 = Convolution3D(nbfeamaps_dwl2, self.size_filter_dwlys[1], activation='relu', padding='same')(hidlayer_dwl2_1)
        hidlayer_dwl2_3 = Convolution3D(nbfeamaps_dwl2, self.size_filter_dwlys[1], activation='relu', padding='same')(hidlayer_dwl2_2)
        #if self.is_dropout:
        #    hidlayer_dwl2_3 = Dropout(self.dropout_rate)(hidlayer_dwl2_3)
        hidlayer_dwl3_1 = MaxPooling3D(pool_size=self.size_pooling_lys[1])(hidlayer_dwl2_3)

        nbfeamaps_dwl3  = 2*nbfeamaps_dwl2
        hidlayer_dwl3_2 = Convolution3D(nbfeamaps_dwl3, self.size_filter_dwlys[2], activation='relu', padding='same')(hidlayer_dwl3_1)
        hidlayer_dwl3_3 = Convolution3D(nbfeamaps_dwl3, self.size_filter_dwlys[2], activation='relu', padding='same')(hidlayer_dwl3_2)
        #if self.is_dropout:
        #    hidlayer_dwl3_3 = Dropout(self.dropout_rate)(hidlayer_dwl3_3)
        #if self.is_batchnormalization:
        #    hidlayer_dwl3_3 = BatchNormalization()(hidlayer_dwl3_3)

        hidlayer_upl2_1 = UpSampling3D(size=self.size_pooling_lys[1])(hidlayer_dwl3_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfeamaps_upl2  = nbfeamaps_dwl2
        hidlayer_upl2_3 = Convolution3D(nbfeamaps_upl2, self.size_filter_uplys[1], activation='relu', padding='same')(hidlayer_upl2_1)
        #hidlayer_upl2_3 = Convolution3D(nbfeamaps_upl2, self.size_filter_uplys[1], activation='relu', padding='same')(hidlayer_upl2_2)
        if self.is_dropout:
            hidlayer_upl2_3 = Dropout(self.dropout_rate)(hidlayer_upl2_3)
        if self.is_batchnormalization:
            hidlayer_upl2_3 = BatchNormalization()(hidlayer_upl2_3)

        hidlayer_upl1_1 = UpSampling3D(size=self.size_pooling_lys[0])(hidlayer_upl2_3)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfeamaps_upl1  = nbfeamaps_dwl1
        hidlayer_upl1_3 = Convolution3D(nbfeamaps_upl1, self.size_filter_uplys[0], activation='relu', padding='same')(hidlayer_upl1_1)
        #hidlayer_upl1_3 = Convolution3D(nbfeamaps_upl1, self.size_filter_uplys[0], activation='relu', padding='same')(hidlayer_upl1_2)
        if self.is_dropout:
            hidlayer_upl1_3 = Dropout(self.dropout_rate)(hidlayer_upl1_3)
        if self.is_batchnormalization:
            hidlayer_upl1_3 = BatchNormalization()(hidlayer_upl1_3)

        outputs = Convolution3D(self.num_classes_out, (1, 1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


# All Available Networks
def DICTAVAILNETWORKS3D(size_image,
                        model_name,
                        num_featmaps_firstlayer,
                        num_classes_out=1):
    if (model_name=='Unet3D'):
        return Unet3D_Tailored(size_image,
                               num_classes_out=num_classes_out,
                               num_featmaps_firstlay=num_featmaps_firstlayer)
    elif (model_name=='Unet3D_Dropout'):
        return Unet3D_Tailored(size_image,
                               num_classes_out=num_classes_out,
                               num_featmaps_firstlay=num_featmaps_firstlayer,
                               is_dropout=True)
    elif (model_name=='Unet3D_BatchNormalization'):
        return Unet3D_Tailored(size_image,
                               num_classes_out=num_classes_out,
                               num_featmaps_firstlay=num_featmaps_firstlayer,
                               is_batchnormalization=True)
    elif (model_name=='Unet3D_Dropout_BatchNormalization'):
        return Unet3D_Tailored(size_image,
                               num_classes_out=num_classes_out,
                               num_featmaps_firstlay=num_featmaps_firstlayer,
                               is_dropout=True,
                               is_batchnormalization=True)
    elif (model_name=='Unet3D_Shallow'):
        return Unet3D_Shallow_Tailored(size_image,
                                       num_classes_out=num_classes_out,
                                       num_featmaps_firstlay=num_featmaps_firstlayer)
    elif (model_name=='Unet3D_Shallow_Dropout'):
        return Unet3D_Shallow_Tailored(size_image,
                                       num_classes_out=num_classes_out,
                                       num_featmaps_firstlay=num_featmaps_firstlayer,
                                       is_dropout=True)
    elif (model_name=='Unet3D_Shallow_BatchNormalization'):
        return Unet3D_Shallow_Tailored(size_image,
                                       num_classes_out=num_classes_out,
                                       num_featmaps_firstlay=num_featmaps_firstlayer,
                                       is_batchnormalization=True)
    elif (model_name=='Unet3D_Shallow_Dropout_BatchNormalization'):
        return Unet3D_Shallow_Tailored(size_image,
                                       num_classes_out=num_classes_out,
                                       num_featmaps_firstlay=num_featmaps_firstlayer,
                                       is_dropout=True,
                                       is_batchnormalization=True)
    else:
        return 0
