#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.ErrorMessages import *
from CommonUtil.FunctionsUtil import *
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

    def getLoadSavedModel(model_saved_path, custom_objects=None):
        return load_model(model_saved_path, custom_objects=custom_objects)


class Unet3D(NeuralNetwork):

    nbfeamaps_first_layer = 18
    size_filter  = (3, 3, 3)
    size_pooling = (2, 2, 2)
    dropout_rate = 0.2

    def __init__(self, (image_nz, image_nx, image_ny), is_dropout=False):
        self.size_input_image = (image_nz, image_nx, image_ny)
        self.is_dropout = is_dropout

    def getModel(self):

        inputs = Input((self.size_input_image[0], self.size_input_image[1], self.size_input_image[2], 1))

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

    def __init__(self, (image_nz, image_nx, image_ny), is_dropout=False):
        self.size_input_image = (image_nz, image_nx, image_ny)
        self.is_dropout = is_dropout

    def getModel(self):

        inputs = Input((self.size_input_image[0], self.size_input_image[1], self.size_input_image[2], 1))

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

    nbfeamaps_first_layer = 16

    size_convfilter_downpath_layers_opt1 = [(3, 3, 3), (3, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)]
    size_convfilter_uppath_layers_opt1   = [(1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)]
    size_pooling_layers_opt1             = [(2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)]
    size_cropping_layers_opt1            = [(0, 90, 90), (0, 41, 41), (0, 16, 16), (0, 4, 4)]
    correct_convLayers_afterpooling_opt1 = [False, False, True, False, False]

    list_size_valid_input_images  = [(64, 256, 256)]
    list_size_valid_output_images = [(48, 216, 216)]

    correct_afterpooling = (0, 1, 1)
    dropout_rate = 0.2


    def __init__(self, (image_nz, image_nx, image_ny)):
        self.size_input_image  = (image_nz, image_nx, image_ny)

        self.type_network = self.find_type_network_from_size_images(self.size_input_image)

        if self.type_network==0:
            self.size_convfilter_downpath_layers = self.size_convfilter_downpath_layers_opt1
            self.size_convfilter_uppath_layers   = self.size_convfilter_uppath_layers_opt1
            self.size_pooling_layers             = self.size_pooling_layers_opt1
            self.size_cropping_layers            = self.size_cropping_layers_opt1
            self.correct_convLayers_afterpooling = self.correct_convLayers_afterpooling_opt1

        self.size_output_image = self.list_size_valid_output_images[self.type_network]

        self.size_convfilter_lay1_downpath_layers = [sumTwoTuples(value, self.correct_afterpooling) if check else value
                                                     for (value, check) in zip(self.size_convfilter_downpath_layers, self.correct_convLayers_afterpooling)]
        self.size_convfilter_lay2_downpath_layers = self.size_convfilter_downpath_layers


    def check_valid_size_input_images(self, (image_nz, image_nx, image_ny)):
        if (image_nz, image_nx, image_ny) not in  self.list_size_valid_input_images:
            message = "\'Unet3D_Shallow_Tailored\' only valid for input images of size: %s..." %(self.list_size_valid_input_images)
            CatchErrorException(message)
        else:
            return True

    def find_type_network_from_size_images(self, (image_nz, image_nx, image_ny)):
        if self.check_valid_size_input_images((image_nz, image_nx, image_ny)):
            return self.list_size_valid_input_images.index((image_nz, image_nx, image_ny))


    def getModel(self):

        inputs = Input((self.size_input_image[0], self.size_input_image[1], self.size_input_image[2], 1))

        nbfeamaps_dwl1  = self.nbfeamaps_first_layer
        hidlayer_dwl1_1 = Convolution3D(nbfeamaps_dwl1, self.size_convfilter_lay1_downpath_layers[0], activation='relu', padding='valid')(inputs)
        hidlayer_dwl1_2 = Convolution3D(nbfeamaps_dwl1, self.size_convfilter_lay2_downpath_layers[0], activation='relu', padding='valid')(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = Dropout(self.dropout_rate)(hidlayer_dwl1_2)
        hidlayer_dwl2_1 = MaxPooling3D(pool_size=self.size_pooling_layers[0])(hidlayer_dwl1_2)

        nbfeamaps_dwl2  = 2*nbfeamaps_dwl1
        hidlayer_dwl2_2 = Convolution3D(nbfeamaps_dwl2, self.size_convfilter_lay1_downpath_layers[1], activation='relu', padding='valid')(hidlayer_dwl2_1)
        hidlayer_dwl2_3 = Convolution3D(nbfeamaps_dwl2, self.size_convfilter_lay2_downpath_layers[1], activation='relu', padding='valid')(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = Dropout(self.dropout_rate)(hidlayer_dwl2_3)
        hidlayer_dwl3_1 = MaxPooling3D(pool_size=self.size_pooling_layers[1])(hidlayer_dwl2_3)

        nbfeamaps_dwl3  = 2*nbfeamaps_dwl2
        hidlayer_dwl3_2 = Convolution3D(nbfeamaps_dwl3, self.size_convfilter_lay1_downpath_layers[2], activation='relu', padding='valid')(hidlayer_dwl3_1)
        hidlayer_dwl3_3 = Convolution3D(nbfeamaps_dwl3, self.size_convfilter_lay2_downpath_layers[2], activation='relu', padding='valid')(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = Dropout(self.dropout_rate)(hidlayer_dwl3_3)
        hidlayer_dwl4_1 = MaxPooling3D(pool_size=self.size_pooling_layers[2])(hidlayer_dwl3_3)

        nbfeamaps_dwl4  = 2*nbfeamaps_dwl3
        hidlayer_dwl4_2 = Convolution3D(nbfeamaps_dwl4, self.size_convfilter_lay1_downpath_layers[3], activation='relu', padding='valid')(hidlayer_dwl4_1)
        hidlayer_dwl4_3 = Convolution3D(nbfeamaps_dwl4, self.size_convfilter_lay2_downpath_layers[3], activation='relu', padding='valid')(hidlayer_dwl4_2)
        hidlayer_dwl4_3 = Dropout(self.dropout_rate)(hidlayer_dwl4_3)
        hidlayer_dwl5_1 = MaxPooling3D(pool_size=self.size_pooling_layers[3])(hidlayer_dwl4_3)

        nbfeamaps_dwl5  = 2*nbfeamaps_dwl4
        hidlayer_dwl5_2 = Convolution3D(nbfeamaps_dwl5, self.size_convfilter_lay1_downpath_layers[4], activation='relu', padding='valid')(hidlayer_dwl5_1)
        hidlayer_dwl5_3 = Convolution3D(nbfeamaps_dwl5, self.size_convfilter_lay2_downpath_layers[4], activation='relu', padding='valid')(hidlayer_dwl5_2)
        hidlayer_dwl5_3 = Dropout(self.dropout_rate)(hidlayer_dwl5_3)

        hidlayer_upl4_1 = UpSampling3D(size=self.size_pooling_layers[3])(hidlayer_dwl5_3)
        hidlayer_dwl4_3 = Cropping3D(cropping=self.size_cropping_layers[3])(hidlayer_dwl4_3)
        hidlayer_upl4_1 = merge([hidlayer_upl4_1, hidlayer_dwl4_3], mode='concat', concat_axis=-1)

        nbfeamaps_upl4  = nbfeamaps_dwl4
        hidlayer_upl4_2 = Convolution3D(nbfeamaps_upl4, self.size_convfilter_uppath_layers[3], activation='relu', padding='valid')(hidlayer_upl4_1)
        hidlayer_upl4_3 = Convolution3D(nbfeamaps_upl4, self.size_convfilter_uppath_layers[3], activation='relu', padding='valid')(hidlayer_upl4_2)
        hidlayer_upl4_3 = Dropout(self.dropout_rate)(hidlayer_upl4_3)

        hidlayer_upl3_1 = UpSampling3D(size=self.size_pooling_layers[2])(hidlayer_upl4_3)
        hidlayer_dwl3_3 = Cropping3D(cropping=self.size_cropping_layers[2])(hidlayer_dwl3_3)
        hidlayer_upl3_1 = merge([hidlayer_upl3_1, hidlayer_dwl3_3], mode='concat', concat_axis=-1)

        nbfeamaps_upl3  = nbfeamaps_dwl3
        hidlayer_upl3_2 = Convolution3D(nbfeamaps_upl3, self.size_convfilter_uppath_layers[2], activation='relu', padding='valid')(hidlayer_upl3_1)
        hidlayer_upl3_3 = Convolution3D(nbfeamaps_upl3, self.size_convfilter_uppath_layers[2], activation='relu', padding='valid')(hidlayer_upl3_2)
        hidlayer_upl3_3 = Dropout(self.dropout_rate)(hidlayer_upl3_3)

        hidlayer_upl2_1 = UpSampling3D(size=self.size_pooling_layers[1])(hidlayer_upl3_3)
        hidlayer_dwl2_3 = Cropping3D(cropping=self.size_cropping_layers[1])(hidlayer_dwl2_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfeamaps_upl2  = nbfeamaps_dwl2
        hidlayer_upl2_2 = Convolution3D(nbfeamaps_upl2, self.size_convfilter_uppath_layers[1], activation='relu', padding='valid')(hidlayer_upl2_1)
        hidlayer_upl2_3 = Convolution3D(nbfeamaps_upl2, self.size_convfilter_uppath_layers[1], activation='relu', padding='valid')(hidlayer_upl2_2)
        hidlayer_upl2_3 = Dropout(self.dropout_rate)(hidlayer_upl2_3)

        hidlayer_upl1_1 = UpSampling3D(size=self.size_pooling_layers[0])(hidlayer_upl2_3)
        hidlayer_dwl1_2 = Cropping3D(cropping=self.size_cropping_layers[0])(hidlayer_dwl1_2)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfeamaps_upl1  = nbfeamaps_dwl1
        hidlayer_upl1_2 = Convolution3D(nbfeamaps_upl1, self.size_convfilter_uppath_layers[0], activation='relu', padding='valid')(hidlayer_upl1_1)
        hidlayer_upl1_3 = Convolution3D(nbfeamaps_upl1, self.size_convfilter_uppath_layers[0], activation='relu', padding='valid')(hidlayer_upl1_2)
        hidlayer_upl1_3 = Dropout(self.dropout_rate)(hidlayer_upl1_3)

        outputs = Convolution3D(1, (1, 1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


class Unet3D_Shallow_Tailored(NeuralNetwork):

    nbfeamaps_first_layer = 16

    size_convfilter_downpath_layers_opt1 = [(3, 3, 3), (1, 3, 3), (1, 3, 3)]
    size_convfilter_uppath_layers_opt1   = [(3, 3, 3), (1, 3, 3), (1, 3, 3)]
    size_pooling_layers_opt1             = [(2, 2, 2), (1, 2, 2)]
    size_cropping_layers_opt1            = [(0, 16, 16), (0, 4, 4)]

    size_convfilter_downpath_layers_opt2 = [(3, 3, 3), (3, 3, 3), (1, 3, 3)]
    size_convfilter_uppath_layers_opt2   = [(3, 3, 3), (1, 3, 3), (1, 3, 3)]
    size_pooling_layers_opt2             = [(2, 2, 2), (1, 2, 2)]
    size_cropping_layers_opt2            = [(4, 16, 16), (0, 4, 4)]

    size_convfilter_downpath_layers_opt3 = [(3, 3, 3), (1, 3, 3), (1, 3, 3)]
    size_convfilter_uppath_layers_opt3   = [(3, 3, 3), (1, 3, 3), (1, 3, 3)]
    size_pooling_layers_opt3             = [(2, 2, 2), (1, 2, 2)]
    size_cropping_layers_opt3            = [(0, 16, 16), (0, 4, 4)]

    size_convfilter_downpath_layers_opt4 = [(3, 3, 3), (3, 3, 3), (1, 3, 3)]
    size_convfilter_uppath_layers_opt4   = [(3, 3, 3), (1, 3, 3), (1, 3, 3)]
    size_pooling_layers_opt4             = [(2, 2, 2), (1, 2, 2)]
    size_cropping_layers_opt4            = [(4, 16, 16), (0, 4, 4)]

    list_size_valid_input_images  = [(16, 256, 256), (32, 256, 256), (16, 352, 240), (32, 352, 240)]
    list_size_valid_output_images = [(4, 216, 216), (16, 216, 216), (8, 312, 200), (16, 312, 200)]

    dropout_rate = 0.2


    def __init__(self, (image_nz, image_nx, image_ny)):
        self.size_input_image  = (image_nz, image_nx, image_ny)

        self.type_network = self.find_type_network_from_size_images(self.size_input_image)

        if self.type_network==0:
            self.size_convfilter_downpath_layers = self.size_convfilter_downpath_layers_opt1
            self.size_convfilter_uppath_layers   = self.size_convfilter_uppath_layers_opt1
            self.size_pooling_layers             = self.size_pooling_layers_opt1
            self.size_cropping_layers            = self.size_cropping_layers_opt1

        if self.type_network==1:
            self.size_convfilter_downpath_layers = self.size_convfilter_downpath_layers_opt2
            self.size_convfilter_uppath_layers   = self.size_convfilter_uppath_layers_opt2
            self.size_pooling_layers             = self.size_pooling_layers_opt2
            self.size_cropping_layers            = self.size_cropping_layers_opt2

        if self.type_network==2:
            self.size_convfilter_downpath_layers = self.size_convfilter_downpath_layers_opt3
            self.size_convfilter_uppath_layers   = self.size_convfilter_uppath_layers_opt3
            self.size_pooling_layers             = self.size_pooling_layers_opt3
            self.size_cropping_layers            = self.size_cropping_layers_opt3

        if self.type_network==3:
            self.size_convfilter_downpath_layers = self.size_convfilter_downpath_layers_opt4
            self.size_convfilter_uppath_layers   = self.size_convfilter_uppath_layers_opt4
            self.size_pooling_layers             = self.size_pooling_layers_opt4
            self.size_cropping_layers            = self.size_cropping_layers_opt4

        self.size_output_image = self.list_size_valid_output_images[self.type_network]


    def check_valid_size_input_images(self, (image_nz, image_nx, image_ny)):
        if (image_nz, image_nx, image_ny) not in  self.list_size_valid_input_images:
            message = "\'Unet3D_Shallow_Tailored\' only valid for input images of size: %s..." %(self.list_size_valid_input_images)
            CatchErrorException(message)
        else:
            return True

    def find_type_network_from_size_images(self, (image_nz, image_nx, image_ny)):
        if self.check_valid_size_input_images((image_nz, image_nx, image_ny)):
            return self.list_size_valid_input_images.index((image_nz, image_nx, image_ny))


    def getModel(self):

        inputs = Input((self.size_input_image[0], self.size_input_image[1], self.size_input_image[2], 1))

        nbfeamaps_dwl1  = self.nbfeamaps_first_layer
        hidlayer_dwl1_1 = Convolution3D(nbfeamaps_dwl1, self.size_convfilter_downpath_layers[0], activation='relu', padding='valid')(inputs)
        hidlayer_dwl1_2 = Convolution3D(nbfeamaps_dwl1, self.size_convfilter_downpath_layers[0], activation='relu', padding='valid')(hidlayer_dwl1_1)
        hidlayer_dwl1_2 = Dropout(self.dropout_rate)(hidlayer_dwl1_2)
        hidlayer_dwl2_1 = MaxPooling3D(pool_size=self.size_pooling_layers[0])(hidlayer_dwl1_2)

        nbfeamaps_dwl2  = 2*nbfeamaps_dwl1
        hidlayer_dwl2_2 = Convolution3D(nbfeamaps_dwl2, self.size_convfilter_downpath_layers[1], activation='relu', padding='valid')(hidlayer_dwl2_1)
        hidlayer_dwl2_3 = Convolution3D(nbfeamaps_dwl2, self.size_convfilter_downpath_layers[1], activation='relu', padding='valid')(hidlayer_dwl2_2)
        hidlayer_dwl2_3 = Dropout(self.dropout_rate)(hidlayer_dwl2_3)
        hidlayer_dwl3_1 = MaxPooling3D(pool_size=self.size_pooling_layers[1])(hidlayer_dwl2_3)

        nbfeamaps_dwl3  = 2*nbfeamaps_dwl2
        hidlayer_dwl3_2 = Convolution3D(nbfeamaps_dwl3, self.size_convfilter_downpath_layers[2], activation='relu', padding='valid')(hidlayer_dwl3_1)
        hidlayer_dwl3_3 = Convolution3D(nbfeamaps_dwl3, self.size_convfilter_downpath_layers[2], activation='relu', padding='valid')(hidlayer_dwl3_2)
        hidlayer_dwl3_3 = Dropout(self.dropout_rate)(hidlayer_dwl3_3)

        hidlayer_upl2_1 = UpSampling3D(size=self.size_pooling_layers[1])(hidlayer_dwl3_3)
        hidlayer_dwl2_3 = Cropping3D(cropping=self.size_cropping_layers[1])(hidlayer_dwl2_3)
        hidlayer_upl2_1 = merge([hidlayer_upl2_1, hidlayer_dwl2_3], mode='concat', concat_axis=-1)

        nbfeamaps_upl2  = nbfeamaps_dwl2
        hidlayer_upl2_2 = Convolution3D(nbfeamaps_upl2, self.size_convfilter_uppath_layers[1], activation='relu', padding='valid')(hidlayer_upl2_1)
        hidlayer_upl2_3 = Convolution3D(nbfeamaps_upl2, self.size_convfilter_uppath_layers[1], activation='relu', padding='valid')(hidlayer_upl2_2)
        hidlayer_upl2_3 = Dropout(self.dropout_rate)(hidlayer_upl2_3)

        hidlayer_upl1_1 = UpSampling3D(size=self.size_pooling_layers[0])(hidlayer_upl2_3)
        hidlayer_dwl1_2 = Cropping3D(cropping=self.size_cropping_layers[0])(hidlayer_dwl1_2)
        hidlayer_upl1_1 = merge([hidlayer_upl1_1, hidlayer_dwl1_2], mode='concat', concat_axis=-1)

        nbfeamaps_upl1  = nbfeamaps_dwl1
        hidlayer_upl1_2 = Convolution3D(nbfeamaps_upl1, self.size_convfilter_uppath_layers[0], activation='relu', padding='valid')(hidlayer_upl1_1)
        hidlayer_upl1_3 = Convolution3D(nbfeamaps_upl1, self.size_convfilter_uppath_layers[0], activation='relu', padding='valid')(hidlayer_upl1_2)
        hidlayer_upl1_3 = Dropout(self.dropout_rate)(hidlayer_upl1_3)

        outputs = Convolution3D(1, (1, 1, 1), activation='sigmoid')(hidlayer_upl1_3)

        model = Model(input=inputs, output=outputs)

        return model


# All Available Networks
def DICTAVAILNETWORKS3D(size_image, option):
    if   (option=="Unet3D"):
        return Unet3D(size_image)
    elif (option=="Unet3D_Dropout"):
        return Unet3D(size_image, is_dropout=True)
    elif (option=="Unet3D_Shallow"):
        return Unet3D_Shallow(size_image)
    elif (option=="Unet3D_Shallow_Dropout"):
        return Unet3D_Shallow(size_image, is_dropout=True)
    elif (option=="Unet3D_Tailored"):
        return Unet3D_Tailored(size_image)
    elif (option=="Unet3D_Shallow_Tailored"):
        return Unet3D_Shallow_Tailored(size_image)
    else:
        return 0