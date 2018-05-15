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


class Unet3D_General(NeuralNetwork):

    num_layers_depth_default                = 5
    num_featuremaps_layers_default          = [16, 32, 64, 128, 256]
    num_convlayers_downpath_default         = [2, 2, 2, 2, 2]
    num_convlayers_uppath_default           = [2, 2, 2, 2, 2]
    size_convfilter_downpath_layers_default = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    size_convfilter_uppath_layers_default   = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    size_pooling_layers_default             = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    type_padding_default                    = 'same'
    activation_hidden_default               = 'relu'
    activation_output_default               = 'sigmoid'
    dropout_rate_default                    = 0.2


    def __init__(self, size_image,
                 num_layers_depth=num_layers_depth_default,
                 num_featuremaps_layers=num_featuremaps_layers_default,
                 num_convlayers_downpath=num_convlayers_downpath_default,
                 num_convlayers_uppath=num_convlayers_uppath_default,
                 size_convfilter_downpath_layers=size_convfilter_downpath_layers_default,
                 size_convfilter_uppath_layers=size_convfilter_uppath_layers_default,
                 size_pooling_layers=size_pooling_layers_default,
                 type_padding=type_padding_default,
                 activation_hidden=activation_hidden_default,
                 activation_output=activation_output_default,
                 isDropout=False,
                 dropout_rate=dropout_rate_default,
                 isBatchNormalize=False):

        self.size_image                     = size_image
        self.num_layers_depth               = num_layers_depth
        self.num_featuremaps_layers         = num_featuremaps_layers
        self.num_convlayers_downpath        = num_convlayers_downpath
        self.num_convlayers_uppath          = num_convlayers_uppath
        self.size_convfilter_downpath_layers= size_convfilter_downpath_layers
        self.size_convfilter_uppath_layers  = size_convfilter_uppath_layers
        self.size_pooling_layers            = size_pooling_layers
        self.type_padding                   = type_padding
        self.activation_hidden              = activation_hidden
        self.activation_output              = activation_output
        self.isDropout                      = isDropout
        self.dropout_rate                   = dropout_rate
        self.isBatchNormalize               = isBatchNormalize


    @staticmethod
    def get_size_output_convLayer_sameConv(size_input, size_filter):
        return size_input

    @staticmethod
    def get_size_output_convLayer_validConv(size_input, size_filter):
        return tuple((s_i - s_f + 1) for (s_i, s_f) in zip(size_input, size_filter))

    @staticmethod
    def get_size_output_transpose_convLayer_sameConv(size_input, size_filter):
        return size_input

    @staticmethod
    def get_size_output_transpose_convLayer_validConv(size_input, size_filter):
        return tuple((s_i + s_f - 1) for (s_i, s_f) in zip(size_input, size_filter))

    def get_size_output_convLayer(self, size_input, size_filter):
        if self.type_padding == 'valid':
            return self.get_size_output_convLayer_validConv(size_input, size_filter)
        elif self.type_padding=='same':
            return self.get_size_output_convLayer_sameConv(size_input, size_filter)

    def get_size_output_transpose_convLayer(self, size_input, size_filter):
        if self.type_padding == 'valid':
            return self.get_size_output_transpose_convLayer_validConv(size_input, size_filter)
        elif self.type_padding=='same':
            return self.get_size_output_transpose_convLayer_sameConv(size_input, size_filter)

    @staticmethod
    def get_size_output_poolingLayer(size_input, size_pooling):
        return tuple((s_i // s_p) for (s_i, s_p) in zip(size_input, size_pooling))

    @staticmethod
    def get_size_output_upsampleLayer(size_input, size_pooling):
        return tuple((s_i * s_u) for (s_i, s_u) in zip(size_input, size_pooling))

    @staticmethod
    def get_limits_CropImage(size_large_input, size_small_input):
        return tuple(((s_li - s_si)/2, (s_li + s_si)/2) for (s_li, s_si) in zip(size_large_input, size_small_input))


    def get_size_output_full_Unet(self):

        if self.type_padding=='valid':

            size_output = self.size_image

            for ilayer in range(self.num_layers_depth-1):
                # Convolutional Layers
                for iconv in range(self.num_convlayers_downpath[ilayer]):
                    size_output = self.get_size_output_convLayer_validConv(size_output, self.size_convfilter_downpath_layers[ilayer])
                #endfor
                # Pooling layer
                size_output = self.get_size_output_poolingLayer(size_output, self.size_pooling_layers[ilayer])
            #endfor

            # Convolutional Layers
            ilayer = self.num_layers_depth-1
            for iconv in range(self.num_convlayers_downpath[ilayer]):
                size_output = self.get_size_output_convLayer_validConv(size_output, self.size_convfilter_downpath_layers[ilayer])
            #endfor

            for ilayer in range(self.num_layers_depth-2, -1, -1):
                # Upsampling Layer
                size_output = self.get_size_output_upsampleLayer(size_output, self.size_pooling_layers[ilayer])
                # Convolutional Layers
                for iconv in range(self.num_convlayers_uppath[ilayer]):
                    size_output = self.get_size_output_convLayer_validConv(size_output, self.size_convfilter_uppath_layers[ilayer])
                #endfor
            # endfor

            return size_output

        elif self.type_padding=='same':
            return self.size_image


    def get_size_output_Unet_depth_validConvs(self, ilayer, size_input):

        size_output = size_input

        # Convolutional Layers
        for iconv in range(self.num_convlayers_downpath[ilayer]):
            size_output = self.get_size_output_convLayer_validConv(size_output, self.size_convfilter_downpath_layers[ilayer])
        # endfor

        if ilayer==self.num_layers_depth-1:
            return size_output

        # Pooling layer
        size_output = self.get_size_output_poolingLayer(size_output, self.size_pooling_layers[ilayer])

        # Compute size_output of Unet of 'depth-1' (Recurrent function)
        size_output = self.get_size_output_Unet_depth_validConvs(ilayer+1, size_output)

        # Upsampling Layer
        size_output = self.get_size_output_upsampleLayer(size_output, self.size_pooling_layers[ilayer])

        # Convolutional Layers
        for iconv in range(self.num_convlayers_uppath[ilayer]):
            size_output = self.get_size_output_convLayer_validConv(size_output, self.size_convfilter_uppath_layers[ilayer])
        # endfor

        return size_output


    def get_size_input_given_output_full_Unet(self, size_output):

        if self.type_padding=='valid':
            # go through the Unet network backwards: transpose convolutions, poolings / upsamplings
            size_input = size_output

            for ilayer in range(self.num_layers_depth-1):
                # Convolutional Layers
                for iconv in range(self.num_convlayers_uppath[ilayer]):
                    size_input = self.get_size_output_transpose_convLayer_validConv(size_input, self.size_convfilter_uppath_layers[ilayer])
                #endfor
                # Pooling layer
                size_input = self.get_size_output_poolingLayer(size_input, self.size_pooling_layers[ilayer])
            #endfor

            # Convolutional Layers
            ilayer = self.num_layers_depth-1
            for iconv in range(self.num_convlayers_uppath[ilayer]):
                size_input = self.get_size_output_transpose_convLayer_validConv(size_input, self.size_convfilter_uppath_layers[ilayer])
            #endfor

            for ilayer in range(self.num_layers_depth-2, -1, -1):
                # Upsampling Layer
                size_input = self.get_size_output_upsampleLayer(size_input, self.size_pooling_layers[ilayer])
                # Convolutional Layers
                for iconv in range(self.num_convlayers_downpath[ilayer]):
                    size_input = self.get_size_output_transpose_convLayer_validConv(size_input, self.size_convfilter_downpath_layers[ilayer])
                #endfor
            # endfor

            return size_input

        elif self.type_padding=='same':
            return self.size_image


    def get_size_input_given_output_Unet_depth_validConvs(self, ilayer, size_output):

        size_input = size_output

        # Convolutional Layers
        for iconv in range(self.num_convlayers_downpath[ilayer]):
            size_input = self.get_size_output_transpose_convLayer_validConv(size_input, self.size_convfilter_downpath_layers[ilayer])
        # endfor

        if ilayer==self.num_layers_depth-1:
            return size_input

        # Pooling layer
        size_input = self.get_size_output_poolingLayer(size_input, self.size_pooling_layers[ilayer])

        # Compute size_output of Unet of 'depth-1' (Recurrent function)
        size_input = self.get_size_input_given_output_Unet_depth_validConvs(ilayer+1, size_input)

        # Upsampling Layer
        size_input = self.get_size_output_upsampleLayer(size_input, self.size_pooling_layers[ilayer])

        # Convolutional Layers
        for iconv in range(self.num_convlayers_uppath[ilayer]):
            size_input = self.get_size_output_transpose_convLayer_validConv(size_input, self.size_convfilter_uppath_layers[ilayer])
        # endfor

        return size_input


    def get_limits_cropImage_merge_downLayer_validConv(self, ilayer):

        # assume size of final upsampling layer
        size_final = tuple([int(pow(2, (self.num_layers_depth-ilayer-1)))] *3)
        size_input = size_final

        # go through the network backwards until the corresponding layer in downsampling path
        # Pooling layer
        size_input = self.get_size_output_poolingLayer(size_input, self.size_pooling_layers[ilayer])

        # Compute size_input of Unet of 'depth-1' (Recurrent function)
        size_input = self.get_size_input_given_output_Unet_depth_validConvs(ilayer+1, size_input)

        # Upsampling layer
        size_input = self.get_size_output_upsampleLayer(size_input, self.size_pooling_layers[ilayer])

        return self.get_limits_CropImage(size_input, size_final)


    @staticmethod
    def check_correct_size_convLayer(size_input, size_filter):
        # check whether the layer is larher than filter size
        return all((s_i > s_f) for (s_i, s_f) in zip(size_input, size_filter))

    @staticmethod
    def check_correct_size_poolingLayer(size_input, size_pooling):
        # check whether layer before pooling has odd dimensions
        return all((s_i % s_p == 0) for (s_i, s_p) in zip(size_input, size_pooling))

    def check_correct_size_all_layers_full_Unet(self):

        if self.type_padding == 'valid':

            size_output = self.size_image

            for ilayer in range(self.num_layers_depth-1):

                # Convolutional Layers
                for iconv in range(self.num_convlayers_downpath[ilayer]):
                    size_output = self.get_size_output_convLayer_validConv(size_output, self.size_convfilter_downpath_layers[ilayer])

                    if self.check_correct_size_convLayer(size_output, self.size_convfilter_downpath_layers[ilayer]):
                        return False, 'wrong size of conv. layer %s in depth %s, of size %s and filter size %s' %(iconv, ilayer, size_output, self.size_convfilter_downpath_layers[ilayer])
                #endfor

                if self.check_correct_size_poolingLayer(size_output):
                    return False, 'wrong size of pooling layer in depth %s, of size %s' %(ilayer, size_output)

                # Pooling layer
                size_output = self.get_size_output_poolingLayer(size_output, self.size_pooling_layers[ilayer])
            #endfor

            # Convolutional Layers
            ilayer = self.num_layers_depth-1
            for iconv in range(self.num_convlayers_downpath[ilayer]):
                size_output = self.get_size_output_convLayer_validConv(size_output, self.size_convfilter_downpath_layers[ilayer])

                if self.check_correct_size_convLayer(size_output, self.size_convfilter_downpath_layers[ilayer]):
                    return False, 'wrong size of conv. layer %s in depth %s, of size %s and filter size %s' %(iconv, ilayer, size_output, self.size_convfilter_downpath_layers[ilayer])
            #endfor

            for ilayer in range(self.num_layers_depth-2, -1, -1):
                # Upsampling Layer
                size_output = self.get_size_output_upsampleLayer(size_output, self.size_pooling_layers[ilayer])

                # Convolutional Layers
                for iconv in range(self.num_convlayers_uppath[ilayer]):
                    size_output = self.get_size_output_convLayer_validConv(size_output, self.size_convfilter_uppath_layers[ilayer])

                    if self.check_correct_size_convLayer(size_output, self.size_convfilter_uppath_layers[ilayer]):
                        return False, 'wrong size of conv. layer %s in depth %s, of size %s and filter size %s' %(iconv, ilayer, size_output, self.size_convfilter_uppath_layers[ilayer])
                #endfor
            # endfor

        elif self.type_padding=='same':
            # All Layers are correct for sure
            return True, 'everything good'


    def getModel(self):

        inputs = Input(shape=self.size_image + (1,))

        # ********** DOWNSAMPLING PATH **********
        last_layer = inputs
        list_last_convlayer_downpath = []

        for ilayer in range(self.num_layers_depth-1):

            # Convolutional layers
            for iconv in range(self.num_convlayers_downpath[ilayer]):

                last_layer = Convolution3D(filters=self.num_featuremaps_layers[ilayer],
                                           kernel_size=self.size_convfilter_downpath_layers[ilayer],
                                           padding=self.type_padding,
                                           activation=self.activation_hidden)(last_layer)
            #endfor

            if self.isDropout:
                last_layer = Dropout(rate=self.dropout_rate)(last_layer)
            if self.isBatchNormalize:
                last_layer = BatchNormalization()(last_layer)

            # Store last convolutional layer needed for upsampling path
            list_last_convlayer_downpath.append(last_layer)

            # Pooling layer
            last_layer = MaxPooling3D(pool_size=self.size_pooling_layers[ilayer],
                                      padding=self.type_padding)(last_layer)
        #endfor
        # ********** DOWNSAMPLING PATH **********

        # Deepest convolutional layers
        ilayer = self.num_layers_depth - 1
        for j in range(self.num_convlayers_downpath[ilayer]):

            last_layer = Convolution3D(filters=self.num_featuremaps_layers[ilayer],
                                       kernel_size=self.size_convfilter_downpath_layers[ilayer],
                                       padding=self.type_padding,
                                       activation=self.activation_hidden)(last_layer)
        #endfor

        if self.isDropout:
            last_layer = Dropout(rate=self.dropout_rate)(last_layer)
        if self.isBatchNormalize:
            last_layer = BatchNormalization()(last_layer)

        # ********** UPSAMPLING PATH **********
        #
        for ilayer in range(self.num_layers_depth-2, -1, -1):

            # Upsampling layer
            last_layer = UpSampling3D(size=self.size_pooling_layers[ilayer])(last_layer)

            # Merge layers
            if self.type_padding=='valid':
                # need to crop the downpath layer to the size of uppath layer
                shape_cropping = self.get_limits_cropImage_merge_downLayer_validConv(ilayer)
                last_layer_downpath = Cropping3D(cropping=shape_cropping)(list_last_convlayer_downpath[ilayer])

            elif self.type_padding=='same':
                last_layer_downpath = list_last_convlayer_downpath[ilayer]

            last_layer = merge(inputs=[last_layer, last_layer_downpath],
                               mode='concat',
                               concat_axis=-1)

            # Convolutional Layers
            for j in range(self.num_convlayers_downpath[ilayer]):

                last_layer = Convolution3D(filters=self.num_featuremaps_layers[ilayer],
                                           kernel_size=self.size_convfilter_uppath_layers[ilayer],
                                           padding=self.type_padding,
                                           activation=self.activation_hidden)(last_layer)
            #endfor

            if self.isDropout:
                last_layer = Dropout(rate=self.dropout_rate)(last_layer)
            if self.isBatchNormalize:
                last_layer = BatchNormalization()(last_layer)
        #endfor
        #  ********** UPSAMPLING PATH **********

        outputs = Convolution3D(filters=1,
                                kernel_size=(1, 1, 1),
                                padding=self.type_padding,
                                activation=self.activation_output)(last_layer)

        # Return complete model
        return Model(input=inputs, output=outputs)


# All Available Networks
def DICTAVAILNETWORKS3D(size_image, option):
    if   (option=='Unet3D'):
        return Unet3D_General(size_image, num_layers_depth=5)
    elif (option=='Unet3D_Dropout'):
        return Unet3D_General(size_image, num_layers_depth=5, isDropout=True)
    elif (option=='Unet3D_Batchnorm'):
        return Unet3D_General(size_image, num_layers_depth=5, isBatchNormalize=True)
    elif (option=='Unet3D_Shallow'):
        return Unet3D_General(size_image, num_layers_depth=3)
    elif (option=='Unet3D_Shallow_Dropout'):
        return Unet3D_General(size_image, num_layers_depth=3, isDropout=True)
    elif (option=='Unet3D_Shallow_Batchnorm'):
        return Unet3D_General(size_image, num_layers_depth=3, isBatchNormalize=True)
    else:
        return 0