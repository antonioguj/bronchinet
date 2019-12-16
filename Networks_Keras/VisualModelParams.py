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
from Common.ErrorMessages import *
#from tensorflow.python.keras import backend as K
import numpy as np



class VisualModelParams(object):

    def __init__(self, model, size_in_images):
        self.model          = model
        self.size_in_images = size_in_images


    def find_layer_index_from_name(self, name_layer):
        for ilay_idx, it_layer in enumerate(self.model.layers):
            if(it_layer.name == name_layer):
                return ilay_idx
        #endfor
        return None

    def is_images_array_list_patches(self, in_array_shape):
        if(len(in_array_shape) == len(self.size_in_images) + 1):
            return False
        elif(len(in_array_shape) == len(self.size_in_images) + 2):
            return True
        else:
            print('ERROR: with dims \'%s\'...' %(in_array_shape))
            return None


    def get_feature_maps(self, in_images_array, name_layer, index_first_featmap=None, max_num_featmaps=None):
        # find index for "name_layer"
        idx_layer = self.find_layer_index_from_name(name_layer)
        if not idx_layer:
            message = 'layer \'%s\' does not exist in model...' %(name_layer)
            CatchErrorException(message)


        model_layer_cls = self.model.layers[idx_layer]
        num_featmaps = model_layer_cls.output.shape[-1].value

        # # compute the limit indexes for feature maps, if needed
        # if max_num_featmaps:
        #     if not index_first_featmap:
        #         index_first_featmap = 0
        #     num_featmaps = min(max_num_featmaps, num_featmaps-index_first_featmap)
        #     index_last_featmap = index_first_featmap + num_featmaps
        # else:
        #     index_first_featmap = 0
        #     index_last_featmap  = num_featmaps

        # define function to retrieve feauture maps for "idx_layer"
        #get_feat_maps_layer_func = K.function([self.model.input], [model_layer_cls.output[..., index_first_featmap:index_last_featmap]])
        get_feat_maps_layer_func = K.function([self.model.input], [model_layer_cls.output])


        if (self.is_images_array_list_patches(in_images_array.shape)):
            #input: list of image patches arrays
            num_images = in_images_array.shape[0]
            out_featmaps_shape = [num_images] + list(self.size_in_images) + [num_featmaps]
            out_featmaps_array = np.zeros(out_featmaps_shape, FORMATPROBABILITYDATA)

            for i, ipatch_images_array in enumerate(in_images_array):
                # compute the feature maps (reformat input image)
                ipatch_featmaps_array = get_feat_maps_layer_func([[ipatch_images_array]])
                out_featmaps_array[i] = ipatch_featmaps_array[0].astype(FORMATPROBABILITYDATA)
            #endfor

            return out_featmaps_array

        else:
            #input: one image array
            # compute the feature maps (reformat input image)
            out_featmaps_array = get_feat_maps_layer_func([[in_images_array]])

            return out_featmaps_array[0].astype(FORMATPROBABILITYDATA)


    def get_feature_maps_all_layers(self, in_images_array, index_first_featmap=None, max_num_featmaps=None):
        # save output in a dictionary
        out_featmaps_array = {}
        for it_layer in self.model.layers:
            out_featmaps_array[it_layer.name] = self.get_feature_maps(in_images_array, it_layer.name,
                                                                      index_first_featmap, max_num_featmaps)
        #endfor
        return out_featmaps_array


    def get_feature_maps_list_layers(self, in_images_array, list_names_layers, index_first_featmap=None, max_num_featmaps=None):
        # save output in a dictionary
        out_featmaps_array = {}
        for it_layer_name in list_names_layers:
            out_featmaps_array[it_layer_name] = self.get_feature_maps(in_images_array, it_layer_name,
                                                                      index_first_featmap, max_num_featmaps)
        #endfor
        return out_featmaps_array