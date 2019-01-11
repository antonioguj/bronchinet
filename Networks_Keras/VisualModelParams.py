#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from keras import backend as K
import numpy as np


class VisualModelParams(object):

    def __init__(self, model, size_image):
        self.model      = model
        self.size_image = size_image


    def find_layer_index_from_name(self, name_layer):

        for ilay_idx, it_layer in enumerate(self.model.layers):
            if(it_layer.name == name_layer):
                return ilay_idx
        #endfor
        print('ERROR: layer "%s" not found...')
        return None

    def is_list_patches_images_array(self, in_array_shape):

        if(len(in_array_shape) == len(self.size_image) + 1):
            return False
        elif(len(in_array_shape) == len(self.size_image) + 2):
            return True
        else:
            print('ERROR: with dims...')
            return None


    def get_feature_maps(self, in_images_array, name_layer, max_num_feat_maps=None, first_feat_maps=None):

        # find index for "name_layer"
        idx_layer = self.find_layer_index_from_name(name_layer)
        if not idx_layer:
            return None

        # define function for "idx_layer"
        layer_cls = self.model.layers[idx_layer]

        num_featmaps_all = layer_cls.output.shape[-1].value

        if max_num_feat_maps:
            if not first_feat_maps:
                first_feat_maps = 0

            num_featmaps = min(max_num_feat_maps, num_featmaps_all-first_feat_maps)
            last_feat_maps = first_feat_maps + num_featmaps

            get_feat_maps_layer_func = K.function([self.model.input], [layer_cls.output[..., first_feat_maps:last_feat_maps]])
        else:
            num_featmaps = num_featmaps_all

            get_feat_maps_layer_func = K.function([self.model.input], [layer_cls.output])


        if (self.is_list_patches_images_array(in_images_array.shape)):
            #input: list of image patches arrays

            num_images = in_images_array.shape[0]

            out_featmaps_shape = [num_images] + list(self.size_image) + [num_featmaps]
            out_featmaps_array = np.zeros(out_featmaps_shape, dtype=in_images_array.dtype)

            for i, ipatch_images_array in enumerate(in_images_array):

                # compute the feature maps (reformat input image)
                ipatch_featmaps_array = get_feat_maps_layer_func([[ipatch_images_array]])
                out_featmaps_array[i] = ipatch_featmaps_array[0]
            #endfor

            return out_featmaps_array.astype(np.float32)

        else:
            #input: one image array
            # compute the feature maps (reformat input image)
            out_featmaps_array = get_feat_maps_layer_func([[in_images_array]])

            return out_featmaps_array[0].astype(np.float32)


    def get_feature_maps_all_layers(self, in_images_array, max_num_feat_maps=None, first_feat_maps=None):

        # save output in a dictionary
        out_featmaps_array = {}
        for it_layer in self.model.layers:
            out_featmaps_array[it_layer.name] = self.get_feature_maps(in_images_array,
                                                                      it_layer.name,
                                                                      max_num_feat_maps=max_num_feat_maps,
                                                                      first_feat_maps=first_feat_maps)
        #endfor

        return out_featmaps_array.astype(np.float32)
