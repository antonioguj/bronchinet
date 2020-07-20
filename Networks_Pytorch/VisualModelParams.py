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
import torch.nn as nn
import torch
import numpy as np



class VisualModelParams(object):

    def __init__(self, model, size_in_images):
        self.model          = model
        self.size_in_images = size_in_images


    def isExist_name_layer_model(self, name_layer):
        for ilay_key, _ in self.model._modules.items():
            if(ilay_key == name_layer):
                return True
        #endfor
        return False


    def is_images_array_list_patches(self, in_array_shape):
        if(len(in_array_shape) == len(self.size_in_images) + 1):
            return False
        elif(len(in_array_shape) == len(self.size_in_images) + 2):
            return True
        else:
            message = 'ERROR: with dims \'%s\'...' % (in_array_shape)
            CatchErrorException(message)


    def get_feature_maps(self, in_images_generator, name_layer, index_first_featmap=None, max_num_featmaps=None):
        # check that "name_layer" exists in model
        if not self.isExist_name_layer_model(name_layer):
            message = 'layer \'%s\' does not exist in model...' %(name_layer)
            CatchErrorException(message)

        size_output_batch = self.model.get_size_output()
        num_featmaps = 8
        size_image_batch = size_output_batch[1:]

        # # compute the limit indexes for feature maps, if needed
        # if max_num_featmaps:
        #     if not index_first_featmap:
        #         index_first_featmap = 0
        #     num_featmaps = min(max_num_featmaps, num_featmaps-index_first_featmap)
        #     index_last_featmap = index_first_featmap + num_featmaps
        # else:
        #     index_first_featmap = 0
        #     index_last_featmap  = num_featmaps


        # define hook to retrieve feature maps of the desired model layer
        def hook(model, input, output):
            global out_featmaps_batch
            out_featmaps_batch = output.detach().cpu()
            # this should be eventually inside Networks.forward()
            out_featmaps_batch = out_featmaps_batch.view(-1, num_featmaps, size_image_batch[0], size_image_batch[1], size_image_batch[2])
            return None

        # attach hook to the corresponding module layer
        self.model._modules[name_layer].register_forward_hook(hook)


        num_batches = len(in_images_generator)
        out_featmaps_shape = [num_batches] + [num_featmaps] + list(size_image_batch)
        out_featmaps_array = np.zeros(out_featmaps_shape, dtype=FORMATPROBABILITYDATA)

        self.model = self.model.eval()
        self.model.preprocess(-1)

        for i_batch, (x_batch, y_batch) in enumerate(in_images_generator):
            x_batch.cuda()

            # only interested in the variable stored in hook defined above
            self.model(x_batch)
            out_featmaps_array[i_batch] = out_featmaps_batch
        #endfor

        # rollaxis to output in "channels_last"
        ndim_out = len(out_featmaps_array.shape)
        return np.rollaxis(out_featmaps_array, 1, ndim_out)
