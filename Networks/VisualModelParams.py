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
        return -1

    def is_list_patches_image_array(self, in_array_shape):

        if(len(in_array_shape) == len(self.size_image) + 1):
            return False
        elif(len(in_array_shape) == len(self.size_image) + 2):
            return True
        else:
            print('ERROR: with dims...')
            return None


    def get_feature_maps(self, in_image_array, name_layer):

        # find index for "name_layer"
        idx_layer = self.find_layer_index_from_name(name_layer)

        # define function for "idx_layer"
        layer_cls = self.model.layers[idx_layer]
        layer_func = K.function([self.model.input], [layer_cls.output])

        num_featmaps = layer_cls.output.shape[-1].value


        if (self.is_list_patches_image_array(in_image_array.shape)):
            #input: list of image patches arrays

            num_images = in_image_array.shape[0]

            out_featmaps_shape = [num_images] + list(self.size_image) + [num_featmaps]
            out_featmaps_array = np.zeros(out_featmaps_shape, dtype=in_image_array.dtype)

            for i, ipatch_image_array in enumerate(in_image_array):

                # compute the feature maps (reformat input image)
                ipatch_featmaps_array = layer_func([[ipatch_image_array]])
                out_featmaps_array[i] = ipatch_featmaps_array[0]
            #endfor

            return out_featmaps_array

        else:
            #input: one image array
            # compute the feature maps (reformat input image)
            out_featmaps_array = layer_func([[in_image_array]])

            return out_featmaps_array[0]


    def get_feature_maps_all_layers(self, in_image_array):

        # save output in a dictionary
        out_featmaps_array = {}
        for it_layer in self.model.layers:
            out_featmaps_array[it_layer.name] = self.get_feature_maps(in_image_array, it_layer.name)
        #endfor

        return out_featmaps_array



def visualize_activation_in_layer_one_plot(model, input_list_images, namedir):
    """
    visualize activations in layers(By Shuai)
    """

    count_img = 1

    for input_image in input_list_images:

        # reformat input image
        input_image = [input_image.tolist()]

        namesubdir_img = 'featmaps_img-%0.2i' %(count_img)
        namesubdir_img = os.path.join(namedir, namesubdir_img)

        if not os.path.exists(namesubdir_img):
            os.makedirs(namesubdir_img)
        count_img = count_img+1


        # iterate across layers
        for layer_index in range(len(model.layers)):

            layer_name = model.layers[layer_index].name

            # define function
            convout1 = model.layers[layer_index]
            convout1_f = K.function([model.input], [convout1.output])

            # compute the activation map
            C1 = convout1_f([input_image])
            C1 = np.squeeze(C1)
            print(layer_index, ' ', layer_name, ' ', 'featmaps dims: ', C1.shape)


            newfigname = 'featmaps_%s' %(layer_name)
            newfigname = os.path.join(namesubdir_img, newfigname)

            # save results
            if len(C1.shape) == 3:  # if only a single filter in the layer (basically to save the input)

                C1 = C1[int(C1.shape[0] * 0.4), :, :]

                # Plot feature maps:
                plt.figure(layer_index, figsize=(3, 3))
                plt.figure(layer_index)

                ax1 = plt.subplot(1, 1, 1)
                # title = 'feature maps ' + str(plt_num+1)
                # plt.title(title)
                plt.axis('off')
                ax1.imshow(C1[:, :], cmap = 'viridis')

                plt.savefig(newfigname)
                plt.close()

            elif len(C1.shape) == 4:  # if several filters in the layer

                C1 = C1[int(C1.shape[0] * 0.4), :, :, :]

                # Plot feature maps:
                plt_row = 4
                plt_col = int(C1.shape[-1] / plt_row)

                if plt_col == 0:
                    plt_row = 1
                    plt_col = C1.shape[-1]

                plt.figure(layer_index, figsize=(3 * plt_col, 3 * plt_row))
                plt.figure(layer_index)

                for plt_num in range (C1.shape[-1]):
                    ax1 = plt.subplot(plt_row, plt_col, plt_num + 1)
                    # title = 'feature maps ' + str(plt_num+1)
                    # plt.title(title)
                    plt.axis('off')
                    ax1.imshow(C1[:, :, plt_num], cmap = 'viridis')
                #endfor

                plt.savefig(newfigname)
                plt.close()

            else:
                print('error: wrong dimensions for C1')
                return False
        #endfor
    #endfor