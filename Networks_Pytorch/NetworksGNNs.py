#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from torch.nn import Conv3d, MaxPool3d, Upsample, ReLU, Sigmoid, Softmax
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn.functional import interpolate
import torch
from Networks_Pytorch.GNNUtil.gnnModules import NodeGNN, NodeGNNwithAttentionLayers
from scipy.sparse import load_npz
import scipy.sparse as sp
from Networks_Pytorch.GNNUtil.graphProcessing import makeAdjacency, compute_ontheflyAdjacency,\
    compute_ontheflyAdjacency_with_attention_layers, GenOntheflyAdjacency_NeighCandit
from Networks_Pytorch.GNNUtil.gnnUtilities import sparse_mx_to_torch_sparse_tensor
from Common.ErrorMessages import *
from Common.FunctionsUtil import *
import numpy as np
import pdb
torch.manual_seed(2017)



source_dir_adjs_default = './Code/GNN/adj/'

class NeuralNetwork(nn.Module):

    nlevel_default = 3
    nfeat_default = 8
    types_module_operations = ['convol', 'pool', 'upsam', 'gnn', 'conv_last']

    def __init__(self, size_image,
                 nlevel= nlevel_default,
                 nfeat= nfeat_default,
                 isUse_valid_convols= False):
        super(NeuralNetwork, self).__init__()
        self.size_image = size_image
        self.nlevel = nlevel
        self.nfeat = nfeat
        self.ipChannels = 1
        self.opChannels = 1
        self.dropout = 0.0

        self.isUse_valid_convols = isUse_valid_convols
        if self.isUse_valid_convols:
            self.gen_list_module_operations()
            self.gen_list_sizes_output_valid()
            self.gen_list_sizes_crop_merge()

        if self.isUse_valid_convols:
            self.size_output = self.get_size_output_valid(self.size_image)
        else:
            self.size_output = self.size_image

        self.build_model()
        self.build_comp_data()


    def get_size_input(self):
        return [self.ipChannels] + list(self.size_image)

    def get_size_output(self):
        return [self.opChannels] + list(self.size_output)

    def count_model_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_arch_desc(self):
        return NotImplemented

    def build_comp_data(self):
        return NotImplemented

    def build_model(self):
        return NotImplemented

    def load_adjacency_matrixes(self, source_dir_adjs=source_dir_adjs_default):
        return NotImplemented

    def preprocess(self, *args, **kwargs):
        pass

    @staticmethod
    def get_create_model(type_model, dict_input_args):
        print("Restarting model \'%s\' with arguments: \'%s\'" % (type_model, dict_input_args))
        if type_model == 'gnnAlterOTF':
            return Unet3DGNN_OTF(**dict_input_args)
        elif type_model == 'gnnOTF':
            del dict_input_args['source_dir_adjs']
            return Unet3DGNN_OTF(**dict_input_args)
        elif type_model == 'UnetGNN_OTF':
            return Unet3DGNN_OTF(**dict_input_args)
        elif type_model == 'Unet3DGNN':
            return Unet3DGNN(**dict_input_args)
        elif type_model == 'Unet3D':
            del dict_input_args['isGNN_with_attention_lays']
            del dict_input_args['source_dir_adjs']
            return Unet3D(**dict_input_args)
        # elif type_model == 'GNN_Skip1_GNN23':
        #     return GNN_Skip1_GNN23(**dict_input_args)
        # elif type_model == 'GNNSkip3':
        #     return GNNSkip3(**dict_input_args)
        # elif type_model == 'GNNSkip2':
        #     return GNNSkip2(**dict_input_args)
        # elif type_model == 'GNNSkip':
        #     return GNNSkip(**dict_input_args)
        # elif type_model == 'GNN3D':
        #     return GNN3D(**dict_input_args)
        else:
            return NotImplemented

    def get_descmodel_sourcecode(self, nodeGNN=None):
        import inspect
        descmodel_txt = str(self.__class__)+'\n'
        descmodel_txt += '...\n'
        descmodel_txt += inspect.getsource(self.build_model)
        descmodel_txt += '\n'
        descmodel_txt += inspect.getsource(self.forward)
        descmodel_txt += '...\n\n'
        if nodeGNN:
            descmodel_txt += str(nodeGNN.__class__)+'\n'
            descmodel_txt += '...\n'
            descmodel_txt += inspect.getsource(nodeGNN.__init__)
            descmodel_txt += '\n'
            descmodel_txt += inspect.getsource(nodeGNN.forward)
            descmodel_txt += '...\n'
        return descmodel_txt


    @staticmethod
    def make_adjacency_matrix(size_image):
        # Process adjacency matrix for GNN; assume 26 neighbourhood.
        return makeAdjacency(size_image, numNgbrs=26).cuda()

    @staticmethod
    def load_adjacency_matrix(list_filename_adjs):
        adjs = []
        for filename_adj in list_filename_adjs:
            adj_i = load_npz(filename_adj)
            adjs.append(sparse_mx_to_torch_sparse_tensor(adj_i).cuda())
        # endfor
        if len(list_filename_adjs)==1:
            return adjs[0]
        else:
            return adjs

    @staticmethod
    def load_adjacency_matrix_with_attention_layers(list_filename_adjs):
        adjs = []
        n2e_ins = []
        n2e_outs = []
        for filename_adj in list_filename_adjs:
            adj_i = load_npz(filename_adj)
            adj_i *= 27
            n2e_in_i = sp.csr_matrix((np.ones(adj_i.nnz), (np.arange(adj_i.nnz), sp.find(adj_i)[1])),
                                     shape=(adj_i.nnz, adj_i.shape[0]))
            n2e_out_i = sp.csr_matrix((np.ones(adj_i.nnz), (np.arange(adj_i.nnz), sp.find(adj_i)[0])),
                                      shape=(adj_i.nnz, adj_i.shape[0]))
            adjs.append(sparse_mx_to_torch_sparse_tensor(adj_i).cuda())
            n2e_ins.append(sparse_mx_to_torch_sparse_tensor(n2e_in_i).cuda())
            n2e_outs.append(sparse_mx_to_torch_sparse_tensor(n2e_out_i).cuda())
        # endfor
        if len(list_filename_adjs)==1:
            return adjs[0], n2e_ins[0], n2e_outs[0]
        else:
            return adjs, n2e_ins, n2e_outs




class Unet3D(NeuralNetwork):
    """ Unet+GNN as base layer """
    nlevel_default = 3
    nfeat_default = 8

    def __init__(self, size_image,
                 nlevel=nlevel_default,
                 nfeat= nfeat_default,
                 isUse_valid_convols= True):
        super(Unet3D, self).__init__(size_image,
                                     nlevel,
                                     nfeat,
                                     isUse_valid_convols= isUse_valid_convols)

    def get_arch_desc(self):
        return ['Unet3D', {'size_image': self.size_image, 'nlevel': self.nlevel, 'nfeat': self.nfeat}]

    def gen_list_module_operations(self):
        if self.nlevel == 1:
            self.list_module_opers = ['convol', 'convol', 'convol', 'convol', 'conv_last']
        elif self.nlevel <= 3:
            self.list_module_opers = (self.nlevel-1) * ['convol', 'convol', 'pool'] + \
                                                       ['convol', 'convol'] + \
                                     (self.nlevel-1) * ['upsam', 'convol', 'convol'] + \
                                                       ['conv_last']
        else: #Assume that last convolutions have padding, to avoid large reduction of image dims
            self.list_module_opers = (self.nlevel_default-1) * ['convol', 'convol', 'pool'] + \
                                                               ['convol', 'convol', 'pool'] + \
                                     (self.nlevel-self.nlevel_default-1) * ['convol_pad', 'convol_pad', 'pool'] + \
                                                               ['convol_pad', 'convol_pad'] + \
                                     (self.nlevel-self.nlevel_default-1) * ['upsam', 'convol_pad', 'convol_pad'] + \
                                                               ['upsam', 'convol', 'convol'] + \
                                     (self.nlevel_default-1) * ['upsam', 'convol', 'convol'] + \
                                                               ['conv_last']

    def get_size_output_deeplevel(self):
        ind_last_conv_deep_level = self.list_module_opers.index('upsam')-1
        size_output = self.list_sizes_output[ind_last_conv_deep_level]
        num_feats = 2**(self.nlevel-1) * self.nfeat
        return [num_feats] + list(size_output)


    def build_model(self):
        if self.isUse_valid_convols:
           padding_val = 0
        else:
           padding_val = 1

        nfeatD1 = self.nfeat
        self.convD11 = Conv3d(self.ipChannels, nfeatD1, kernel_size= 3, padding= padding_val)
        self.convD12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_val)
        self.mpoolD1 = MaxPool3d(kernel_size= 2, padding= 0)

        nfeatD2 = 2 * nfeatD1
        self.convD21 = Conv3d(nfeatD1, nfeatD2, kernel_size= 3, padding= padding_val)
        self.convD22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_val)
        self.mpoolD2 = MaxPool3d(kernel_size= 2, padding= 0)

        nfeatD3 = 2 * nfeatD2
        self.convD31 = Conv3d(nfeatD2, nfeatD3, kernel_size= 3, padding= padding_val)
        self.convD32 = Conv3d(nfeatD3, nfeatD3, kernel_size= 3, padding= padding_val)
        # self.mpoolD3 = MaxPool3d(kernel_size= 2, padding= 0)
        #
        # nfeatD4 = 2 * nfeatD3
        # self.convD41 = Conv3d(nfeatD3, nfeatD4, kernel_size= 3, padding= 1)
        # self.convD42 = Conv3d(nfeatD4, nfeatD4, kernel_size= 3, padding= 1)
        # self.mpoolD4 = MaxPool3d(kernel_size= 2, padding= 0)
        #
        # nfeatD5 = 2 * nfeatD4
        # self.convD51 = Conv3d(nfeatD4, nfeatD5, kernel_size= 3, padding= 1)
        # self.convD52 = Conv3d(nfeatD5, nfeatD5, kernel_size= 3, padding= 1)

        # self.upoolU5 = Upsample(scale_factor= 2, mode= 'nearest')
        # nfeatD45 = nfeatD4 + nfeatD5
        # #nfeatD44 = nfeatD5
        # self.convU41 = Conv3d(nfeatD45, nfeatD4, kernel_size= 3, padding= 1)
        # self.convU42 = Conv3d(nfeatD4, nfeatD4, kernel_size= 3, padding= 1)
        #
        # self.upoolU4 = Upsample(scale_factor= 2, mode= 'nearest')
        # nfeatD34 = nfeatD3 + nfeatD4
        # #nfeatD34 = nfeatD4
        # self.convU31 = Conv3d(nfeatD34, nfeatD3, kernel_size= 3, padding= padding_val)
        # self.convU32 = Conv3d(nfeatD3, nfeatD3, kernel_size= 3, padding= padding_val)

        self.upoolU3 = Upsample(scale_factor= 2, mode= 'nearest')
        nfeatD23 = nfeatD2 + nfeatD3
        #nfeatD23 = nfeatD3
        self.convU21 = Conv3d(nfeatD23, nfeatD2, kernel_size= 3, padding= padding_val)
        self.convU22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_val)

        self.upoolU2 = Upsample(scale_factor= 2, mode= 'nearest')
        nfeatD12 = nfeatD1 + nfeatD2
        #nfeatD12 = nfeatD2
        self.convU11 = Conv3d(nfeatD12, nfeatD1, kernel_size= 3, padding= padding_val)
        self.convU12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_val)

        self.classify = Conv3d(nfeatD1, self.opChannels, kernel_size= 1, padding= 0)

        self.relu = ReLU(inplace=True)
        self.activation_output = Sigmoid()

    def forward(self, input):

        x = self.relu(self.convD11(input))
        x = self.relu(self.convD12(x))
        # # if self.isUse_valid_convols:
        # #     x_skip_lev1 = self.crop_image(x, self.list_sizes_crop_merge[0])
        # # else:
        # #     x_skip_lev1 = x
        x_skip_lev1 = self.crop_image(x, self.list_sizes_crop_merge[0])
        x = self.mpoolD1(x)

        x = self.relu(self.convD21(x))
        x = self.relu(self.convD22(x))
        # # if self.isUse_valid_convols:
        # #     x_skip_lev2 = self.crop_image(x, self.list_sizes_crop_merge[1])
        # # else:
        # #     x_skip_lev2 = x
        x_skip_lev2 = self.crop_image(x, self.list_sizes_crop_merge[1])
        x = self.mpoolD2(x)

        x = self.relu(self.convD31(x))
        x = self.relu(self.convD32(x))
        # # if self.isUse_valid_convols:
        # #     x_skip_lev3 = self.crop_image(x, self.list_sizes_crop_merge[2])
        # # else:
        # #     x_skip_lev3 = x
        # x_skip_lev3 = self.crop_image(x, self.list_sizes_crop_merge[2])
        # x = self.mpoolD3(x)
        #
        # x = self.relu(self.convD41(x))
        # x = self.relu(self.convD42(x))
        # # if self.isUse_valid_convols:
        # #     x_skip_lev4 = self.crop_image(x, self.list_sizes_crop_merge[3])
        # # else:
        # #     x_skip_lev4 = x
        # x_skip_lev4 = self.crop_image(x, self.list_sizes_crop_merge[3])
        # x = self.mpoolD4(x)

        # x = self.relu(self.convD51(x))
        # x = self.relu(self.convD52(x))

        # x = self.upoolU5(x)
        # x = torch.cat([x, x_skip_lev4], dim=1)
        # x = self.relu(self.convU41(x))
        # x = self.relu(self.convU42(x))
        #
        # x = self.upoolU4(x)
        # x = torch.cat([x, x_skip_lev3], dim=1)
        # x = self.relu(self.convU31(x))
        # x = self.relu(self.convU32(x))

        x = self.upoolU3(x)
        x = torch.cat([x, x_skip_lev2], dim=1)
        x = self.relu(self.convU21(x))
        x = self.relu(self.convU22(x))

        x = self.upoolU2(x)
        x = torch.cat([x, x_skip_lev1], dim=1)
        x = self.relu(self.convU11(x))
        x = self.relu(self.convU12(x))

        output = self.classify(x)
        output = self.activation_output(output)

        return output




class Unet3DGNN(NeuralNetwork):
    """ Unet with GNN as base layer """
    nlevel_default = 3
    nfeat_default = 8

    def __init__(self, size_image,
                 nlevel=nlevel_default,
                 nfeat=nfeat_default,
                 isUse_valid_convols= True,
                 isGNN_with_attention_lays= False,
                 source_dir_adjs= source_dir_adjs_default):
        self.isGNN_with_attention_lays = isGNN_with_attention_lays
        self.source_dir_adjs = source_dir_adjs
        super(Unet3DGNN, self).__init__(size_image,
                                        nlevel,
                                        nfeat,
                                        isUse_valid_convols=isUse_valid_convols)

    def get_arch_desc(self):
        return ['Unet3DGNN', {'size_image': self.size_image, 'nlevel': self.nlevel, 'nfeat': self.nfeat}]

    def gen_list_module_operations(self):
        if self.nlevel == 1:
            self.list_module_opers = ['convol', 'convol', 'gnn', 'conv_last']
        else:
            self.list_module_opers = (self.nlevel - 1) * ['convol', 'convol', 'pool'] + \
                                     ['gnn'] + \
                                     (self.nlevel - 1) * ['upsam', 'convol', 'convol'] + \
                                     ['conv_last']

    def get_size_output_deeplevel(self):
        ind_last_conv_deep_level = self.list_module_opers.index('upsam')-1
        size_output = self.list_sizes_output[ind_last_conv_deep_level]
        num_feats = 2**(self.nlevel-2) * self.nfeat
        return [num_feats] + list(size_output)


    def build_comp_data(self):
        if self.isUse_valid_convols:
            input_shape_gnnadj = self.list_sizes_output[self.list_module_opers.index('gnn')-1]
        else:
            input_shape_gnnadj = tuple(s//2**(self.nlevel-1) for s in self.size_image)

        filename_adjs = 'adj_z%i_x%i_y%i.npz'%(input_shape_gnnadj)
        list_filename_adjs = [filename_adjs]
        list_filename_adjs = [joinpathnames(self.source_dir_adjs, filename) for filename in list_filename_adjs]

        print("Loading adjacency matrix(es): \'%s\'..." % ('\' \''.join(list_filename_adjs)))
        if self.isGNN_with_attention_lays:
            print("Loading matrixes for attention layers in GNN module...")
            (self.adj, self.n2e_in, self.n2e_out) = self.load_adjacency_matrix_with_attention_layers(list_filename_adjs)

            self.nGNN._preprocess((self.adj, self.n2e_in, self.n2e_out))
        else:
            self.adj = self.load_adjacency_matrix(list_filename_adjs)

            self.nGNN._preprocess(self.adj)


    def get_descmodel_sourcecode(self):
        return super(Unet3DGNN, self).get_descmodel_sourcecode(self.nGNN)

    def modify_state_dict_restartGNN_fromUnet(self, in_state_dict):
        # delete params not used from Unet
        del in_state_dict['convD31.weight']
        del in_state_dict['convD31.bias']
        del in_state_dict['convD32.weight']
        del in_state_dict['convD32.bias']
        # add params corresponding to GNN
        state_dict_nodeGNN = self.nGNN.alloc_state_dict_vars(basename_module='nGNN', reset_parameters=True)
        in_state_dict.update(state_dict_nodeGNN)


    def build_model(self):
        if self.isUse_valid_convols:
           padding_val = 0
        else:
           padding_val = 1

        nfeatD1 = self.nfeat
        self.convD11 = Conv3d(self.ipChannels, nfeatD1, kernel_size= 3, padding= padding_val)
        self.convD12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_val)
        self.mpoolD1 = MaxPool3d(kernel_size= 2, padding= 0)

        nfeatD2 = 2 * nfeatD1
        self.convD21 = Conv3d(nfeatD1, nfeatD2, kernel_size= 3, padding= padding_val)
        self.convD22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_val)
        self.mpoolD2 = MaxPool3d(kernel_size= 2, padding= 0)

        nfeatD3 = 2 * nfeatD2
        if self.isGNN_with_attention_lays:
            self.nGNN = NodeGNNwithAttentionLayers(nfeatD2, nfeatD3, nfeatD2, self.dropout)
        else:
            self.nGNN = NodeGNN(nfeatD2, nfeatD3, nfeatD2, self.dropout)

        self.upoolU3 = Upsample(scale_factor= 2, mode= 'nearest')
        nfeatD23 = nfeatD2 + nfeatD2
        #nfeatD23 = nfeatD2
        self.convU21 = Conv3d(nfeatD23, nfeatD2, kernel_size= 3, padding= padding_val)
        self.convU22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_val)

        self.upoolU2 = Upsample(scale_factor= 2, mode= 'nearest')
        nfeatD12 = nfeatD1 + nfeatD2
        #nfeatD12 = nfeatD2
        self.convU11 = Conv3d(nfeatD12, nfeatD1, kernel_size= 3, padding= padding_val)
        self.convU12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_val)

        self.classify = Conv3d(nfeatD1, self.opChannels, kernel_size=1, padding=0)

        self.relu = ReLU(inplace=True)
        self.activation_output = Sigmoid()

    def forward(self, input):

        x = self.relu(self.convD11(input))
        x = self.relu(self.convD12(x))
        # if self.isUse_valid_convols:
        #     x_skip_lev1 = self.crop_image(x, self.list_sizes_crop_merge[0])
        # else:
        #     x_skip_lev1 = x
        x_skip_lev1 = self.crop_image(x, self.list_sizes_crop_merge[0])
        x = self.mpoolD1(x)

        x = self.relu(self.convD21(x))
        x = self.relu(self.convD22(x))
        # if self.isUse_valid_convols:
        #     x_skip_lev2 = self.crop_image(x, self.list_sizes_crop_merge[1])
        # else:
        #     x_skip_lev2 = x
        x_skip_lev2 = self.crop_image(x, self.list_sizes_crop_merge[1])
        x = self.mpoolD2(x)

        # Node GNN here # see gnnModules.py for further details
        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
        x = self.nGNN(x)
        x = x.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])
        torch.cuda.empty_cache()

        x = self.upoolU3(x)
        x = torch.cat([x, x_skip_lev2], dim=1)
        x = self.relu(self.convU21(x))
        x = self.relu(self.convU22(x))

        x = self.upoolU2(x)
        x = torch.cat([x, x_skip_lev1], dim=1)
        x = self.relu(self.convU11(x))
        x = self.relu(self.convU12(x))

        output = self.classify(x)
        output = self.activation_output(output)

        return output




class Unet3DGNN_OTF(NeuralNetwork):
    """ Unet+GNN as base layer with on-the-fly adjacency matrix computation.
    Alternating i) on-the-fly adjacency matrix computation, and ii) precomputed adjacency
    Unet layers have 2 CNNs per layer.
    """
    nlevel_default = 3
    nfeat_default = 8
    freq_epoch_adj_onthefly_default = 1
    dist_neigh_max_onthefly_adj = 5


    def __init__(self, size_image,
                 nlevel=nlevel_default,
                 nfeat=nfeat_default,
                 isUse_valid_convols= True,
                 freq_epoch_adj_onthefly=freq_epoch_adj_onthefly_default,
                 is_limit_neighs_onthefly_adj= True,
                 isGNN_with_attention_lays= False,
                 source_dir_adjs = source_dir_adjs_default):
        self.freq_epoch_adj_onthefly = freq_epoch_adj_onthefly
        if self.freq_epoch_adj_onthefly>1:
            print("Alternate between i) \'On-the-fly\' and ii) \'Precomputed\' adjacency matrix, every \'%s\' epochs..."
                  %(self.freq_epoch_adj_onthefly))
            self.is_alter_onthefly_precalc_adj = True
            self.is_onthefly_adjacency_inepoch = False
        else:
            print("Every epoch with \'On-the-fly\' computation of adjacency matrix...")
            self.is_alter_onthefly_precalc_adj = False
            self.is_onthefly_adjacency_inepoch = True

        self.is_limit_neighs_onthefly_adj = is_limit_neighs_onthefly_adj
        self.isGNN_with_attention_lays = isGNN_with_attention_lays
        self.source_dir_adjs = source_dir_adjs
        super(Unet3DGNN_OTF, self).__init__(size_image,
                                            nlevel,
                                            nfeat,
                                            isUse_valid_convols=isUse_valid_convols)

    def get_arch_desc(self):
        return ['UnetGNN_OTF', {'size_image': self.size_image, 'nlevel': self.nlevel, 'nfeat': self.nfeat}]

    def gen_list_module_operations(self):
        if self.nlevel == 1:
            self.list_module_opers = ['convol', 'convol', 'gnn', 'conv_last']
        else:
            self.list_module_opers = (self.nlevel-1) * ['convol', 'convol', 'pool'] + \
                                     ['gnn'] + \
                                     (self.nlevel-1) * ['upsam', 'convol', 'convol'] + \
                                     ['conv_last']

    def get_size_output_deeplevel(self):
        ind_last_conv_deep_level = self.list_module_opers.index('upsam')-1
        size_output = self.list_sizes_output[ind_last_conv_deep_level]
        num_feats = 2**(self.nlevel-2) * self.nfeat
        return [num_feats] + list(size_output)


    def build_comp_data(self):
        if self.isUse_valid_convols:
            input_shape_gnnadj = self.list_sizes_output[self.list_module_opers.index('gnn')-1]
        else:
            input_shape_gnnadj = tuple(s//2**(self.nlevel-1) for s in self.size_image)

        filename_adjs = 'adj_z%i_x%i_y%i.npz'%(input_shape_gnnadj)
        list_filename_adjs = [filename_adjs]
        list_filename_adjs = [joinpathnames(self.source_dir_adjs, filename) for filename in list_filename_adjs]

        if self.is_alter_onthefly_precalc_adj:
            print("Loading adjacency matrix(es): \'%s\'..." %('\' \''.join(list_filename_adjs)))
            if self.isGNN_with_attention_lays:
                print("Loading matrixes for attention layers in GNN module...")
                (self.adj_sto, self.n2e_in_sto, self.n2e_out_sto) = self.load_adjacency_matrix_with_attention_layers(list_filename_adjs)
                self.list_matrix_adjs_gnn_sto = (self.adj_sto, self.n2e_in_sto, self.n2e_out_sto)
            else:
                self.adj_sto = self.load_adjacency_matrix(list_filename_adjs)
                self.list_matrix_adjs_gnn_sto = (self.adj_sto)

        if self.is_limit_neighs_onthefly_adj:
            print("Limiting neighbourhood of candidates nodes in computation of"
                  "adjacency matrix to \'%s\' levels..." %(self.dist_neigh_max_onthefly_adj))
            self.genOntheflyAdjacency = GenOntheflyAdjacency_NeighCandit(input_shape_gnnadj,
                                                                         dist_neigh_max= self.dist_neigh_max_onthefly_adj)
            if self.isGNN_with_attention_lays:
                print("Setting to load matrixes for attention layers in GNN module...")
                self.funCalc_onthefly_adjacency = self.genOntheflyAdjacency.compute_with_attention_layers
            else:
                self.funCalc_onthefly_adjacency = self.genOntheflyAdjacency.compute
        else:
            if self.isGNN_with_attention_lays:
                print("Setting to load matrixes for attention layers in GNN module...")
                self.funCalc_onthefly_adjacency = compute_ontheflyAdjacency_with_attention_layers
            else:
                self.funCalc_onthefly_adjacency = compute_ontheflyAdjacency


    def get_descmodel_sourcecode(self):
        return super(Unet3DGNN_OTF, self).get_descmodel_sourcecode(self.nGNN)

    def modify_state_dict_restartGNN_fromUnet(self, in_state_dict):
        # delete params not used from Unet
        del in_state_dict['convD31.weight']
        del in_state_dict['convD31.bias']
        del in_state_dict['convD32.weight']
        del in_state_dict['convD32.bias']
        # add params corresponding to GNN
        state_dict_nodeGNN = self.nGNN.alloc_state_dict_vars(basename_module='nGNN', reset_parameters=True)
        in_state_dict.update(state_dict_nodeGNN)


    def preprocess(self, epoch_count):
        if self.is_alter_onthefly_precalc_adj:
            self.is_onthefly_adjacency_inepoch = (epoch_count+1) % self.freq_epoch_adj_onthefly == 0
            if self.is_onthefly_adjacency_inepoch:
                print "Using on-the-fly computed adjacency matrix in this epoch..."
            else:
                print "Using precomputed and stored adjacency matrix in this epoch..."
                # Depending on the GNN module: list_matrix_adjs_gnn_sto := i) adj_sto, ii) adj_sto, n2e_in, n2e_out
                self.nGNN._preprocess(self.list_matrix_adjs_gnn_sto)
        else:
            print "Using every epoch on-the-fly computed adjacency matrix in this epoch..."


    def build_model(self):
        if self.isUse_valid_convols:
           padding_val = 0
        else:
           padding_val = 1

        nfeatD1 = self.nfeat
        self.convD11 = Conv3d(self.ipChannels, nfeatD1, kernel_size= 3, padding= padding_val)
        self.convD12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_val)
        self.mpoolD1 = MaxPool3d(kernel_size= 2, padding= 0)

        nfeatD2 = 2 * nfeatD1
        self.convD21 = Conv3d(nfeatD1, nfeatD2, kernel_size= 3, padding= padding_val)
        self.convD22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_val)
        self.mpoolD2 = MaxPool3d(kernel_size= 2, padding= 0)

        nfeatD3 = 2 * nfeatD2
        if self.isGNN_with_attention_lays:
            self.nGNN = NodeGNNwithAttentionLayers(nfeatD2, nfeatD3, nfeatD2, self.dropout)
        else:
            self.nGNN = NodeGNN(nfeatD2, nfeatD3, nfeatD2, self.dropout)

        self.upoolU3 = Upsample(scale_factor= 2, mode= 'nearest')
        nfeatD23 = nfeatD2 + nfeatD2
        #nfeatD23 = nfeatD2
        self.convU21 = Conv3d(nfeatD23, nfeatD2, kernel_size= 3, padding= padding_val)
        self.convU22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_val)

        self.upoolU2 = Upsample(scale_factor= 2, mode= 'nearest')
        nfeatD12 = nfeatD1 + nfeatD2
        #nfeatD12 = nfeatD2
        self.convU11 = Conv3d(nfeatD12, nfeatD1, kernel_size= 3, padding= padding_val)
        self.convU12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_val)

        self.classify = Conv3d(nfeatD1, self.opChannels, kernel_size= 1, padding= 0)

        self.relu = ReLU(inplace=True)
        self.activation_output = Sigmoid()

    def forward(self, input):

        x = self.relu(self.convD11(input))
        x = self.relu(self.convD12(x))
        # if self.isUse_valid_convols:
        #     x_skip_lev1 = self.crop_image(x, self.list_sizes_crop_merge[0])
        # else:
        #     x_skip_lev1 = x
        x_skip_lev1 = self.crop_image(x, self.list_sizes_crop_merge[0])
        x = self.mpoolD1(x)

        x = self.relu(self.convD21(x))
        x = self.relu(self.convD22(x))
        # if self.isUse_valid_convols:
        #     x_skip_lev2 = self.crop_image(x, self.list_sizes_crop_merge[1])
        # else:
        #     x_skip_lev2 = x
        x_skip_lev2 = self.crop_image(x, self.list_sizes_crop_merge[1])
        x = self.mpoolD2(x)

        # Node GNN here # see gnnModules.py for further details
        xSh = x.shape
        # pdb.set_trace()
        x = x.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
        if self.is_onthefly_adjacency_inepoch:
            list_matrix_adjs_gnn_otf = self.funCalc_onthefly_adjacency(x.data.cpu().numpy(), numNgbrs=10)
            # Depending on the GNN module: list_matrix_adjs_gnn := i) adj, ii) adj, n2e_in, n2e_out
            self.nGNN._preprocess(list_matrix_adjs_gnn_otf)
            # adj = otfAdjTorch(x.cpu())
            # x = x.attach()
        x = self.nGNN(x)
        x = x.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])
        torch.cuda.empty_cache()

        x = self.upoolU3(x)
        x = torch.cat([x, x_skip_lev2], dim=1)
        x = self.relu(self.convU21(x))
        x = self.relu(self.convU22(x))

        x = self.upoolU2(x)
        x = torch.cat([x, x_skip_lev1], dim=1)
        x = self.relu(self.convU11(x))
        x = self.relu(self.convU12(x))

        output = self.classify(x)
        output = self.activation_output(output)

        return output




# all available metrics
def DICTAVAILMODELSGNNS(option, size_image,
                        nlevel=Unet3D.nlevel_default,
                        nfeat=Unet3D.nfeat_default,
                        isUse_valid_convols= True,
                        isGNN_with_attention_lays= False,
                        source_dir_adjs=source_dir_adjs_default):
    list_models_avail = ['UnetGNN_OTF', 'UnetGNN', 'Unet']

    if not isUse_valid_convols:
        message = 'Neet to set use of Valid Convolutions. Models are reimplemented with this option'
        CatchErrorException(message)

    print("Building model \'%s\' with nfeats \'%s\' in first layer" %(option, nfeat))
    if (option == 'Unet'):
        return Unet3D(size_image,
                      nlevel=nlevel,
                      nfeat=nfeat,
                      isUse_valid_convols=isUse_valid_convols)
    elif (option == 'UnetGNN'):
        return Unet3DGNN(size_image,
                         nlevel=nlevel,
                         nfeat=nfeat,
                         isUse_valid_convols=isUse_valid_convols,
                         isGNN_with_attention_lays=isGNN_with_attention_lays,
                         source_dir_adjs=source_dir_adjs)
    elif (option == 'UnetGNN_OTF'):
        return Unet3DGNN_OTF(size_image,
                             nlevel=nlevel,
                             nfeat=nfeat,
                             isUse_valid_convols=isUse_valid_convols,
                             isGNN_with_attention_lays=isGNN_with_attention_lays,
                             freq_epoch_adj_onthefly=1,
                             is_limit_neighs_onthefly_adj=True,
                             source_dir_adjs=source_dir_adjs)
    else:
        message = 'Model chosen not found. Models available: (%s)' %(', '.join(list_models_avail))
        CatchErrorException(message)
        return NotImplemented
