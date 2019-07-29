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
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn.functional import interpolate
from Networks_Pytorch.GNNUtil.gnnModules import GraphConvolution, GraphConvolutionFirstOrder, SparseMM, LayerNorm
from Networks_Pytorch.NetworksGNNs import NeuralNetwork, source_dir_adjs_default
from Common.ErrorMessages import *
import numpy as np
import pdb



class ModelRestartPlugin(NeuralNetwork):
    """ Need to copy the model definition from the readme.txt files accompaying the models dir
    """
    nfeat_default = 8
    freq_epoch_adj_onthefly = 5

    def __init__(self, size_image,
                 nfeat=nfeat_default,
                 source_dir_adjs=source_dir_adjs_default):
        super(ModelRestartPlugin, self).__init__(size_image, nfeat)
        self.dropout = 0.0
        self.build_model()
        (self.adj1, self.adj2, self.adj3) = self.load_series_adjacency_matrixes(size_image,
                                                                                res0=2, numRes=3,
                                                                                source_dir=source_dir_adjs)

    @staticmethod
    def get_create_model(type_model, dict_input_args):
        return ModelRestartPlugin(**dict_input_args)

    def build_model(self):
        nfeatD1 = self.nfeat

        self.mpool = MaxPool3d(kernel_size=2, padding=0)
        self.convD11 = Conv3d(self.ipChannels, nfeatD1, kernel_size=3, padding=1)
        self.convD12 = Conv3d(nfeatD1, nfeatD1, kernel_size=3, padding=1)

        nfeatD2 = 2 * nfeatD1
        self.convD21 = Conv3d(nfeatD1, nfeatD2, kernel_size=3, padding=1)
        self.convD22 = Conv3d(nfeatD2, nfeatD2, kernel_size=3, padding=1)

        self.nGNN1 = nodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)
        self.nGNN2 = nodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)
        self.nGNN3 = nodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)

        self.upool = Upsample(scale_factor=2, mode='nearest')
        self.convU21 = Conv3d(nfeatD2 * 3, nfeatD2, kernel_size=3, padding=1)
        self.convU22 = Conv3d(nfeatD2, nfeatD2, kernel_size=3, padding=1)

        self.convU11 = Conv3d(nfeatD2, nfeatD1, kernel_size=3, padding=1)
        self.convU12 = Conv3d(nfeatD1, nfeatD1, kernel_size=3, padding=1)

        self.classify = Conv3d(nfeatD1, self.opChannels, kernel_size=1, padding=0)

        self.relu = ReLU(inplace=True)
        self.activation_output = Sigmoid()

    def forward(self, input):
        # pdb.set_trace()
        x = self.relu(self.convD11(input))
        x = self.relu(self.convD12(x))
        x = self.mpool(x)

        x = self.relu(self.convD21(x))
        x = self.relu(self.convD22(x))
        x = self.mpool(x)
        x2 = x
        # Node GNN here # see gnnModules.py for further details

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
        x = self.nGNN1(x, self.adj1)
        x = x.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])
        x_skip_1 = x

        # Node GNN here # see gnnModules.py for further details
        x = self.mpool(x2)
        x3 = x

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
        x = self.nGNN2(x, self.adj2)
        x = x.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])
        x_skip_2 = x

        # Node GNN here # see gnnModules.py for further details
        x = self.mpool(x3)

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
        x = self.nGNN2(x, self.adj3)
        x = x.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])

        x = self.upool(x)
        x = torch.cat([x, x_skip_2], dim=1)
        x = self.upool(x)
        x = torch.cat([x, x_skip_1], dim=1)

        x = self.upool(x)

        x = self.relu(self.convU21(x))
        x = self.relu(self.convU22(x))
        x = self.upool(x)

        x = self.relu(self.convU11(x))
        x = self.relu(self.convU12(x))

        output = self.classify(x)
        output = self.activation_output(output)

        return output


class nodeGNN(nn.Module):
    def __init__(self, nfeat, nhid, opFeat, dropout):
        super(nodeGNN, self).__init__()

        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        #self.fc3 = nn.Linear(nhid, nhid)
        #self.fc4 = nn.Linear(nhid, nhid)

        #self.gc1 = GraphConvolutionFirstOrder(nfeat, nhid)
        self.gc2 = GraphConvolutionFirstOrder(nhid, nhid)
        #self.gc3 = GraphConvolutionFirstOrder(nhid, nhid)
        #self.gc4 = GraphConvolutionFirstOrder(nhid, nhid)
        #self.gc5 = GraphConvolutionFirstOrder(nhid, nhid)
        #self.gc6 = GraphConvolutionFirstOrder(nhid, nhid)
        #self.gc7 = GraphConvolutionFirstOrder(nhid, nhid)
        self.gc8 = GraphConvolutionFirstOrder(nhid, opFeat)

        self.ln1 = LayerNorm(nhid)
        #self.ln2 = LayerNorm(nhid)
        #self.dropout = dropout

    def forward(self, x, adj):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.ln1(x)

        x = F.relu(self.gc2(x, adj))

        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc2(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc3(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc4(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc5(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc6(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.gc7(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc8(x, adj)

        return x
