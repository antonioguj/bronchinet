
from torch.nn import Conv3d, MaxPool3d, Upsample, ReLU, Sigmoid
#from torch.nn.functional import interpolate
import torch
from models.pytorch.gnn_util import NodeGNN
from models.pytorch.gnn_util import compute_ontheflyAdjacency
from models.pytorch.gnn_util.NetworksGNNs import NeuralNetwork, source_dir_adjs_default
torch.manual_seed(2017)

from common.functionutil import join_path_names
from common.exceptionmanager import catch_error_exception


class Unet3DGNN_AllskipGNNs(NeuralNetwork):
    """ Unet with GNN as base layer. All skip connections in Unet replaced by GNNs """
    nlevel_default = 3
    nfeat_default = 8

    def __init__(self, size_image,
                 nlevel=nlevel_default,
                 nfeat=nfeat_default,
                 isUse_valid_convs= True,
                 source_dir_adjs=source_dir_adjs_default):
        super(Unet3DGNN_AllskipGNNs, self).__init__(size_image,
                                                    nlevel,
                                                    nfeat,
                                                    isUse_valid_convs=isUse_valid_convs)
        if isUse_valid_convs:
            input_shape_gnnadj1 = self.list_sizes_output[self.list_module_opers.index('gnn')-5]
            input_shape_gnnadj2 = self.list_sizes_output[self.list_module_opers.index('gnn')-2]
            input_shape_gnnadj3 = self.list_sizes_output[self.list_module_opers.index('gnn')-1]
        else:
            input_shape_gnnadj = tuple(s//2**(self.nlevel-1) for s in size_image)
        filename_adjs_1 = 'adj_z%i_x%i_y%i.npz'%(input_shape_gnnadj1)
        filename_adjs_2 = 'adj_z%i_x%i_y%i.npz'%(input_shape_gnnadj2)
        filename_adjs_3 = 'adj_z%i_x%i_y%i.npz'%(input_shape_gnnadj3)
        list_filename_adjs = [filename_adjs_1, filename_adjs_2, filename_adjs_3]
        list_filename_adjs = [join_path_names(source_dir_adjs, filename) for filename in list_filename_adjs]

        print "Loading adjacency matrix(es): \'%s\'..." % ('\' \''.join(list_filename_adjs))
        self.adj1, self.adj2, self.adj3 = self.load_adjacency_matrix(list_filename_adjs)

    def get_arch_desc(self):
        return ['Unet3DGNN', {'size_image': self.size_image,
                              'nlevel': self.nlevel,
                              'nfeat': self.nfeat}]

    def gen_list_module_operations(self):
        if self.nlevel == 1:
            self.list_module_opers = ['convol', 'convol', 'gnn', 'conv_last']
        else:
            self.list_module_opers = (self.nlevel - 1) * ['convol', 'convol', 'pool'] + \
                                     ['gnn'] + \
                                     (self.nlevel - 1) * ['upsam', 'convol', 'convol'] + \
                                     ['conv_last']

    def build_model(self):
        if self.isUse_valid_convs:
           padding_convs = 0
        else:
           padding_convs = 1

        nfeatD1 = self.nfeat
        self.convD11 = Conv3d(self.ipChannels, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convD12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.mpoolD1 = MaxPool3d(kernel_size= 2, padding= 0)

        nfeatD2 = 2 * nfeatD1
        self.convD21 = Conv3d(nfeatD1, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convD22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.mpoolD2 = MaxPool3d(kernel_size= 2, padding= 0)

        nfeatD3 = 2 * nfeatD2
        self.nGNN1 = NodeGNN(nfeatD1, nfeatD1, nfeatD1, self.dropout)
        self.nGNN2 = NodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)
        self.nGNN3 = NodeGNN(nfeatD2, nfeatD3, nfeatD3, self.dropout)

        self.upoolU3 = Upsample(scale_factor= 2, mode= 'nearest')
        nfeatD23 = nfeatD2 + nfeatD3
        self.convU21 = Conv3d(nfeatD23, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convU22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)

        self.upoolU2 = Upsample(scale_factor= 2, mode= 'nearest')
        nfeatD12 = nfeatD1 + nfeatD2
        self.convU11 = Conv3d(nfeatD12, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convU12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)

        self.classify = Conv3d(nfeatD1, self.opChannels, kernel_size=1, padding=0)

        self.relu = ReLU(inplace=True)
        self.activation_output = Sigmoid()

    def forward(self, input):

        x = self.relu(self.convD11(input))
        x = self.relu(self.convD12(x))
        x_gnn1 = x
        x = self.mpoolD1(x)

        x = self.relu(self.convD21(x))
        x = self.relu(self.convD22(x))
        x_gnn2 = x
        x = self.mpoolD2(x)

        # Node GNN here # see gnn.py for further details
        xSh = x_gnn1.shape
        x_gnn1 = x_gnn1.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
        x_gnn1 = self.nGNN1(x_gnn1, self.adj1)
        x_gnn1 = x_gnn1.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])

        xSh = x_gnn2.shape
        x_gnn2 = x_gnn2.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
        x_gnn2 = self.nGNN2(x_gnn2, self.adj2)
        x_gnn2 = x_gnn2.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
        x = self.nGNN3(x, self.adj3)
        x = x.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])

        # if self.isUse_valid_convs:
        #     x_skip_lev1 = self.crop_image(x_gnn1, self.list_sizes_crop_merge[0])
        # else:
        #     x_skip_lev1 = x_gnn1
        x_skip_lev1 = self.crop_image(x_gnn1, self.list_sizes_crop_merge[0])
        # if self.isUse_valid_convs:
        #     x_skip_lev2 = self.crop_image(x_gnn2, self.list_sizes_crop_merge[1])
        # else:
        #     x_skip_lev2 = x_gnn2
        x_skip_lev2 = self.crop_image(x_gnn2, self.list_sizes_crop_merge[1])

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




class gnnAlterOTF_AllskipGNNs(NeuralNetwork):
    """ Unet+GNN as base layer, alternating i) on-the-fly adjacency matrix computation, and ii) precomputed adjacency
    Unet layers have 2 CNNs per layer.
    """
    nlevel_default = 3
    nfeat_default = 8
    freq_epoch_adj_onthefly = 4

    def __init__(self, size_image,
                 nlevel=nlevel_default,
                 nfeat=nfeat_default,
                 isUse_valid_convs= True,
                 is_limit_neighs_otfAdj= True,
                 source_dir_adjs= source_dir_adjs_default):
        super(gnnAlterOTF_AllskipGNNs, self).__init__(size_image,
                                                      nlevel,
                                                      nfeat,
                                                      isUse_valid_convs=isUse_valid_convs)
        if isUse_valid_convs:
            input_shape_gnnadj1 = self.list_sizes_output[self.list_module_opers.index('gnn')-5]
            input_shape_gnnadj2 = self.list_sizes_output[self.list_module_opers.index('gnn')-2]
            input_shape_gnnadj3 = self.list_sizes_output[self.list_module_opers.index('gnn')-1]
        else:
            input_shape_gnnadj = tuple(s//2**(self.nlevel-1) for s in size_image)
        filename_adjs_1 = 'adj_z%i_x%i_y%i.npz'%(input_shape_gnnadj1)
        filename_adjs_2 = 'adj_z%i_x%i_y%i.npz'%(input_shape_gnnadj2)
        filename_adjs_3 = 'adj_z%i_x%i_y%i.npz'%(input_shape_gnnadj3)
        list_filename_adjs = [filename_adjs_1, filename_adjs_2, filename_adjs_3]
        list_filename_adjs = [join_path_names(source_dir_adjs, filename) for filename in list_filename_adjs]

        print "Loading adjacency matrix(es): \'%s\'..." %('\' \''.join(list_filename_adjs))
        self.adj_sto1, self.adj_sto2, self.adj_sto3 = self.load_adjacency_matrix(list_filename_adjs)

        if is_limit_neighs_otfAdj:
            self.genOtfAdj_NeighCandit1 = GenerateOntheflyAdjacency_NeighCandit(input_shape_gnnadj1, dist_neigh_max= 5)
            self.funCalc_otfAdj1 = self.genOtfAdj_NeighCandit1.compute

            self.genOtfAdj_NeighCandit2 = GenerateOntheflyAdjacency_NeighCandit(input_shape_gnnadj2, dist_neigh_max= 5)
            self.funCalc_otfAdj2 = self.genOtfAdj_NeighCandit2.compute

            self.genOtfAdj_NeighCandit3 = GenerateOntheflyAdjacency_NeighCandit(input_shape_gnnadj3, dist_neigh_max= 5)
            self.funCalc_otfAdj3 = self.genOtfAdj_NeighCandit3.compute
        else:
            self.funCalc_otfAdj = compute_ontheflyAdjacency

    def get_arch_desc(self):
        return ['gnnAlterOTF', {'size_image': self.size_image,
                                'nlevel': self.nlevel,
                                'nfeat': self.nfeat}]

    def gen_list_module_operations(self):
        if self.nlevel == 1:
            self.list_module_opers = ['convol', 'convol', 'gnn', 'conv_last']
        else:
            self.list_module_opers = (self.nlevel-1) * ['convol', 'convol', 'pool'] + \
                                     ['gnn'] + \
                                     (self.nlevel-1) * ['upsam', 'convol', 'convol'] + \
                                     ['conv_last']

    def preprocess(self, epoch_count):
        self.is_adj_onthefly_inepoch = (epoch_count+1) % self.freq_epoch_adj_onthefly == 0
        if self.is_adj_onthefly_inepoch:
            print "Using on-the-fly computed adjacency matrix in this epoch..."
        else:
            print "Using precomputed and stored adjacency matrix in this epoch..."

    def build_model(self):
        if self.isUse_valid_convs:
           padding_convs = 0
        else:
           padding_convs = 1

        nfeatD1 = self.nfeat
        self.convD11 = Conv3d(self.ipChannels, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convD12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.mpoolD1 = MaxPool3d(kernel_size= 2, padding= 0)

        nfeatD2 = 2 * nfeatD1
        self.convD21 = Conv3d(nfeatD1, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convD22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.mpoolD2 = MaxPool3d(kernel_size= 2, padding= 0)

        nfeatD3 = 2 * nfeatD2
        self.nGNN1 = NodeGNN(nfeatD1, nfeatD1, nfeatD1, self.dropout)
        self.nGNN2 = NodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)
        self.nGNN3 = NodeGNN(nfeatD2, nfeatD3, nfeatD3, self.dropout)

        self.upoolU3 = Upsample(scale_factor= 2, mode= 'nearest')
        nfeatD23 = nfeatD2 + nfeatD3
        #nfeatD23 = nfeatD3
        self.convU21 = Conv3d(nfeatD23, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convU22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)

        self.upoolU2 = Upsample(scale_factor= 2, mode= 'nearest')
        nfeatD12 = nfeatD1 + nfeatD2
        #nfeatD12 = nfeatD2
        self.convU11 = Conv3d(nfeatD12, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convU12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)

        self.classify = Conv3d(nfeatD1, self.opChannels, kernel_size= 1, padding= 0)

        self.relu = ReLU(inplace=True)
        self.activation_output = Sigmoid()

    def forward(self, input):

        x = self.relu(self.convD11(input))
        x = self.relu(self.convD12(x))
        x_gnn1 = x
        x = self.mpoolD1(x)

        x = self.relu(self.convD21(x))
        x = self.relu(self.convD22(x))
        x_gnn2 = x
        x = self.mpoolD2(x)

        if self.is_adj_onthefly_inepoch:
            # Node GNN here # see gnn.py for further details
            xSh = x_gnn1.shape
            # pdb.set_trace()
            x_gnn1 = x_gnn1.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
            adj_otf = self.funCalc_otfAdj1(x_gnn1.data.cpu().numpy())
            x_gnn1 = self.nGNN1(x_gnn1, adj_otf)
            x_gnn1 = x_gnn1.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])

            xSh = x_gnn2.shape
            # pdb.set_trace()
            x_gnn2 = x_gnn2.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
            adj_otf = self.funCalc_otfAdj2(x_gnn2.data.cpu().numpy())
            x_gnn2 = self.nGNN2(x_gnn2, adj_otf)
            x_gnn2 = x_gnn2.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])

            xSh = x.shape
            # pdb.set_trace()
            x = x.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
            adj_otf = self.funCalc_otfAdj2(x.data.cpu().numpy())
            x = self.nGNN2(x, adj_otf)
            x = x.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])
        else:
            # Node GNN here # see gnn.py for further details
            xSh = x_gnn1.shape
            x_gnn1 = x_gnn1.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
            x_gnn1 = self.nGNN1(x_gnn1, self.adj_sto1)
            x_gnn1 = x_gnn1.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])

            xSh = x_gnn2.shape
            x_gnn2 = x_gnn2.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
            x_gnn2 = self.nGNN2(x_gnn2, self.adj_sto2)
            x_gnn2 = x_gnn2.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])

            xSh = x.shape
            x = x.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
            x = self.nGNN3(x, self.adj_sto3)
            x = x.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])

        # if self.isUse_valid_convs:
        #     x_skip_lev1 = self.crop_image(x_gnn1, self.list_sizes_crop_merge[0])
        # else:
        #     x_skip_lev1 = x_gnn1
        x_skip_lev1 = self.crop_image(x_gnn1, self.list_sizes_crop_merge[0])
        # if self.isUse_valid_convs:
        #     x_skip_lev2 = self.crop_image(x_gnn2, self.list_sizes_crop_merge[1])
        # else:
        #     x_skip_lev2 = x_gnn2
        x_skip_lev2 = self.crop_image(x_gnn2, self.list_sizes_crop_merge[1])

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




class GNN_Skip1_GNN23(NeuralNetwork):
    """ Unet with 3 layers. 1st level: skip connection. 2nd,3rd levels: GNN modules."""
    nfeat_default = 8
    nlevels = 3

    def __init__(self, size_image,
                 nfeat= nfeat_default,
                 isUse_valid_convs= False,
                 source_dir_adjs= source_dir_adjs_default):
        super(GNN_Skip1_GNN23, self).__init__(size_image,
                                              nfeat,
                                              isUse_valid_convs= isUse_valid_convs)
        self.load_adjacency_matrixes(source_dir_adjs)

    def get_arch_desc(self):
        return ['GNN_Skip1_GNN23', {'size_image': self.size_image, 'nfeat': self.nfeat}]

    def gen_list_module_operations(self):
        self.list_module_opers = (self.nlevels-1) * ['convol', 'convol', 'pool'] + \
                                 ['gnn'] + \
                                 (self.nlevels-1) * ['upsam', 'convol', 'convol'] + \
                                 ['conv_last']

    def load_adjacency_matrixes(self, source_dir_adjs=source_dir_adjs_default):
        if self.isUse_valid_convs:
            list_filename_adjs = ['adj_z82_x170_y114.npz',
                                  'adj_z41_x85_y57.npz']
        else:
            list_filename_adjs = ['y240_2_adj176.npz',
                                  'y240_4_adj176.npz']
        list_filename_adjs = [join_path_names(source_dir_adjs, filename) for filename in list_filename_adjs]

        (self.adj1, self.adj2) = self.load_adjacency_matrix(list_filename_adjs)

    def build_model(self):
        if self.isUse_valid_convs:
            padding_convs = 0
        else:
            padding_convs = 1

        nfeatD1 = self.nfeat
        self.convD11 = Conv3d(self.ipChannels, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convD12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.mpool = MaxPool3d(kernel_size= 2, padding= 0)

        nfeatD2 = 2 * nfeatD1
        self.convD21 = Conv3d(nfeatD1, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convD22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)

        self.nGNN1 = NodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)
        self.nGNN2 = NodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)

        self.convU21 = Conv3d(nfeatD2*2, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convU22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)

        self.convU11 = Conv3d(nfeatD2, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convU12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.upool = Upsample(scale_factor=2, mode='nearest')

        self.classify = Conv3d(nfeatD1, self.opChannels, kernel_size= 1, padding= 0)

        self.relu = ReLU(inplace=True)
        self.activation_output = Sigmoid()

    def forward(self, input):

        x = self.relu(self.convD11(input))
        x = self.relu(self.convD12(x))
        # if self.isUse_valid_convs:
        #     x_skip_1 = self.crop_image(x, self.list_sizes_crop_merge[0])
        # else:
        #     x_skip_1 = x
        x = self.mpool(x)

        x = self.relu(self.convD21(x))
        x = self.relu(self.convD22(x))
        x_prev_2 = x

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
        x = self.nGNN1(x, self.adj1)
        x = x.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])

        if self.isUse_valid_convs:
            x_skip_2 = self.crop_image(x, self.list_sizes_crop_merge[1])
        else:
            x_skip_2 = x

        # Node GNN here # see gnn.py for further details
        x = self.mpool(x_prev_2)

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1, xSh[1])
        x = self.nGNN2(x, self.adj2)
        x = x.view(xSh[0], -1, xSh[2], xSh[3], xSh[4])

        x = self.upool(x)

        x = torch.cat([x, x_skip_2], dim=1)
        x = self.relu(self.convU21(x))
        x = self.relu(self.convU22(x))
        x = self.upool(x)

        #x = torch.cat([x, x_skip_1], dim=1)
        x = self.relu(self.convU11(x))
        x = self.relu(self.convU12(x))

        output = self.classify(x)
        output = self.activation_output(output)

        return output



class GNNSkip3(NeuralNetwork):
    """ Unet with 6 layers. GNN as base layer, base+1, base+2, +3 skip connection."""
    nfeat_default = 8
    nlevels = 3

    def __init__(self, size_image,
                 nfeat= nfeat_default,
                 isUse_valid_convs= False,
                 source_dir_adjs= source_dir_adjs_default):
        super(GNNSkip3, self).__init__(size_image,
                                       nfeat,
                                       isUse_valid_convs= isUse_valid_convs)
        self.load_adjacency_matrixes(source_dir_adjs)

    def get_arch_desc(self):
        return ['GNNSkip3', {'size_image': self.size_image, 'nfeat': self.nfeat}]

    def gen_list_module_operations(self):
        self.list_module_opers = (self.nlevels -1) * ['convol', 'convol', 'pool'] + \
                                 ['pool', 'pool', 'pool', 'gnn', 'upsam', 'upsam', 'upsam'] + \
                                 (self.nlevels -1) * ['upsam', 'convol', 'convol'] + \
                                 ['conv_last']

    def load_adjacency_matrixes(self, source_dir_adjs=source_dir_adjs_default):
        return NotImplemented

    def build_model(self):
        if self.isUse_valid_convs:
            padding_convs = 0
        else:
            padding_convs = 1

        nfeatD1 = self.nfeat
        self.mpool = MaxPool3d(kernel_size= 2, padding= 0)
        self.convD11 = Conv3d(self.ipChannels, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convD12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)

        nfeatD2 = 2 * nfeatD1
        self.convD21 = Conv3d(nfeatD1, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convD22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)

        self.nGNN1 = NodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)
        self.nGNN2 = NodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)
        self.nGNN3 = NodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)
        self.nGNN4 = NodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)

        self.upool = Upsample(scale_factor= 2, mode= 'nearest')
        self.convU21 = Conv3d(nfeatD2*4, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convU22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)

        self.convU11 = Conv3d(nfeatD2, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convU12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)

        self.classify = Conv3d(nfeatD1, self.opChannels, kernel_size= 1, padding= 0)

        self.relu = ReLU(inplace=True)
        self.activation_output = Sigmoid()

    def forward(self, input):

        x = self.relu(self.convD11(input))
        x = self.relu(self.convD12(x))
        x = self.mpool(x)

        x = self.relu(self.convD21(x))
        x = self.relu(self.convD22(x))
        x = self.mpool(x)
        x2 = x
        # Node GNN here # see gnn.py for further details

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1 ,xSh[1])
        x = self.nGNN1(x, self.adj1)
        x = x.view(xSh[0] ,-1 ,xSh[2] ,xSh[3] ,xSh[4])

        if self.isUse_valid_convs:
            x_skip_1 = self.crop_image(x, self.list_sizes_crop_merge[-3])
        else:
            x_skip_1 = x

        # Node GNN here # see gnn.py for further details
        x = self.mpool(x2)
        x3 = x

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1 ,xSh[1])
        x = self.nGNN2(x, self.adj2)
        x = x.view(xSh[0] ,-1 ,xSh[2] ,xSh[3] ,xSh[4])

        if self.isUse_valid_convs:
            x_skip_2 = self.crop_image(x, self.list_sizes_crop_merge[-2])
        else:
            x_skip_2 = x

        # Node GNN here # see gnn.py for further details
        x = self.mpool(x3)
        x4 = x

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1 ,xSh[1])
        x = self.nGNN3(x, self.adj3)
        x = x.view(xSh[0] ,-1 ,xSh[2] ,xSh[3] ,xSh[4])

        if self.isUse_valid_convs:
            x_skip_3 = self.crop_image(x, self.list_sizes_crop_merge[-1])
        else:
            x_skip_3 = x

        # Node GNN here # see gnn.py for further details
        x = self.mpool(x4)

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1 ,xSh[1])
        x = self.nGNN4(x, self.adj4)
        x = x.view(xSh[0] ,-1 ,xSh[2] ,xSh[3] ,xSh[4])

        x = self.upool(x)
        x = torch.cat([x, x_skip_3], dim=1)
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





class GNNSkip2(NeuralNetwork):
    """ Unet with 5 layers. GNN as base layer, base+1, base+2 skip connection."""
    nfeat_default = 8
    nlevels = 3

    def __init__(self, size_image,
                 nfeat= nfeat_default,
                 isUse_valid_convs= False,
                 source_dir_adjs= source_dir_adjs_default):
        super(GNNSkip2, self).__init__(size_image,
                                       nfeat,
                                       isUse_valid_convs= isUse_valid_convs)
        self.load_adjacency_matrixes(source_dir_adjs)

    def get_arch_desc(self):
        return ['GNNSkip2', {'size_image': self.size_image, 'nfeat': self.nfeat}]

    def gen_list_module_operations(self):
        self.list_module_opers = (self.nlevels -1) * ['convol', 'convol', 'pool'] + \
                                 ['pool', 'pool', 'gnn', 'upsam', 'upsam'] + \
                                 (self.nlevels -1) * ['upsam', 'convol', 'convol'] + \
                                 ['conv_last']

    def load_adjacency_matrixes(self, source_dir_adjs=source_dir_adjs_default):
        return NotImplemented

    def build_model(self):
        if self.isUse_valid_convs:
            padding_convs = 0
        else:
            padding_convs = 1

        nfeatD1 = self.nfeat
        self.mpool = MaxPool3d(kernel_size= 2, padding= 0)
        self.convD11 = Conv3d(self.ipChannels, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convD12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)

        nfeatD2 = 2 * nfeatD1
        self.convD21 = Conv3d(nfeatD1, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convD22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)

        self.nGNN1 = NodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)
        self.nGNN2 = NodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)
        self.nGNN3 = NodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)

        self.upool = Upsample(scale_factor= 2, mode= 'nearest')
        self.convU21 = Conv3d(nfeatD2*3, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convU22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)

        self.convU11 = Conv3d(nfeatD2, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convU12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)

        self.classify = Conv3d(nfeatD1, self.opChannels, kernel_size= 1, padding= 0)

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
        # Node GNN here # see gnn.py for further details

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1 ,xSh[1])
        x = self.nGNN1(x, self.adj1)
        x = x.view(xSh[0] ,-1 ,xSh[2] ,xSh[3] ,xSh[4])

        if self.isUse_valid_convs:
            x_skip_1 = self.crop_image(x, self.list_sizes_crop_merge[-2])
        else:
            x_skip_1 = x

        # Node GNN here # see gnn.py for further details
        x = self.mpool(x2)
        x3 = x

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1 ,xSh[1])
        x = self.nGNN2(x, self.adj2)
        x = x.view(xSh[0] ,-1 ,xSh[2] ,xSh[3] ,xSh[4])

        if self.isUse_valid_convs:
            x_skip_2 = self.crop_image(x, self.list_sizes_crop_merge[-1])
        else:
            x_skip_2 = x

        # Node GNN here # see gnn.py for further details
        x = self.mpool(x3)

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1 ,xSh[1])
        x = self.nGNN2(x, self.adj3)
        x = x.view(xSh[0] ,-1 ,xSh[2] ,xSh[3] ,xSh[4])

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





class GNNSkip(NeuralNetwork):
    """ Unet with 4 layers. GNN as base layer and base+1 skip connection."""
    nfeat_default = 8
    nlevels = 3

    def __init__(self, size_image,
                 nfeat= nfeat_default,
                 isUse_valid_convs= False,
                 source_dir_adjs= source_dir_adjs_default):
        super(GNNSkip, self).__init__(size_image,
                                      nfeat,
                                      isUse_valid_convs= isUse_valid_convs)
        self.load_adjacency_matrixes(source_dir_adjs)

    def get_arch_desc(self):
        return ['GNNSkip', {'size_image': self.size_image, 'nfeat': self.nfeat}]

    def gen_list_module_operations(self):
        self.list_module_opers = (self.nlevels -1) * ['convol', 'convol', 'pool'] + \
                                 ['pool', 'gnn', 'upsam'] + \
                                 (self.nlevels -1) * ['upsam', 'convol', 'convol'] + \
                                 ['conv_last']

    def load_adjacency_matrixes(self, source_dir_adjs=source_dir_adjs_default):
        return NotImplemented

    def build_model(self):
        if self.isUse_valid_convs:
            padding_convs = 0
        else:
            padding_convs = 1

        nfeatD1 = self.nfeat
        self.mpool = MaxPool3d(kernel_size= 2, padding= 0)
        self.convD11 = Conv3d(self.ipChannels, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convD12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)

        nfeatD2 = 2 * nfeatD1
        self.convD21 = Conv3d(nfeatD1, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convD22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)

        self.nGNN1 = NodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)
        self.nGNN2 = NodeGNN(nfeatD2, nfeatD2, nfeatD2, self.dropout)

        self.upool = Upsample(scale_factor= 2, mode= 'nearest')
        self.convU21 = Conv3d(nfeatD2*2, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convU22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)

        self.convU11 = Conv3d(nfeatD2, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convU12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)

        self.classify = Conv3d(nfeatD1, self.opChannels, kernel_size= 1, padding= 0)

        self.relu = ReLU(inplace=True)
        self.activation_output = Sigmoid()

    def forward(self, input):

        x = self.relu(self.convD11(input))
        x = self.relu(self.convD12(x))
        x = self.mpool(x)

        x = self.relu(self.convD21(x))
        x = self.relu(self.convD22(x))
        x = self.mpool(x)
        x2 = x
        # Node GNN here # see gnn.py for further details

        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1 ,xSh[1])
        x = self.nGNN1(x, self.adj1)
        x = x.view(xSh[0] ,-1 ,xSh[2] ,xSh[3] ,xSh[4])

        if self.isUse_valid_convs:
            x_skip_1 = self.crop_image(x, self.list_sizes_crop_merge[-1])
        else:
            x_skip_1 = x

        # Node GNN here # see gnn.py for further details
        x = self.mpool(x2)
        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1 ,xSh[1])
        x = self.nGNN2(x, self.adj2)
        x = x.view(xSh[0] ,-1 ,xSh[2] ,xSh[3] ,xSh[4])

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





class GNN3D(NeuralNetwork):
    """ Unet with GNN as base layer and no skip connections"""
    nfeat_default = 8
    nlevels = 3

    def __init__(self, size_image,
                 nfeat= nfeat_default,
                 isUse_valid_convs= False,
                 source_dir_adjs= source_dir_adjs_default):
        super(GNN3D, self).__init__(size_image,
                                    nfeat,
                                    isUse_valid_convs= isUse_valid_convs)
        self.load_adjacency_matrixes(source_dir_adjs)

    def get_arch_desc(self):
        return ['GNN3D', {'size_image': self.size_image, 'nfeat': self.nfeat}]

    def gen_list_module_operations(self):
        self.list_module_opers = (self.nlevels -1) * ['convol', 'convol', 'pool'] + \
                                 ['gnn'] + \
                                 (self.nlevels -1) * ['upsam', 'convol', 'convol'] + \
                                 ['conv_last']

    def load_adjacency_matrixes(self, source_dir_adjs=source_dir_adjs_default):
        if self.isUse_valid_convs:
            list_filename_adjs = ['adj_z41_x85_y57.npz']
        else:
            list_filename_adjs = ['y240_4_adj176.npz']
        list_filename_adjs = [join_path_names(source_dir_adjs, filename) for filename in list_filename_adjs]

        self.adj = self.load_adjacency_matrix(list_filename_adjs)

    def build_model(self):
        if self.isUse_valid_convs:
            padding_convs = 0
        else:
            padding_convs = 1

        nfeatD1 = self.nfeat
        self.mpool = MaxPool3d(kernel_size= 2, padding= 0)
        self.convD11 = Conv3d(self.ipChannels, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convD12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)

        nfeatD2 = 2 * nfeatD1
        self.convD21 = Conv3d(nfeatD1, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convD22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)

        nfeatD3 = 2 * nfeatD2
        gnnHid = 2 * nfeatD2
        self.nGNN = NodeGNN(nfeatD2, gnnHid, nfeatD3, self.dropout)

        self.upool = Upsample(scale_factor= 2, mode= 'nearest')
        self.convU21 = Conv3d(nfeatD3, nfeatD2, kernel_size= 3, padding= padding_convs)
        self.convU22 = Conv3d(nfeatD2, nfeatD2, kernel_size= 3, padding= padding_convs)

        self.convU11 = Conv3d(nfeatD2, nfeatD1, kernel_size= 3, padding= padding_convs)
        self.convU12 = Conv3d(nfeatD1, nfeatD1, kernel_size= 3, padding= padding_convs)

        self.classify = Conv3d(nfeatD1, self.opChannels, kernel_size= 1, padding= 0)

        self.relu = ReLU(inplace=True)
        self.activation_output = Sigmoid()

    def forward(self, input):

        x = self.relu(self.convD11(input))
        x = self.relu(self.convD12(x))
        x = self.mpool(x)

        x = self.relu(self.convD21(x))
        x = self.relu(self.convD22(x))
        x = self.mpool(x)

        # Node GNN here # see gnn.py for further details
        xSh = x.shape
        x = x.view(xSh[0], xSh[1], -1).view(-1 ,xSh[1])
        x = self.nGNN(x, self.adj)
        x = x.view(xSh[0] ,-1 ,xSh[2] ,xSh[3] ,xSh[4])

        x = self.upool(x)

        x = self.relu(self.convU21(x))
        x = self.relu(self.convU22(x))
        x = self.upool(x)

        x = self.relu(self.convU11(x))
        x = self.relu(self.convU12(x))

        output = self.classify(x)
        output = self.activation_output(output)

        return output



# all available metrics
def DICTAVAILMODELS(option, size_image,
                    nfeat=GNN3D.nfeat_default,
                    isUse_valid_convs=False,
                    source_dir_adjs=source_dir_adjs_default):
    list_models_avail = ['GNN-Skip1-GNN23', 'GNNSkip3', 'GNNSkip2', 'GNNSkip', 'GNN']

    print("Building model \'%s\' with nfeats \'%s\' in first layer" %(option, nfeat))
    if (option == 'GNN-Skip1-GNN23'):
        return GNN_Skip1_GNN23(size_image, nfeat=nfeat,
                               isUse_valid_convs=isUse_valid_convs,
                               source_dir_adjs=source_dir_adjs)
    elif (option == 'GNNSkip3'):
        return GNNSkip3(size_image, nfeat=nfeat,
                        isUse_valid_convs=isUse_valid_convs,
                        source_dir_adjs=source_dir_adjs)
    elif (option == 'GNNSkip2'):
        return GNNSkip2(size_image, nfeat=nfeat,
                        isUse_valid_convs=isUse_valid_convs,
                        source_dir_adjs=source_dir_adjs)
    elif (option == 'GNNSkip'):
        return GNNSkip(size_image, nfeat=nfeat,
                       isUse_valid_convs=isUse_valid_convs,
                       source_dir_adjs=source_dir_adjs)
    elif (option == 'GNN'):
        return GNN3D(size_image, nfeat=nfeat,
                     isUse_valid_convs=isUse_valid_convs,
                     source_dir_adjs=source_dir_adjs)
    else:
        message = 'Model chosen not found. Models available: (%s)' %(', '.join(list_models_avail))
        catch_error_exception(message)
        return NotImplemented