
from typing import Tuple, Dict, Union, Any

import numpy as np
import scipy.sparse as sp
from torch.nn import Conv3d, MaxPool3d, Upsample, ReLU, LeakyReLU, Sigmoid
import torch

from common.exceptionmanager import catch_error_exception
from models.pytorch.networks import UNet
from models.pytorch.gnns.gnnmodules import NodeGNN, NodeGNNwithAttention
from models.pytorch.gnns.gnnutil import sparse_matrix_to_torch_sparse_tensor
from models.pytorch.gnns.graphprocessing import build_adjacency, compute_onthefly_adjacency, \
    compute_onthefly_adjacency_with_attention, OntheflyAdjacencyLimitCanditNeighbours

torch.manual_seed(2017)


class UNetGNN(UNet):
    _num_neighs_fixed_adjacency_default = 26
    _num_neighs_onthefly_adjacency_default = 10
    _dist_max_candit_neighs_otfadj_default = 5

    def __init__(self,
                 size_image_in: Union[Tuple[int, int, int], Tuple[int, int]],
                 num_levels: int,
                 num_featmaps_in: int,
                 num_channels_in: int,
                 num_classes_out: int,
                 is_use_valid_convols: bool = False,
                 is_gnn_onthefly_adjacency: bool = False,
                 is_gnn_limit_candit_neighs_otfadj: bool = False,
                 is_gnn_with_attention: bool = False
                 ) -> None:
        super(UNetGNN, self).__init__(size_image_in,
                                      num_levels,
                                      num_featmaps_in,
                                      num_channels_in,
                                      num_classes_out,
                                      is_use_valid_convols=is_use_valid_convols)
        self._is_gnn_onthefly_adjacency = is_gnn_onthefly_adjacency
        self._is_gnn_limit_candit_neighs_otfadj = is_gnn_limit_candit_neighs_otfadj
        self._is_gnn_with_attention = is_gnn_with_attention
        self._num_neighs_fixed_adjacency = self._num_neighs_fixed_adjacency_default
        self._num_neighs_onthefly_adjacency = self._num_neighs_onthefly_adjacency_default
        self._dist_max_candit_neighs_adj = self._dist_max_candit_neighs_otfadj_default
        self._gnn_module = None

        if self._is_use_valid_convols:
            index_input_gnn_module = self._names_operations_layers_all.index('gnn_module') - 1
            self._size_input_gnn_module = self._sizes_output_all_layers[index_input_gnn_module]
        else:
            self._size_input_gnn_module = tuple([elem // (self._num_levels - 1)**2 for elem in self._size_image_in])

        print("Total input volume to GNN module: \'%s\'..." % (str(self._size_input_gnn_module)))

        if self._is_gnn_onthefly_adjacency:
            print("Computing the adjacency \'on-the-fly\' in every epoch, with \'%s\' neighbours for each node..."
                  % (self._num_neighs_onthefly_adjacency))
            if self._is_gnn_limit_candit_neighs_otfadj:
                print("Limit the neighbourhood of candidates when computing the \'on-the-fly\' adjacency, to nodes "
                      "within max. distance \'%s\' around each node..." % (self._dist_max_candit_neighs_adj))
        else:
            print("Computing the fixed adjacency at the start of training, with \'%s\' neighbours for each node..."
                  % (self._num_neighs_fixed_adjacency))

        if self._is_gnn_with_attention:
            print("Using graph convolutions with Attention layers...")

        self._func_compute_onthefly_adjacency = None
        if self._is_gnn_onthefly_adjacency:
            self.set_func_compute_onthefly_adjacency()

    def _build_model(self) -> None:
        raise NotImplementedError

    def count_model_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _gnn_module_forward(self, input: torch.Tensor) -> torch.Tensor:
        shape_in = input.shape
        input = input.view(shape_in[0], shape_in[1], -1).view(-1, shape_in[1])

        if self._is_gnn_onthefly_adjacency:
            if self._is_gnn_with_attention:
                (adjacency, node2edge_in, node2edge_out) = \
                    self._func_compute_onthefly_adjacency(input.data.cpu().numpy(),
                                                          num_neighs=self._num_neighs_onthefly_adjacency)
                self._gnn_module.set_adjacency(adjacency, node2edge_in, node2edge_out)

            else:
                adjacency = self._func_compute_onthefly_adjacency(input.data.cpu().numpy(),
                                                                  num_neighs=self._num_neighs_onthefly_adjacency)
                self._gnn_module.set_adjacency(adjacency)

        output = self._gnn_module.forward(input)

        output = output.view(shape_in[0], -1, shape_in[2], shape_in[3], shape_in[4])
        torch.cuda.empty_cache()

        return output

    def set_build_adjacency_data(self, filename_adjacency: str) -> None:
        print("Build adjacency matrix...")
        adjacency = build_adjacency(self._size_input_gnn_module, num_neighs=self._num_neighs_fixed_adjacency)

        # store the built adjacency matrix
        print("Save the adjacency matrix in \'%s\'..." % (filename_adjacency))
        self._save_adjacency_matrix(filename_adjacency, adjacency)

        # load the built adjacency matrix in torch format
        self.set_load_adjacency_data(filename_adjacency)

    def set_load_adjacency_data(self, filename_adjacency: str) -> None:
        if self._is_gnn_with_attention:
            print("Load the adjacency and matrices for attention layers, from file: \'%s\'..." % (filename_adjacency))
            (adjacency, node2edge_in, node2edge_out) = self._load_adjacency_matrix_with_attention(filename_adjacency)
            self._gnn_module.set_adjacency(adjacency, node2edge_in, node2edge_out)

        else:
            print("Load the adjacency matrix, from file: \'%s\'..." % (filename_adjacency))
            adjacency = self._load_adjacency_matrix(filename_adjacency)
            self._gnn_module.set_adjacency(adjacency)

    def set_func_compute_onthefly_adjacency(self) -> None:
        if self._is_gnn_limit_candit_neighs_otfadj:
            self._onthefly_adjacency_cls = \
                OntheflyAdjacencyLimitCanditNeighbours(self._size_input_gnn_module,
                                                       dist_max_candit_neighs=self._dist_max_candit_neighs_adj)

            if self._is_gnn_with_attention:
                print("Set function to compute \'on-the-fly\' the adjacency and matrices for attention layers...")
                self._func_compute_onthefly_adjacency = self._onthefly_adjacency_cls.compute_with_attention
            else:
                print("Set function to compute \'on-the-fly\' the adjacency matrix...")
                self._func_compute_onthefly_adjacency = self._onthefly_adjacency_cls.compute
        else:
            if self._is_gnn_with_attention:
                print("Set function to compute \'on-the-fly\' the adjacency and matrices for attention layers...")
                self._func_compute_onthefly_adjacency = compute_onthefly_adjacency_with_attention
            else:
                print("Set function to compute \'on-the-fly\' the adjacency matrix...")
                self._func_compute_onthefly_adjacency = compute_onthefly_adjacency

    def _save_adjacency_matrix(self, filename_adjacency: str, adjacency: torch.Tensor) -> None:
        sp.save_npz(filename_adjacency, adjacency, compressed=True)

    def _load_adjacency_matrix(self, filename_adjacency: str) -> torch.Tensor:
        adjacency = sp.load_npz(filename_adjacency)
        adjacency = sparse_matrix_to_torch_sparse_tensor(adjacency)
        return adjacency.cuda()

    def _load_adjacency_matrix_with_attention(self, filename_adjacency: str
                                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        adjacency = sp.load_npz(filename_adjacency)
        adjacency = adjacency * (self._num_neighs_fixed_adjacency + 1)
        num_nonzero = adjacency.nnz
        num_nodes = adjacency.shape[0]
        node2edge_in = sp.csr_matrix((np.ones(num_nonzero), (np.arange(num_nonzero), sp.find(adjacency)[1])),
                                     shape=(num_nonzero, num_nodes))
        node2edge_out = sp.csr_matrix((np.ones(num_nonzero), (np.arange(num_nonzero), sp.find(adjacency)[0])),
                                      shape=(num_nonzero, num_nodes))

        adjacency = sparse_matrix_to_torch_sparse_tensor(adjacency)
        node2edge_in = sparse_matrix_to_torch_sparse_tensor(node2edge_in)
        node2edge_out = sparse_matrix_to_torch_sparse_tensor(node2edge_out)
        return (adjacency.cuda(), node2edge_in.cuda(), node2edge_out.cuda())

    def _build_names_operation_layers(self) -> None:
        if self._num_levels == 1:
            self._names_operations_layers_all = ['convols'] * 2 + ['gnn_module'] + ['convols'] * 2 + ['classify']

        elif self._is_use_valid_convols \
                and self._num_levels > self._num_levels_valid_convols:
            num_levels_nonpadded = self._num_levels_valid_convols
            num_levels_padded_exclast = self._num_levels - num_levels_nonpadded - 1
            self._names_operations_layers_all = \
                num_levels_nonpadded * (['convols'] * 2 + ['pooling']) \
                + num_levels_padded_exclast * (['convols_padded'] * 2 + ['pooling']) \
                + ['gnn_module'] \
                + num_levels_padded_exclast * (['upsample'] + ['convols_padded'] * 2) \
                + num_levels_nonpadded * (['upsample'] + ['convols'] * 2) \
                + ['classify']
        else:
            self._names_operations_layers_all = \
                (self._num_levels - 1) * (['convols'] * 2 + ['pooling']) \
                + ['gnn_module'] \
                + (self._num_levels - 1) * (['upsample'] + ['convols'] * 2) \
                + ['classify']


class UNet3DGNNPlugin(UNetGNN):
    _num_levels_fixed = 3
    _num_featmaps_in_default = 16
    _num_channels_in_default = 1
    _num_classes_out_default = 1
    _dropout_rate_default = 0.2
    _type_activate_hidden_default = 'relu'
    _type_activate_output_default = 'sigmoid'

    def __init__(self,
                 size_image_in: Tuple[int, int, int],
                 num_featmaps_in: int = _num_featmaps_in_default,
                 num_channels_in: int = _num_channels_in_default,
                 num_classes_out: int = _num_classes_out_default,
                 is_use_valid_convols: bool = False,
                 is_gnn_onthefly_adjacency: bool = False,
                 is_gnn_limit_candit_neighs_otfadj: bool = False,
                 is_gnn_with_attention: bool = False
                 ) -> None:
        super(UNet3DGNNPlugin, self).__init__(size_image_in,
                                              self._num_levels_fixed,
                                              num_featmaps_in,
                                              num_channels_in,
                                              num_classes_out,
                                              is_use_valid_convols=is_use_valid_convols,
                                              is_gnn_onthefly_adjacency=is_gnn_onthefly_adjacency,
                                              is_gnn_limit_candit_neighs_otfadj=is_gnn_limit_candit_neighs_otfadj,
                                              is_gnn_with_attention=is_gnn_with_attention)
        self._type_activate_hidden = self._type_activate_hidden_default
        self._type_activate_output = self._type_activate_output_default

        self._build_model()

    def get_network_input_args(self) -> Dict[str, Any]:
        return {'size_image_in': self._size_image_in,
                'num_featmaps_in': self._num_featmaps_in,
                'num_channels_in': self._num_channels_in,
                'num_classes_out': self._num_classes_out,
                'is_use_valid_convols': self._is_use_valid_convols,
                'is_gnn_onthefly_adjacency': self._is_gnn_onthefly_adjacency,
                'is_gnn_limit_candit_neighs_otfadj': self._is_gnn_limit_candit_neighs_otfadj,
                'is_gnn_with_attention': self._is_gnn_with_attention}

    def _build_model(self):
        value_padding = 0 if self._is_use_valid_convols else 1

        num_featmaps_lev1 = self._num_featmaps_in
        self._convolution_down_lev1_1 = Conv3d(self._num_channels_in, num_featmaps_lev1, kernel_size=3,
                                               padding=value_padding)
        self._convolution_down_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3,
                                               padding=value_padding)
        self._pooling_down_lev1 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev2 = 2 * num_featmaps_lev1
        self._convolution_down_lev2_1 = Conv3d(num_featmaps_lev1, num_featmaps_lev2, kernel_size=3,
                                               padding=value_padding)
        self._convolution_down_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3,
                                               padding=value_padding)
        self._pooling_down_lev2 = MaxPool3d(kernel_size=2, padding=0)

        num_featmaps_lev3 = 2 * num_featmaps_lev2
        if self._is_gnn_with_attention:
            self._gnn_module = NodeGNNwithAttention(num_featmaps_lev2, num_featmaps_lev3, num_featmaps_lev3,
                                                    is_dropout=False)
        else:
            self._gnn_module = NodeGNN(num_featmaps_lev2, num_featmaps_lev3, num_featmaps_lev3,
                                       is_dropout=False)
        self._upsample_up_lev3 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lev2pl3 = num_featmaps_lev2 + num_featmaps_lev3
        self._convolution_up_lev2_1 = Conv3d(num_feats_lev2pl3, num_featmaps_lev2, kernel_size=3,
                                             padding=value_padding)
        self._convolution_up_lev2_2 = Conv3d(num_featmaps_lev2, num_featmaps_lev2, kernel_size=3,
                                             padding=value_padding)
        self._upsample_up_lev2 = Upsample(scale_factor=2, mode='nearest')

        num_feats_lay1pl2 = num_featmaps_lev1 + num_featmaps_lev2
        self._convolution_up_lev1_1 = Conv3d(num_feats_lay1pl2, num_featmaps_lev1, kernel_size=3,
                                             padding=value_padding)
        self._convolution_up_lev1_2 = Conv3d(num_featmaps_lev1, num_featmaps_lev1, kernel_size=3,
                                             padding=value_padding)

        self._classification_last = Conv3d(num_featmaps_lev1, self._num_classes_out, kernel_size=1, padding=0)

        if self._type_activate_hidden == 'relu':
            self._activation_hidden = ReLU(inplace=True)
        elif self._type_activate_hidden == 'leaky_relu':
            self._activation_hidden = LeakyReLU(inplace=True)
        elif self._type_activate_hidden == 'linear':
            def func_activation_linear(input: torch.Tensor) -> torch.Tensor:
                return input
            self._activation_hidden = func_activation_linear
        else:
            message = 'Type activation hidden not existing: \'%s\'' % (self._type_activate_hidden)
            catch_error_exception(message)

        if self._type_activate_output == 'sigmoid':
            self._activation_last = Sigmoid()
        elif self._type_activate_output == 'linear':
            def func_activation_linear(input: torch.Tensor) -> torch.Tensor:
                return input
            self._activation_last = func_activation_linear
        else:
            message = 'Type activation output not existing: \'%s\' ' % (self._type_activate_output)
            catch_error_exception(message)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        hidden_nxt = self._activation_hidden(self._convolution_down_lev1_1(input))
        hidden_nxt = self._activation_hidden(self._convolution_down_lev1_2(hidden_nxt))
        hidden_skip_lev1 = hidden_nxt
        hidden_nxt = self._pooling_down_lev1(hidden_nxt)

        hidden_nxt = self._activation_hidden(self._convolution_down_lev2_1(hidden_nxt))
        hidden_nxt = self._activation_hidden(self._convolution_down_lev2_2(hidden_nxt))
        hidden_skip_lev2 = hidden_nxt
        hidden_nxt = self._pooling_down_lev2(hidden_nxt)

        hidden_nxt = self._gnn_module_forward(hidden_nxt)
        hidden_nxt = self._upsample_up_lev3(hidden_nxt)

        if self._is_use_valid_convols:
            hidden_skip_lev2 = self._crop_image_3d(hidden_skip_lev2, self._sizes_crop_where_merge[1])
        hidden_nxt = torch.cat([hidden_nxt, hidden_skip_lev2], dim=1)
        hidden_nxt = self._activation_hidden(self._convolution_up_lev2_1(hidden_nxt))
        hidden_nxt = self._activation_hidden(self._convolution_up_lev2_2(hidden_nxt))
        hidden_nxt = self._upsample_up_lev2(hidden_nxt)

        if self._is_use_valid_convols:
            hidden_skip_lev1 = self._crop_image_3d(hidden_skip_lev1, self._sizes_crop_where_merge[0])
        hidden_nxt = torch.cat([hidden_nxt, hidden_skip_lev1], dim=1)
        hidden_nxt = self._activation_hidden(self._convolution_up_lev1_1(hidden_nxt))
        hidden_nxt = self._activation_hidden(self._convolution_up_lev1_2(hidden_nxt))

        output = self._activation_last(self._classification_last(hidden_nxt))
        return output
