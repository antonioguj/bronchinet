
from typing import Tuple, Dict, Union, Any

import numpy as np
import scipy.sparse as sp
from torch.nn import Conv3d, MaxPool3d, Upsample, ReLU, LeakyReLU, Sigmoid
import torch

from common.exceptionmanager import catch_error_exception
from models.pytorch.networks import UNet
from models.pytorch.gnn.gnnmodules import NodeGNN, NodeGNNwithAttentionLayers
from models.pytorch.gnn.gnnutil import sparse_matrix_to_torch_sparse_tensor
from models.pytorch.gnn.graphprocessing import build_adjacency, compute_onthefly_adjacency, \
    compute_onthefly_adjacency_with_attention, OntheflyAdjacencyLimitCanditsGenerator

torch.manual_seed(2017)


class UNetGNN(UNet):

    def __init__(self,
                 size_image_in: Union[Tuple[int, int, int], Tuple[int, int]],
                 num_levels: int,
                 num_featmaps_in: int,
                 num_channels_in: int,
                 num_classes_out: int,
                 is_use_valid_convols: bool = False,
                 num_levels_valid_convols: int = UNet._num_levels_valid_convols_default,
                 is_onthefly_adjacency: bool = False,
                 is_gnn_with_attention: bool = False,
                 is_limit_candits_onthefly_adjacency: bool = True
                 ) -> None:
        super(UNetGNN, self).__init__(size_image_in,
                                      num_levels,
                                      num_featmaps_in,
                                      num_channels_in,
                                      num_classes_out,
                                      is_use_valid_convols=is_use_valid_convols,
                                      num_levels_valid_convols=num_levels_valid_convols)
        self._is_onthefly_adjacency = is_onthefly_adjacency
        self._is_gnn_with_attention = is_gnn_with_attention
        self._is_limit_candits_onthefly_adjacency = is_limit_candits_onthefly_adjacency

        self._gnn_module = None
        self._func_calc_onthefly_adjacency = None

        if self._is_limit_candits_onthefly_adjacency:
            self.set_calcdata_onthefly_adjacency()

        self._build_model()

    def _build_model(self) -> None:
        raise NotImplementedError

    def count_model_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_build_adjacency_data(self, filename_adjacency: str) -> None:
        print("Build adjacency matrix...")
        adjacency = build_adjacency(self._size_image_in, num_neighs=26).cuda()

        # store the built adjacency matrix
        print("Store the built adjacency matrix in \'%s\'..." % (filename_adjacency))
        self._save_adjacency_matrix(filename_adjacency, adjacency)

        if self._is_gnn_with_attention:
            print("And build matrices for attention layers...")
            (adjacency, node2edge_in, node2edge_out) = self._load_adjacency_matrix_with_attention(filename_adjacency)
            self._gnn_module.preprocess(adjacency, node2edge_in, node2edge_out)
        else:
            self._gnn_module.preprocess(adjacency)

    def set_load_adjacency_data(self, filename_adjacency: str) -> None:
        if self._is_gnn_with_attention:
            print("Loading adjacency and matrices for attention layers, from file: \'%s\'..." % (filename_adjacency))
            (adjacency, node2edge_in, node2edge_out) = self._load_adjacency_matrix_with_attention(filename_adjacency)
            self._gnn_module.preprocess(adjacency, node2edge_in, node2edge_out)
        else:
            print("Loading adjacency matrix, from file: \'%s\'..." % (filename_adjacency))
            adjacency = self._load_adjacency_matrix(filename_adjacency)
            self._gnn_module.preprocess(adjacency)

    def set_calcdata_onthefly_adjacency(self) -> None:
        index_input_gnn_module = self._list_operation_names_layers_all.index('gnn_module') - 1
        shape_input_gnn_module = self._list_sizes_output_all_layers[index_input_gnn_module]

        if self._is_limit_candits_onthefly_adjacency:
            print("Limit the neighbourhood of candidates when computing the adjacency to max. distance \'5\'...")
            self._onthefly_adjacency_generator = \
                OntheflyAdjacencyLimitCanditsGenerator(shape_input_gnn_module, dist_max_candits_neighs=5)
            if self._is_gnn_with_attention:
                print("Set on-the-fly calculator of adjacency and matrices for attention layers...")
                self._func_calc_onthefly_adjacency = self._onthefly_adjacency_generator.compute_with_attention
            else:
                print("Set on-the-fly calculator of adjacency matrix...")
                self._func_calc_onthefly_adjacency = self._onthefly_adjacency_generator.compute
        else:
            if self._is_gnn_with_attention:
                print("Set on-the-fly calculator of adjacency and matrices for attention layers...")
                self._func_calc_onthefly_adjacency = compute_onthefly_adjacency_with_attention
            else:
                print("Set on-the-fly calculator of adjacency matrix...")
                self._func_calc_onthefly_adjacency = compute_onthefly_adjacency

    def _load_adjacency_matrix(self, filename_adjacency: str) -> torch.Tensor:
        adjacency = sparse_matrix_to_torch_sparse_tensor(sp.load_npz(filename_adjacency))
        return adjacency.cuda()

    def _save_adjacency_matrix(self, filename_adjacency: str, adjacency: torch.Tensor) -> None:
        sp.save_npz(filename_adjacency, adjacency)

    def _load_adjacency_matrix_with_attention(self, filename_adjacency: str
                                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_neighs = 26
        adjacency = sp.load_npz(filename_adjacency)
        adjacency = adjacency * (num_neighs + 1)
        num_nonzero = adjacency.nnz
        num_nodes = adjacency.shape[0]
        node2edge_in = sp.csr_matrix((np.ones(num_nonzero), (np.arange(num_nonzero), sp.find(adjacency)[1])),
                                     shape=(num_nonzero, num_nodes))
        node2edge_out = sp.csr_matrix((np.ones(num_nonzero), (np.arange(num_nonzero), sp.find(adjacency)[0])),
                                      shape=(num_nonzero, num_nodes))
        return (adjacency.cuda(), node2edge_in.cuda(), node2edge_out.cuda())

    def _build_list_operation_names_layers(self) -> None:
        if self._num_levels == 1:
            self._list_operation_names_layers_all = ['convols'] * 2 + ['gnn_module'] + ['convols'] * 2 + ['classify']

        elif self._is_use_valid_convols \
                and self._num_levels > self._num_levels_valid_convols:
            num_levels_nonpadded = self._num_levels_valid_convols
            num_levels_padded_exclast = self._num_levels - num_levels_nonpadded - 1
            self._list_operation_names_layers_all = \
                num_levels_nonpadded * (['convols'] * 2 + ['pooling']) \
                + num_levels_padded_exclast * (['convols_padded'] * 2 + ['pooling']) \
                + ['gnn_module'] \
                + num_levels_padded_exclast * (['upsample'] + ['convols_padded'] * 2) \
                + num_levels_nonpadded * (['upsample'] + ['convols'] * 2) \
                + ['classify']
        else:
            self._list_operation_names_layers_all = \
                (self._num_levels - 1) * (['convols'] * 2 + ['pooling']) \
                + ['gnn_module'] \
                + (self._num_levels - 1) * (['upsample'] + ['convols'] * 2) \
                + ['classify']


class Unet3DGNNPlugin(UNetGNN):
    _num_levels_fixed = 3
    _num_levels_valid_convols_fixed = 3
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
                 is_onthefly_adjacency: bool = False,
                 is_gnn_with_attention: bool = False,
                 is_limit_candits_onthefly_adjacency: bool = True
                 ) -> None:
        super(Unet3DGNNPlugin, self).__init__(size_image_in,
                                              self._num_levels_fixed,
                                              num_featmaps_in,
                                              num_channels_in,
                                              num_classes_out,
                                              is_use_valid_convols=is_use_valid_convols,
                                              num_levels_valid_convols=self._num_levels_valid_convols_fixed,
                                              is_onthefly_adjacency=is_onthefly_adjacency,
                                              is_gnn_with_attention=is_gnn_with_attention,
                                              is_limit_candits_onthefly_adjacency=is_limit_candits_onthefly_adjacency)
        self._type_activate_hidden = self._type_activate_hidden_default
        self._type_activate_output = self._type_activate_output_default

    def get_network_input_args(self) -> Dict[str, Any]:
        return {'size_image_in': self._size_image_in,
                'num_featmaps_in': self._num_featmaps_in,
                'num_channels_in': self._num_channels_in,
                'num_classes_out': self._num_classes_out,
                'is_use_valid_convols': self._is_use_valid_convols,
                'is_onthefly_adjacency': self._is_onthefly_adjacency,
                'is_gnn_with_attention': self._is_gnn_with_attention,
                'is_limit_candits_onthefly_adjacency': self._is_limit_candits_onthefly_adjacency}

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
            self._gnn_module = NodeGNNwithAttentionLayers(num_featmaps_lev2, num_featmaps_lev3, num_featmaps_lev2,
                                                          is_dropout=False)
        else:
            self._gnn_module = NodeGNN(num_featmaps_lev2, num_featmaps_lev3, num_featmaps_lev2,
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

        hidden_next = self._activation_hidden(self._convolution_down_lev1_1(input))
        hidden_next = self._activation_hidden(self._convolution_down_lev1_2(hidden_next))
        skipconn_lev1 = hidden_next
        hidden_next = self._pooling_down_lev1(hidden_next)

        hidden_next = self._activation_hidden(self._convolution_down_lev2_1(hidden_next))
        hidden_next = self._activation_hidden(self._convolution_down_lev2_2(hidden_next))
        skipconn_lev2 = hidden_next
        hidden_next = self._pooling_down_lev2(hidden_next)

        hid_shape = hidden_next.shape
        hidden_next = hidden_next.view(hid_shape[0], hid_shape[1], -1).view(-1, hid_shape[1])
        if self._is_onthefly_adjacency:
            adjacency_data = self._func_calc_onthefly_adjacency(hidden_next.data.cpu().numpy(), num_neighs=10)
            self._gnn_module.preprocess(adjacency_data)
        hidden_next = self._gnn_module(hidden_next)
        hidden_next = hidden_next.view(hid_shape[0], -1, hid_shape[2], hid_shape[3], hid_shape[4])
        torch.cuda.empty_cache()

        if self._is_use_valid_convols:
            skipconn_lev2 = self._crop_image_3d(skipconn_lev2, self._list_sizes_crop_where_merge[1])
        hidden_next = torch.cat([hidden_next, skipconn_lev2], dim=1)
        hidden_next = self._activation_hidden(self._convolution_up_lev2_1(hidden_next))
        hidden_next = self._activation_hidden(self._convolution_up_lev2_2(hidden_next))
        hidden_next = self._upsample_up_lev2(hidden_next)

        if self._is_use_valid_convols:
            skipconn_lev1 = self._crop_image_3d(skipconn_lev1, self._list_sizes_crop_where_merge[0])
        hidden_next = torch.cat([hidden_next, skipconn_lev1], dim=1)
        hidden_next = self._activation_hidden(self._convolution_up_lev1_1(hidden_next))
        hidden_next = self._activation_hidden(self._convolution_up_lev1_2(hidden_next))

        output = self._activation_last(self._classification_last(hidden_next))
        return output
