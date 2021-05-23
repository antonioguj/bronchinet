
from typing import Tuple, Dict

from torch.autograd import Variable
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

_EPSILON = 1.0e-6


def allocate_memory_parameters_module(module: nn.Module) -> Dict[str, nn.Parameter]:
    out_params = {}
    for params in module.named_parameters():
        key = params[0]
        val_tensor_shape = params[1].shape
        value = nn.Parameter(torch.zeros(val_tensor_shape, device='cuda:0'), requires_grad=False)
        out_params[key] = value
    return out_params


def reset_parameters_graph_convolution(in_params: Dict[str, nn.Parameter]) -> None:
    stdv = 1.0 / math.sqrt(in_params['weight'].size(1))
    in_params['weight'].data.uniform_(-stdv, stdv)
    in_params['bias'].data.uniform_(-stdv, stdv)


def reset_parameters_graph_convolution_first_order(in_params: Dict[str, nn.Parameter]) -> None:
    stdv = 1.0 / math.sqrt(in_params['weight_neighbour'].size(1))
    in_params['weight_neighbour'].data.uniform_(-stdv, stdv)
    in_params['weight_self'].data.uniform_(-stdv, stdv)
    in_params['bias'].data.uniform_(-stdv, stdv)


def reset_parameters_linear(in_params: Dict[str, nn.Parameter]) -> None:
    "source taken from torch.nn"
    init.kaiming_uniform_(in_params['weight'], a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(in_params['weight'])
    bound = 1.0 / math.sqrt(fan_in)
    init.uniform_(in_params['bias'], -bound, bound)


def reset_parameters_layer_norm(in_params: Dict[str, nn.Parameter]) -> None:
    in_params['gamma'].data.fill_(1.0)
    in_params['beta'].data.zero_()


class NodeGNN(nn.Module):

    def __init__(self, num_feats: int, num_hidden: int, num_out_feats: int, is_dropout: bool = False) -> None:
        super(NodeGNN, self).__init__()
        self._full_connect_1 = nn.Linear(num_feats, num_hidden)
        self._full_connect_2 = nn.Linear(num_hidden, num_hidden)
        self._graph_convol_1 = GraphConvolutionFirstOrder(num_hidden, num_hidden)
        self._graph_convol_2 = GraphConvolutionFirstOrder(num_hidden, num_hidden)
        self._graph_convol_3 = GraphConvolutionFirstOrder(num_hidden, num_hidden)
        self._graph_convol_4 = GraphConvolutionFirstOrder(num_hidden, num_out_feats)
        self._layer_norm_1 = LayerNorm(num_hidden)
        # self._is_dropout = is_dropout

    def set_adjacency(self, adjacency: torch.Tensor) -> None:
        self._adjacency = adjacency

    def _create_dict_internal_modules(self) -> Dict[str, nn.Module]:
        out_dict = {'full_connect_1': self._full_connect_1,
                    'full_connect_2': self._full_connect_2,
                    'graph_convol_1': self._graph_convol_1,
                    'graph_convol_2': self._graph_convol_2,
                    'graph_convol_3': self._graph_convol_3,
                    'graph_convol_4': self._graph_convol_4,
                    'layer_norm_1': self._layer_norm_1}
        return out_dict

    def allocate_state_variables_modules(self, basename_module: str = '',
                                         reset_parameters: bool = False
                                         ) -> Dict[str, nn.Parameter]:
        dict_internal_modules = self._create_dict_internal_modules()

        out_state_variables_modules = {}
        for (name_module, in_module) in dict_internal_modules.items():
            in_params = allocate_memory_parameters_module(in_module)

            if reset_parameters:
                # reset the allocated tensors in the same way as in the corresponding classes
                name_class_module = type(in_module).__name__
                if name_class_module == 'GraphConvolution':
                    reset_parameters_graph_convolution(in_params)
                elif name_class_module == 'GraphConvolutionFirstOrder':
                    reset_parameters_graph_convolution_first_order(in_params)
                elif name_class_module == 'Linear':
                    reset_parameters_linear(in_params)
                elif name_class_module == 'LayerNorm':
                    reset_parameters_layer_norm(in_params)
                else:
                    raise NotImplementedError

            # update output dict with parameters of internal modules
            for (name_param, value_param) in in_params.items():
                out_key = '.'.join([basename_module, name_module, name_param])
                out_state_variables_modules[out_key] = value_param

        return out_state_variables_modules

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        hidden_nxt = F.relu(self._full_connect_1(input))
        hidden_nxt = F.relu(self._full_connect_2(hidden_nxt))
        hidden_nxt = self._layer_norm_1(hidden_nxt)
        hidden_nxt = F.relu(self._graph_convol_1(hidden_nxt, self._adjacency))
        hidden_nxt = F.relu(self._graph_convol_2(hidden_nxt, self._adjacency))
        hidden_nxt = F.relu(self._graph_convol_3(hidden_nxt, self._adjacency))
        hidden_nxt = F.relu(self._graph_convol_4(hidden_nxt, self._adjacency))
        return hidden_nxt


class NodeGNNwithAttention(nn.Module):

    def __init__(self, num_feats: int, num_hidden: int, num_out_feats: int, is_dropout: bool = False) -> None:
        super(NodeGNNwithAttention, self).__init__()
        # self._full_connect_1 = nn.Linear(num_feats, num_hidden)
        # self._full_connect_2 = nn.Linear(num_hidden, num_hidden)
        self._graph_convol_1 = GraphConvolutionFirstOrderWithAttention(num_feats, num_hidden)
        self._graph_convol_2 = GraphConvolutionFirstOrderWithAttention(num_hidden, num_out_feats)
        # self._layer_norm_1 = LayerNorm(num_hidden)
        # self._is_dropout = is_dropout

    def set_adjacency(self, adjacency: torch.Tensor, node2edge_in: torch.Tensor, node2edge_out: torch.Tensor) -> None:
        self._adjacency = adjacency
        self._node2edge_in = node2edge_in
        self._node2edge_out = node2edge_out

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # hidden_nxt = F.relu(self._full_connect_1(input))
        # hidden_nxt = F.relu(self._full_connect_2(hidden_nxt))
        # hidden_nxt = self._layer_norm_1(hidden_nxt)
        hidden_nxt = F.relu(self._graph_convol_1(input, self._adjacency, self._node2edge_in, self._node2edge_out))
        hidden_nxt = F.relu(self._graph_convol_2(hidden_nxt, self._adjacency, self._node2edge_in, self._node2edge_out))
        return hidden_nxt


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, num_in_feats: int, num_out_feats: int, is_bias: bool = True) -> None:
        super(GraphConvolution, self).__init__()
        self._num_in_feats = num_in_feats
        self._num_out_feats = num_out_feats
        self._weight = nn.Parameter(torch.Tensor(num_in_feats, num_out_feats))
        if is_bias:
            self._bias = nn.Parameter(torch.Tensor(num_out_feats))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

        self._matrix_multiply = SparseMatrixMultiply()

    def _reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self._weight.size(1))
        self._weight.data.uniform_(-stdv, stdv)
        if self._bias is not None:
            self._bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = torch.mm(input, self._weight)
        output = self._matrix_multiply.forward(adjacency, support)
        if self._bias is not None:
            return output + self._bias
        else:
            return output


class GraphConvolutionFirstOrder(nn.Module):
    """
    Simple GCN layer, with separate processing of self-connection
    """

    def __init__(self, num_in_feats: int, num_out_feats: int, is_bias: bool = True) -> None:
        super(GraphConvolutionFirstOrder, self).__init__()
        self._num_in_feats = num_in_feats
        self._num_out_feats = num_out_feats
        self._weight_neighbour = nn.Parameter(torch.Tensor(num_in_feats, num_out_feats))
        self._weight_self = nn.Parameter(torch.Tensor(num_in_feats, num_out_feats))
        if is_bias:
            self._bias = nn.Parameter(torch.Tensor(num_out_feats))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

        self._matrix_multiply = SparseMatrixMultiply()

    def _reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self._weight_neighbour.size(1))
        self._weight_neighbour.data.uniform_(-stdv, stdv)
        self._weight_self.data.uniform_(-stdv, stdv)
        if self._bias is not None:
            self._bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        activ_self = torch.mm(input, self._weight_self)
        support_neighbour = torch.mm(input, self._weight_neighbour)
        activ_neighbour = self._matrix_multiply.forward(adjacency, support_neighbour)
        output = activ_self + activ_neighbour
        if self._bias is not None:
            return output + self._bias
        else:
            return output


class GraphConvolutionFirstOrderWithAttention(GraphConvolutionFirstOrder):
    """
    Simple Graph Attention Layer, with separate processing of self-connection.
    Equation format from https://docs.dgl.ai/en/latest/tutorials/models/1_gnn/9_gat.html
    """

    def __init__(self, num_in_feats: int, num_out_feats: int, is_bias: bool = True) -> None:
        super(GraphConvolutionFirstOrderWithAttention, self).__init__(num_in_feats, num_out_feats, is_bias)
        self._attention = nn.Parameter(torch.Tensor(2 * num_out_feats, 1))
        nn.init.xavier_uniform_(self._attention.data, gain=1.414)
        self._alpha = 0.2
        self._leaky_relu = nn.LeakyReLU(self._alpha, inplace=True)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, adjacency: torch.Tensor,
                node2edge_in: torch.Tensor, node2edge_out: torch.Tensor) -> torch.Tensor:
        num_nodes = adjacency.shape[0]

        # Transform node activations Eq. (1)
        hidden = torch.mm(input, self._weight_neighbour)

        # Compute pairwise edge features (Terms inside Eq. (2))
        hidden_in = Variable(torch.mm(node2edge_in, hidden), requires_grad=False)
        hidden_out = Variable(torch.mm(node2edge_out, hidden), requires_grad=False)
        hidden_edge = Variable(torch.cat([hidden_in, hidden_out], 1), requires_grad=False)

        # Apply leakyReLU and weights for attention coefficients Eq.(2)
        hidden_edge = self._leaky_relu(torch.matmul(hidden_edge, self._attention))

        # Apply Softmax per node Eq.(3) (Sparse implementation)
        num_neighs = (adjacency.coalesce().indices()[0] == 0).sum()
        attention = Variable(self._softmax(hidden_edge.view(-1, num_neighs)).view(-1), requires_grad=False)

        indexes = adjacency.coalesce().indices()
        values = adjacency.coalesce().values()

        # Weigh nodes with attention; done by weighting the adj entries
        adjacency = torch.sparse.FloatTensor(indexes, values * attention, (num_nodes, num_nodes))

        # Compute node updates with attention Eq. (4)
        output = self._matrix_multiply.forward(adjacency, hidden)
        if self._bias is not None:
            return output + self._bias
        else:
            return output


class SparseMatrixMultiply(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def __init__(self) -> None:
        super(SparseMatrixMultiply, self).__init__()

    def forward(self, matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
        self.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)

    def backward(self, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        matrix1, matrix2 = self.saved_tensors
        grad_matrix1 = grad_matrix2 = None
        if self.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())
        if self.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)
        return (grad_matrix1, grad_matrix2)


class LayerNorm(nn.Module):

    def __init__(self, num_in_feats: int) -> None:
        super(LayerNorm, self).__init__()
        self._gamma = nn.Parameter(torch.ones(num_in_feats))
        self._beta = nn.Parameter(torch.zeros(num_in_feats))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)
        return self._gamma * (input - mean) / (std + _EPSILON) + self._beta
