
from torch.autograd import Variable
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

SIGMA = 1
EPSILON = 1e-5


def alloc_memory_params_module(module):
    out_dict_params = {}
    for params in module.named_parameters():
        key = params[0]
        val_tensor_shape = params[1].shape
        value = nn.Parameter(torch.zeros(val_tensor_shape, device='cuda:0'), requires_grad=False)
        out_dict_params[key] = value
    return out_dict_params

def reset_parameters_graph_convolution(in_dict_params):
    stdv = 1. / math.sqrt(in_dict_params['weight'].size(1))
    in_dict_params['weight'].data.uniform_(-stdv, stdv)
    in_dict_params['bias'].data.uniform_(-stdv, stdv)

def reset_parameters_graph_convolution_first_order(in_dict_params):
    stdv = 1. / math.sqrt(in_dict_params['weight_neighbor'].size(1))
    in_dict_params['weight_neighbor'].data.uniform_(-stdv, stdv)
    in_dict_params['weight_self'].data.uniform_(-stdv, stdv)
    in_dict_params['bias'].data.uniform_(-stdv, stdv)

def reset_parameters_linear(in_dict_params):
    #source taken from torch.nn"
    init.kaiming_uniform_(in_dict_params['weight'], a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(in_dict_params['weight'])
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(in_dict_params['bias'], -bound, bound)

def reset_parameters_layer_norm(in_dict_params):
    in_dict_params['gamma'].data.fill_(1.0)
    in_dict_params['beta'].data.zero_()


class NodeGNN(nn.Module):
    def __init__(self, nfeat, nhid, opfeat, dropout):
        super(NodeGNN, self).__init__()
        self._fc1 = nn.Linear(nfeat, nhid)
        self._fc2 = nn.Linear(nhid, nhid)
        self._gc1 = GraphConvolutionFirstOrder(nhid, nhid)
        self._gc2 = GraphConvolutionFirstOrder(nhid, nhid)
        self._gc3 = GraphConvolutionFirstOrder(nhid, nhid)
        self._gc4 = GraphConvolutionFirstOrder(nhid, opfeat)
        self._ln1 = LayerNorm(nhid)
        self._dropout = dropout

    def _preprocess(self, in_adj):
        self._adj = in_adj

    def forward(self, x):
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._ln1(x)
        x = F.relu(self._gc1(x, self._adj))
        x = F.relu(self._gc2(x, self._adj))
        x = F.relu(self._gc3(x, self._adj))
        x = F.relu(self._gc4(x, self._adj))
        return x

    def _create_dict_params_module(self):
        out_dict = {'fc1':self._fc1, 'fc2':self._fc2,
                    'gc1':self._gc1, 'gc2':self._gc2, 'gc3':self._gc3, 'gc4':self._gc4,
                    'ln1':self._ln1}
        return out_dict

    def alloc_state_dict_vars(self, basename_module='', reset_parameters=False):
        dict_params_module = self._create_dict_params_module()

        out_state_dict = {}
        for (key_mod, val_module) in dict_params_module.items():
            i_dict_params = alloc_memory_params_module(val_module)

            if reset_parameters:
                #reset the allocated tensors in the same way as in the corresponding classes
                name_class_module = type(val_module).__name__
                if name_class_module=='GraphConvolution':
                    reset_parameters_graph_convolution(i_dict_params)
                elif name_class_module=='GraphConvolutionFirstOrder':
                    reset_parameters_graph_convolution_first_order(i_dict_params)
                elif name_class_module=='Linear':
                    reset_parameters_linear(i_dict_params)
                elif name_class_module=='LayerNorm':
                    reset_parameters_layer_norm(i_dict_params)
                else:
                    raise NotImplementedError

            #update out dict. Need to complete the the key
            for (key_par, val_params) in i_dict_params.items():
                out_key = '.'.join([basename_module, key_mod, key_par])
                out_state_dict[out_key] = val_params

        return out_state_dict


class NodeGNNwithAttentionLayers(nn.Module):
    def __init__(self, nfeat, nhid, opFeat, dropout):
        super(NodeGNNwithAttentionLayers, self).__init__()
        #self._fc1 = nn.Linear(nfeat, nhid)
        #self._fc2 = nn.Linear(nhid, nhid)
        #self._ln1 = LayerNorm(nhid)
        self._gc1 = GraphAttentionLayer(nfeat, nhid)
        self._gc2 = GraphAttentionLayer(nhid, opFeat)
        #self._dropout = dropout

    def _preprocess(self, adj, n2e_in, n2e_out):
        self._adj = adj
        self._n2e_in = n2e_in
        self._n2e_out = n2e_out

    def forward(self, x):
        #x = F.relu(self._fc1(x))
        #x = F.relu(self._fc2(x))
        #x = self._ln1(x)
        x = F.relu(self._gc1(x, self._adj, self._n2e_in, self._n2e_out))
        x = F.relu(self._gc2(x, self._adj, self._n2e_in, self._n2e_out))
        return x


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self._weight.size(1))
        self._weight.data.uniform_(-stdv, stdv)
        if self._bias is not None:
            self._bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self._weight)
        output = SparseMM()(adj, support)
        if self._bias is not None:
            return output + self._bias
        else:
            return output


class GraphAttentionLayer(nn.Module):
    """
    Simple Graph Attention Layer, with separate processing of self-connection.
    Equation format from https://docs.dgl.ai/en/latest/tutorials/models/1_gnn/9_gat.html
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphAttentionLayer, self).__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._weight_neighbor = nn.Parameter(torch.Tensor(in_features, out_features))
        self._weight_self = nn.Parameter(torch.Tensor(in_features, out_features))
        self._a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        nn.init.xavier_uniform_(self._a.data, gain=1.414)
        self._alpha = 0.2
        self._leakyRelu = nn.LeakyReLU(self._alpha, inplace=True)
        self._softmax = nn.Softmax(dim=1)

        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self._weight_neighbor.size(1))
        self._weight_neighbor.data.uniform_(-stdv, stdv)
        self._weight_self.data.uniform_(-stdv, stdv)
        if self._bias is not None:
            self._bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, n2e_in, n2e_out):
        N = adj.shape[0]

        # Transform node activations Eq. (1)
        h = torch.mm(input, self._weight_neighbor)

        # Compute pairwise edge features (Terms inside Eq. (2))
        h_in = Variable(torch.mm(n2e_in, h), requires_grad=False)
        h_out = Variable(torch.mm(n2e_out,h), requires_grad=False)
        h_edge = Variable(torch.cat([h_in, h_out],1), requires_grad=False)

        # Apply leakyReLU and weights for attention coefficients Eq.(2)
        e = self._leakyRelu(torch.matmul(h_edge, self._a))

        # Apply Softmax per node Eq.(3) (Sparse implementation)
        num_ngbrs = (adj.coalesce().indices()[0] == 0).sum()
        attention = Variable(self._softmax(e.view(-1, num_ngbrs)).view(-1), requires_grad=False)

        idx = adj.coalesce().indices()
        val = adj.coalesce().values()

        # Weigh nodes with attention; done by weighting the adj entries
        adj = torch.sparse.FloatTensor(idx,val*attention,(N,N))

        # Compute node updates with attention Eq. (4)
        output = SparseMM()(adj, h)
        if self._bias is not None:
            return output + self._bias
        else:
            return output


class GraphConvolutionFirstOrder(nn.Module):
    """
    Simple GCN layer, with separate processing of self-connection
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionFirstOrder, self).__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._weight_neighbor = nn.Parameter(torch.Tensor(in_features, out_features))
        self._weight_self = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self._bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self._weight_neighbor.size(1))
        self._weight_neighbor.data.uniform_(-stdv, stdv)
        self._weight_self.data.uniform_(-stdv, stdv)
        if self._bias is not None:
            self._bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        act_self = torch.mm(input, self._weight_self)
        support_neighbor = torch.mm(input, self._weight_neighbor)
        act_neighbor = SparseMM()(adj, support_neighbor)
        output = act_self + act_neighbor
        if self._bias is not None:
            return output + self._bias
        else:
            return output


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.

    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def forward(self, matrix1, matrix2):
        self.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)

    def backward(self, grad_output):
        matrix1, matrix2 = self.saved_tensors
        grad_matrix1 = grad_matrix2 = None
        if self.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())
        if self.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self._gamma = nn.Parameter(torch.ones(features))
        self._beta = nn.Parameter(torch.zeros(features))
        self._eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self._gamma * (x - mean) / (std + self._eps) + self._beta