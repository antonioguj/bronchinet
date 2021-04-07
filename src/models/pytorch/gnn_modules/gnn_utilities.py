
import numpy as np
import scipy.sparse as sp
import torch

SMOOTH = 1.0


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def to_linear_idx(x_idx, y_idx, num_cols):
    assert num_cols > np.max(x_idx)
    x_idx = np.array(x_idx, dtype=np.int32)
    y_idx = np.array(y_idx, dtype=np.int32)
    return y_idx * num_cols + x_idx

def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx

def dice_loss(preds, labels):
    "Return dice score. "
    preds_sq = preds**2
    return 1 - (2. * (torch.sum(preds * labels)) + SMOOTH) / (preds_sq.sum() + labels.sum() + SMOOTH)

def focalCE(preds, labels, gamma):
    "Return focal cross entropy"
    loss = -torch.mean( ( ((1-preds)**gamma) * labels * torch.log(preds) ) + ( ((preds)**gamma) * (1-labels) * torch.log(1-preds) ) )
    return loss

def dice(preds, labels):
    "Return dice score"
    preds_bin = (preds > 0.5).type_as(labels)
    return 2. * torch.sum(preds_bin * labels) / (preds_bin.sum() + labels.sum())

def wBCE(preds, labels, w):
    "Return weighted CE loss."
    return -torch.mean( w*labels*torch.log(preds) + (1-w)*(1-labels)*torch.log(1-preds) )