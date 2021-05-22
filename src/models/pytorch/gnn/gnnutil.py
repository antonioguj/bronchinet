
from typing import List

import numpy as np
import scipy.sparse as sp
import torch

_SMOOTH = 1.0


def row_normalize(sparse_matrix: np.array) -> np.array:
    """Row-normalize sparse matrix"""
    row_sum = np.array(sparse_matrix.sum(1), dtype=np.float32)
    row_inv = np.power(row_sum, -1).flatten()
    row_inv[np.isinf(row_inv)] = 0.0
    row_matdiag_inv = sp.diags(row_inv)
    sparse_matrix = row_matdiag_inv.dot(sparse_matrix)
    return sparse_matrix


def sparse_matrix_to_torch_sparse_tensor(sparse_matrix: np.array) -> np.array:
    """Convert a scipy sparse matrix to a torch sparse tensor"""
    sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_matrix.row, sparse_matrix.col))).long()
    values = torch.from_numpy(sparse_matrix.data)
    shape = torch.Size(sparse_matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def to_linear_indexes(indexes_x: List[int], indexes_y: List[int], num_cols: int) -> np.array:
    assert num_cols > np.max(indexes_x)
    indexes_x = np.array(indexes_x, dtype=np.int32)
    indexes_y = np.array(indexes_y, dtype=np.int32)
    return indexes_y * num_cols + indexes_x


def to_2d_indexes(indexes: List[int], num_cols: int) -> np.array:
    indexes = np.array(indexes, dtype=np.int64)
    indexes_y = np.array(np.floor(indexes / float(num_cols)), dtype=np.int64)
    indexes_x = indexes % num_cols
    return (indexes_x, indexes_y)


def dice_loss(preds: np.array, labels: np.array) -> np.array:
    "Return dice score"
    preds_squared = preds**2
    return 1.0 - (2.0 * (torch.sum(preds * labels)) + _SMOOTH) / (preds_squared.sum() + labels.sum() + _SMOOTH)


def focal_bce(preds: np.array, labels: np.array, gamma: float) -> np.array:
    "Return focal cross entropy"
    return - torch.mean((((1.0 - preds)**gamma) * labels * torch.log(preds))
                        + (((preds)**gamma) * (1.0 - labels) * torch.log(1.0 - preds)))


def dice(preds: np.array, labels: np.array) -> np.array:
    "Return dice score"
    preds_bin = (preds > 0.5).type_as(labels)
    return 2.0 * torch.sum(preds_bin * labels) / (preds_bin.sum() + labels.sum())


def weighted_bce(preds: np.array, labels: np.array, weight: float) -> np.array:
    "Return weighted BCE loss"
    return - torch.mean(weight * labels * torch.log(preds)
                        + (1.0 - weight) * (1.0 - labels) * torch.log(1.0 - preds))
