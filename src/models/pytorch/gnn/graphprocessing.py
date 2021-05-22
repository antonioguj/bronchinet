
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import pairwise_distances

# from models.pytorch.gnn.gnnutil import row_normalize, sparse_matrix_to_torch_sparse_tensor

torch.manual_seed(2017)


def build_adjacency(shape_volume, num_neighs: int = 26, resolution: int = 4) -> np.array:
    """
    Given the input volume size, constructs an adjacency matrix either of 6/26 neighbors.
    shape_volume: Tuple with 3D shape
    neighbours: 6/26
    resolution: reduction in resolution factor: 2/4/8/16/32
    """
    shape_volume = np.array((np.array(shape_volume)), dtype=np.int)
    # shape_volume = np.array((np.array(shape_volume) / resolution), dtype=np.int)

    (zdim, xdim, ydim) = shape_volume
    num_nodes = shape_volume.prod()
    indexes_nodes = np.arange(num_nodes).reshape(xdim, ydim, zdim)

    # Construct the neighborhood indices
    # Perhaps can be done faster, but will be used only once/run.
    if num_neighs == 6:
        # neighbours = np.zeros((num_nodes, num_neighs), dtype=np.int)
        row = np.array([1, -1, 0, 0, 0, 0])
        col = np.array([0, 0, 1, -1, 0, 0])
        pag = np.array([0, 0, 0, 0, 1, -1])
        neighbours_offset = np.array([row, col, pag])

    elif num_neighs == 26:
        neighbours_offset = np.zeros((3, num_neighs), dtype=np.int)
        index = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if (i | j | k):
                        neighbours_offset[:, index] = [i, j, k]
                        index += 1

    elif num_neighs == 124:
        neighbours_offset = np.zeros((3, num_neighs), dtype=np.int)
        index = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    if (i | j | k):
                        neighbours_offset[:, index] = [i, j, k]
                        index += 1

    elif num_neighs == 342:
        neighbours_offset = np.zeros((3, num_neighs), dtype=np.int)
        index = 0
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    if (i | j | k):
                        neighbours_offset[:, index] = [i, j, k]
                        index += 1

    else:
        return 0  # if neighbourhood size != 6/26 return zero

    # Construct NxN adjacency matrix (N : num elems) based on 3d neighbourhood
    neighbours = np.zeros((num_nodes, num_neighs), dtype=np.int)
    index = 0
    for i in range(xdim):
        for j in range(ydim):
            for k in range(zdim):
                indexes_x = np.mod(neighbours_offset[0, :] + i, xdim)
                indexes_y = np.mod(neighbours_offset[1, :] + j, ydim)
                indexes_z = np.mod(neighbours_offset[2, :] + k, zdim)
                neighbours[index, :] = indexes_nodes[indexes_x, indexes_y, indexes_z]
                index += 1

    adjacency = neighbours_to_adjacency(neighbours)
    return adjacency


def neighbours_to_adjacency(neighbours: np.array) -> np.array:
    """
    Create an adjacency matrix, given the neighbour indices
    Input: Nxd neighbourhood, where N is number of nodes
    Output: NxN sparse torch adjacency matrix
    """
    num_nodes, num_neighs = neighbours.shape
    valid_neighbours = (neighbours >= 0)                # Mask for valid neighbours amongst the d-neighbours
    row = np.repeat(np.arange(num_nodes), num_neighs)   # Row indices like in sparse matrix formats
    row = row[valid_neighbours.reshape(-1)]             # Remove non-neighbour row indices
    col = (neighbours * valid_neighbours).reshape(-1)   # Obtain neighbour col indices
    col = col[valid_neighbours.reshape(-1)]             # Remove non-neighbour col indices
    # data = np.ones(col.size)
    # Definition adjacency matrix
    adjacency = sp.csr_matrix((np.ones(col.size, dtype=np.float16), (row, col)), shape=(num_nodes, num_nodes))
    adjacency = adjacency + sp.eye(num_nodes)           # Self connections
    adjacency = adjacency / (num_neighs + 1)
    return adjacency


def compute_onthefly_adjacency(in_features: np.array, num_neighs: int = 26, is_normalise: bool = False
                               ) -> torch.Tensor:
    """
    Given an NxD feature vector, and adjacency matrix in sparse form
    is returned with 'num_neighs' neighbours.
    """
    num_nodes, num_feats = in_features.shape
    if is_normalise:
        in_features = (in_features - in_features.min(0)) * 2.0 / (in_features.max(0) - in_features.min(0)) - 1.0

    pair_dists = pairwise_distances(in_features)
    neighbours = np.argsort(pair_dists, axis=1)[:, :num_neighs]

    # Create sparse torch adjacency from neighbours, similar to 'neighbours_to_adjacency'
    row = torch.LongTensor(np.arange(num_nodes))
    row = row.view(-1, 1).repeat(1, num_neighs).view(-1)
    col = torch.LongTensor(neighbours.reshape(-1))
    indexes = torch.stack((row, col))
    values = torch.FloatTensor(np.ones(len(row)))
    adjacency = torch.sparse.FloatTensor(indexes, values, (num_nodes, num_nodes))
    adjacency = adjacency / (num_neighs + 1)
    return adjacency.cuda()


def compute_onthefly_adjacency_torch(in_features: torch.Tensor, num_neighs: int = 26, is_normalise: bool = False
                                     ) -> torch.Tensor:
    """
    Same as 'compute_onthefly_adjacency', but using torch functions
    """
    num_nodes, num_feats = in_features.shape
    if is_normalise:
        in_features = (in_features - in_features.min(0)[0]) * 2. / (in_features.max(0)[0] - in_features.min(0)[0]) - 1.0

    # Compute Euclidean distance in D-dimensional feature space
    # pair_dists = torch.sum((x.view(-1, 1, num_feats) - x.view(1, -1, num_feats))**2, dim=2)
    x_norm = (in_features ** 2).sum(1).view(-1, 1)
    pair_dists = x_norm + x_norm.view(1, -1) - 2.0 * torch.mm(in_features, torch.transpose(in_features, 0, 1))
    _, indexes_neighs = torch.sort(pair_dists)
    neighbours = indexes_neighs[:, :num_neighs]      # Obtain 'num_neighs' nearest neighbours / node in this space

    # Create sparse torch adjacency from neighbours, similar to 'neighbours_to_adjacency'
    row = torch.LongTensor(np.arange(num_nodes))
    row = row.view(-1, 1).repeat(1, num_neighs).view(-1)
    col = neighbours.contiguous().view(-1)
    indexes = torch.stack((row, col))
    values = torch.FloatTensor(np.ones(len(row)))
    adjacency = torch.sparse.FloatTensor(indexes, values, (num_nodes, num_nodes))
    adjacency = adjacency / (num_neighs + 1)
    return adjacency.cuda()


def compute_onthefly_adjacency_with_attention(in_features: np.array, num_neighs: int = 26, is_normalise: bool = False
                                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given an NxD feature vector, and adjacency matrix in sparse form
    is returned with num_ngbrs neighbours, including attention.
    """
    num_nodes, num_feats = in_features.shape
    if is_normalise:
        in_features = (in_features - in_features.min(0)) * 2.0 / (in_features.max(0) - in_features.min(0)) - 1.0

    pair_dists = pairwise_distances(in_features)
    neighbours = np.argsort(pair_dists, axis=1)[:, :num_neighs]

    # Create Sparse torch adjacency from neighbours, similar to 'neighbours_to_adjacency'
    row = torch.LongTensor(np.arange(num_nodes))
    row = row.view(-1, 1).repeat(1, num_neighs).view(-1)
    col = torch.LongTensor(neighbours.reshape(-1))
    indexes = torch.stack((row, col))
    values = torch.FloatTensor(np.ones(len(row)))
    adjacency = torch.sparse.FloatTensor(indexes, values, (num_nodes, num_nodes))

    num_nonzero = adjacency._nnz()
    values = torch.ones(num_nonzero)
    indexes_0 = torch.arange(num_nonzero)
    indexes = torch.stack((indexes_0, col))
    node2edge_in = torch.sparse.FloatTensor(indexes, values, (num_nonzero, num_nodes))
    indexes = torch.stack((indexes_0, row))
    node2edge_out = torch.sparse.FloatTensor(indexes, values, (num_nonzero, num_nodes))

    return (adjacency.cuda(), node2edge_in.cuda(), node2edge_out.cuda())


class OntheflyAdjacencyLimitCanditsGenerator(object):
    _dist_max_candits_neighs_default = 5
    _dist_jump_nodes_candits_default = None

    def __init__(self, shape_volume: Tuple[int, int, int],
                 dist_max_candits_neighs: int = _dist_max_candits_neighs_default,
                 dist_jump_nodes_candits: int = _dist_jump_nodes_candits_default
                 ) -> None:
        # self._shape_volume = shape_volume
        # self._dist_max_candits_neighs = dist_max_candits_neighs
        # self._dist_jump_nodes_candits = dist_jump_nodes_candits
        self._indexes_neighbours_candits = \
            self._get_indexes_neighbours_candits(shape_volume, dist_max_candits_neighs, dist_jump_nodes_candits)

    def _get_indexes_neighbours_candits(self, shape_volume: Tuple[int, int, int],
                                        dist_max_candits_neighs: int,
                                        dist_jump_nodes_candits: int
                                        ) -> np.array:
        """
        Get the candidate indexes around each node of the image volume,
        within a cubic neighbourhood of size 'dist_max_candits_neighs' in each dir
        """
        (zdim, xdim, ydim) = shape_volume
        num_nodes_total = zdim * xdim * ydim

        if dist_jump_nodes_candits:
            dim_1d_neighs = 2 * dist_max_candits_neighs // (dist_jump_nodes_candits + 1) + 1
        else:
            dim_1d_neighs = 2 * dist_max_candits_neighs + 1

        zdim_neighs = min(dim_1d_neighs, zdim)
        xdim_neighs = min(dim_1d_neighs, xdim)
        ydim_neighs = min(dim_1d_neighs, ydim)
        max_nodes_neighs = zdim_neighs * xdim_neighs * ydim_neighs

        # 32-integer enough to store indexes of volume (512, 512, 512). Vol = 512^3 = 2^27 < 2^31 (max val. 32-int)
        indexes_neighbours_candits = np.full((num_nodes_total, max_nodes_neighs), 0, dtype=np.uint32)

        for iz in range(zdim):
            z_neigh_min = max(0, iz - dist_max_candits_neighs)
            z_neigh_max = min(zdim, iz + dist_max_candits_neighs + 1)
            z_neigh_inds = np.arange(z_neigh_min, z_neigh_max)

            for ix in range(xdim):
                x_neigh_min = max(0, ix - dist_max_candits_neighs)
                x_neigh_max = min(xdim, ix + dist_max_candits_neighs + 1)
                x_neigh_inds = np.arange(x_neigh_min, x_neigh_max)

                for iy in range(ydim):
                    y_neigh_min = max(0, iy - dist_max_candits_neighs)
                    y_neigh_max = min(ydim, iy + dist_max_candits_neighs + 1)
                    y_neigh_inds = np.arange(y_neigh_min, y_neigh_max)

                    inode = (iz * xdim + ix) * ydim + iy
                    num_nodes_neighs = len(z_neigh_inds) * len(x_neigh_inds) * len(y_neigh_inds)

                    indexes_neighbours_candits[inode, :num_nodes_neighs] = \
                        ((z_neigh_inds[:, None] * xdim + x_neigh_inds)[:, :, None] * ydim + y_neigh_inds).reshape(-1)

                    # unused and point to a dummy value stored in the last row of x
                    indexes_neighbours_candits[inode, num_nodes_neighs:] = num_nodes_total
                # endfor
            # endfor
        # endfor

        return indexes_neighbours_candits
        # return indexes_neighbours_candits.T

    def compute(self, in_features: np.array, num_neighs: int = 26, is_normalise: bool = False
                ) -> torch.Tensor:
        """
        Given an NxD feature vector, and adjacency matrix in sparse form
        is returned with 'num_neighs' neighbours.
        """
        num_nodes, num_feats = in_features.shape
        if is_normalise:
            in_features = (in_features - in_features.min(0)) * 2.0 / (in_features.max(0) - in_features.min(0)) - 1.0

        max_dummy_val = 1.0e+06
        max_nodes_neighs = self._indexes_neighbours_candits.shape[1]

        in_features = np.vstack((in_features, np.full((num_feats), max_dummy_val)))

        pair_dists = np.zeros((num_nodes, max_nodes_neighs), dtype=np.float)
        for i in range(num_nodes):
            feats_candits = in_features[self._indexes_neighbours_candits[i], :]
            pair_dists[i, :] = np.sum((in_features[i] - feats_candits) ** 2, axis=1)

        neighbours = np.argsort(pair_dists, axis=1)[:, :num_neighs]
        for i in range(num_nodes):
            neighbours[i, :] = self._indexes_neighbours_candits[i, neighbours[i, :]]

        # Create sparse torch adjacency from neighbours, similar to 'neighbours_to_adjacency'
        row = torch.LongTensor(np.arange(num_nodes))
        row = row.view(-1, 1).repeat(1, num_neighs).view(-1)
        col = torch.LongTensor(neighbours.reshape(-1))
        indexes = torch.stack((row, col))
        values = torch.FloatTensor(np.ones(len(row)))
        adjacency = torch.sparse.FloatTensor(indexes, values, (num_nodes, num_nodes))
        adjacency = adjacency / (num_neighs + 1)
        return adjacency.cuda()

    def compute_with_attention(self, in_features: np.array, num_neighs: int = 26, is_normalise: bool = False
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given an NxD feature vector, and adjacency matrix in sparse form
        is returned with num_ngbrs neighbours, including attention.
        """
        num_nodes, num_feats = in_features.shape
        if is_normalise:
            in_features = (in_features - in_features.min(0)) * 2.0 / (in_features.max(0) - in_features.min(0)) - 1.0

        max_dummy_val = 1.0e+06
        max_nodes_neighs = self._indexes_neighbours_candits.shape[1]

        in_features = np.vstack((in_features, np.full((num_feats), max_dummy_val)))

        pair_dists = np.zeros((num_nodes, max_nodes_neighs), dtype=np.float)
        for i in range(num_nodes):
            feats_candits = in_features[self._indexes_neighbours_candits[i], :]
            pair_dists[i, :] = np.sum((in_features[i] - feats_candits) ** 2, axis=1)

        neighbours = np.argsort(pair_dists, axis=1)[:, :num_neighs]
        for i in range(num_nodes):
            neighbours[i, :] = self._indexes_neighbours_candits[i, neighbours[i, :]]

        # Create sparse torch adjacency from neighbours, similar to 'neighbours_to_adjacency'
        row = torch.LongTensor(np.arange(num_nodes))
        row = row.view(-1, 1).repeat(1, num_neighs).view(-1)
        col = torch.LongTensor(neighbours.reshape(-1))
        indexes = torch.stack((row, col))
        values = torch.FloatTensor(np.ones(len(row)))
        adjacency = torch.sparse.FloatTensor(indexes, values, (num_nodes, num_nodes))

        num_nonzero = adjacency._nnz()
        values = torch.ones(num_nonzero)
        indexes_0 = torch.arange(num_nonzero)
        indexes = torch.stack((indexes_0, col))
        node2edge_in = torch.sparse.FloatTensor(indexes, values, (num_nonzero, num_nodes))
        indexes = torch.stack((indexes_0, row))
        node2edge_out = torch.sparse.FloatTensor(indexes, values, (num_nonzero, num_nodes))

        return (adjacency.cuda(), node2edge_in.cuda(), node2edge_out.cuda())


def vol2nodes(volume: np.array) -> np.array:
    """
    Given a 3D volume with voxel level features, transform it
    into a 2D feature matrix to be used for GNNs.
    volume: X x Y x Z x F
    output: M x F, where M=prod(X,Y,Z)
    """
    pass
