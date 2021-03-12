
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch
from sklearn.metrics.pairwise import pairwise_distances
torch.manual_seed(2017)

#from models.pytorch.gnn_util.gnn_utilities import row_normalize, sparse_mx_to_torch_sparse_tensor


def ngbrs2Adj(ngbrs):
    """ Create an adjacency matrix, given the neighbour indices
    Input: Nxd neighbourhood, where N is number of nodes
    Output: NxN sparse torch adjacency matrix
    """
    N, d = ngbrs.shape
    valid_ngbrs = (ngbrs >= 0) # Mask for valid neighbours amongst the d-neighbours
    row = np.repeat(np.arange(N),d) # Row indices like in sparse matrix formats
    row = row[valid_ngbrs.reshape(-1)] #Remove non-neighbour row indices
    col = (ngbrs*valid_ngbrs).reshape(-1) # Obtain nieghbour col indices
    col = col[valid_ngbrs.reshape(-1)] # Remove non-neighbour col indices
    #data = np.ones(col.size)
    adj = sp.csr_matrix((np.ones(col.size, dtype=np.float16),(row, col)), shape=(N, N)) # Make adj matrix
    adj = adj + sp.eye(N) # Self connections
    adj = adj/(d+1)

    return adj


def compute_onthefly_adjacency(x, num_ngbrs=26, normalise=False):
    """
    Given an NxD feature vector, and adjacency matrix in sparse form
    is returned with num_ngbrs neighbours.
    """
    N,d = x.shape
    if normalise:
        x = (x - x.min(0)) * 2 / (x.max(0) - x.min(0)) - 1

    dist = pairwise_distances(x)
    ngbrs = np.argsort(dist, axis=1)[:, :num_ngbrs]

    # Create Sparse torch adjacency from neighbours, similar to ngbrs2Adj()
    row = torch.LongTensor(np.arange(N))
    row = row.view(-1,1).repeat(1, num_ngbrs).view(-1)
    col = torch.LongTensor(ngbrs.reshape(-1))
    idx = torch.stack((row,col))
    val = torch.FloatTensor(np.ones(len(row)))
    adj = torch.sparse.FloatTensor(idx,val,(N,N))/(num_ngbrs + 1)

    return adj.cuda()


def compute_onthefly_adjacency_torch(x, num_ngbrs=26, normalise=False):
    """
    Same as 'compute_onthefly_adjacency' but implemented in PyTorch
    """
    N,d = x.shape
    if normalise:
        x = (x - x.min(0)[0]) * 2 / (x.max(0)[0] - x.min(0)[0]) - 1

    # Compute Euclidean distance in d- dimensional feature space
    # dist = torch.sum( (x.view(-1,1,d) - x.view(1,-1,d))**2, dim=2)
    x_norm = (x**2).sum(1).view(-1,1)
    dist = x_norm + x_norm.view(1,-1) - 2.0*torch.mm(x, torch.transpose(x, 0, 1))
    _, idx = torch.sort(dist)
    ngbrs = idx[:, :num_ngbrs] # Obtain num_ngbrs nearest neighbours/node in this space

    # Create Sparse torch adjacency from neighbours, similar to ngbrs2Adj()
    row = torch.LongTensor(np.arange(N))
    row = row.view(-1,1).repeat(1, num_ngbrs).view(-1)
    col = ngbrs.contiguous().view(-1)
    idx = torch.stack((row,col))
    val = torch.FloatTensor(np.ones(len(row)))
    adj = torch.sparse.FloatTensor(idx,val,(N,N))/(num_ngbrs + 1)

    return adj.cuda()


def compute_onthefly_adjacency_with_attention(x, num_ngbrs=26, normalise=False):
    """
    Given an NxD feature vector, and adjacency matrix in sparse form
    is returned with num_ngbrs neighbours.
    """
    N, d = x.shape
    if normalise:
        x = (x - x.min(0)) * 2 / (x.max(0) - x.min(0)) - 1

    dist = pairwise_distances(x)
    ngbrs = np.argsort(dist, axis=1)[:, :num_ngbrs]

    # Create Sparse torch adjacency from neighbours, similar to ngbrs2Adj()
    row = torch.LongTensor(np.arange(N))
    row = row.view(-1,1).repeat(1, num_ngbrs).view(-1)
    col = torch.LongTensor(ngbrs.reshape(-1))
    idx = torch.stack((row,col))
    val = torch.FloatTensor(np.ones(len(row)))
    adj = torch.sparse.FloatTensor(idx,val,(N,N))

    nnz = adj._nnz()
    val = torch.ones(nnz)
    idx0 = torch.arange(nnz)

    idx = torch.stack((idx0,col))
    n2e_in = torch.sparse.FloatTensor(idx,val,(nnz,N))

    idx = torch.stack((idx0,row))
    n2e_out = torch.sparse.FloatTensor(idx,val,(nnz,N))

    return adj.cuda(), n2e_in.cuda(), n2e_out.cuda()


class GenOntheflyAdjacencyNeighCandits(object):
    _dist_neigh_max_default = 5
    _dist_jump_nodes_default = None

    def __init__(self, image_shape,
                 dist_neigh_max=_dist_neigh_max_default,
                 dist_jump_nodes=_dist_jump_nodes_default):
        # self._image_shape = image_shape
        # self._dist_neigh_max = dist_neigh_max
        # self._dist_jump_nodes = dist_jump_nodes
        self._indexes_neigh_candits = self.indexes_nodes_neighcube_around_node(image_shape, dist_neigh_max, dist_jump_nodes)

    def compute(self, x, num_ngbrs=26, normalise=False):
        """
        Adjacency matrix is computed with pairwise distances with a maximum of candidate nodes.
        Candidate nodes are within a cube around of the given node, with a max distance.
        It's possible to indicate a distance (or jump) in between the candidate nodes.
        """
        N, d = x.shape
        if normalise:
            x = (x - x.min(0)) * 2 / (x.max(0) - x.min(0)) - 1

        max_dummy_val = 1.0e+06
        num_max_neigh_candits = self._indexes_neigh_candits.shape[1]
        #num_max_neigh_candits = self._indexes_neigh_candits.shape[0]

        x = np.vstack((x, np.full((d), max_dummy_val)))

        dist = np.zeros((N, num_max_neigh_candits), dtype=float)
        #dist = np.zeros((num_max_neigh_candits, N), dtype=float)
        for i in range(N):
            x_candit = x[self._indexes_neigh_candits[i], :]
            dist[i, :] = np.sum((x[i] - x_candit)**2, axis=1)

        # for i in range(num_max_neigh_candits):
        #     x_candit = x[self._indexes_neigh_candits[:, i], :]
        #     dist[:, i] = np.sum((x[:-1] - x_candit)**2, axis=1)

        # for i in range(num_max_neigh_candits):
        #     x_candit = x[self._indexes_neigh_candits[i], :]
        #     dist[i, :] = np.sum((x[:-1] - x_candit)**2, axis=1)

        # for i in range(N):
        #     x_candit = x[self._indexes_neigh_candits[:, i], :]
        #     dist[:, i] = np.sum((x[i] - x_candit)**2, axis=1)
        # #endfor

        ngbrs = np.argsort(dist, axis=1)[:, :num_ngbrs]
        for i in range(N):
            ngbrs[i, :] = self._indexes_neigh_candits[i, ngbrs[i, :]]
        #endfor

        # Create Sparse torch adjacency from neighbours, similar to ngbrs2Adj()
        row = torch.LongTensor(np.arange(N))
        row = row.view(-1, 1).repeat(1, num_ngbrs).view(-1)
        col = torch.LongTensor(ngbrs.reshape(-1))
        idx = torch.stack((row, col))
        val = torch.FloatTensor(np.ones(len(row)))
        adj = torch.sparse.FloatTensor(idx, val, (N, N))/(num_ngbrs + 1)

        return adj.cuda()

    def compute_with_attention(self, x, num_ngbrs=26, normalise=False):
        N, d = x.shape
        if normalise:
            x = (x - x.min(0)) * 2 / (x.max(0) - x.min(0)) - 1

        max_dummy_val = 1.0e+06
        num_max_neigh_candits = self._indexes_neigh_candits.shape[1]

        x = np.vstack((x, np.full((d), max_dummy_val)))

        dist = np.zeros((N, num_max_neigh_candits), dtype=float)
        for i in range(N):
            x_candit = x[self._indexes_neigh_candits[i], :]
            dist[i, :] = np.sum((x[i] - x_candit)**2, axis=1)

        ngbrs = np.argsort(dist, axis=1)[:, :num_ngbrs]
        for i in range(N):
            ngbrs[i, :] = self._indexes_neigh_candits[i, ngbrs[i, :]]

        # Create Sparse torch adjacency from neighbours, similar to ngbrs2Adj()
        row = torch.LongTensor(np.arange(N))
        row = row.view(-1, 1).repeat(1, num_ngbrs).view(-1)
        col = torch.LongTensor(ngbrs.reshape(-1))
        idx = torch.stack((row, col))
        val = torch.FloatTensor(np.ones(len(row)))
        adj = torch.sparse.FloatTensor(idx, val, (N, N))

        nnz = adj._nnz()
        val = torch.ones(nnz)
        idx0 = torch.arange(nnz)

        idx = torch.stack((idx0, col))
        n2e_in = torch.sparse.FloatTensor(idx, val, (nnz, N))

        idx = torch.stack((idx0, row))
        n2e_out = torch.sparse.FloatTensor(idx, val, (nnz, N))

        return adj.cuda(), n2e_in.cuda(), n2e_out.cuda()


    def indexes_nodes_neighcube_around_node(self, image_shape, dist_neigh_max, dist_jump_nodes):
        (zdim, xdim, ydim) = image_shape
        vol_img = zdim*xdim*ydim

        if dist_jump_nodes:
            dim1D_neigh = 2*dist_neigh_max // (dist_jump_nodes+1) + 1
        else:
            dim1D_neigh = 2*dist_neigh_max + 1

        zdim_neigh = min(dim1D_neigh, zdim)
        xdim_neigh = min(dim1D_neigh, xdim)
        ydim_neigh = min(dim1D_neigh, ydim)
        max_vol_neigh = zdim_neigh*xdim_neigh*ydim_neigh

        # if dist_jump_nodes:
        #     def fun_calc_indexes_neighs_nodes(x_min, x_max, jump):
        #         return np.arange(x_min, x_max, jump)
        # else:
        #     def fun_calc_indexes_neighs_nodes(x_min, x_max):
        #         return np.arange(x_min, x_max)

        # 32-bit INTEGER ENOUGH TO STORE INDEXES OF IMAGE (512, 512, 512) := vol = 512^3, 2^27 < 2^31 (max val. 32-bit int)
        indexes_neigh_candits = np.full((vol_img, max_vol_neigh), 0, dtype=np.uint32)

        for iz in range(zdim):
            z_neigh_min = max(0, iz - dist_neigh_max)
            z_neigh_max = min(zdim, iz + dist_neigh_max + 1)
            z_neigh_inds = np.arange(z_neigh_min, z_neigh_max)
            # z_neigh_inds = fun_calc_indexes_neighs_nodes(z_neigh_min, z_neigh_max, dist_jump_nodes+1)

            for ix in range(xdim):
                x_neigh_min = max(0, ix - dist_neigh_max)
                x_neigh_max = min(xdim, ix + dist_neigh_max + 1)
                x_neigh_inds = np.arange(x_neigh_min, x_neigh_max)
                # x_neigh_index = fun_calc_indexes_neighs_nodes(x_neigh_min, x_neigh_max, dist_jump_nodes+1)

                for iy in range(ydim):
                    y_neigh_min = max(0, iy - dist_neigh_max)
                    y_neigh_max = min(ydim, iy + dist_neigh_max + 1)
                    y_neigh_inds = np.arange(y_neigh_min, y_neigh_max)
                    # y_neigh_inds = fun_calc_indexes_neighs_nodes(y_neigh_min, y_neigh_max, dist_jump_nodes+1)

                    inode = (iz * xdim + ix) * ydim + iy
                    max_vol_neigh = len(z_neigh_inds)*len(x_neigh_inds)*len(y_neigh_inds)

                    indexes_neigh_candits[inode, :max_vol_neigh] = ((z_neigh_inds[:,None] * xdim + x_neigh_inds)[:,:,None]
                                                                      * ydim + y_neigh_inds).reshape(-1)
                    # unused and point to a dummy value stored in the last row of x
                    indexes_neigh_candits[inode, max_vol_neigh:] = vol_img

        return indexes_neigh_candits
        #return indexes_neigh_candits.T


def make_adjacency(vol_shape, num_ngbrs=26):    #, resolution=4):
    """
    Given the input volume size, constructs an adjacency matrix either of 6/26 neighbors.
    vol_shape: Tuple with 3D shape
    neighbours: 6/26
    resolution: reduction in resolution factor: 2/4/8/16/32
    """
    vol_shape = np.array((np.array(vol_shape)), dtype=int)
    # vol_shape = np.array((np.array(vol_shape)/resolution), dtype=int)
    ydim = vol_shape[2]
    xdim = vol_shape[1]
    zdim = vol_shape[0]
    num_elems = vol_shape.prod()
    idxs_vol = np.arange(num_elems).reshape(xdim, ydim, zdim)

    # Construct the neighborhood indices
    # Perhaps can be done faster, but will be used only once/run.
    if num_ngbrs == 6:
        ngbrs = np.zeros((num_elems, num_ngbrs), dtype=int)
        row = np.array([1, -1, 0, 0, 0, 0])
        col = np.array([0, 0, 1, -1, 0, 0])
        pag = np.array([0, 0, 0, 0, 1, -1])
        ngbr_offset = np.array([row, col, pag])

    elif num_ngbrs == 26:
        ngbr_offset = np.zeros((3, num_ngbrs), dtype=int)
        idx = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if (i | j | k):
                        ngbr_offset[:, idx] = [i, j, k]
                        idx += 1

    elif num_ngbrs == 124:
        ngbr_offset = np.zeros((3, num_ngbrs), dtype=int)
        idx = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    if (i | j | k):
                        ngbr_offset[:, idx] = [i, j, k]
                        idx += 1

    elif num_ngbrs == 342:
        ngbr_offset = np.zeros((3, num_ngbrs), dtype=int)
        idx = 0
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    if (i | j | k):
                        ngbr_offset[:, idx] = [i, j, k]
                        idx += 1

    else:
        return 0  # if neighbourhood size != 6/26 return zero

    # Construct num_elems x num_elems adj matrix based on 3d neighbourhood
    ngbrs = np.zeros((num_elems, num_ngbrs), dtype=int)
    idx = 0
    for i in range(xdim):
        for j in range(ydim):
            for k in range(zdim):
                xIdx = np.mod(ngbr_offset[0, :] + i, xdim)
                yIdx = np.mod(ngbr_offset[1, :] + j, ydim)
                zIdx = np.mod(ngbr_offset[2, :] + k, zdim)
                ngbrs[idx, :] = idxs_vol[xIdx, yIdx, zIdx]
                idx += 1

    adj = ngbrs2Adj(ngbrs)

    return adj

def vol2nodes(volume):
    """
    Given a 3D volume with voxel level features, transform it
    into a 2D feature matrix to be used for GNNs.
    volume: X x Y x Z x F
    output: M x F, where M=prod(X,Y,Z)
    """