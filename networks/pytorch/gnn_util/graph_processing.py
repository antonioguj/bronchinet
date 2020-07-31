
#from networks.pytorch.gnn_util.gnn_utilities import row_normalize, sparse_mx_to_torch_sparse_tensor
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch
from sklearn.metrics.pairwise import pairwise_distances
torch.manual_seed(2017)



def ngbrs2Adj(ngbrs):
    """ Create an adjacency matrix, given the neighbour indices
    Input: Nxd neighbourhood, where N is number of nodes
    Output: NxN sparse torch adjacency matrix
    """
    N, d = ngbrs.shape
    validNgbrs = (ngbrs >= 0) # Mask for valid neighbours amongst the d-neighbours
    row = np.repeat(np.arange(N),d) # Row indices like in sparse matrix formats
    row = row[validNgbrs.reshape(-1)] #Remove non-neighbour row indices
    col = (ngbrs*validNgbrs).reshape(-1) # Obtain nieghbour col indices
    col = col[validNgbrs.reshape(-1)] # Remove non-neighbour col indices
    #data = np.ones(col.size)
    adj = sp.csr_matrix((np.ones(col.size, dtype=np.float16),(row, col)), shape=(N, N)) # Make adj matrix
    adj = adj + sp.eye(N) # Self connections
    adj = adj/(d+1)

    return adj



def compute_ontheflyAdjacency_Torch(x, numNgbrs=26, normalise=False):
    """
    Same as otfAdj but implemented in PyTorch
    """
    N,d = x.shape
    if normalise:
        x = (x - x.min(0)[0]) * 2 / (x.max(0)[0] - x.min(0)[0]) - 1

    # Compute Euclidean distance in d- dimensional feature space
    # dist = torch.sum( (x.view(-1,1,d) - x.view(1,-1,d))**2, dim=2)
    x_norm = (x**2).sum(1).view(-1,1)
    dist = x_norm + x_norm.view(1,-1) - 2.0*torch.mm(x, torch.transpose(x, 0, 1))
    _, idx = torch.sort(dist)
    ngbrs = idx[:,:numNgbrs] # Obtain numNgbrs nearest neighbours/node in this space

    # Create Sparse torch adjacency from neighbours, similar to ngbrs2Adj()
    row = torch.LongTensor(np.arange(N))
    row = row.view(-1,1).repeat(1,numNgbrs).view(-1)
    col = ngbrs.contiguous().view(-1)
    idx = torch.stack((row,col))
    val = torch.FloatTensor(np.ones(len(row)))
    adj = torch.sparse.FloatTensor(idx,val,(N,N))/(numNgbrs+1)

    return adj.cuda()



def compute_ontheflyAdjacency(x, numNgbrs=26, normalise=False):
    """
    Given an NxD feature vector, and adjacency matrix in sparse form 
    is returned with numNgbrs neighbours.
    """
    N,d = x.shape
    if normalise:
        x = (x - x.min(0)) * 2 / (x.max(0) - x.min(0)) - 1

    dist = pairwise_distances(x)
    ngbrs = np.argsort(dist, axis=1)[:, :numNgbrs]

    # Create Sparse torch adjacency from neighbours, similar to ngbrs2Adj()
    row = torch.LongTensor(np.arange(N))
    row = row.view(-1,1).repeat(1,numNgbrs).view(-1)
    col = torch.LongTensor(ngbrs.reshape(-1))
    idx = torch.stack((row,col))
    val = torch.FloatTensor(np.ones(len(row)))
    adj = torch.sparse.FloatTensor(idx,val,(N,N))/(numNgbrs+1)

    return adj.cuda()



def compute_ontheflyAdjacency_with_attention_layers(x, numNgbrs=26, normalise=False):
    """
    Given an NxD feature vector, and adjacency matrix in sparse form
    is returned with numNgbrs neighbours.
    """
    N, d = x.shape
    if normalise:
        x = (x - x.min(0)) * 2 / (x.max(0) - x.min(0)) - 1

    dist = pairwise_distances(x)
    ngbrs = np.argsort(dist, axis=1)[:, :numNgbrs]

    # Create Sparse torch adjacency from neighbours, similar to ngbrs2Adj()
    row = torch.LongTensor(np.arange(N))
    row = row.view(-1,1).repeat(1,numNgbrs).view(-1)
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



class GenOntheflyAdjacency_NeighCandit(object):
    dist_neigh_max_default = 5
    dist_jump_nodes_default = None

    def __init__(self, image_shape,
                 dist_neigh_max= dist_neigh_max_default,
                 dist_jump_nodes= dist_jump_nodes_default):
        # self.image_shape = image_shape
        # self.dist_neigh_max = dist_neigh_max
        # self.dist_jump_nodes = dist_jump_nodes
        self.indexs_neighcands = self.indexes_nodes_neighcube_around_node(image_shape,
                                                                          dist_neigh_max,
                                                                          dist_jump_nodes)

    def compute(self, x, numNgbrs=26, normalise=False):
        """
        Adjacency matrix is computed with pairwise distances with a maximum of candidate nodes.
        Candidate nodes are within a cube around of the given node, with a max distance.
        It's possible to indicate a distance (or jump) in between the candidate nodes.
        """
        N, d = x.shape
        if normalise:
            x = (x - x.min(0)) * 2 / (x.max(0) - x.min(0)) - 1

        max_dummy_val = 1.0e+06
        nummax_neighcands = self.indexs_neighcands.shape[1]
        #nummax_neighcands = self.indexs_neighcands.shape[0]

        x = np.vstack((x, np.full((d), max_dummy_val)))

        dist = np.zeros((N, nummax_neighcands), dtype=float)
        #dist = np.zeros((nummax_neighcands, N), dtype=float)
        for i in range(N):
            x_candit = x[self.indexs_neighcands[i], :]
            dist[i, :] = np.sum((x[i] - x_candit)**2, axis=1)
        #endfor
        # for i in range(nummax_neighcands):
        #     x_candit = x[self.indexs_neighcands[:, i], :]
        #     dist[:, i] = np.sum((x[:-1] - x_candit)**2, axis=1)
        # # endfor
        # for i in range(nummax_neighcands):
        #     x_candit = x[self.indexs_neighcands[i], :]
        #     dist[i, :] = np.sum((x[:-1] - x_candit)**2, axis=1)
        # #endfor
        # for i in range(N):
        #     x_candit = x[self.indexs_neighcands[:, i], :]
        #     dist[:, i] = np.sum((x[i] - x_candit)**2, axis=1)
        # #endfor

        ngbrs = np.argsort(dist, axis=1)[:, :numNgbrs]
        for i in range(N):
            ngbrs[i, :] = self.indexs_neighcands[i, ngbrs[i,:]]
        #endfor

        # Create Sparse torch adjacency from neighbours, similar to ngbrs2Adj()
        row = torch.LongTensor(np.arange(N))
        row = row.view(-1, 1).repeat(1, numNgbrs).view(-1)
        col = torch.LongTensor(ngbrs.reshape(-1))
        idx = torch.stack((row, col))
        val = torch.FloatTensor(np.ones(len(row)))
        adj = torch.sparse.FloatTensor(idx, val, (N, N))/(numNgbrs+1)

        return adj.cuda()


    def compute_with_attention_layers(self, x, numNgbrs=26, normalise=False):
        N, d = x.shape
        if normalise:
            x = (x - x.min(0)) * 2 / (x.max(0) - x.min(0)) - 1

        max_dummy_val = 1.0e+06
        nummax_neighcands = self.indexs_neighcands.shape[1]

        x = np.vstack((x, np.full((d), max_dummy_val)))

        dist = np.zeros((N, nummax_neighcands), dtype=float)
        for i in range(N):
            x_candit = x[self.indexs_neighcands[i], :]
            dist[i, :] = np.sum((x[i] - x_candit)**2, axis=1)
        #endfor

        ngbrs = np.argsort(dist, axis=1)[:, :numNgbrs]
        for i in range(N):
            ngbrs[i, :] = self.indexs_neighcands[i, ngbrs[i,:]]
        #endfor

        # Create Sparse torch adjacency from neighbours, similar to ngbrs2Adj()
        row = torch.LongTensor(np.arange(N))
        row = row.view(-1, 1).repeat(1, numNgbrs).view(-1)
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


    def indexes_nodes_neighcube_around_node(self, image_shape,
                                            dist_neigh_max,
                                            dist_jump_nodes):
        (zdim, xdim, ydim) = image_shape
        volimg = zdim*xdim*ydim

        if dist_jump_nodes:
            dim1D_neigh = 2*dist_neigh_max // (dist_jump_nodes+1) + 1
        else:
            dim1D_neigh = 2*dist_neigh_max + 1

        zdim_neigh = min(dim1D_neigh, zdim)
        xdim_neigh = min(dim1D_neigh, xdim)
        ydim_neigh = min(dim1D_neigh, ydim)
        volmax_neigh = zdim_neigh*xdim_neigh*ydim_neigh

        if dist_jump_nodes:
            def funCalc_indexes_neighs_nodes(x_min, x_max, jump):
                return np.arange(x_min, x_max, jump)
        else:
            def funCalc_indexes_neighs_nodes(x_min, x_max):
                return np.arange(x_min, x_max)

        # 32-bit INTEGER ENOUGH TO STORE INDEXES OF IMAGE (512, 512, 512) := vol = 512^3, 2^27 < 2^31 (max val. 32-bit int)
        indexs_neighcanditsnodes = np.full((volimg, volmax_neigh), 0, dtype=np.uint32)

        for iz in range(zdim):
            z_neigh_min = max(0, iz - dist_neigh_max)
            z_neigh_max = min(zdim, iz + dist_neigh_max + 1)
            z_neigh_inds = np.arange(z_neigh_min, z_neigh_max)
            #z_neigh_inds = funCalc_indexes_neighs_nodes(z_neigh_min, z_neigh_max, dist_jump_nodes+1)

            for ix in range(xdim):
                x_neigh_min = max(0, ix - dist_neigh_max)
                x_neigh_max = min(xdim, ix + dist_neigh_max + 1)
                x_neigh_inds = np.arange(x_neigh_min, x_neigh_max)
                #x_neigh_index = funCalc_indexes_neighs_nodes(x_neigh_min, x_neigh_max, dist_jump_nodes+1)

                for iy in range(ydim):
                    y_neigh_min = max(0, iy - dist_neigh_max)
                    y_neigh_max = min(ydim, iy + dist_neigh_max + 1)
                    y_neigh_inds = np.arange(y_neigh_min, y_neigh_max)
                    #y_neigh_inds = funCalc_indexes_neighs_nodes(y_neigh_min, y_neigh_max, dist_jump_nodes+1)

                    inode = (iz * xdim + ix) * ydim + iy
                    volmax_neigh = len(z_neigh_inds)*len(x_neigh_inds)*len(y_neigh_inds)

                    indexs_neighcanditsnodes[inode, :volmax_neigh] = ((z_neigh_inds[:,None] * xdim + x_neigh_inds)[:,:,None]
                                                                      * ydim + y_neigh_inds).reshape(-1)
                    # unused and point to a dummy value stored in the last row of x
                    indexs_neighcanditsnodes[inode, volmax_neigh:] = volimg
                # endfor
            # endfor
        # endfor

        return indexs_neighcanditsnodes
        #return indexs_neighcanditsnodes.T



def makeAdjacency(volShape, numNgbrs=26):#, resolution=4):
	"""
	Given the input volume size, constructs
	an adjacency matrix either of 6/26 neighbors.
	volShape: Tuple with 3D shape
	neighbours: 6/26
    resolution: reduction in resolution factor: 2/4/8/16/32
	"""	
	volShape = np.array((np.array(volShape)), dtype=int)
	#volShape = np.array((np.array(volShape)/resolution), dtype=int)
	ydim = volShape[2]
	xdim = volShape[1]
	zdim = volShape[0]
	numEl = volShape.prod()
	
	idxVol = np.arange(numEl).reshape(xdim,ydim,zdim)
 
	# Construct the neighborhood indices
	# Perhaps can be done faster, but will be used only once/run.
	if numNgbrs == 6:

		ngbrs = np.zeros((numEl, numNgbrs), dtype=int)
		row = np.array([1,-1,0,0,0,0])
		col = np.array([0,0,1,-1,0,0])
		pag = np.array([0,0,0,0,1,-1])
		ngbrOffset = np.array([row,col,pag])

	elif numNgbrs == 26:
		idx = 0
		ngbrOffset = np.zeros((3,numNgbrs),dtype=int) 
		for i in range(-1,2):
		    for j in range(-1,2):
		        for k in range(-1,2):
            			if(i | j | k):
			                ngbrOffset[:,idx] = [i,j,k]
			                idx+=1
	elif numNgbrs == 124:
		idx = 0
		ngbrOffset = np.zeros((3,numNgbrs),dtype=int) 
		for i in range(-2,3):
		    for j in range(-2,3):
		        for k in range(-2,3):
            			if(i | j | k):
			                ngbrOffset[:,idx] = [i,j,k]
			                idx+=1
	elif numNgbrs == 342:
		idx = 0
		ngbrOffset = np.zeros((3,numNgbrs),dtype=int) 
		for i in range(-3,4):
		    for j in range(-3,4):
		        for k in range(-3,4):
            			if(i | j | k):
			                ngbrOffset[:,idx] = [i,j,k]
			                idx+=1
	else:
		return 0 # if neighbourhood size != 6/26 return zero

	# Construct numEl x numEl adj matrix based on 3d neighbourhood	
	idx = 0
	ngbrs = np.zeros((numEl, numNgbrs), dtype=int)
	
	for i in range(xdim):
		for j in range(ydim):
			for k in range(zdim):
				xIdx = np.mod(ngbrOffset[0,:]+i,xdim)
				yIdx = np.mod(ngbrOffset[1,:]+j,ydim)
				zIdx = np.mod(ngbrOffset[2,:]+k,zdim)
				ngbrs[idx,:] = idxVol[xIdx, yIdx, zIdx]
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
