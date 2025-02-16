import torch
import pyproj
import numpy as np

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def project_array(coordinates, from_epsg=4326, to_epsg=3338, always_xy=True):
    """
    Project a numpy (n,2) array from <from_epsg> to <to_epsg>
    Returns the projected numpy (n,2) array.
    """
    tform = pyproj.Transformer.from_crs(crs_from=from_epsg, crs_to=to_epsg, always_xy=always_xy)
    fx, fy = tform.transform(coordinates[:,0], coordinates[:,1])
    # Re-create (n,2) coordinates
    return np.dstack([fx, fy])[0]

def batch_vec(M):
    return torch.permute(M,[0,2,1]).reshape(M.shape[0],-1)
    
def batch_mat(v,shape):
    return torch.permute(v.reshape(v.shape[0],shape[1],shape[0]),[0,2,1]) 

def batch_mm(matrix, vector_batch):
    return matrix.mm(vector_batch.T).T

def batch_kron(A,B,z):
    return batch_vec(B @ batch_mat(z,(B.shape[1],A.shape[1])) @ A.T)




