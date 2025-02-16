import torch
from scipy.spatial import KDTree
import firedrake as df

import pickle
import os

from functools import singledispatchmethod

from .util import *

class NystromBasis:
    def __init__(self,kernel,mean_function,mesh,rank=1000):
        self.kernel = kernel
        self.mesh = mesh
        self.mean_function = mean_function

        self.rank = rank

        V_dg = df.VectorFunctionSpace(mesh,"DG",0)
        X_cell = df.interpolate(mesh.coordinates,V_dg)
        self.X_ = torch.from_numpy(X_cell.dat.data[:])#.to(torch.float)
        
        self.n = self.X_.shape[0]
        F = torch.zeros(self.n,rank)
        d = torch.ones(self.n)*kernel.amplitude
        self.pivots = []
        for i in range(self.rank):
            pivot = torch.argmax(d)
            self.pivots.append(pivot)
            g = self.kernel(self.X_,self.X_[pivot].reshape(1,2),return_amplitude_root=False)
            g -= (F[:,:i] @ F[pivot,:i].T).reshape(-1,1)
            F[:,i] = (g/torch.sqrt(g[pivot]+1e-16)).ravel()
            d -= F[:,i]**2
            d[d<0.] = 0.
            print(d.max())

        print(f'maximum diagonal residual: {d.max()}')

        Kstar = self.kernel(self.X_,self.X_[self.pivots])
        Kss = self.kernel(self.X_[self.pivots],self.X_[self.pivots])
        self.X_ = self.X_.to(torch.float)
        l,u = torch.linalg.eigh(Kss + 1e-10*torch.eye(Kss.shape[0]))
        self.F = (Kstar @ u/torch.sqrt(l) @ u.T).to(torch.float)
        self.h = self.mean_function(self.X_)
        self.Psi = torch.hstack((self.F,self.h))


    def set_training_data(self,X_train,Z_train,sigma2_obs):
        self.sigma2_obs = sigma2_obs

        self.M,self.mask = self.get_fem_map(X_train)
        self.X_train = X_train[self.mask]
        self.Z_train = Z_train[self.mask]

        self.Psi_train = self.M @ self.Psi

    def get_fem_map(self,X):
        indices = [[],[]]
        values = []
        mask = []
        count = 0
        for i,x in enumerate(X):
            c = self.mesh.locate_cell(x,tolerance=1e-4)
            if c is not None:
                indices[0].append(count)
                indices[1].append(c)
                values.append(1.)
                mask.append(i)
                count += 1

        M = torch.sparse_coo_tensor(torch.tensor(indices),torch.tensor(values),(count,self.X_.shape[0]))
        return M,mask

    def condition(self):
        Iv = torch.ones(self.rank + self.mean_function.n_dofs)
        Iv[self.rank:] = 0.0
        I = torch.diag(Iv)

        Tau_post = self.Psi_train.T  /self.sigma2_obs @ self.Psi_train + I

        L = torch.linalg.cholesky(Tau_post)

        self.R = torch.linalg.solve_triangular(L,torch.eye(L.shape[0]),upper=False).T
        self.w_post = self.R @ (self.R.T @ (self.Psi_train.T @ (self.Z_train/self.sigma2_obs).unsqueeze(-1) ))

    def sample(self,mode='prior',mean=True,test=True):
        if test:
            Psi = self.Psi_test
        else:
            Psi = self.Psi_train
        return Psi @ self.w_post

    def marginal_variance(self,mode='prior',test=True):
        if test:
            Psi = self.Psi_test
        else:
            Psi = self.Psi_train
        return ((Psi @ self.R)**2).sum(axis=1)**0.5

    @singledispatchmethod
    def set_test_data(self,_):
        raise NotImplementedError("torch tensor or firedrake mesh + function space")
    
    @set_test_data.register
    def _(self,X_test: torch.Tensor):
        self.X_test = X_test
        M,mask = self.get_fem_map(X_test)
        self.Psi_test = M @ self.Psi

    @set_test_data.register
    def _(self,mesh: df.mesh.MeshGeometry):

        self.X_test = torch.tensor(df.interpolate(df.SpatialCoordinate(mesh),df.VectorFunctionSpace(mesh,"DG",0)).dat.data[:],dtype=torch.float)
        M,mask = self.get_fem_map(self.X_test)
        self.Psi_test = M @ self.Psi

    def save(self,path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pickle.dump(self,open(path,'wb'))
