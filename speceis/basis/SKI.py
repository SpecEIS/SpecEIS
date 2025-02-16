import torch
from scipy.spatial import KDTree
import firedrake as df

import pickle
import os

from functools import singledispatchmethod

from .util import *



class StructuredKernelBasis:
    def __init__(self,kernel,mean_function,mesh,resolution,buffer,mask=True):
        self.kernel = kernel
        #self.mesh = mesh
         
        xmin,ymin = mesh.coordinates.dat.data[:].min(axis=0)
        xmax,ymax = mesh.coordinates.dat.data[:].max(axis=0)

        self.x_bounds = (xmin-buffer,xmax+buffer)
        self.y_bounds = (ymin-buffer,ymax+buffer)
 
        self.res = resolution

        self.nx = int((self.x_bounds[1] - self.x_bounds[0])/self.res)
        self.ny = int((self.y_bounds[1] - self.y_bounds[0])/self.res)

        self.x_ = torch.linspace(self.x_bounds[0],self.x_bounds[1],self.nx)
        self.y_ = torch.linspace(self.y_bounds[0],self.y_bounds[1],self.ny)
      

        self.Lx,x_cols = self.get_lowrank_factor(self.x_,kernel)
        self.Ly,y_cols = self.get_lowrank_factor(self.y_,kernel)

        X__,Y__ = torch.meshgrid(self.x_[x_cols],self.y_[y_cols])
        self.X_ = torch.hstack((reshape_fortran(X__,(-1,1)),reshape_fortran(Y__,(-1,1))))
        if mask:
            tree = KDTree(mesh.coordinates.dat.data[:])
            self.near = tree.query(self.X_,k=1)[0] < buffer
        else:
            self.near = torch.ones(self.X_.shape[0],dtype=torch.bool)

        i = [[],[]]
        v = []
        counter = 0
        for index,value in enumerate(self.near):
            if value:
                i[0].append(index)
                i[1].append(counter)
                v.append(1.)
                counter+=1

        self.M = torch.sparse_coo_tensor(i,v,(self.near.shape[0],self.near.sum()))

        self.n_grid = self.Lx.shape[1]*self.Ly.shape[1]
        self.n_gp = self.M.shape[1]
        self.n_mean = mean_function.n_dofs
        self.n = self.n_gp + self.n_mean

        self.mean_function = mean_function

    def get_lowrank_factor(self,x_,kernel,compression_factor=2.5):
        Kx = kernel(x_,x_)
        n_cols = int(compression_factor*(x_.max() - x_.min())/kernel.l)
        cols = [int(j) for j in (np.round(np.linspace(0,len(x_)-1,n_cols)))]
        Ks = Kx[cols,:][:,cols]
        s,u = torch.linalg.eigh(Ks)
        Lx = Kx[:,cols] @ u * 1./np.sqrt(s) @ u.T
        return Lx,cols

    def set_training_data(self,X_train,Z_train,sigma2_obs,sparse_operator=True):
        self.X_train = X_train
        self.Z_train = Z_train
        self.W_train,_ = self.build_interpolation_matrix(self.X_train,self.x_,self.y_)

        self.h_train = self.mean_function(self.X_train)

        if sparse_operator:
            self.Psi_train = MixedBasisOperator(self.Lx,self.Ly,self.W_train,self.h_train,self.M)
        else:
            self.Psi_train = torch.hstack((self.W_train @ torch.kron(self.Ly,self.Lx) @ self.M,self.h_train))
      
        self.sigma2_obs = sigma2_obs

    def condition(self):
        Iv = torch.ones(self.n)
        Iv[self.n_gp:] = 0.0
        I = torch.diag(Iv)

        Tau_post = self.Psi_train.T @ (self.Psi_train @ torch.eye(self.n)/self.sigma2_obs) + I
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
    def _(self,X_test: torch.Tensor,sparse_operator=True):
        self.X_test = X_test
        self.W_test,_ = self.build_interpolation_matrix(X_test,self.x_,self.y_)

        self.h_test = self.mean_function(X_test)
        if sparse_operator:
            self.Psi_test = MixedBasisOperator(self.Lx,self.Ly,self.W_test,self.h_test,self.M)
        else:
            self.Psi_test = torch.hstack((self.W_test @ torch.kron(self.Ly,self.Lx) @ self.M,self.h_test))

    @set_test_data.register
    def _(self,mesh: df.mesh.MeshGeometry, interpolation_degree: int = 2, sparse_operator=True):
        average_space = df.FunctionSpace(mesh,"DG",0)
        interpolation_space = df.FunctionSpace(mesh,"CG",interpolation_degree)

        X_interp = torch.tensor(df.interpolate(df.SpatialCoordinate(mesh),df.VectorFunctionSpace(mesh,"CG",interpolation_degree)).dat.data[:],dtype=torch.float)
        self.X_test = torch.tensor(df.interpolate(df.SpatialCoordinate(mesh),df.VectorFunctionSpace(mesh,"DG",0)).dat.data[:],dtype=torch.float)
        W_test_,_ = self.build_interpolation_matrix(X_interp,self.x_,self.y_)

        phi_dg = df.TestFunction(average_space)
        w_dg = df.TrialFunction(average_space)
        phi_cg2 = df.TrialFunction(interpolation_space)
        M_dg = df.assemble(phi_dg*w_dg*df.dx)
        M_m = df.assemble(phi_dg * phi_cg2 * df.dx)

        petsc_mat = M_m.M.handle
        indptr,indics,data = petsc_mat.getValuesCSR()
        M_m_ = torch.sparse_csr_tensor(indptr,indics,data,dtype=torch.float).to_sparse_coo()

        petsc_mat = M_dg.M.handle
        indptr,indics,data = petsc_mat.getValuesCSR()
        M_dg = torch.sparse_csr_tensor(indptr,indics,1./data,dtype=torch.float).to_sparse_coo()

        self.W_test = M_dg @ M_m_ @ W_test_

        self.h_test = self.mean_function(self.X_test)
        if sparse_operator:
            self.Psi_test = MixedBasisOperator(self.Lx,self.Ly,self.W_test,self.h_test,self.M)
        else:
            self.Psi_test = torch.hstack((self.W_test @ torch.kron(self.Ly,self.Lx) @ self.M,self.h_test))

    @staticmethod
    def build_interpolation_matrix(X,x_,y_):
        rows = []
        cols = []
        vals = []

        delta_x = x_[1] - x_[0]
        delta_y = y_[1] - y_[0]

        nx = len(x_)
        ny = len(y_)
        m = nx*ny
        
        xmin = x_.min()
        ymin = y_.min()

        neighbors = torch.zeros(4,2)

        for ii,xx in enumerate(X):
            
            x_low = int(torch.floor((xx[0] - xmin)/delta_x))
            x_high = x_low + 1

            y_low = int(torch.floor((xx[1] - ymin)/delta_y))
            y_high = y_low + 1
            
            ll = x_low + y_low*nx
            ul = x_low + y_high*nx
            lr = x_high + y_low*nx
            ur = x_high + y_high*nx
            bbox = [ll,ul,lr,ur]

            neighbors[0,0],neighbors[0,1] = x_[x_low],y_[y_low]
            neighbors[1,0],neighbors[1,1] = x_[x_low],y_[y_high]
            neighbors[2,0],neighbors[2,1] = x_[x_high],y_[y_low]
            neighbors[3,0],neighbors[3,1] = x_[x_high],y_[y_high]

            dist = torch.sqrt(((xx - neighbors)**2).sum(axis=1))

            w = 1./dist
            w/=w.sum()

            rows.append(torch.ones((4))*ii)
            cols.append(torch.tensor(bbox))
            vals.append(w) 

        inds = torch.vstack((torch.hstack(rows),torch.hstack(cols)))
        tens = torch.sparse_coo_tensor(inds,torch.hstack(vals),(X.shape[0],m))
        return tens.coalesce(),torch.transpose(tens,1,0).coalesce()

    def save(self,path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pickle.dump(self,open(path,'wb'))

class MixedBasisOperator:
    def __init__(self,Lx,Ly,W,h,M):
        self.Lx = Lx
        self.Ly = Ly
        self.n_gp = M.shape[1]
        self.n_mean = h.shape[1]
        self.W = W
        self.h = h
        self.M = M

    def __matmul__(self,z):
        z_L = z[:self.n_gp]
        z_h = z[self.n_gp:]
        r1 = self.M @ z_L
        r2 = batch_kron(self.Ly,self.Lx,r1.T)
        r3a = self.W @ r2.T
        r3b = self.h @ z_h
        return r3a + r3b

    @property
    def T(self):
        return MixedBasisOperatorTranspose(self.Lx,self.Ly,self.W,self.h,self.M)

class MixedBasisOperatorTranspose(MixedBasisOperator):
    def __matmul__(self,z):
        r1 = self.W.T @ z
        r2a = self.M.T @ batch_kron(self.Ly.T,self.Lx.T,r1.T).T
        r2b = self.h.T @ z
        return torch.vstack((r2a,r2b))

    @property
    def T(self):
        return MixedBasisOperator(self.Lx,self.Ly,self.W,self.h,self.M)
