import torch
#torch.set_default_dtype(torch.float64)

import sys
sys.path.append('../../../src/')

import os

import rasterio
import firedrake as df
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import path
import pyproj
import fiona
import scipy
import pandas as pd
import verde as vd

from functools import singledispatchmethod
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import scipy.sparse as sp

device = 'cuda'

# define a method to convert between coordinate systems - for most of our data, the input epsg will be 4326, and we'll want to project to 3338
from_epsg = 4326
to_epsg = 3338

def reshape_fortran(x, shape):
  if len(x.shape) > 0:
    x = x.permute(*reversed(range(len(x.shape))))
  return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

class PolynomialMean:
    def __init__(self,degree=1):
        self.degree = degree

        if degree==1:
            self.n_dofs = 3
        elif degree==2:
            self.n_dofs = 6
        else:
            print('only 1 or 2 supported')

    def __call__(self,X,orthogonalize=False):
        if self.degree==1:
            h = torch.hstack((torch.ones(X.shape[0],1),X))
        else:
            h = torch.hstack((torch.ones(X.shape[0],1),X,X**2,(X[:,0]*X[:,1]).reshape(-1,1)))

        if orthogonalize:
            h,_ = torch.linalg.qr(h)

        return h

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

class KernelSquaredExponential:
    def __init__(self,l,amplitude):
        self.l = l
        self.amplitude = amplitude

    def __call__(self,x1,x2,return_amplitude_root=True):
        D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
        if return_amplitude_root:
            p = 0.5
        else:
            p = 1.0
        return self.amplitude**p*torch.exp(-D**2/(self.l**2))

class PointDataset:
    def __init__(self,xyz,from_epsg=4326):
        self.xyz = xyz
        self.from_epsg = from_epsg

    def query_by_mesh(self,mesh,length_offset,length_scale,vertical_offset,vertical_scale,mesh_buffer=1000,target_resolution=200,epsg=3338,filter_by_distance=True,return_torch=True):
        X_ = mesh.coordinates.dat.data[:]*length_scale + length_offset
        xy = project_array(self.xyz[:,:2],from_epsg=self.from_epsg,to_epsg=epsg)

        if filter_by_distance:
            tree = KDTree(X_)
            near = tree.query(xy,k=1)[0] < mesh_buffer
            self.X_train = xy[near]
            self.Z_train = self.xyz[near,2]

        if target_resolution is not None:
            reducer = vd.BlockReduce(reduction=np.median,spacing=target_resolution)
            coords,self.Z_train = reducer.filter((self.X_train[:,0],self.X_train[:,1]),data=self.Z_train.squeeze())
            self.X_train = np.c_[coords]

        self.X_train -= length_offset
        self.X_train /= length_scale
        self.Z_train -= vertical_offset
        self.Z_train /= vertical_scale

        if return_torch:
            self.X_train = torch.tensor(self.X_train,dtype=torch.float)
            self.Z_train = torch.tensor(self.Z_train,dtype=torch.float)
        
        return self.X_train,self.Z_train

class DEMDataset:
    def __init__(self,dem_path):
        self.dem_path = dem_path
        self.dem = rasterio.open(dem_path)

    def query_by_mesh(self,mesh,length_offset,length_scale,vertical_offset,vertical_scale,bbox_buffer=1000,mesh_buffer=1000,target_resolution=200,nodatavalue=3.4028234663852886e+38,glims_polygon_file=None,epsg=3338,force_recompute_mask=False,filter_by_distance=True,return_torch=True):
        X_ = mesh.coordinates.dat.data[:]*length_scale + length_offset
        bounds = X_[:,0].min()-bbox_buffer,X_[:,0].max()+bbox_buffer,X_[:,1].min()-bbox_buffer,X_[:,1].max()+bbox_buffer
        self.X_train, self.Z_train = self.get_training_data_from_bbox(self.dem,bounds,200,nodatavalue)

        if filter_by_distance:
            tree = KDTree(X_)
            near = tree.query(self.X_train,k=1)[0] < mesh_buffer
            self.X_train = self.X_train[near]
            self.Z_train = self.Z_train[near]   

        if glims_polygon_file is not None:
            mask = self.build_mask_from_glims(glims_polygon_file,epsg,force_recompute_mask)

            self.X_train = self.X_train[mask]
            self.Z_train = self.Z_train[mask] 

        self.X_train -= length_offset
        self.X_train /= length_scale
        self.Z_train -= vertical_offset
        self.Z_train /= vertical_scale

        if return_torch:
            self.X_train = torch.tensor(self.X_train,dtype=torch.float)
            self.Z_train = torch.tensor(self.Z_train,dtype=torch.float)

        return self.X_train, self.Z_train

    def build_mask_from_glims(self,glims_polygon_file,epsg,force_recompute_mask):
        glims_dir = os.path.dirname(glims_polygon_file)

        if os.path.exists(f'{glims_dir}/cached_mask.p') and not force_recompute_mask:
            mask = pickle.load(open(f'{glims_dir}/cached_mask.p','rb'))
            if len(mask)==len(self.X_train):
                return mask
    
        print("Cached mask not found or not correct shape - recomputing")
        n_pts = self.X_train.shape[0]
        mask = np.zeros(n_pts).astype(bool)
        data = fiona.open(glims_polygon_file)
        q = [d for d in data if d['properties']['line_type']=='glac_bound']
        for glacier in q:
            print(glacier['properties']['glac_name'])
            l_mask = np.zeros(n_pts).astype(bool)
            coords = glacier['geometry']['coordinates']
            try:
                outline = np.array(coords[0])[:,:2]
                outline = project_array(outline,from_epsg=4326, to_epsg=epsg, always_xy=True)
                outer_ice = path.Path(outline)
                within_outer = outer_ice.contains_points(self.X_train)
                l_mask[within_outer] = True
                holes = []
                for c in coords[1:]:
                    hole = np.array(c)[:,:2]
                    hole = project_array(hole,from_epsg=4326, to_epsg=epsg, always_xy=True)
                    inner_ice = path.Path(hole)
                    within_inner = inner_ice.contains_points(self.X_train)
                    l_mask[within_inner] = False
                mask += l_mask
            except ValueError:
                pass

        mask = np.invert(mask)
        pickle.dump(mask,open(f'{glims_dir}/cached_mask.p','wb'))
        return mask

    @staticmethod
    def get_training_data_from_bbox(dem,bounds,target_resolution,nodatavalue):

        # Get DEM elevations
        Z_dem = dem.read().squeeze()[::-1].astype(float)

        # Get edge coordinates
        x_dem_ = np.linspace(dem.bounds.left,dem.bounds.right,dem.width+1)
        y_dem_ = np.linspace(dem.bounds.bottom,dem.bounds.top,dem.height+1)

        # Get cell center coordinates
        x_dem = 0.5*(x_dem_[:-1] + x_dem_[1:])
        y_dem = 0.5*(y_dem_[:-1] + y_dem_[1:])

        # Get extremal locations
        x_min,x_max,y_min,y_max = bounds

        # 
        x_in = (x_dem > (x_min-target_resolution)) & (x_dem < (x_max+target_resolution))
        y_in = (y_dem > (y_min-target_resolution)) & (y_dem < (y_max+target_resolution))

        # Keep valid locations
        x_dem = x_dem[x_in]
        y_dem = y_dem[y_in]
        Z_dem = Z_dem[y_in][:,x_in]

        # Downsample DEM
        dx = abs(x_dem[1] - x_dem[0])
        dy = abs(y_dem[1] - y_dem[0])

        skip_x = int(target_resolution // dx)
        skip_y = int(target_resolution // dy)

        x_ = x_dem[::skip_x].reshape(-1,1)
        y_ = y_dem[::skip_y].reshape(-1,1)

        X_dem,Y_dem = np.meshgrid(x_,y_)
        V = Z_dem.copy()
        V[Z_dem==nodatavalue] = 0
        VV = gaussian_filter(V,[skip_x,skip_y],mode='nearest')

        W = np.ones_like(V)
        W[Z_dem==nodatavalue] = 0
        WW = gaussian_filter(W,[skip_x,skip_y],mode='nearest')
        ZZ = VV/WW
        ZZ[Z_dem==nodatavalue] = nodatavalue
        Z_dem = ZZ[::skip_y,::skip_x]

        z = Z_dem.ravel(order='F')
        inds = (z!=nodatavalue)
        Z_train = z[inds]

        X_train = np.c_[X_dem.ravel(order='F'),Y_dem.ravel(order='F')][inds]

        return X_train,Z_train

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

vertical_scale = 1000.0
vertical_offset = 0.0

length_offset = np.array([574192.92588861, 1337135.10319813])
length_scale = 10000.

mesh_directory = '../meshes/mesh_1000/'
mesh = df.Mesh(f'{mesh_directory}/mesh.msh',name='mesh')

dem_path = '../data/dem/kennicott_3338.tif'
dem_dataset = DEMDataset(dem_path)
X_train_dem, Z_train_dem = dem_dataset.query_by_mesh(mesh,length_offset,length_scale,vertical_offset,vertical_scale,glims_polygon_file='../data/outline/glims_download_46178/glims_polygons.shp')

dataframe = pd.read_csv('../data/bed/kennicott_root_merged.csv', delimiter=',')
dataframe = dataframe[dataframe.bed_elev.notnull()]
xyz = dataframe.loc[:,['lon','lat','bed_elev']].to_numpy()
point_dataset = PointDataset(xyz)

X_train_radar,Z_train_radar = point_dataset.query_by_mesh(mesh,length_offset,length_scale,vertical_offset,vertical_scale)
X_train = torch.vstack((X_train_dem,X_train_radar))
Z_train = torch.hstack((Z_train_dem,Z_train_radar))

l = 1500/length_scale
a = 1000/vertical_scale
kernel = KernelSquaredExponential(l,a)
mean = PolynomialMean(degree=2)

ski_grid_spacing = 100/length_scale
ski_buffer = 2000/length_scale

sigma2_obs = (50/vertical_scale)**2

ski = StructuredKernelBasis(kernel,mean,mesh,ski_grid_spacing,ski_buffer,mask=True)
ski.set_training_data(X_train,Z_train,sigma2_obs,sparse_operator=True)
ski.condition()

ski.set_test_data(mesh,sparse_operator=True)

fig,axs = plt.subplots(nrows=2,ncols=2)
df.triplot(mesh,axes=axs[0,0])
axs[0,0].scatter(*ski.X_test.T,c=ski.sample(test=True),vmin=0,vmax=3)
axs[0,0].axis('equal')
df.triplot(mesh,axes=axs[0,1])
axs[0,1].scatter(*ski.X_train.T,c=ski.sample(test=False),vmin=0,vmax=3)
axs[0,1].axis('equal')
df.triplot(mesh,axes=axs[1,0])
axs[1,0].scatter(*ski.X_train.T,c=ski.Z_train,vmin=0,vmax=3)
axs[1,0].axis('equal')
df.triplot(mesh,axes=axs[1,1])
axs[1,1].scatter(*ski.X_test.T,c=ski.marginal_variance(),vmin=0,vmax=0.2)
axs[1,1].axis('equal')

ski.save(f'{mesh_directory}/maps/bed.p')


