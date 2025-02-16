import sys
sys.path.append('../../../')

import firedrake as df
import torch
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from speceis.data.Datasets import *
from speceis.basis.SKI import StructuredKernelBasis
from speceis.basis.kernel import SquaredExponential
from speceis.basis.mean import PolynomialMean

from_epsg = 4326
to_epsg = 3338

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

l = 2000/length_scale
a = 1000/vertical_scale
kernel = SquaredExponential(l,a)
mean = PolynomialMean(degree=2)

ski_grid_spacing = 100/length_scale
ski_buffer = 2000/length_scale

sigma2_obs = (50/vertical_scale)**2

ski = StructuredKernelBasis(kernel,mean,mesh,ski_grid_spacing,ski_buffer,mask=True)
ski.set_training_data(X_train,Z_train,sigma2_obs,sparse_operator=True)
ski.condition()

ski.set_test_data(mesh,sparse_operator=False)

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

#ski.save(f'{mesh_directory}/maps/bed.p')


