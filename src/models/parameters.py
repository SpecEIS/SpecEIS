import os
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import firedrake as fd
import numpy as np
from dataclasses import dataclass

class Geometry:
    def __init__(self,mesh, thklim, bed=None, thickness= None):
        self.mesh = mesh

        E_thk = self.E_thk = fd.FiniteElement('DG',mesh.ufl_cell(),0)
        Q_thk = self.Q_thk = fd.FunctionSpace(mesh,E_thk)
        V_thk = self.V_thk = fd.VectorFunctionSpace(self.mesh,self.E_thk)

        self.B = fd.Function(Q_thk,name='B')
        self.H = fd.Function(Q_thk,name='H')

        self.bed_initialized = False
        self.thk_initialized = False

        self.X = fd.interpolate(mesh.coordinates,V_thk)

        self.element_area = fd.interpolate(fd.CellVolume(mesh),Q_thk)
        one = fd.Function(Q_thk)
        one.dat.data[:] = 1.
        self.area = fd.assemble(one*fd.dx)
        #self.area = self.element_area.dat.data[:].sum()

        self.thklim = thklim

        if bed is not None:
            self.bed = bed

        if thickness is not None:
            self.thickness = thickness

    @property
    def bed(self):
        return self.B.dat.data[:]

    @bed.setter
    def bed(self,b_in):
        if type(b_in) == float:
            self.B.dat.data[:] = b_in
        elif type(b_in) == np.ndarray:
            try:
                self.B.dat.data[:] = b_in
            except ValueError:
                print(f'Incorrect array length: Expected an array of length {len(self.B.dat.data[:])}, got one of length {len(b_in)}')
        elif type(b_in) == fd.Function:
            self.B.dat.data[:] = b_in.dat.data[:]
        else:
            try:
                self.B.dat.data[:] = b_in(self.X.dat.data_ro[:,0],self.X.dat.data_ro[:,1])
            except:
                print('failed')

        self.bed_initialized = True


    @property
    def thickness(self):
        return self.H.dat.data[:]

    @thickness.setter
    def thickness(self,h_in):
        if type(h_in) == float:
            self.H.dat.data[:] = h_in
        elif type(h_in) == np.ndarray:
            try:
                self.H.dat.data[:] = h_in
            except ValueError:
                print(f'Incorrect array length: Expected an array of length {len(self.H.dat.data[:])}, got one of length {len(h_in)}')
        elif type(b_in) == fd.Function:
            self.H.dat.data[:] = h_in.dat.data[:]
        else:
            try:
                self.H.dat.data[:] = h_in(self.X.dat.data_ro[:,0],self.X.dat.data_ro[:,1],grid=False)
            except:
                print('failed')

        self.thk_initialized = True

    @property
    def surface(self):
        return self.B.dat.data[:] + self.H.dat.data[:]

@dataclass
class Scales():
    velocity: float = 1.0
    thickness: float = 1.0
    length: float = 1.0
    beta: float = 1.0
    time: float = 1.0
    pressure: float = 1.0

@dataclass
class PhysicalConstants():
    g: float = 9.81 
    rho_i: float = 917.
    rho_w: float = 1000.
    n: float = 3.0
    A: float = 1e-16
    eps_reg: float = 1e-6
    c: float = 0.7
    z_sea: float = -1000
    calving_velocity: float = 0.0
    flotation_scale: float = 1.0













        
        
        



        
