import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.append('../../src/')

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("once",category=DeprecationWarning)
    import firedrake as fd
    from firedrake.petsc import PETSc
    from models.molho import Model, SolverConfig, ModelConfig
    from models.parameters import Geometry
    from models.parameters import Scales
    from models.parameters import PhysicalConstants

import logging
logging.captureWarnings(True)

import pickle

results_dir = './results/'
data_dir = './data/'

mesh = fd.Mesh(f'{data_dir}/mesh.msh',name='mesh')
  
thklim = 1e-3

constants = PhysicalConstants(z_sea=2.033,calving_velocity=1.0,flotation_scale=1e-2)
scales = Scales(velocity=100,thickness=1000,length=108000,beta=1000)
solver_config = SolverConfig(petsc_parameters='krylov_default')
model_config = ModelConfig(sliding_law='Budd',alpha_flux=1000,alpha_thick=1000.0)

interpolant = pickle.load(open(f'{data_dir}/interpolant.pkl','rb'))
interp = lambda x,y: interpolant(x,y,grid=False)/1000.
geometry = Geometry(mesh,thklim,bed=interp,thickness=1e-3)

model = Model(geometry,scales,constants,model_config,solver_config)
model.traction = 5.0

z_ela = 2.45
lapse_rate = 5/1000
time_step_factor = 1.05
model.specific_balance = ((geometry.surface - z_ela)*lapse_rate)

S_file = fd.File(f'{results_dir}/S.pvd')
B_file = fd.File(f'{results_dir}/B.pvd')
Us_file = fd.File(f'{results_dir}/U_s.pvd')
H_file = fd.File(f'{results_dir}/H.pvd')
N_file = fd.File(f'{results_dir}/N.pvd')
adot_file = fd.File(f'{results_dir}/adot.pvd')

Q_cg2 = fd.VectorFunctionSpace(mesh,"CG",3)
S_out = fd.Function(model.Q_thk,name='S')
N_out = fd.Function(model.Q_thk,name='N')
U_s = fd.Function(Q_cg2,name='U_s')

S_out.interpolate(model.S)
N_out.interpolate(model.N)
U_s.interpolate(model.Ubar0 - 1./4*model.Udef0)

S_file.write(S_out,time=0.)
H_file.write(model.H0,time=0.)
B_file.write(model.B,time=0.)
Us_file.write(U_s,time=0.)
adot_file.write(model.adot,time=0.)

t = 0.0
t_end = 2000
dt = 2.5
max_step = 2.5

with fd.CheckpointFile(f"{results_dir}/functions.h5", 'w') as afile:
    
    afile.save_mesh(mesh)

    i = 0
    while t<t_end:
        dt = min(dt*time_step_factor,max_step)

        z_ela = 2.45 + max(0,(t-700)/3000)

        model.adot.dat.data[:] = (((model.B.dat.data[:] + model.H0.dat.data[:])
                                  - z_ela)*lapse_rate)

        converged = model.step(t,
                               dt,
                               picard_tol=1e-3,
                               momentum=0.5,
                               max_iter=20,
                               convergence_norm='l2')

        if not converged:
            dt*=0.5
            continue
        t += dt
        PETSc.Sys.Print(t,dt,fd.assemble(model.H0*fd.dx))
        S_out.interpolate(model.S)
        N_out.interpolate(model.N)
        U_s.interpolate((model.Ubar0 - 1./4*model.Udef0)*(model.H0>model.thklim))

        afile.save_function(model.H0, idx=i)
        afile.save_function(S_out, idx=i)
        afile.save_function(N_out, idx=i)
        afile.save_function(U_s, idx=i)

        S_file.write(S_out,time=t)
        H_file.write(model.H0,time=t)
        B_file.write(model.B,time=t)
        Us_file.write(U_s,time=t)
        N_file.write(N_out,time=t)
        adot_file.write(model.adot,time=t)
        i += 1
