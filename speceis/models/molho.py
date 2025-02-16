import os
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import firedrake as df
    from firedrake.petsc import PETSc
import numpy as np
import time

from dataclasses import dataclass, field


class Model:
    def __init__(
            self, geometry, scales, constants, model_config, solver_config):
        
        # Remember inputs
        self.geometry = geometry
        self.scales = scales
        self.constants = constants
        self.model_config = model_config
        self.solver_config = solver_config

        # Get the spatial mesh
        self.mesh = mesh = geometry.mesh

        # Set the time and timestep variables (values don't matter - they are overwritten later)
        self.t = df.Constant(0.0)
        dt = self.dt = df.Constant(1.0)

        # Get theta-method parameter (1 ->  backward euler, 0.5 -> Crank-Nicholson)
        theta = self.theta = df.Constant(model_config.theta)
       
        # Get some geometric variables
        nhat = self.nhat = df.FacetNormal(mesh)
        area = self.area = geometry.area

         # Get physical constants
        g = self.g = df.Constant(constants.g)
        rho_i = self.rho_i = df.Constant(constants.rho_i)
        rho_w = self.rho_w = df.Constant(constants.rho_w)
        n = self.n = df.Constant(constants.n)
        A = self.A = df.Constant(constants.A)
        eps_reg = self.eps_reg = df.Constant(constants.eps_reg)
        thklim = self.thklim = df.Constant(geometry.thklim)
        z_sea = self.z_sea = df.Constant(constants.z_sea)
        p = self.p = model_config.p
        c = self. c = df.Constant(constants.c)
        alpha_flux = self.alpha_flux = df.Constant(model_config.alpha_flux)
        alpha_thick = self.alpha_thick = df.Constant(model_config.alpha_thick)
        calving_velocity = self.calving_velocity = df.Constant(constants.calving_velocity)
        flotation_scale = self.flotation_scale = df.Constant(constants.flotation_scale)

        # Define non-dimensional constants
        eta_star = self.eta_star = df.Constant(constants.A**(-1./constants.n)
                                               * (scales.velocity/scales.thickness)**((1-constants.n)/constants.n))

        delta = self.delta = df.Constant(scales.thickness/scales.length)

        gamma = self.gamma = df.Constant(scales.beta*scales.thickness/eta_star)

        omega = self.omega = df.Constant(constants.rho_i*constants.g*scales.thickness**3
                                         / (eta_star*scales.length*scales.velocity))

        zeta = self.zeta = df.Constant(scales.time*scales.velocity/scales.length)
        
        ### BUILD FUNCTION SPACES ###

        # Specify the thickness function space
        E_thk = self.E_thk = geometry.E_thk
        Q_thk = self.Q_thk = geometry.Q_thk

        # Specify the traction function space (CG1)
        E_tra = self.E_tra = df.FiniteElement('CG',mesh.ufl_cell(),1)
        Q_tra = self.Q_tra = df.FunctionSpace(mesh,E_tra)

        # Specify the depth-averaged velocity function space (MTW)
        E_bar = self.E_bar = df.FiniteElement('MTW',mesh.ufl_cell(),3)
        Q_bar = self.Q_bar = df.FunctionSpace(mesh,E_bar)

        # Specify the deformation velocity function space (RT)
        E_def = self.E_def = df.FiniteElement('RT',mesh.ufl_cell(),1)
        Q_def = self.Q_def = df.FunctionSpace(mesh,E_def)
        
        # Specify the function space for thickness and bed gradients (RT, this is only used in sigma-coordinate terms) 
        E_grd = self.E_grd = df.FiniteElement('RT',mesh.ufl_cell(),1)
        Q_grd = self.Q_grd = df.FunctionSpace(mesh,E_grd)

        # Specify the mixed function space for the velocity-thickness solve
        E = self.E = df.MixedElement(E_bar,E_def,E_thk)
        V = self.V = df.FunctionSpace(mesh,E)
        
        ### BUILD FUNCTIONS FOR COUPLED SOLVE ### 

        # Specify the coupled function that stores the velocity and thickness values
        W = self.W = df.Function(V,name='W')      # 
        W_i = self.W_i = df.Function(V)           # Frozen function for Picard iteration
        Psi = self.Psi = df.TestFunction(V)       # Test Function
        dW = self.dW = df.TrialFunction(V)        # Trial Function

        # Split W into components
        Ubar,Udef,H = self.Ubar,self.Udef,self.H = df.split(W)
        ubar,vbar = Ubar                          # Split velocity vectors into components  
        udef,vdef = Udef

        #
        Ubar_i,Udef_i,H_i = self.Ubar_i,self.Udef_i,self.H_i = df.split(W_i)
        ubar_i,vbar_i = Ubar_i
        udef_i,vdef_i = Udef_i

        # Do the same for the test function
        Phibar,Phidef,xsi = self.Phibar,self.Phidef,self.xsi = df.split(Psi)
        phibar_x,phibar_y = Phibar
        phidef_x,phidef_y = Phidef
        
        H0 = self.H0 = geometry.H
        Hmid = self.Hmid = theta*H + (1-theta)*H0
        Hmid_i = self.Hmid_i = theta*H_i + (1-theta)*H0
        self.H_temp = df.Function(self.Q_thk)
        
        B = self.B = geometry.B
        Bhat = self.Bhat =  df.max_value(B, z_sea - rho_i/rho_w*Hmid_i)  # Maximum of ice base and bedrock
        
        S = self.S = Bhat + H
        S0 = self.S0 = Bhat + H0
        Smid = self.Smid = theta*S + (1-theta)*S0

        ### BUILD VARIABLES FOR BED/THICKNESS GRADIENT PROJECTION

        # Set up variables to store bed and thickness gradients
        S_grad = self.S_grad = df.Function(Q_grd)
        B_grad = self.B_grad = df.Function(Q_grd)
        Chi = self.Chi = df.TestFunction(Q_grd)
        dS = self.dS = df.TrialFunction(Q_grd)

        # Combine depth-averaged and deformational velocity functions via the MOLHO ansatz
        u = self.u = MOLHOBasis([ubar,udef],Hmid_i,S_grad,B_grad,p=p)
        v = self.v = MOLHOBasis([vbar,vdef],Hmid_i,S_grad,B_grad,p=p)
        u_i = self.u_i = MOLHOBasis([ubar_i,udef_i],Hmid_i,S_grad,B_grad,p=p)
        v_i = self.v_i = MOLHOBasis([vbar_i,vdef_i],Hmid_i,S_grad,B_grad,p=p)
        phi_x = self.phi_x = MOLHOBasis([phibar_x,phidef_x],Hmid_i,S_grad,B_grad,p=p)
        phi_y = self.phi_y = MOLHOBasis([phibar_y,phidef_y],Hmid_i,S_grad,B_grad,p=p)

        # Set up variables to store previous solution values and derived geometric quantities        
        self.Ubar0 = df.Function(Q_bar)
        self.Udef0 = df.Function(Q_def)
        
        # Functions to hold a linear thickness and bed gradient (this is necessary for
        # implementation of periodic boundary conditions when using the Blatter-Pattyn approx.
        S_lin = self.S_lin = df.Function(Q_thk)  
        B_lin = self.B_lin = df.Function(Q_thk)  
        S_grad_lin = self.S_grad_lin = df.Constant([0.0,0.0])
        B_grad_lin = self.B_grad_lin = df.Constant([0.0,0.0])

        # Build functions to hold parameters
        adot = self.adot = df.Function(Q_thk) 
        beta = self.beta = df.Function(Q_tra)

        # Compute the residual of the Blatter-Pattyn eqns.
        R_stress = self.build_stress_form()

        # Compute the residual of the transport equation
        R_transport = self.build_transport_form()

        # Compute the joint residual
        R = self.R = R_stress + R_transport

        # Convert joint residual to a bilinear form 
        # by replacing the Function W with a TrialFunction
        R_lin = self.R_lin = df.replace(R,{W:dW})
        
        # Define the linear system corresponding to a Picard iteration
        coupled_problem = df.LinearVariationalProblem(df.lhs(R_lin),df.rhs(R_lin),W)

        # Build a coupled linear solver
        self.coupled_solver = df.LinearVariationalSolver(
            coupled_problem,
            solver_parameters=solver_config.petsc_parameters)


        # Compute forms for gradient terms used in sigma-coordinate derivatives 
        R_S,R_B = self.build_gradient_forms()
        
        # Define associated solvers
        projection_parameters = {'ksp_type':'cg','mat_type':'matfree'}
        S_grad_problem = df.LinearVariationalProblem(df.lhs(R_S),df.rhs(R_S),S_grad)
        self.S_grad_solver = df.LinearVariationalSolver(
            S_grad_problem,
            solver_parameters=projection_parameters)

        B_grad_problem = df.LinearVariationalProblem(df.lhs(R_B),df.rhs(R_B),B_grad)
        self.B_grad_solver = df.LinearVariationalSolver(
            B_grad_problem,
            solver_parameters=projection_parameters)

        
        self.beta_initialized = False
        self.adot_initialized = False

    def build_stress_form(self):
        # Compute the residual form for the Blatter-Pattyn Equation

        # We treat the vertical dimension explicitly, while firedrake handles 
        # the horizontal part - these classes us Gauss-Legendre quadrature 
        # rules to integrate a function of $\varsigma over the unit interval,
        # where $\varsigma=0 is the surface and 1 is the bed.  
        vi_x = VerticalIntegrator(2)
        vi_z = VerticalIntegrator(3) 
        
        # The viscous stress is the combination of horizontal and
        # vertical deformation terms, plus a boundary term that keeps
        # ice from flowing through the boundaries if desired.  
        viscous_stress = -(vi_x.intz(self.membrane_form) 
                + vi_z.intz(self.shear_form) 
                + vi_x.intz(self.membrane_boundary_form_nopen)) 

        # The total momentum balance is the sum of the viscous and basal shear stress
        # minus the driving stress
        return viscous_stress + self.basal_shear_stress() - self.driving_stress()

    def basal_shear_stress(self):
        # Compute the basal shear stress

        # Compute basal velocity and test function
        U_b = df.as_vector([self.u(1),self.v(1)])
        Phi_b = df.as_vector([self.phi_x(1),self.phi_y(1)])
       
        Hmid_i = self.Hmid_i
        gamma = self.gamma
        c = self.c
        rho_w = self.rho_w
        rho_i = self.rho_i
        z_sea = self.z_sea
        Bhat = self.Bhat

        beta = self.beta

        N = self.N = (df.min_value(c*Hmid_i + df.Constant(1e-2),
                          Hmid_i-rho_w/rho_i*(z_sea - Bhat)) 
                          + df.Constant(1e-4))


        # This is where a new sliding law ought to be implemented
        if self.model_config.sliding_law == 'linear':
            basal_stress = -gamma*beta*df.dot(U_b,Phi_b)*df.dx
        elif self.model_config.sliding_law == 'Budd':

            basal_stress = -gamma*beta*N**df.Constant(1./3.)*df.dot(U_b,Phi_b)*df.dx
        else:
            basal_stress = df.Constant(0.0)

        return basal_stress

    def driving_stress(self):
        # Compute the driving stress.  This looks complicated because there
        # are two extra things going on - first, we use integration by parts
        # to deal with S being piecewise constant, and second, we allow
        # for an arbitrary constant slope, which facilites periodic boundary
        # conditions (which are otherwise problematic).  

        omega = self.omega
        S_grad_lin = self.S_grad_lin
        S_lin = self.S_lin

        Phibar = self.Phibar
        Bhat = self.Bhat

        nhat = self.nhat
        Hmid = self.Hmid
        Hmid_i = self.Hmid_i

        driving_stress = (omega*Hmid*df.dot(S_grad_lin,Phibar)*df.dx
                          - omega*df.div(Phibar*Hmid)*(Bhat - S_lin)*df.dx
                          - omega*df.div(Phibar*Hmid_i)*Hmid*df.dx 
                          + omega*df.jump(Phibar*Hmid,nhat)*df.avg(Bhat - S_lin)*df.dS 
                          + omega*df.jump(Phibar*Hmid_i,nhat)*df.avg(Hmid)*df.dS 
                          + omega*df.dot(Phibar*Hmid,nhat)*(Bhat - S_lin)*df.ds 
                          + omega*df.dot(Phibar*Hmid_i,nhat)*(Hmid)*df.ds)

        return driving_stress

    def strain_rate_second_invariant(self,s):
        delta = self.delta
        eps_reg = self.eps_reg
        u = self.u_i
        v = self.v_i
        return (delta**2*(u.dx(s,0))**2 
                    + delta**2*(v.dx(s,1))**2 
                    + delta**2*(u.dx(s,0))*(v.dx(s,1)) 
                    + delta**2*0.25*((u.dx(s,1)) + (v.dx(s,0)))**2 
                    +0.25*(u.dz(s))**2 + 0.25*(v.dz(s))**2 
                    + eps_reg)

    def eta(self,s):
        n = self.n
        return 0.5*self.strain_rate_second_invariant(s)**((1-n)/(2*n))

    def phi_grad_membrane(self,s):
        delta = self.delta
        phi_x = self.phi_x
        phi_y = self.phi_y
        return np.array([[delta*phi_x.dx(s,0), delta*phi_x.dx(s,1)],
                         [delta*phi_y.dx(s,0), delta*phi_y.dx(s,1)]])

    def phi_grad_shear(self,s):
        phi_x = self.phi_x
        phi_y = self.phi_y
        return np.array([[phi_x.dz(s)],
                         [phi_y.dz(s)]])

    def eps_membrane(self,s):
        delta = self.delta
        u = self.u
        v = self.v
        return np.array([[2*delta*u.dx(s,0) + delta*v.dx(s,1), 
                          0.5*delta*u.dx(s,1) + 0.5*delta*v.dx(s,0)],
                         [0.5*delta*u.dx(s,1) + 0.5*delta*v.dx(s,0),
                          delta*u.dx(s,0) + 2*delta*v.dx(s,1)]])

    def eps_shear(self,s):
        u = self.u
        v = self.v
        return np.array([[0.5*u.dz(s)],
                        [0.5*v.dz(s)]])

    def membrane_form(self,s):  
        Hmid_i = self.Hmid_i
        return (2*self.eta(s)*(self.eps_membrane(s)
                * self.phi_grad_membrane(s)).sum()*Hmid_i*df.dx(degree=9))

    def shear_form(self,s):
        Hmid_i = self.Hmid_i
        return (2*self.eta(s)*(self.eps_shear(s)
                * self.phi_grad_shear(s)).sum()*Hmid_i*df.dx(degree=9))

    def membrane_boundary_form_nopen(self,s):
        nhat = self.nhat
        alpha_flux = self.alpha_flux
        un = self.u(s)*nhat[0] + self.v(s)*nhat[1]
        phin = self.phi_x(s)*nhat[0] + self.phi_y(s)*nhat[1]
        if self.model_config.boundary_markers is not None:
            return alpha_flux*phin*un*df.ds(self.model_config.boundary_markers[0])
        else:
            return alpha_flux*phin*un*df.ds

    def build_transport_form(self):
        dt = self.dt

        Ubar = self.Ubar
        Ubar_i = self.Ubar_i

        Hmid = self.Hmid
        Hmid_i = self.Hmid_i
        
        H = self.H
        H0 = self.H0

        adot = self.adot

        nhat = self.nhat

        xsi = self.xsi

        rho_w = self.rho_w
        rho_i = self.rho_i
        z_sea = self.z_sea
        B = self.B

        zeta = self.zeta

        H_avg = 0.5*(Hmid_i('+') + Hmid_i('-'))
        H_jump = Hmid('+')*nhat('+') + Hmid('-')*nhat('-')
        xsi_jump = xsi('+')*nhat('+') + xsi('-')*nhat('-')

        unorm_i = df.dot(Ubar_i,Ubar_i)**0.5

        if self.model_config.flux_type=='centered':
            uH = df.avg(Ubar)*H_avg

        elif self.model_config.flux_type=='lax-friedrichs':
            uH = df.avg(Ubar)*H_avg + df.Constant(0.5)*df.avg(unorm_i)*H_jump

        elif self.model_config.flux_type=='upwind':
            uH = df.avg(Ubar)*H_avg + 0.5*abs(df.dot(df.avg(Ubar_i),nhat('+')))*H_jump

        else:
            print('Invalid flux')

        if self.model_config.boundary_markers is not None:
            R_transport = ((H - H0)/dt - adot)*xsi*df.dx + zeta*df.dot(uH,xsi_jump)*df.dS + self.model_config.alpha_flux*xsi*df.dot((H-thklim)*nhat,nhat)*df.ds(self.model_config.boundary_markers[1])
        else:
            R_transport = ((H - H0)/dt - adot)*xsi*df.dx + zeta*df.dot(uH,xsi_jump)*df.dS

        calving_velocity = self.calving_velocity
        flotation_scale = self.flotation_scale

        flotation_criterion = rho_w/rho_i*(z_sea - B) - Hmid_i
        self.floating = df.min_value(df.max_value(flotation_criterion/flotation_scale,0),1)
        
        indicator = self.floating('+')*self.floating('-')
        u_c = calving_velocity*nhat
        R_transport += zeta*indicator*2*df.avg(df.dot(u_c,nhat)*Hmid*xsi)*df.dS

        return R_transport


    def build_gradient_forms(self):
        R_S = (df.dot(self.Chi,self.dS)*df.dx 
              - df.dot(self.Chi,self.S_grad_lin)*df.dx 
              + df.div(self.Chi)*(self.Smid - self.S_lin)*df.dx 
              - df.dot(self.Chi,self.nhat)*(self.Smid - self.S_lin)*df.ds)

        R_B = (df.dot(self.Chi,self.dS)*df.dx 
              - df.dot(self.Chi,self.B_grad_lin)*df.dx 
              + df.div(self.Chi)*(self.Bhat - self.B_lin)*df.dx
              - df.dot(self.Chi,self.nhat)*(self.Bhat - self.B_lin)*df.ds)

        return R_S,R_B


    def step(
            self,
            t,
            dt,
            picard_tol=1e-6,
            max_iter=50,
            momentum=0.0,
            error_on_nonconvergence=False,
            convergence_norm='linf',
            update=True,
            enforce_positivity=True):

        self.W.sub(0).assign(self.Ubar0)
        self.W.sub(1).assign(self.Udef0)
        self.W.sub(2).assign(self.H0)

        self.W_i.assign(self.W)
        self.dt.assign(dt)

        eps = 1.0
        i = 0
        
        while eps>picard_tol and i<max_iter:
            t_ = time.time()
            self.S_grad_solver.solve()
            self.B_grad_solver.solve()
            self.coupled_solver.solve()
            if enforce_positivity:
                self.H_temp.interpolate(df.max_value(self.W.sub(2),self.thklim))
                self.W.sub(2).assign(self.H_temp)
            
            if convergence_norm=='linf':
                with self.W_i.dat.vec_ro as w_i:
                    with self.W.dat.vec_ro as w:
                        eps = abs(w_i - w).max()[1]
            else:
                eps = (np.sqrt(
                       df.assemble((self.W_i.sub(2) - self.W.sub(2))**2*df.dx))
                       / self.area)

            PETSc.Sys.Print(i,eps,time.time()-t_)


            self.W_i.assign((1-momentum)*self.W + momentum*self.W_i)
            i+=1

        if i==max_iter and eps>picard_tol:
            converged=False
        else:
            converged=True

        if error_on_nonconvergence and not converged:
            return converged
            

        if update:
            self.Ubar0.assign(self.W.sub(0))
            self.Udef0.assign(self.W.sub(1))
            self.H0.assign(self.W.sub(2))
            self.t.assign(t+dt)

        return converged

    @property
    def traction(self):
        return self.beta.dat.data[:]

    @traction.setter
    def traction(self,b_in):
        if type(b_in) == float:
            self.beta.dat.data[:] = b_in
        elif type(b_in) == np.ndarray:
            try:
                self.beta.dat.data[:] = b_in
            except ValueError:
                print(f'Incorrect array length: Expected an array of length {len(self.beta.dat.data[:])}, got one of length {len(b_in)}')
        elif type(b_in) == fd.Function:
            self.beta.dat.data[:] = b_in.dat.data[:]
        else:
            try:
                self.beta.dat.data[:] = b_in(self.X.dat.data_ro[:,0],self.X.dat.data_ro[:,1])
            except:
                print('failed')

        self.beta_initialized = True

    @property
    def specific_balance(self):
        return self.adot.dat.data[:]

    @specific_balance.setter
    def specific_balance(self,a_in):
        if type(a_in) == float:
            self.adot.dat.data[:] = a_in
        elif type(a_in) == np.ndarray:
            try:
                self.adot.dat.data[:] = a_in
            except ValueError:
                print(f'Incorrect array length: Expected an array of length {len(self.beta.dat.data[:])}, got one of length {len(b_in)}')
        elif type(a_in) == fd.Function:
            self.adot.dat.data[:] = a_in.dat.data[:]
        else:
            try:
                self.adot.dat.data[:] = a_in(self.X.dat.data_ro[:,0],self.X.dat.data_ro[:,1])
            except:
                print('failed')

        self.adot_initialized = True


class CoupledModelAdjoint:
    def __init__(self,model):
        self.model = model

        Lambda = self.Lambda = df.Function(model.V)
        delta = self.delta = df.Function(model.V)

        w_H0 = df.TestFunction(model.Q_thk)
        w_B = df.TestFunction(model.Q_thk)
        w_beta = df.TestFunction(model.Q_cg1)
        w_adot = df.TestFunction(model.Q_thk)

        R_full = self.R_full = df.replace(self.model.R,{model.W_i:model.W, 
                                                        model.Psi:Lambda})

        R_adjoint = df.derivative(R_full,model.W,model.Psi)
        R_adjoint_linear = df.replace(R_adjoint,{Lambda:model.dW})
        self.A_adjoint = df.lhs(R_adjoint_linear)
        self.b_adjoint = df.rhs(R_adjoint_linear)

        G_H0 = self.G_H0 = df.derivative(R_full,model.H0,w_H0)
        G_B = self.G_B = df.derivative(R_full,model.B,w_B)
        G_beta = self.G_beta = df.derivative(R_full,model.beta,w_beta)
        G_adot = self.G_adot = df.derivative(R_full,model.adot,w_adot)

        self.g_H0 = None
        self.g_B = None
        self.g_beta = None
        self.g_adot = None


        self.ksp = PETSc.KSP().create()
        if self.model.solver_type=='direct':
            self.ksp.setType('preonly')
            pc = self.ksp.getPC()
            pc.setType('lu')
            pc.setFactorSolverType('mumps')
        else:
            self.ksp.setType('gmres')
            pc = self.ksp.getPC()
            self.ksp.setInitialGuessNonzero(True)
            pc.setType('ilu')
            self.ksp.setTolerances(1e-6)

        self.A = None

    def backward(self,delta):
        t1 = time.time()
        if self.A:
            df.assemble(self.A_adjoint,tensor=self.A,mat_type='aij')
        else:
            self.A = df.assemble(self.A_adjoint,tensor=self.A,mat_type='aij')
            self.ksp.setOperators(self.A.M.handle)
        print(f'assemble: {time.time() - t1}')

        for d in self.delta.dat.data[:]:
            d*=-1

        with self.delta.dat.vec_ro as vec:
            with self.Lambda.dat.vec as sol:
                t1 = time.time()
                self.ksp.solve(vec,sol)
                print(f'adjoint solve: {time.time() - t1}')

        
        t1 = time.time()
        if self.g_H0:
            df.assemble(self.G_H0,tensor=self.g_H0)
            df.assemble(self.G_B,tensor=self.g_B)
            df.assemble(self.G_beta,tensor=self.g_beta)
            df.assemble(self.G_adot,tensor=self.g_adot)
        else:
            self.g_H0 = df.assemble(self.G_H0)
            self.g_B = df.assemble(self.G_B)
            self.g_beta = df.assemble(self.G_beta)
            self.g_adot = df.assemble(self.G_adot)
        print(f'gradient: {time.time() - t1}')


class MOLHOBasis(object):
    def __init__(self,u,H,S_grad,B_grad,p=4):
        self.u = u
        self.coef = [lambda s:1.0, lambda s:1./p*((p+1)*s**p - 1)]
        self.dcoef = [lambda s:0, lambda s:(p+1)*s**(p-1)]
        
        self.H = H
        self.S_grad = S_grad
        self.B_grad = B_grad

    def __call__(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.coef)])

    def ds(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.dcoef)])

    def dz(self,s):
        return self.ds(s)*self.dsdz(s)

    def dx_(self,s,x):
        return sum([u.dx(x)*c(s) for u,c in zip(self.u,self.coef)])

    def dx(self,s,x):
        return self.dx_(s,x) + self.ds(s)*self.dsdx(s,x)

    def dsdx(self,s,x):
        return 1./self.H*(self.S_grad[x] - s*(self.S_grad[x] - self.B_grad[x]))

    def dsdz(self,x):
        return -1./self.H

class VerticalIntegrator(object):
    def __init__(self,order):
        self.points,self.weights = np.polynomial.legendre.leggauss(order)
        self.points = (self.points+1)/2.
        self.weights/=2

    def intz(self,f):
        return sum([w*f(s) for s,w in zip(self.points,self.weights)])  

@dataclass
class SolverConfig():
    petsc_parameters: str

    def __init__(self,petsc_parameters=None):
        self.petsc_parameters = petsc_parameters

    @property
    def petsc_parameters(self):
        return self._petsc_parameters

    @petsc_parameters.setter
    def petsc_parameters(self, params: dict):
        if params == 'direct_default' or params is None:
            self._petsc_parameters = {"ksp_type": "preonly",
                                  "pmat_type":"aij",
                                  "pc_type": "lu",  
                                  "pc_factor_mat_solver_type": "mumps"} 

        elif params == 'krylov_default':
            self._petsc_parameters = {'ksp_type': 'gmres',
                                      'pc_type':'bjacobi',
                                      "ksp_rtol":1e-6,
                                      'ksp_initial_guess_nonzero': True}
        else:
            self._petsc_parameters = params  

@dataclass
class ModelConfig():
    sliding_law: str = 'linear'
    theta: float = 1.0
    alpha_flux: float = 0.0
    alpha_thick: float = 0.0
    p: float = 4.0
    flux_type: str = 'lax-friedrichs'
    boundary_markers: list = None



