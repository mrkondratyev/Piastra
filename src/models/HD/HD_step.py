# -*- coding: utf-8 -*-
"""
HD2D - Container class for 2D compressible hydrodynamics routines.

Hybrid approach:
- Functions remain modular and pedagogically simple
- Lightweight container class provides a clean interface for timestepping
  and groups related routines for compressible 2D hydrodynamics

This class handles:
- CFL-limited timestep calculation
- Single-step Runge-Kutta updates (RK1, RK2, RK3)
- Residual computation via Godunov-type finite-volume method
- Approximate Riemann solver flux evaluation in 2D
- Primitive-to-conservative and conservative-to-primitive transformations
- Boundary condition handling

The underlying methods are suitable for explicit, finite-volume hydrodynamics
simulations with compressible flows. The code assumes that the following objects
are provided:

Attributes
----------
g : object
    Grid object containing domain size, spacing, volumes, and face areas.
HD : object
    Fluid state object containing primitive (density, velocities, pressure) and
    conservative (mass, momentum, total energy) variables.
eos : object
    Equation of state object providing methods such as sound_speed, etc.
par : object
    Simulation parameters including:
        - CFL : Courant number
        - RK_order : 'RK1', 'RK2', or 'RK3'
        - solver_type : type of approximate Riemann solver
        - rec_type : reconstruction type
        - BC : boundary condition type
        - phystime, phystimefin : current and final physical time

Example usage
-------------
>>> hydro = HD2D(g, HD, eos, par)
>>> HD = hydro.step_RK()  # advances the solution by one RK timestep
"""


import numpy as np
import copy
from src.models.HD.HD_phys import (
    prim2cons_HD,
    cons2prim_HD, 
    boundCond_HD,
    Riemann_HD)
from src.common.high_order_rec import VarReconstruct
from src.gravity import body_force_dt


class HD2D:
    """
    Container class for 2D compressible hydrodynamics routines.

    Attributes
    ----------
    g : object
        Grid object with domain sizes, spacing, volumes, and face areas.
    HD : object
        FluidState object containing primitive and conservative variables.
    par : object
        Simulation parameters including CFL, RK_order, flux_type, rec_type, phystime, phystimefin.
    eos : object
        Equation of state object.
    """

    def __init__(self, g, HD, eos, par):
        """
        Initialize the Hydro2D container.

        Parameters
        ----------
        g : object
            Grid object.
        HD : object
            FluidState object.
        eos : object
            Equation of state object.
        par : object
            Simulation parameters object.
        """
        self.g = g
        self.HD = HD
        self.eos = eos
        self.par = par

    def step_RK(self):
        """
        Perform a single Runge-Kutta timestep.

        Returns
        -------
        HD : object
            Updated FluidState object.
        """
        dt = min(CFLcondition_HD(self.g, self.HD, self.eos, self.par.CFL),
                 self.par.timefin - self.par.timenow)
        self.HD = oneStep_HD_RK(self.g, self.HD, self.eos, self.par, dt)
        self.par.timenow += dt
        return self.HD



# -------------------------
# Small helper: one RK stage applied to all five conservative variables
# -------------------------
def _rk_stage(HD_out, HD_a, HD_b, ResM, Res1, Res2, Res3, ResE, dt, a, b, c):
    """
    Set HD_out.* = a * HD_a.* + b * HD_b.* + c * dt * Res*
 
    For SSP-RK, the standard combinations are:
      Stage 1 (predictor): a=1,    b=0,    c=-1     -> HD_h = HD - dt*R(HD)
      RK2 corrector:       a=0.5,  b=0.5,  c=-0.5
      RK3 stage 2:         a=0.75, b=0.25, c=-0.25
      RK3 stage 3 (final): a=1/3,  b=2/3,  c=-2/3
    """
    HD_out.mass = a * HD_a.mass + b * HD_b.mass + c * dt * ResM
    HD_out.mom1 = a * HD_a.mom1 + b * HD_b.mom1 + c * dt * Res1
    HD_out.mom2 = a * HD_a.mom2 + b * HD_b.mom2 + c * dt * Res2
    HD_out.mom3 = a * HD_a.mom3 + b * HD_b.mom3 + c * dt * Res3
    HD_out.etot = a * HD_a.etot + b * HD_b.etot + c * dt * ResE


# -------------------------
# Function definitions
# -------------------------
def CFLcondition_HD(g, HD, eos, CFL):
    """
    Compute the maximum stable timestep for 2D compressible hydrodynamics
    according to the CFL (Courant-Friedrichs-Lewy) condition.
    
    The CFL condition ensures that during a timestep, the fastest wave in the system
    does not propagate more than one cell.
    
    Notes
    -----
    - This function accounts for both advection velocities and local sound speed.
    - Based on the local cell size in each direction.
    - For compressible flows, sound speed is computed using the EOS.
    
    Parameters
    ----------
    g : object
        Grid object with attributes dx1, dx2 (cell spacings) and Ngc (ghost cells).
    HD : object
        Fluid state object with attributes dens, vel1, vel2 (density and velocities).
    eos : object
        Equation of state object providing sound_speed(density, pressure).
    CFL : float
        CFL number (0 < CFL <= 1) controlling timestep size.
    
    Returns
    -------
    dt : float
        Maximum stable timestep according to CFL condition.
    
    Notes
    -------
    Lame coefficient hx2 is included in the CFL calculation 
    in order to adjust the correct timestep for the simulations in the polar coordinates, 
    e.g. dt ~ rdφ for the cylindrical polar geometry
    """
    #make copy of ghost cells number to simplify indexing below 
    Ngc = g.Ngc
    
    dens = HD.dens[Ngc:-Ngc, Ngc:-Ngc]
    vel1 = HD.vel1[Ngc:-Ngc, Ngc:-Ngc]
    vel2 = HD.vel2[Ngc:-Ngc, Ngc:-Ngc]
    pres = HD.pres[Ngc:-Ngc, Ngc:-Ngc]
    
    #sound speed calculation for whole domain
    sound = eos.sound_speed_nr(dens, pres)
    
    #FIRST APPROACH
    #maximal possible timestep in each direction
    #dt1 = np.min(g.dx1[Ngc:-Ngc, Ngc:-Ngc] / (np.abs(vel1) + sound))
    #dt2 = np.min(g.dx2[Ngc:-Ngc, Ngc:-Ngc]*g.hx2[Ngc:-Ngc, Ngc:-Ngc] / (np.abs(vel2) + sound))
    #return CFL * min(dt1, dt2)
    
    #SECOND APPROACH 
    dt_inv = np.max((np.abs(vel1) + sound)/g.dx1[Ngc:-Ngc, Ngc:-Ngc] + \
        (np.abs(vel2) + sound)/(g.dx2[Ngc:-Ngc, Ngc:-Ngc]* g.hx2[Ngc:-Ngc, Ngc:-Ngc]))
        
    return min(CFL/dt_inv, body_force_dt(g, HD, CFL))




def oneStep_HD_RK(g, HD, eos, par, dt):
    """
    Perform a single Runge-Kutta timestep for 2D compressible hydrodynamics.

    This function implements first-, second-, and third-order explicit Runge-Kutta
    schemes for updating the conservative fluid variables, including
    primitive variable recovery using the EOS.

    Notes
    -----
    - RK1: simple forward Euler update (1st order)
    - RK2: predictor-corrector scheme (2nd order)
    - RK3: 3-stage scheme (3rd order)
    - Fluxes are calculated using approximate Riemann solvers (Godunov-type)
    - Residuals are computed via finite-volume integral form:
        U_t + RES = 0
        RES = (1/Volume) * sum(flux * face_area)
    - At each stage:
        1. Fill ghost zones according to boundary conditions.
        2. Reconstruct primitive variables at cell faces.
        3. Compute fluxes via Riemann solver.
        4. Compute residuals.
        5. Update conservative variables.
        6. Recover primitive variables for next stage.
    - Stages 1-4 are done for 2 directions in 2D 
        
    Parameters
    ----------
    g : object
        Grid object with attributes Nx1, Nx2, Ngc, dx1, dx2, fS1, fS2, cVol.
    HD : object
        Fluid state object containing:
            - dens, vel1, vel2, vel3, pres : primitive variables
            - mass, mom1, mom2, mom3, etot : conservative variables
    eos : object
        Equation of state object.
    par : object
        Simulation parameters including:
            - CFL : CFL number
            - RK_order : 'RK1', 'RK2', or 'RK3'
            - phystime, phystimefin : current and final simulation time
    dt : float
        Suggested timestep (bounded by CFL condition).

    Returns
    -------
    HD : object
        Updated FluidState object after one Runge-Kutta timestep.
    """
    
    #define local copy of ghost cells number to simplify array indexing
    Ngc = g.Ngc
    
    #here we define the copy for the auxilary fluid state
    HD_h = copy.deepcopy(HD)
    
    #conservative variables at the beginning of timestep
    (HD.mass, HD.mom1, HD.mom2, HD.mom3, HD.etot) = \
        prim2cons_HD(
            HD.dens[Ngc:-Ngc,Ngc:-Ngc], 
            HD.vel1[Ngc:-Ngc,Ngc:-Ngc], 
            HD.vel2[Ngc:-Ngc,Ngc:-Ngc], 
            HD.vel3[Ngc:-Ngc,Ngc:-Ngc], 
            HD.pres[Ngc:-Ngc,Ngc:-Ngc], eos)
    
    #residuals for conservative variables calculation
    #1st Runge-Kutta iteration - predictor stage
    ResM, Res1, Res2, Res3, ResE = \
        flux_calc_HD(g, HD, par, eos)
    
    # Conservative update - 1st RK iteration (predictor stage)
    _rk_stage(HD_h, HD, HD, \
        ResM, Res1, Res2, Res3, ResE, dt, 1.0, 0.0, -1.0)
    
    #first order Runge-Kutta scheme
    if (par.RK_order == 'RK1'): 
        
        #simply rewrite the conservative state here for clarity
        HD.mass = HD_h.mass
        HD.mom1 = HD_h.mom1; HD.mom2 = HD_h.mom2; HD.mom3 = HD_h.mom3
        HD.etot = HD_h.etot
    
    #second-order Runge-Kutta scheme
    elif (par.RK_order == 'RK2'):
        
        #Primitive variables recovery after predictor stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        _prim_recovery(HD_h, Ngc, eos)
            
        #2nd Runge-Kutta iteration - corrector stage
        ResM, Res1, Res2, Res3, ResE = \
            flux_calc_HD(g, HD_h, par, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy
        _rk_stage(HD, HD, HD_h, \
            ResM, Res1, Res2, Res3, ResE, dt, 1.0/2.0, 1.0/2.0, -1.0/2.0)
    
    elif (par.RK_order == 'RK3'):
        
        #Primitive variables recovery after 1st RK stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        _prim_recovery(HD_h, Ngc, eos)
        
        #residuals for conservative variables calculation
        #2nd Runge-Kutta iteration 
        ResM, Res1, Res2, Res3, ResE = \
            flux_calc_HD(g, HD_h, par, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy        
        _rk_stage(HD_h, HD, HD_h, \
            ResM, Res1, Res2, Res3, ResE, dt, 1.0/4.0, 3.0/4.0, -1.0/4.0)
    
        # Primitive variables recovery after second stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        _prim_recovery(HD_h, Ngc, eos)
        
        ResM, Res1, Res2, Res3, ResE = \
            flux_calc_HD(g, HD_h, par, eos)
        
        # Conservative update - final 3rd RK iteration
        # update mass, three components of momentum and total energy
        _rk_stage(HD, HD, HD_h, \
            ResM, Res1, Res2, Res3, ResE, dt, 2.0/3.0, 1.0/3.0, -2.0/3.0)
            
    else:
        
        raise ValueError(
            f"Invalid RK_order: '{par.RK_order}'. "
            f"Expected one of ['RK1', 'RK2', 'RK3'].")
        
    # Primitive variables recovery at the end of the timestep
    #density, 3 components of velocity and pressure are evaluated 
    _prim_recovery(HD, Ngc, eos)
    
    #return the updated class object of the fluid state on the next timestep 
    return HD


# ============================================================================
# Helper: call cons2prim_nr_hydro for a SimState object
# ============================================================================
def _prim_recovery(state, Ngc, eos):
    """
    Call cons2prim_nr_hydro and write results back into
    state.{dens,vel*,pres}.

    Parameters
    ----------
    state   : SimState  with conservative vars populated
    Ngc     : int       number of ghost cells
    eos     : EOSdata
    """
    (state.dens[Ngc:-Ngc, Ngc:-Ngc],
     state.vel1[Ngc:-Ngc, Ngc:-Ngc],
     state.vel2[Ngc:-Ngc, Ngc:-Ngc],
     state.vel3[Ngc:-Ngc, Ngc:-Ngc],
     state.pres[Ngc:-Ngc, Ngc:-Ngc]) = \
        cons2prim_HD(
            state.mass, state.mom1, state.mom2, state.mom3, state.etot, eos)



def flux_calc_HD(g, HD, par, eos):
    """
    Compute residuals for conservative variables in 2D compressible hydrodynamics.
    
    Notes
    ----------
    Residuals are calculated using a Godunov-type method:
    - boundary conditions are taken into account via ghost cells,
    - primitive variables are reconstructed to cell faces,
    - fluxes are computed via approximate Riemann solvers,
    - source terms are calculated, if needed, 
    - residuals are obtained via finite-volume integral form.    
    
    Parameters
    ----------
    g : object
        Grid object with attributes Nx1, Nx2, Ngc, fS1, fS2, cVol.
    HD : object
        Fluid state object at current time step.
    par : object
        Simulation parameters including reconstruction type (rec_type) and flux_type.
    eos : object
        Equation of state object.

    Returns
    -------
    ResM : np.ndarray
        Residual array for mass density.
    Res1, Res2, Res3 : np.ndarray
        Residual arrays for momentum.
    ResE : np.ndarray
        Residual array for total energy.
    """
    #fill the ghost cells
    HD = boundCond_HD(g, par.BC, HD, par.BC_fixed)

    #re-evaluate a state- or time-dependent body force for THIS RK stage
    #(self-gravity, Coriolis, an orbiting perturber -- see gravity.py).
    #A static force leaves body_force = None and simply keeps the F1/F2
    #the initial condition wrote.
    if HD.body_force is not None:
        HD.body_force(g, HD, par)

    #make copies of ghost cell numbers to simplify indexing below
    Ngc = g.Ngc

    #residuals initialization (only for real cells)
    ResM = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res1 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res2 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res3 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    ResE = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    
    #fluxes in 1-dimension 
    if (g.Nx1 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 1-dim
        #here we reconstruct density, 3 components of velocity and pressure
        dens_L, dens_R = VarReconstruct(HD.dens, g, par.rec_type, 1)
        vel1_L, vel1_R = VarReconstruct(HD.vel1, g, par.rec_type, 1)
        vel2_L, vel2_R = VarReconstruct(HD.vel2, g, par.rec_type, 1)
        vel3_L, vel3_R = VarReconstruct(HD.vel3, g, par.rec_type, 1)
        pres_L, pres_R = VarReconstruct(HD.pres, g, par.rec_type, 1)

        #fluxes calculation with approximate Riemann solver (see flux_type) in 1-dim
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            Riemann_HD(dens_L, dens_R, \
                vel1_L, vel1_R, vel2_L, vel2_R, vel3_L, vel3_R, \
                pres_L, pres_R, eos, par.solver_type, 1)
        
        #residuals calculation for mass, 3 components of momentum and total energy in 1-dim
        ResM = ( Fmass[1:,:]*g.fS1[1:,:] - Fmass[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        Res1 = ( Fmomx[1:,:]*g.fS1[1:,:] - Fmomx[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        Res2 = ( Fmomy[1:,:]*g.fS1[1:,:] - Fmomy[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        Res3 = ( Fmomz[1:,:]*g.fS1[1:,:] - Fmomz[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        ResE = ( Fetot[1:,:]*g.fS1[1:,:] - Fetot[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        
        
    #fluxes in 2-dimension
    if (g.Nx2 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 2-dim
        #here we reconstruct density, 3 components of velocity and pressure
        dens_L, dens_R = VarReconstruct(HD.dens, g, par.rec_type, 2)
        pres_L, pres_R = VarReconstruct(HD.pres, g, par.rec_type, 2)
        vel1_L, vel1_R = VarReconstruct(HD.vel1, g, par.rec_type, 2)
        vel2_L, vel2_R = VarReconstruct(HD.vel2, g, par.rec_type, 2)
        vel3_L, vel3_R = VarReconstruct(HD.vel3, g, par.rec_type, 2)
     
        #fluxes calculation with approximate Riemann solver (see flux_type) in 2-dim
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            Riemann_HD(dens_L, dens_R, \
                vel1_L, vel1_R, vel2_L, vel2_R, vel3_L, vel3_R, \
                pres_L, pres_R, eos, par.solver_type, 2)
        
        #residuals calculation for mass, 3 components of momentum and total energy in 2-dim
        #here we add the fluxes differences to the residuals after 1-dim calculation
        ResM += ( Fmass[:,1:]*g.fS2[:,1:] - Fmass[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        Res1 += ( Fmomx[:,1:]*g.fS2[:,1:] - Fmomx[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        Res2 += ( Fmomy[:,1:]*g.fS2[:,1:] - Fmomy[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        Res3 += ( Fmomz[:,1:]*g.fS2[:,1:] - Fmomz[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResE += ( Fetot[:,1:]*g.fS2[:,1:] - Fetot[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        
    #curvature source terms for different curvilinear coordinates
    ST1, ST2, ST3 = curv_source_HD(g, HD)
    
    #finally, here we add the external force + curvature source terms
    #source term for momentum residual 
    Res1 += - HD.dens[Ngc:-Ngc, Ngc:-Ngc] * HD.F1[:,:] - ST1
    Res2 += - HD.dens[Ngc:-Ngc, Ngc:-Ngc] * HD.F2[:,:] - ST2
    Res3 += - ST3
    #source term for energy residual 
    ResE += - HD.dens[Ngc:-Ngc, Ngc:-Ngc]*\
        (HD.F1[:,:] * HD.vel1[Ngc:-Ngc, Ngc:-Ngc] + \
         HD.F2[:,:] * HD.vel2[Ngc:-Ngc, Ngc:-Ngc])
             
    #return the residuals for mass, 3 components of momentum and total energy
    return ResM, Res1, Res2, Res3, ResE



    
def curv_source_HD(g, HD):
    """
    Compute geometric source terms for the hydrodynamic equations 
    in curvilinear coordinates (finite-volume formulation).

    In Cartesian coordinates, the Euler equations are source-free, but in 
    curvilinear geometries (e.g., cylindrical, spherical) additional terms 
    appear due to the divergence operator expressed in non-Cartesian bases.
    This function evaluates those terms for momentum equations.

    Parameters
    ----------
    g : object
        Grid object containing:
        - ``geom`` : str, geometry type ('cart', 'cyl', 'pol', or 'sph').
        - ``cx1`` : ndarray, radial cell-center positions.
        - ``Ngc`` : int, number of ghost cells.
        - ``Nx1, Nx2`` : int, number of grid points.
    HD : object
        Fluid state containing:
        - ``dens`` : ndarray, density field.
        - ``pres`` : ndarray, pressure field.
        - ``vel1``, ``vel2``, ``vel3`` : ndarray, velocity field

    Returns
    -------
    ST1, ST2, ST3 : ndarray, shape (Nx1, Nx2)
        Momentum source terms on interior cells (no ghost cells; matches
        Res1/Res2/Res3 in flux_calc_HD, which these are subtracted from
        directly). Identically zero for 'cart' (the Euler equations are
        source-free in Cartesian coordinates).

    Notes
    -----
    - Zero for 'cart'; geometric curvature terms for 'cyl', 'pol', 'sph'.
    """
    Ngc = g.Ngc 
    ST1 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    ST2 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    ST3 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    
    # source-free; nothing further to do
    if g.geom == 'cart':
        return ST1, ST2, ST3
    
    r    = g.cx1[Ngc:-Ngc,Ngc:-Ngc]
    dens = HD.dens[Ngc:-Ngc,Ngc:-Ngc]
    pres = HD.pres[Ngc:-Ngc,Ngc:-Ngc]
    v1   = HD.vel1[Ngc:-Ngc,Ngc:-Ngc]
    v2   = HD.vel2[Ngc:-Ngc,Ngc:-Ngc]
    v3   = HD.vel3[Ngc:-Ngc,Ngc:-Ngc]
    
    #cylindrical (R,Z) geometry
    if (g.geom == 'cyl'):
        ST1 = ( pres + dens * v3**2 ) / r
        ST3 = -dens * v3 * v1 / r
    
    #polar (R,phi) geometry
    if (g.geom == 'pol'):
        ST1 = ( pres + dens * v2**2 ) / r
        ST2 =  -dens * v2 * v1 / r
            
    #spherical polar (r,theta) geometry
    if (g.geom == 'sph'):
        
        #cotangent of theta
        if g.Nx2 > 1: 
            sin_theta = np.sin(g.fx2[Ngc:g.Nx1+Ngc, Ngc:g.Nx2+Ngc+1])
            cos_theta = np.cos(g.fx2[Ngc:g.Nx1+Ngc, Ngc:g.Nx2+Ngc+1])
            cot = (sin_theta[:,1:]-sin_theta[:,:-1]) / \
                (cos_theta[:,:-1]-cos_theta[:,1:])
        else:
            cot = np.zeros_like(r)
            
        ST1 = ( 2.0 * pres + dens * (v2**2 + v3**2) ) / r
        ST2 = ( pres + dens * v3**2 ) * cot / r  - dens * v1 * v2 / r
        ST3 = - ( dens * v2 * v3 ) * cot / r - dens * v1 * v3 / r
            
    return ST1, ST2, ST3


