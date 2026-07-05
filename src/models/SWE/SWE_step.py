# -*- coding: utf-8 -*-
"""
===============================================================================
SWE_one_step.py
===============================================================================

Container class and time-stepping routines for the 2D Shallow Water Equations.

Time integration uses TVD Runge-Kutta schemes combined
with Strang operator splitting to handle the source terms from bathymetry
and the Coriolis force at second-order accuracy in time:

    U* = U^n + (dt/2) S(U^n)          -- half-step source
    U** = U* - dt ∇·F(U*)             -- full hyperbolic step (RK2)
    U^{n+1} = U** + (dt/2) S(U**)     -- half-step source

The hyperbolic step uses the MUSCL-Hancock PLM reconstruction from SWE_phys.py
and the HLL Riemann solver, giving second-order accuracy in smooth regions.

The bathymetry gradient (b_x, b_y) and Coriolis parameter (f_c) are stored
as arrays on the SimState object and set once by the initial condition function.
They do not change during the simulation.

Author: mrkondratyev

Example usage
-------------
>>> solver = SWE2D(g, SWE, par)
>>> SWE = solver.step_RK()  # advances the solution by one RK timestep
"""


import numpy as np
import copy
from src.models.SWE.SWE_phys import ( 
    boundCond_SWE, 
    Riemann_SWE)
from src.common.high_order_rec import VarReconstruct


class SWE2D:
    """
    Container class for 2D shallow water hydrodynamics routines.

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

    def __init__(self, g, SWE, par):
        """
        Initialize the SWE2D container.

        Parameters
        ----------
        g : object
            Grid object.
        SWE : object
            FluidState object.
        par : object
            Simulation parameters object.
        """
        if g.geom != 'cart':
            raise ValueError(
                f"Invalid geometry for SWE: '{g.geom}'. "
                f"Expected 'cart'.")
        self.g = g
        self.SWE = SWE
        self.par = par

    def step_RK(self):
        """
        Perform a single Runge-Kutta timestep.

        Returns
        -------
        SWE : object
            Updated FluidState object.
        """
        dt = min(CFLcondition_SWE(self.g, self.SWE, self.par.CFL),
                 self.par.timefin - self.par.timenow)
        
        #Strang splitting for the source terms (Coriolis + variable bottom)
        self.SWE = _apply_strang_source(self.SWE, dt)
        
        #uniform system without sources (RK)
        self.SWE = oneStep_SWE_RK(self.g, self.SWE, self.par, dt)
        
        #Strang splitting for the source terms (Coriolis + variable bottom)
        self.SWE = _apply_strang_source(self.SWE, dt)
        
        self.par.timenow += dt
        return self.SWE



# -------------------------
# Function definitions
# -------------------------
def CFLcondition_SWE(g, SWE, CFL):
    """
    Compute the maximum stable timestep for 2D SWE
    according to the CFL (Courant-Friedrichs-Lewy) condition.
    
    The CFL condition ensures that during a timestep, the fastest wave in the system
    does not propagate more than one cell.
    
    Notes
    -----
    - This function accounts for both advection velocities and local "sound speed".
    - Based on the local cell size in each direction.
    
    Parameters
    ----------
    g : object
        Grid object with attributes dx1, dx2 (cell spacings) and Ngc (ghost cells).
    SWE : object
        Fluid state object with attributes h, vel1, vel2 (height and velocities).
    CFL : float
        CFL number (0 < CFL <= 1) controlling timestep size.
    
    Returns
    -------
    dt : float
        Maximum stable timestep according to CFL condition.
    """
    #make copy of ghost cells number to simplify indexing below 
    Ngc = g.Ngc
    
    #gravity wavespeed calculation for whole domain
    gwave = np.sqrt(SWE.h[Ngc:-Ngc, Ngc:-Ngc] * SWE.g_ff)
    
    #FIRST APPROACH
    #maximal possible timestep in each direction
    #dt1 = np.min( g.dx1[Ngc:-Ngc, Ngc:-Ngc] / (np.abs(HD.vel1[Ngc:-Ngc, Ngc:-Ngc]) + gwave) )
    #dt2 = np.min( g.dx2[Ngc:-Ngc, Ngc:-Ngc] / (np.abs(HD.vel2[Ngc:-Ngc, Ngc:-Ngc]) + gwave) )
    #return CFL * min(dt1, dt2)
    
    #SECOND APPROACH 
    dt_inv = \
        np.max((np.abs(SWE.vel1[Ngc:-Ngc, Ngc:-Ngc]) + gwave)/g.dx1[Ngc:-Ngc, Ngc:-Ngc] + \
        (np.abs(SWE.vel2[Ngc:-Ngc, Ngc:-Ngc]) + gwave)/g.dx2[Ngc:-Ngc, Ngc:-Ngc])
    return CFL/dt_inv




# ============================================================================
# Source term application (bathymetry + Coriolis)
# ============================================================================

def _apply_strang_source(state, dt):
    """
    Apply half-step bathymetry and Coriolis source terms to vel1 and vel2.

    The source terms are:
        dv₁/dt = -g ∂b/∂x₁ + f_c v₂
        dv₂/dt = -g ∂b/∂x₂ - f_c v₁

    Applied as an explicit half-step (dt/2) as part of Strang splitting.
    The height h is unaffected by these source terms.

    Parameters
    ----------
    state : SimState  -- modified in place; g_ff, b_x, b_y, f_c read from it
    dt    : float     -- full timestep (half-step = dt/2 is applied here)

    Returns
    -------
    state : SimState
    """
    
    # Store current velocities before updating (needed for Coriolis cross-terms)
    v1_old = state.vel1.copy()
    v2_old = state.vel2.copy()
    
    #here we add bathimetry (curvature of the bottom) and Coriolis force 
    state.vel1 += 0.5 * dt * (-state.g_ff * state.b_x + state.f_c * v2_old)
    state.vel2 += 0.5 * dt * (-state.g_ff * state.b_y - state.f_c * v1_old)

    return state



# ============================================================================
# SWE solver for the system without source terms (i.e. uniform case)
# ============================================================================

def oneStep_SWE_RK(g, SWE, par, dt):
    """
    Perform a single Runge-Kutta timestep for 2D shallow water model.

    This function implements first-, second-, and third-order explicit Runge-Kutta
    schemes for updating the conservative fluid variables

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
    SWE : object
        Fluid state object containing:
            - h, vel1, vel2, : primitive variables
            - H, mom1, mom2 : conservative variables
    par : object
        Simulation parameters including:
            - CFL : CFL number
            - RK_order : 'RK1', 'RK2', or 'RK3'
            - phystime, phystimefin : current and final simulation time
    dt : float
        Suggested timestep (bounded by CFL condition).

    Returns
    -------
    SWE : object
        Updated FluidState object after one Runge-Kutta timestep.
    """
    
    #define local copy of ghost cells number to simplify array indexing
    Ngc = g.Ngc
    
    #here we define the copy for the auxilary fluid state
    SWE_h = copy.deepcopy(SWE)
    
    #conservative variables at the beginning of timestep
    SWE.H    = SWE.h[Ngc:-Ngc,Ngc:-Ngc] 
    SWE.mom1 = SWE.h[Ngc:-Ngc,Ngc:-Ngc] * SWE.vel1[Ngc:-Ngc,Ngc:-Ngc]
    SWE.mom2 = SWE.h[Ngc:-Ngc,Ngc:-Ngc] * SWE.vel2[Ngc:-Ngc,Ngc:-Ngc]
    
    #residuals for conservative variables calculation
    #1st Runge-Kutta iteration - predictor stage
    ResH, Res1, Res2 = flux_calc_SWE(g, SWE, par)
    
    # Conservative update - 1st RK iteration (predictor stage)
    SWE_h.H    = SWE.H    - dt * ResH
    SWE_h.mom1 = SWE.mom1 - dt * Res1 
    SWE_h.mom2 = SWE.mom2 - dt * Res2 
    
    #first order Runge-Kutta scheme
    if (par.RK_order == 'RK1'): 
        
        #simply rewrite the conservative state here for clarity
        SWE.H    = SWE_h.H
        SWE.mom1 = SWE_h.mom1
        SWE.mom2 = SWE_h.mom2
    
    #second-order Runge-Kutta scheme
    elif (par.RK_order == 'RK2'):
        
        #Primitive variables recovery after predictor stage
        SWE_h.h[Ngc:-Ngc, Ngc:-Ngc]    = SWE_h.H
        SWE_h.vel1[Ngc:-Ngc, Ngc:-Ngc] = SWE_h.mom1/SWE_h.H
        SWE_h.vel2[Ngc:-Ngc, Ngc:-Ngc] = SWE_h.mom2/SWE_h.H 
            
        #2nd Runge-Kutta iteration - corrector stage
        ResH, Res1, Res2 = flux_calc_SWE(g, SWE_h, par)
        
        # Conservative update - 2nd RK iteration
        SWE.H    = (SWE_h.H    + SWE.H   ) / 2.0 - dt * ResH / 2.0
        SWE.mom1 = (SWE_h.mom1 + SWE.mom1) / 2.0 - dt * Res1 / 2.0 
        SWE.mom2 = (SWE_h.mom2 + SWE.mom2) / 2.0 - dt * Res2 / 2.0 
    
    elif (par.RK_order == 'RK3'):
        
        #Primitive variables recovery after 1st RK stage
        SWE_h.h[Ngc:-Ngc, Ngc:-Ngc]    = SWE_h.H
        SWE_h.vel1[Ngc:-Ngc, Ngc:-Ngc] = SWE_h.mom1/SWE_h.H
        SWE_h.vel2[Ngc:-Ngc, Ngc:-Ngc] = SWE_h.mom2/SWE_h.H 
        
        #residuals for conservative variables calculation
        #2nd Runge-Kutta iteration 
        ResH, Res1, Res2 = flux_calc_SWE(g, SWE_h, par)
        
        # Conservative update - 2nd RK iteration
        SWE_h.H    = (SWE_h.H    + 3.0 * SWE.H   ) / 4.0 - dt * ResH / 4.0
        SWE_h.mom1 = (SWE_h.mom1 + 3.0 * SWE.mom1) / 4.0 - dt * Res1 / 4.0 
        SWE_h.mom2 = (SWE_h.mom2 + 3.0 * SWE.mom2) / 4.0 - dt * Res2 / 4.0 
    
        # Primitive variables recovery after second stage
        #auxilary fluid height and 2 components of velocity are evaluated 
        SWE_h.h[Ngc:-Ngc, Ngc:-Ngc]    = SWE_h.H
        SWE_h.vel1[Ngc:-Ngc, Ngc:-Ngc] = SWE_h.mom1/SWE_h.H
        SWE_h.vel2[Ngc:-Ngc, Ngc:-Ngc] = SWE_h.mom2/SWE_h.H 
        
        #3rd Runge-Kutta iteration 
        ResH, Res1, Res2 = flux_calc_SWE(g, SWE_h, par)
        
        # Conservative update - final 3rd RK iteration        
        SWE.H    = (2.0 * SWE_h.H    + SWE.H   ) / 3.0 - 2.0 * dt * ResH / 3.0
        SWE.mom1 = (2.0 * SWE_h.mom1 + SWE.mom1) / 3.0 - 2.0 * dt * Res1 / 3.0 
        SWE.mom2 = (2.0 * SWE_h.mom2 + SWE.mom2) / 3.0 - 2.0 * dt * Res2 / 3.0 
        
    else:
        
        raise ValueError(
            f"Invalid RK_order: '{par.RK_order}'. "
            f"Expected one of ['RK1', 'RK2', 'RK3'].")
        
    # Primitive variables recovery at the end of the timestep
    SWE.h[Ngc:-Ngc, Ngc:-Ngc]    = SWE.H
    SWE.vel1[Ngc:-Ngc, Ngc:-Ngc] = SWE.mom1/SWE.H
    SWE.vel2[Ngc:-Ngc, Ngc:-Ngc] = SWE.mom2/SWE.H 
    
    #return the updated class object of the SWE state on the next timestep 
    return SWE



def flux_calc_SWE(g, SWE, par):
    """
    Compute residuals for conservative variables in 2D shallow water equations.
    
    Notes
    ----------
    Residuals are calculated using a Godunov-type method:
    - boundary conditions are taken into account via ghost cells,
    - primitive variables are reconstructed to cell faces,
    - fluxes are computed via (approximate) Riemann solvers,
    - source terms are calculated, if needed, 
    - residuals are obtained via finite-volume integral form.    
    
    Parameters
    ----------
    g : object
        Grid object with attributes Nx1, Nx2, Ngc, fS1, fS2, cVol.
    SWE : object
        Fluid state object at current time step.
    par : object
        Simulation parameters including reconstruction type (rec_type) and flux_type.
    

    Returns
    -------
    ResH : np.ndarray
        Residual array for fluid height.
    Res1 : np.ndarray
        Residual array for x-momentum.
    Res2 : np.ndarray
        Residual array for y-momentum.
    """
    #fill the ghost cells
    SWE = boundCond_SWE(g, par.BC, SWE)
    
    #residuals initialization (only for real cells)
    ResH = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res1 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res2 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    
    #fluxes in 1-dimension 
    if (g.Nx1 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 1-dim
        h_L, h_R       = VarReconstruct(SWE.h   , g, par.rec_type, 1)
        vel1_L, vel1_R = VarReconstruct(SWE.vel1, g, par.rec_type, 1)
        vel2_L, vel2_R = VarReconstruct(SWE.vel2, g, par.rec_type, 1)

        #fluxes calculation with approximate Riemann solver (see flux_type) in 1-dim
        Fh, Fx, Fy = \
            Riemann_SWE(h_L, h_R, \
            vel1_L, vel1_R, vel2_L, vel2_R, SWE.g_ff, par.solver_type, 1)
        
        #residuals calculation for mass, 3 components of momentum and total energy in 1-dim
        ResH = ( Fh[1:,:]*g.fS1[1:,:] - Fh[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        Res1 = ( Fx[1:,:]*g.fS1[1:,:] - Fx[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        Res2 = ( Fy[1:,:]*g.fS1[1:,:] - Fy[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        
        
    #fluxes in 2-dimension
    if (g.Nx2 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 2-dim
        h_L, h_R       = VarReconstruct(SWE.h   , g, par.rec_type, 2)
        vel1_L, vel1_R = VarReconstruct(SWE.vel1, g, par.rec_type, 2)
        vel2_L, vel2_R = VarReconstruct(SWE.vel2, g, par.rec_type, 2)
     
        #fluxes calculation with approximate Riemann solver (see flux_type) in 2-dim
        Fh, Fx, Fy = \
            Riemann_SWE(h_L, h_R, \
            vel1_L, vel1_R, vel2_L, vel2_R, SWE.g_ff, par.solver_type, 2)
        
        #residuals calculation for mass, 3 components of momentum and total energy in 2-dim
        #here we add the fluxes differences to the residuals after 1-dim calculation
        ResH += ( Fh[:,1:]*g.fS2[:,1:] - Fh[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        Res1 += ( Fx[:,1:]*g.fS2[:,1:] - Fx[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        Res2 += ( Fy[:,1:]*g.fS2[:,1:] - Fy[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        
    #return the residuals for height and two components of momentum
    return ResH, Res1, Res2


