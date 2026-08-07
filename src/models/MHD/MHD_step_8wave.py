# -*- coding: utf-8 -*-
"""
===============================================================================
MHD_step_8wave.py
===============================================================================

2D Magnetohydrodynamics (MHD) Finite-volume Solver
====================================================================

This module provides routines for solving the 2D compressible
magnetohydrodynamics (MHD) equations using finite-volume Godunov-type
methods. The solver employs high-order reconstruction, approximate Riemann
solvers, and Runge-Kutta (RK) timestepping schemes. Magnetic divergence
is controlled via the 8-wave method.

Main Components
---------------
- ``MHD2D_8wave`` : container class managing grid, state, EOS, and parameters.
- ``CFLcondition_MHD`` : compute timestep from CFL stability condition.
- ``oneStep_MHD_RK_8wave`` : advance MHD state by one timestep (RK1/RK2/RK3).
- ``flux_calc_MHD_8wave`` : compute residuals of conservative variables.
- ``curv_source_MHD_8wave`` : evaluate curvature source terms.

Features
--------
- Spatial accuracy via piecewise constant, linear, PPM, MP5, or WENO-type reconstructions.
- Riemann solvers: Local Lax-Friedrichs (LLF), HLL, HLLC, HLLD.
- Temporal accuracy: TVD Runge-Kutta methods (1st, 2nd, or 3rd order).
- Divergence of the magnetic field is treated via simple Powell's 8-wave approach.
- Curvilinear support.

@author:
    mrkondratyev
"""

from src.models.MHD.MHD_phys import (
    prim2cons_MHD,
    _prim_recovery,
    max_wavespeed_MHD, 
    boundCond_MHD, 
    Riemann_MHD)
from src.grid.grid_misc import div_cell_vector 
from src.common.high_order_rec import VarReconstruct 
import numpy as np 
from src.gravity import body_force_dt
import copy 


class MHD2D_8wave:
    """
    Container class for 2D compressible 8-wave MHD routines.

    Attributes
    ----------
    g : object
        Grid object with domain sizes, spacing, volumes, and face areas.
    fluid : object
        FluidState object containing primitive and conservative variables.
    par : object
        Simulation parameters including CFL, RK_order, flux_type, rec_type, phystime, phystimefin.
    eos : object
        Equation of state object.
    """

    def __init__(self, g, MHD, eos, par):
        """
        Initialize the MHD2D_8wave container.

        Parameters
        ----------
        g : object
            Grid object.
        fluid : object
            FluidState object.
        eos : object
            Equation of state object.
        par : object
            Simulation parameters object.
        """
        self.g = g
        self.MHD = MHD
        self.eos = eos
        self.par = par

    def step_RK(self):
        """
        Perform a single Runge-Kutta timestep.

        Returns
        -------
        fluid : object
            Updated FluidState object.
        """
        dt = min(CFLcondition_MHD(self.g, self.MHD, self.eos, self.par.CFL),
                 self.par.timefin - self.par.timenow)
        self.MHD = oneStep_MHD_RK_8wave(self.g, self.MHD, self.eos, self.par, dt)
        self.par.timenow += dt
        return self.MHD


# -------------------------
# Small helper: one RK stage applied to all five conservative variables
# -------------------------
def _rk_stage(MHD_out, MHD_a, MHD_b, \
    ResM, Res1, Res2, Res3, ResE, ResB1, ResB2, ResB3, dt, a, b, c):
    """
    Set HD_out.* = a * HD_a.* + b * HD_b.* + c * dt * Res*
 
    For SSP-RK, the standard combinations are:
      Stage 1 (predictor): a=1,    b=0,    c=-1     -> HD_h = HD - dt*R(HD)
      RK2 corrector:       a=0.5,  b=0.5,  c=-0.5
      RK3 stage 2:         a=0.75, b=0.25, c=-0.25
      RK3 stage 3 (final): a=1/3,  b=2/3,  c=-2/3
    """
    MHD_out.mass = a * MHD_a.mass + b * MHD_b.mass + c * dt * ResM
    MHD_out.mom1 = a * MHD_a.mom1 + b * MHD_b.mom1 + c * dt * Res1
    MHD_out.mom2 = a * MHD_a.mom2 + b * MHD_b.mom2 + c * dt * Res2
    MHD_out.mom3 = a * MHD_a.mom3 + b * MHD_b.mom3 + c * dt * Res3
    MHD_out.etot = a * MHD_a.etot + b * MHD_b.etot + c * dt * ResE
    MHD_out.bcon1 = a * MHD_a.bcon1 + b * MHD_b.bcon1 + c * dt * ResB1
    MHD_out.bcon2 = a * MHD_a.bcon2 + b * MHD_b.bcon2 + c * dt * ResB2
    MHD_out.bcon3 = a * MHD_a.bcon3 + b * MHD_b.bcon3 + c * dt * ResB3
    

def CFLcondition_MHD(g, MHD, eos, CFL):
    """
    Compute timestep based on CFL stability condition for MHD.
    
    The CFL condition states that the fastest wave in the system must not
    travel more than one cell per timestep.
    
    Parameters
    ----------
    g : object
        Grid object.
    MHD : object
        Fluid state object.
    eos : object
        Equation of state object.
    CFL : float
        CFL number (stability factor, < 1).
    
    Returns
    -------
    dt : float
        Stable timestep satisfying CFL condition.
        
    Notes
    -------
    Lame coefficient hx2 is included in the CFL calculation 
    in order to adjust the correct timestep for the simulations in the polar coordinates, 
    e.g. dt ~ rdφ for the cylindrical polar geometry
    """
    Ngc = g.Ngc
    
    dens = MHD.dens[Ngc:-Ngc, Ngc:-Ngc]
    vel1 = MHD.vel1[Ngc:-Ngc, Ngc:-Ngc]
    vel2 = MHD.vel2[Ngc:-Ngc, Ngc:-Ngc]
    pres = MHD.pres[Ngc:-Ngc, Ngc:-Ngc]
    B1   = MHD.bfi1[Ngc:-Ngc, Ngc:-Ngc]
    B2   = MHD.bfi2[Ngc:-Ngc, Ngc:-Ngc]
    B3   = MHD.bfi3[Ngc:-Ngc, Ngc:-Ngc]
    
    #sound speed calculation for whole domain
    csound = eos.sound_speed_nr(dens, pres)
    
    #fast magnetosonic speed calculation for whole domain 
    cfast = max_wavespeed_MHD(csound, B1,B2,B3, dens)
    
    #FIRST APPROACH
    #dt1 = np.min( g.dx1[Ngc:-Ngc, Ngc:-Ngc] / (np.abs(vel1) + cfast) )
    #dt2 = np.min( g.dx2[Ngc:-Ngc, Ngc:-Ngc] * g.hx2[Ngc:-Ngc, Ngc:-Ngc] / (np.abs(vel2) + cfast) )    
    #return  CFL * min(dt1, dt2)
    
    #SECOND APPROACH 
    dt_inv = np.max((np.abs(vel1) + cfast)/g.dx1[Ngc:-Ngc, Ngc:-Ngc] + \
        (np.abs(vel2) + cfast)/(g.dx2[Ngc:-Ngc, Ngc:-Ngc] * g.hx2[Ngc:-Ngc, Ngc:-Ngc]))
        
    return min(CFL/dt_inv, body_force_dt(g, MHD, CFL))



def oneStep_MHD_RK_8wave(g, MHD, eos, par, dt):
    """
    Advance the MHD state by one timestep using RK1, RK2, or RK3 schemes.

    Parameters
    ----------
    g : object
        Computational grid with geometry and metric data.
    MHD : object
        Fluid state containing primitive and conservative variables.
    eos : object
        Equation of state object.
    par : object
        Simulation parameters (RK order, reconstruction type, flux type, etc.).
    dt : float
        Timestep size.

    Returns
    -------
    MHD : object
        Updated fluid state after one timestep.

    Notes
    -----
    - Implements TVD Runge–Kutta timestepping:
      - RK1 (Forward Euler)
      - RK2 (2nd-order TVD RK, Shu & Osher 1988)
      - RK3 (3rd-order TVD RK, Shu & Osher 1988)
    - Residuals are computed via ``flux_calc_MHD_8wave``.
    - After each substep, the updated conservative variables are converted
      back to primitive form 
    - This function controls the global structure of one timestep.
    
    - For RK timestepping one can see (Shu and Osher (1988))

    for a given timestep dt and a primitive fluid state, we calculate conservative state and the residuals for them 
    if RK method is beyond the first order, we additionally introduce the intermediate conservative and primitive states
    on the predictor stage, we update the initial fluid state to the intermediate one on each stage, 
    and after the final stage, we update the fluid state itself, using the information from the intermediate stages 
    """
    
    #define local copy of ghost cells number to simplify array indexing
    Ngc = g.Ngc
    
    #here we define the copy for the auxilary fluid state
    MHD_h = copy.deepcopy(MHD)
    
    #conservative variables at the beginning of timestep
    (MHD.mass, MHD.mom1, MHD.mom2, MHD.mom3, MHD.etot,
     MHD.bcon1, MHD.bcon2, MHD.bcon3) = \
        prim2cons_MHD(
            MHD.dens[Ngc:-Ngc, Ngc:-Ngc],
            MHD.vel1[Ngc:-Ngc, Ngc:-Ngc],
            MHD.vel2[Ngc:-Ngc, Ngc:-Ngc],
            MHD.vel3[Ngc:-Ngc, Ngc:-Ngc],
            MHD.pres[Ngc:-Ngc, Ngc:-Ngc],
            MHD.bfi1[Ngc:-Ngc, Ngc:-Ngc],
            MHD.bfi2[Ngc:-Ngc, Ngc:-Ngc],
            MHD.bfi3[Ngc:-Ngc, Ngc:-Ngc],
            eos)
    
    #residuals for conservative variables calculation
    #1st Runge-Kutta iteration - predictor stage
    ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = \
        flux_calc_MHD_8wave(g, MHD, par, eos)
    
    # Conservative update - 1st RK stage (predictor)
    _rk_stage(MHD_h, MHD, MHD, \
       ResM, ResV1, ResV2, ResV3, ResE, \
       ResB1, ResB2, ResB3, dt, 1.0, 0.0, -1.0)
    
    #first order Runge-Kutta scheme
    if (par.RK_order == 'RK1'): 
        
        #simply rewrite the conservative state here for clarity
        MHD.mass  = MHD_h.mass
        MHD.mom1  = MHD_h.mom1; MHD.mom2  = MHD_h.mom2; MHD.mom3  = MHD_h.mom3
        MHD.etot  = MHD_h.etot
        MHD.bcon1 = MHD_h.bcon1; MHD.bcon2 = MHD_h.bcon2; MHD.bcon3 = MHD_h.bcon3
    
    
    #second-order Runge-Kutta scheme
    elif (par.RK_order == 'RK2'):
        
        #Primitive variables recovery after predictor stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        _prim_recovery(MHD_h, Ngc, eos)
            
        #2nd Runge-Kutta stage - corrector
        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = \
            flux_calc_MHD_8wave(g, MHD_h, par, eos)
        
        # Conservative update - 2nd RK iteration
        _rk_stage(MHD, MHD_h, MHD, \
           ResM, ResV1, ResV2, ResV3, ResE, \
           ResB1, ResB2, ResB3, dt, 0.5, 0.5, -0.5)
        
    elif (par.RK_order == 'RK3'):
        
        #Primitive variables recovery after predictor stage
        _prim_recovery(MHD_h, Ngc, eos)
            
        #2nd Runge-Kutta stage
        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = \
            flux_calc_MHD_8wave(g, MHD_h, par, eos)
        
        # Conservative update - 2nd RK iteration
        _rk_stage(MHD_h, MHD_h, MHD, \
           ResM, ResV1, ResV2, ResV3, ResE, \
           ResB1, ResB2, ResB3, dt, 1.0/4.0, 3.0/4.0, -1.0/4.0)
        
        # Primitive variables recovery after the second stage
        _prim_recovery(MHD_h, Ngc, eos)
        
        #3rd Runge-Kutta stage
        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = \
            flux_calc_MHD_8wave(g, MHD_h, par, eos)
        
        # Conservative update - 3rd RK iteration
        _rk_stage(MHD, MHD_h, MHD, \
           ResM, ResV1, ResV2, ResV3, ResE, \
           ResB1, ResB2, ResB3, dt, 2.0/3.0, 1.0/3.0, -2.0/3.0)
            
    else:
        
        raise ValueError(
            f"Invalid RK_order: '{par.RK_order}'. "
            f"Expected one of ['RK1', 'RK2', 'RK3'].")
        
    # Primitive variables recovery at the end of the timestep
    _prim_recovery(MHD, Ngc, eos)
    
    #evaluate the divergence of B field 
    MHD.divB = div_cell_vector(g, MHD.bfi1, MHD.bfi2)
    
    #return the updated class object of the fluid state on the next timestep 
    return MHD



def flux_calc_MHD_8wave(g, MHD, par, eos):
    """
    Compute residuals (flux divergences + sources) of conservative MHD vars.

    Parameters
    ----------
    g : object
        Computational grid object.
    MHD : object
        Fluid state (primitive + conservative variables).
    eos : object
        Equation of state.
    par : object
        Simulation parameters (flux solver, reconstruction method, etc.).

    Returns
    -------
    ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 : ndarrays
        Array of residuals for conservative variables.
        shape (Nx1,Nx2) for all residuals, i.e. only real cells are included.

    Notes
    -----
    - Governing update equation:
      
      .. math::
          \\frac{du}{dt} = -\\nabla \\cdot F(u) + S(u)

    - Steps:
      1. fill the ghost cells according to boundary conditions.
      2. Reconstruct states at faces (piecewise constant/linear/PPM/WENO).
      3. Solve Riemann problems to compute face fluxes.
          The latter is the key ingredient of Godunov-type methods (Godunov 1959), where
          the conservative states in neighbouring cells shares the fluxes between each other.
          The general idea here, that the flux can be calculated, using the solution of the Riemann problem, 
          because the states in adjusement cells represent the arbitrary discontinuity of the fluid. 
          #################################################################
          (see E.F. Toro "Riemann Solvers and Numerical Methods for Fluid Dynamics: A practical introduction" (2009))
          #################################################################
      4. Compute divergence of fluxes across each cell.
      5. Compute Powell's source terms for the momentum, energy and induction equations
      6. Add curvature source terms (for curvilinear grids) for momentum.
      
    - Returns residuals in conservative form, ready for RK update.
    """
    #fill the ghost cells
    MHD = boundCond_MHD(g, par.BC, par.BCm, MHD, par.BC_fixed)

    #re-evaluate a state- or time-dependent body force for THIS RK stage
    #(self-gravity, Coriolis, an orbiting perturber -- see gravity.py).
    #A static force leaves body_force = None and simply keeps the F1/F2
    #the initial condition wrote.
    if MHD.body_force is not None:
        MHD.body_force(g, MHD, par)
    
    #make copies of ghost cell and real cell numbers in each direction
    #to simplify indexing below 
    Ngc = g.Ngc 
    Nx1 = g.Nx1
    Nx2 = g.Nx2
    
    #nulifying the divergence of the magnetic field 
    MHD.divB[:,:] = 0.0
    
    #residuals initialization (only for real cells)
    ResM  = np.zeros((Nx1, Nx2))
    ResV1 = np.zeros((Nx1, Nx2))
    ResV2 = np.zeros((Nx1, Nx2))
    ResV3 = np.zeros((Nx1, Nx2))
    ResE  = np.zeros((Nx1, Nx2))
    ResB1 = np.zeros((Nx1, Nx2))
    ResB2 = np.zeros((Nx1, Nx2))
    ResB3 = np.zeros((Nx1, Nx2))
    
    #fluxes in 1-dimension 
    if (g.Nx1 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 1-dim
        #here we reconstruct density, 3 components of velocity and pressure
        dens_rec_L, dens_rec_R = VarReconstruct(MHD.dens, g, par.rec_type, 1)
        vel1_rec_L, vel1_rec_R = VarReconstruct(MHD.vel1, g, par.rec_type, 1)
        vel2_rec_L, vel2_rec_R = VarReconstruct(MHD.vel2, g, par.rec_type, 1)
        vel3_rec_L, vel3_rec_R = VarReconstruct(MHD.vel3, g, par.rec_type, 1)
        pres_rec_L, pres_rec_R = VarReconstruct(MHD.pres, g, par.rec_type, 1)
        bfi1_rec_L, bfi1_rec_R = VarReconstruct(MHD.bfi1, g, par.rec_type, 1)
        bfi2_rec_L, bfi2_rec_R = VarReconstruct(MHD.bfi2, g, par.rec_type, 1)
        bfi3_rec_L, bfi3_rec_R = VarReconstruct(MHD.bfi3, g, par.rec_type, 1)
        
        #fluxes calculation with approximate Riemann solver (see flux_type) in 1-dim
        Fmass, Fmom1, Fmom2, Fmom3, Fetot, \
        Fbfi1, Fbfi2, Fbfi3 = \
            Riemann_MHD(dens_rec_L, dens_rec_R, \
            vel1_rec_L, vel1_rec_R, vel2_rec_L, vel2_rec_R, vel3_rec_L, vel3_rec_R, \
            pres_rec_L, pres_rec_R,\
            bfi1_rec_L, bfi1_rec_R, bfi2_rec_L, bfi2_rec_R, bfi3_rec_L, bfi3_rec_R, \
            eos, par.solver_type, 1)
        
        #residuals calculation for mass, 3 components of momentum, 
        #total energy and normal components of magnetic field in 1-dim
        ResM  = ( Fmass[1:,:]*g.fS1[1:,:] - Fmass[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        ResV1 = ( Fmom1[1:,:]*g.fS1[1:,:] - Fmom1[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        ResV2 = ( Fmom2[1:,:]*g.fS1[1:,:] - Fmom2[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        ResV3 = ( Fmom3[1:,:]*g.fS1[1:,:] - Fmom3[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        ResE  = ( Fetot[1:,:]*g.fS1[1:,:] - Fetot[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        ResB2 = ( Fbfi2[1:,:]*g.fS1[1:,:] - Fbfi2[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        ResB3 = ( Fbfi3[1:,:]*g.fS1[1:,:] - Fbfi3[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        
        #calculation of magnetic field divergence for Powell (1999) 8-wave approach
        MHD.divB = ( (bfi1_rec_L[1:,:] + bfi1_rec_R[1:,:]) / 2.0 * g.fS1[1:,:] - \
            ( bfi1_rec_L[:-1,:] + bfi1_rec_R[:-1,:]) / 2.0 * g.fS1[:-1,:] ) / g.cVol[:,:]
        
    #fluxes in 2-dimension
    if (g.Nx2 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 2-dim
        #here we reconstruct density, 3 components of velocity and pressure
        dens_rec_L, dens_rec_R = VarReconstruct(MHD.dens, g, par.rec_type, 2)
        pres_rec_L, pres_rec_R = VarReconstruct(MHD.pres, g, par.rec_type, 2)
        vel1_rec_L, vel1_rec_R = VarReconstruct(MHD.vel1, g, par.rec_type, 2)
        vel2_rec_L, vel2_rec_R = VarReconstruct(MHD.vel2, g, par.rec_type, 2)
        vel3_rec_L, vel3_rec_R = VarReconstruct(MHD.vel3, g, par.rec_type, 2)
        bfi1_rec_L, bfi1_rec_R = VarReconstruct(MHD.bfi1, g, par.rec_type, 2)
        bfi2_rec_L, bfi2_rec_R = VarReconstruct(MHD.bfi2, g, par.rec_type, 2)
        bfi3_rec_L, bfi3_rec_R = VarReconstruct(MHD.bfi3, g, par.rec_type, 2)
     
        #fluxes calculation with approximate Riemann solver (see flux_type) in 2-dim
        Fmass, Fmom1, Fmom2, Fmom3, Fetot, \
        Fbfi1, Fbfi2, Fbfi3 = \
            Riemann_MHD(dens_rec_L, dens_rec_R, \
            vel1_rec_L, vel1_rec_R, vel2_rec_L, vel2_rec_R, vel3_rec_L, vel3_rec_R, \
            pres_rec_L, pres_rec_R, \
            bfi1_rec_L, bfi1_rec_R, bfi2_rec_L, bfi2_rec_R, bfi3_rec_L, bfi3_rec_R, \
            eos, par.solver_type, 2)
        
        #residuals calculation for mass, 3 components of momentum, 
        #total energy and normal components of magnetic field in 2-dim
        #here we add the fluxes differences to the residuals after 1-dim calculation
        ResM  += ( Fmass[:,1:]*g.fS2[:,1:] - Fmass[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResV1 += ( Fmom1[:,1:]*g.fS2[:,1:] - Fmom1[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResV2 += ( Fmom2[:,1:]*g.fS2[:,1:] - Fmom2[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResV3 += ( Fmom3[:,1:]*g.fS2[:,1:] - Fmom3[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResE  += ( Fetot[:,1:]*g.fS2[:,1:] - Fetot[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResB1 += ( Fbfi1[:,1:]*g.fS2[:,1:] - Fbfi1[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResB3 += ( Fbfi3[:,1:]*g.fS2[:,1:] - Fbfi3[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
         
        #calculation of magnetic field divergence for Powell (1999) 8-wave approach
        MHD.divB += ( (bfi2_rec_L[:,1:] + bfi2_rec_R[:,1:]) / 2.0 * g.fS2[:,1:] - \
            ( bfi2_rec_L[:,:-1] + bfi2_rec_R[:,:-1]) / 2.0 * g.fS2[:,:-1] ) / g.cVol[:,:]     
    
    #finally, here we add the external force terms
    #we add forces in momentum res, while in energy we add Power = Force*Vel 
    ResV1 += - MHD.dens[Ngc:-Ngc, Ngc:-Ngc] * MHD.F1 
    ResV2 += - MHD.dens[Ngc:-Ngc, Ngc:-Ngc] * MHD.F2 
    ResE  += - MHD.dens[Ngc:-Ngc, Ngc:-Ngc] * \
        (MHD.F1 * MHD.vel1[Ngc:-Ngc, Ngc:-Ngc] + \
        MHD.F2 * MHD.vel2[Ngc:-Ngc, Ngc:-Ngc])
    
    #curvature source terms 
    STv1, STv2, STv3, STm1, STm2, STm3 = curv_source_MHD_8wave(g, MHD)
    
    #Powell 8-wave divB cleaning method (simply add the sources in RHS)
    #curvature sources are also added here
    ResV1 += MHD.bfi1[Ngc:-Ngc, Ngc:-Ngc] * MHD.divB - STv1
    ResV2 += MHD.bfi2[Ngc:-Ngc, Ngc:-Ngc] * MHD.divB - STv2
    ResV3 += MHD.bfi3[Ngc:-Ngc, Ngc:-Ngc] * MHD.divB - STv3
    
    ResE  += MHD.divB * \
        (MHD.vel1[Ngc:-Ngc, Ngc:-Ngc] * MHD.bfi1[Ngc:-Ngc, Ngc:-Ngc] + \
        MHD.vel2[Ngc:-Ngc, Ngc:-Ngc] * MHD.bfi2[Ngc:-Ngc, Ngc:-Ngc] + \
        MHD.vel3[Ngc:-Ngc, Ngc:-Ngc] * MHD.bfi3[Ngc:-Ngc, Ngc:-Ngc])
        
    ResB1 += MHD.vel1[Ngc:-Ngc, Ngc:-Ngc] * MHD.divB - STm1
    ResB2 += MHD.vel2[Ngc:-Ngc, Ngc:-Ngc] * MHD.divB - STm2
    ResB3 += MHD.vel3[Ngc:-Ngc, Ngc:-Ngc] * MHD.divB - STm3
    
    #return the residuals for mass, 3 components of momentum, total energy and magnetic field
    return ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3




def curv_source_MHD_8wave(g, MHD):
    """
    Compute geometric source terms for the MHD equations 
    in curvilinear coordinates (finite-volume formulation).

    In Cartesian coordinates, the Euler equations are source-free, but in 
    curvilinear geometries (e.g., cylindrical, spherical) additional terms 
    appear due to the divergence operator expressed in non-Cartesian bases.
    This function evaluates those terms for momentum and induction equations.

    Parameters
    ----------
    g : object
        Grid object
    MHD : object
        Fluid state containing:
        - ``dens`` : ndarray, density field.
        - ``pres`` : ndarray, pressure field.
        - ``vel1, vel2, vel3`` : ndarray, velocity.
        - ``Bfi1, Bfi2, Bfi3`` : ndarray, magnetic field.

    Returns
    -------
    STv1, STv2, STv3 : ndarray
        Momentum source terms.
    
    STm1, STm2, STm3 : ndarray
        Magnetic field source terms.

    Notes
    -----
    - Arrays are allocated inside the real grid (excluding ghost cells).
    """
    Ngc = g.Ngc 
    STv1 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    STv2 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    STv3 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    STm1 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    STm2 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    STm3 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    
    # source-free; nothing further to do
    if g.geom == 'cart':
        return STv1, STv2, STv3, STm1, STm2, STm3
    
    r    = g.cx1[Ngc:-Ngc,Ngc:-Ngc]
    dens = MHD.dens[Ngc:-Ngc,Ngc:-Ngc]
    pres = MHD.pres[Ngc:-Ngc,Ngc:-Ngc]
    v1   = MHD.vel1[Ngc:-Ngc,Ngc:-Ngc]
    v2   = MHD.vel2[Ngc:-Ngc,Ngc:-Ngc]
    v3   = MHD.vel3[Ngc:-Ngc,Ngc:-Ngc]
    b1   = MHD.bfi1[Ngc:-Ngc,Ngc:-Ngc]
    b2   = MHD.bfi2[Ngc:-Ngc,Ngc:-Ngc]
    b3   = MHD.bfi3[Ngc:-Ngc,Ngc:-Ngc]
    
    #cylindrical (R,Z) geometry
    if (g.geom == 'cyl'):
        
        STv1 = (pres + (b1**2 + b2**2 - b3**2) / 2.0 + dens * v3**2) / r
        STv3 = (b3 * b1 - dens * v3 * v1) / r
        STm3 = (b3 * v1 - b1 * v3) / r
      
    #polar (R,phi) geometry
    if (g.geom == 'pol'):
        
        STv1 = (pres + (b1**2 + b3**2 - b2**2) / 2.0 + dens * v2**2) / r
        STv2 = (b2 * b1 - dens * v2 * v1) / r
        STm2 = (b2 * v1 - b1 * v2) / r
    
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
            
        STv1 = ( 2.0 * pres + b1**2 + dens * (v2**2 + v3**2) ) / r
        STv2 = ( pres + (b1**2 + b2**2 - b3**2)/2.0 + dens * v3**2) * cot / r  - \
            (dens * v1 * v2 - b1 * b2) / r
        STv3 = (b2 * b3 - dens * v2 * v3 ) * cot / r + (b1 * b3 - dens * v1 * v3) / r
        STm2 = (b2 * v1 - b1 * v2) / r
        STm3 = (b3 * v1 - b1 * v3) / r + (b3 * v2 - b2 * v3) * cot / r  
            
    return STv1, STv2, STv3, STm1, STm2, STm3


