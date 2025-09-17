# -*- coding: utf-8 -*-
"""
===============================================================================
MHD_one_step_CT.py
===============================================================================

2D Magnetohydrodynamics (MHD) Solver with Constrained Transport (CT)
====================================================================

This module provides routines for solving the 2D compressible
magnetohydrodynamics (MHD) equations using finite-volume Godunov-type
methods. The solver employs high-order reconstruction, approximate Riemann
solvers, and Runge-Kutta (RK) timestepping schemes. Magnetic divergence
is controlled via the Constrained Transport (CT) method 'flux-CT'.

Main Components
---------------
- ``MHD2D_CT`` : container class managing grid, state, EOS, and parameters.
- ``CFLcondition_MHD`` : compute timestep from CFL stability condition.
- ``oneStep_MHD_RK_CT`` : advance MHD state by one timestep (RK1/RK2/RK3).
- ``flux_calc_MHD_CT`` : compute residuals of conservative variables.
- ``MHD_curv_sources`` : evaluate curvature source terms (e.g., cylindrical).
- ``boundCond_electric_field`` : Fill ghost cells for face-centered electric field E3 along x1 and x2.
      (E3 = Ez for MHD in 2D XY coordinates, for instance)

Features
--------
- Spatial accuracy via piecewise constant, linear, PPM, or WENO-type reconstructions.
- Riemann solvers: Local Lax-Friedrichs (LLF), HLL, HLLD.
- Temporal accuracy: TVD Runge-Kutta methods (1st, 2nd, or 3rd order).
- Divergence-free magnetic field evolution with CT.
- Curvilinear support (currently cylindrical geometry implemented).

@author:
    mrkondratyev
"""

from MHD_phys import *
from reconstruction import VarReconstruct 
import numpy as np 
import copy 
from grid_misc import interp_face_to_cell, div_face_vector


class MHD2D_CT:
    """
    Container class for 2D compressible hydrodynamics routines.

    Attributes
    ----------
    grid : object
        Grid object with domain sizes, spacing, volumes, and face areas.
    MHD : object
        SimState object containing primitive and conservative variables.
    par : object
        Simulation parameters including CFL, RK_order, flux_type, rec_type, phystime, phystimefin.
    eos : object
        Equation of state object.
    """

    def __init__(self, grid, MHD, eos, par):
        """
        Initialize the Hydro2D container.

        Parameters
        ----------
        grid : object
            Grid object.
        MHD : object
            SimState object.
        eos : object
            Equation of state object.
        par : object
            Simulation parameters object.
        """
        self.grid = grid
        self.MHD = MHD
        self.eos = eos
        self.par = par

    def step_RK(self):
        """
        Perform a single Runge-Kutta timestep.

        Returns
        -------
        fluid : object
            Updated SimState object.
        """
        dt = min(CFLcondition_MHD(self.grid, self.MHD, self.eos, self.par.CFL),
                 self.par.timefin - self.par.timenow)
        self.fluid = oneStep_MHD_RK_CT(self.grid, self.MHD, self.eos, self.par, dt)
        self.par.timenow += dt
        return self.fluid



def CFLcondition_MHD(grid, MHD, eos, CFL):
    """
    Compute timestep based on CFL stability condition for MHD.
    
    The CFL condition states that the fastest wave in the system must not
    travel more than one cell per timestep.
    
    Parameters
    ----------
    grid : object
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
    """
    Ngc = grid.Ngc
    
    #sound speed calculation for whole domain
    csound = eos.sound_speed(MHD.dens[Ngc:-Ngc, Ngc:-Ngc], MHD.pres[Ngc:-Ngc, Ngc:-Ngc])
    
    #fast magnetosonic speed calculation for whole domain 
    cfast = max_wavespeed_MHD(csound, \
        MHD.bfi1[Ngc:-Ngc, Ngc:-Ngc], MHD.bfi2[Ngc:-Ngc, Ngc:-Ngc], MHD.bfi3[Ngc:-Ngc, Ngc:-Ngc], MHD.dens[Ngc:-Ngc, Ngc:-Ngc])
    
    #FIRST APPROACH
    #dt1 = np.min( grid.dx1[Ngc:-Ngc, Ngc:-Ngc] / (1e-14 + np.abs(MHD.vel1[Ngc:-Ngc, Ngc:-Ngc]) + cfast) )
    #dt2 = np.min( grid.dx2[Ngc:-Ngc, Ngc:-Ngc] / (1e-14 + np.abs(MHD.vel2[Ngc:-Ngc, Ngc:-Ngc]) + cfast) )    
    #return  CFL * min(dt1, dt2)
    
    #SECOND APPROACH 
    dt_inv = np.max((np.abs(MHD.vel1[Ngc:-Ngc, Ngc:-Ngc]) + cfast)/grid.dx1[Ngc:-Ngc, Ngc:-Ngc] + \
        (np.abs(MHD.vel2[Ngc:-Ngc, Ngc:-Ngc]) + cfast)/grid.dx2[Ngc:-Ngc, Ngc:-Ngc])
    return CFL/dt_inv


 
def oneStep_MHD_RK_CT(grid, MHD, eos, par, dt):
    """
    Advance the MHD state by one timestep using RK1, RK2, or RK3 schemes.

    Parameters
    ----------
    grid : object
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
    - Residuals are computed via ``flux_calc_MHD_CT``.
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
    Ngc = grid.Ngc
    
    #here we define the copy for the auxilary fluid state
    MHD_h = copy.deepcopy(MHD)
    
    #conservative variables at the beginning of timestep
    MHD.mass, MHD.mom1, MHD.mom2, MHD.mom3, MHD.etot, MHD.bcon1, MHD.bcon2, MHD.bcon3 = \
        prim2cons_nr_MHD(MHD.dens[Ngc:-Ngc,Ngc:-Ngc], 
        MHD.vel1[Ngc:-Ngc,Ngc:-Ngc], MHD.vel2[Ngc:-Ngc,Ngc:-Ngc],  
        MHD.vel3[Ngc:-Ngc,Ngc:-Ngc], MHD.pres[Ngc:-Ngc,Ngc:-Ngc], 
        MHD.bfi1[Ngc:-Ngc,Ngc:-Ngc], MHD.bfi2[Ngc:-Ngc,Ngc:-Ngc], 
        MHD.bfi3[Ngc:-Ngc,Ngc:-Ngc], eos)
    
    #residuals for conservative variables calculation
    #1st Runge-Kutta iteration - predictor stage
    ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = flux_calc_MHD_CT(grid, MHD, par, eos)
    
    # Conservative update - 1st RK stage (predictor)
    MHD_h.mass = MHD.mass - dt * ResM 
    MHD_h.mom1 = MHD.mom1 - dt * ResV1 
    MHD_h.mom2 = MHD.mom2 - dt * ResV2 
    MHD_h.mom3 = MHD.mom3 - dt * ResV3 
    MHD_h.etot = MHD.etot - dt * ResE 
    MHD_h.fb1 = MHD.fb1 - dt * ResB1
    MHD_h.fb2 = MHD.fb2 - dt * ResB2
    MHD_h.bcon3 = MHD.bcon3 - dt * ResB3
    
    #first order Runge-Kutta scheme
    if (par.RK_order == 'RK1'): 
        
        #simply rewrite the conservative state here for clarity
        MHD.mass = MHD_h.mass
        MHD.mom1 = MHD_h.mom1
        MHD.mom2 = MHD_h.mom2
        MHD.mom3 = MHD_h.mom3
        MHD.etot = MHD_h.etot
        MHD.fb1 = MHD_h.fb1
        MHD.fb2 = MHD_h.fb2
        MHD.bcon3 = MHD_h.bcon3
    
    #second-order Runge-Kutta scheme
    if (par.RK_order == 'RK2'):
        
        #interpolation from staggererd to cell-centered fields
        MHD_h.bcon1, MHD_h.bcon2 = interp_face_to_cell(grid, MHD_h.fb1, MHD_h.fb2)
        
        #Primitive variables recovery after predictor stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        MHD_h.dens[Ngc:-Ngc, Ngc:-Ngc], MHD_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            MHD_h.vel2[Ngc:-Ngc, Ngc:-Ngc], MHD_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            MHD_h.pres[Ngc:-Ngc, Ngc:-Ngc], MHD_h.bfi1[Ngc:-Ngc, Ngc:-Ngc],  \
            MHD_h.bfi2[Ngc:-Ngc, Ngc:-Ngc], MHD_h.bfi3[Ngc:-Ngc, Ngc:-Ngc] = \
            cons2prim_nr_MHD(MHD_h.mass, MHD_h.mom1, MHD_h.mom2, MHD_h.mom3, MHD_h.etot, \
            MHD_h.bcon1, MHD_h.bcon2, MHD_h.bcon3, eos) 
            
        #2nd Runge-Kutta stage - corrector
        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = flux_calc_MHD_CT(grid, MHD_h, par, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy
        MHD.mass = (MHD_h.mass + MHD.mass) / 2.0 - dt * ResM / 2.0
        MHD.mom1 = (MHD_h.mom1 + MHD.mom1) / 2.0 - dt * ResV1 / 2.0 
        MHD.mom2 = (MHD_h.mom2 + MHD.mom2) / 2.0 - dt * ResV2 / 2.0 
        MHD.mom3 = (MHD_h.mom3 + MHD.mom3) / 2.0 - dt * ResV3 / 2.0  
        MHD.etot = (MHD_h.etot + MHD.etot) / 2.0 - dt * ResE / 2.0 
        MHD.fb1 = (MHD_h.fb1 + MHD.fb1) / 2.0 - dt * ResB1 / 2.0 
        MHD.fb2 = (MHD_h.fb2 + MHD.fb2) / 2.0 - dt * ResB2 / 2.0 
        MHD.bcon3 = (MHD_h.bcon3 + MHD.bcon3) / 2.0 - dt * ResB3 / 2.0  
        
    if (par.RK_order == 'RK3'):
        
        #interpolation from staggererd to cell-centered fields
        MHD_h.bcon1, MHD_h.bcon2 = interp_face_to_cell(grid, MHD_h.fb1, MHD_h.fb2)
        
        #Primitive variables recovery after predictor stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        MHD_h.dens[Ngc:-Ngc, Ngc:-Ngc], MHD_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            MHD_h.vel2[Ngc:-Ngc, Ngc:-Ngc], MHD_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            MHD_h.pres[Ngc:-Ngc, Ngc:-Ngc], MHD_h.bfi1[Ngc:-Ngc, Ngc:-Ngc],  \
            MHD_h.bfi2[Ngc:-Ngc, Ngc:-Ngc], MHD_h.bfi3[Ngc:-Ngc, Ngc:-Ngc] = \
            cons2prim_nr_MHD(MHD_h.mass, MHD_h.mom1, MHD_h.mom2, MHD_h.mom3, MHD_h.etot, \
            MHD_h.bcon1, MHD_h.bcon2, MHD_h.bcon3, eos) 
            
        #2nd Runge-Kutta stage
        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = flux_calc_MHD_CT(grid, MHD_h, par, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy
        MHD_h.mass = (MHD_h.mass + 3.0 * MHD.mass) / 4.0 - dt * ResM / 4.0
        MHD_h.mom1 = (MHD_h.mom1 + 3.0 * MHD.mom1) / 4.0 - dt * ResV1 / 4.0 
        MHD_h.mom2 = (MHD_h.mom2 + 3.0 * MHD.mom2) / 4.0 - dt * ResV2 / 4.0 
        MHD_h.mom3 = (MHD_h.mom3 + 3.0 * MHD.mom3) / 4.0 - dt * ResV3 / 4.0  
        MHD_h.etot = (MHD_h.etot + 3.0 * MHD.etot) / 4.0 - dt * ResE / 4.0 
        MHD_h.fb1 = (MHD_h.fb1 + 3.0 * MHD.fb1) / 4.0 - dt * ResB1 / 4.0 
        MHD_h.fb2 = (MHD_h.fb2 + 3.0 * MHD.fb2) / 4.0 - dt * ResB2 / 4.0 
        MHD_h.bcon3 = (MHD_h.bcon3 + 3.0 * MHD.bcon3) / 4.0 - dt * ResB3 / 4.0  
        
        #interpolation from staggererd to cell-centered fields
        MHD_h.bcon1, MHD_h.bcon2 = interp_face_to_cell(grid, MHD_h.fb1, MHD_h.fb2)
        
        # Primitive variables recovery after the second stage
        #density, 3 components of velocity and pressure are evaluated 
        MHD_h.dens[Ngc:-Ngc, Ngc:-Ngc], MHD_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            MHD_h.vel2[Ngc:-Ngc, Ngc:-Ngc], MHD_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            MHD_h.pres[Ngc:-Ngc, Ngc:-Ngc], MHD_h.bfi1[Ngc:-Ngc, Ngc:-Ngc],  \
            MHD_h.bfi2[Ngc:-Ngc, Ngc:-Ngc], MHD_h.bfi3[Ngc:-Ngc, Ngc:-Ngc] = \
            cons2prim_nr_MHD(MHD_h.mass, MHD_h.mom1, MHD_h.mom2, MHD_h.mom3, MHD_h.etot, \
            MHD_h.bcon1, MHD_h.bcon2, MHD_h.bcon3, eos) 
        
        #3rd Runge-Kutta stage
        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = flux_calc_MHD_CT(grid, MHD_h, par, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, 3 components of momentum, total energy and 3 comps of magnetic field
        MHD.mass = (2.0 * MHD_h.mass + MHD.mass) / 3.0 - 2.0 * dt * ResM / 3.0
        MHD.mom1 = (2.0 * MHD_h.mom1 + MHD.mom1) / 3.0 - 2.0 * dt * ResV1 / 3.0 
        MHD.mom2 = (2.0 * MHD_h.mom2 + MHD.mom2) / 3.0 - 2.0 * dt * ResV2 / 3.0 
        MHD.mom3 = (2.0 * MHD_h.mom3 + MHD.mom3) / 3.0 - 2.0 * dt * ResV3 / 3.0  
        MHD.etot = (2.0 * MHD_h.etot + MHD.etot) / 3.0 - 2.0 * dt * ResE / 3.0 
        MHD.fb1 = (2.0 * MHD_h.fb1 + MHD.fb1) / 3.0 - 2.0 * dt * ResB1 / 3.0 
        MHD.fb2 = (2.0 * MHD_h.fb2 + MHD.fb2) / 3.0 - 2.0 * dt * ResB2 / 3.0 
        MHD.bcon3 = (2.0 * MHD_h.bcon3 + MHD.bcon3) / 3.0 - 2.0 * dt * ResB3 / 3.0  
        
    #interpolation from staggererd to cell-centered fields
    MHD.bcon1, MHD.bcon2 = interp_face_to_cell(grid, MHD.fb1, MHD.fb2)
    
    # Primitive variables recovery at the end of the timestep
    #density, 3 components of velocity and pressure are evaluated 
    MHD.dens[Ngc:-Ngc, Ngc:-Ngc], MHD.vel1[Ngc:-Ngc, Ngc:-Ngc], \
        MHD.vel2[Ngc:-Ngc, Ngc:-Ngc], MHD.vel3[Ngc:-Ngc, Ngc:-Ngc], \
        MHD.pres[Ngc:-Ngc, Ngc:-Ngc], MHD.bfi1[Ngc:-Ngc, Ngc:-Ngc],  \
        MHD.bfi2[Ngc:-Ngc, Ngc:-Ngc], MHD.bfi3[Ngc:-Ngc, Ngc:-Ngc] = \
        cons2prim_nr_MHD(MHD.mass, MHD.mom1, MHD.mom2, MHD.mom3, MHD.etot, \
        MHD.bcon1, MHD.bcon2, MHD.bcon3, eos) 
    
    MHD.divB = div_face_vector(grid, MHD.fb1, MHD.fb2)
    
    #return the updated class object of the fluid state on the next timestep 
    return MHD



def flux_calc_MHD_CT(grid, MHD, par, eos):
    """
    Compute residuals (flux divergences + sources) of conservative MHD vars.

    Parameters
    ----------
    grid : object
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
        shape (Nx1,Nx2) for all residuals except ResB1, ResB2, i.e. only real cells are included.
        ResB1.shape = (Nx1+1,Nx2) (real faces along 1-direction)
        ResB2.shape = (Nx1,Nx2+1) (real faces along 2-direction)

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
      4. Compute divergence of fluxes across each cell (mass, momentum and energy).
      5. Compute Faraday's law for Bfield along 3-axis
      6. Compute electric fields at edges through averaging face fluxes 
      7. Add curvature source terms (for curvilinear grids) for momentum.
      
    - Returns residuals in conservative form, ready for RK update.
    """
    #fill the ghost cells
    MHD = boundCond_MHD(grid, par.BC, MHD)
    
    #make copies of ghost cell and real cell numbers to simplify indexing 
    Ngc = grid.Ngc 
    Nx1 = grid.Nx1
    Nx2 = grid.Nx2
    Nx1r = grid.Nx1r
    Nx2r = grid.Nx2r
    
    #nulifying the divergence of the magnetic field 
    MHD.divB[:,:] = 0.0
    
    #residuals initialization (only for real cells)
    ResM = np.zeros((Nx1, Nx2))
    ResV1 = np.zeros((Nx1, Nx2))
    ResV2 = np.zeros((Nx1, Nx2))
    ResV3 = np.zeros((Nx1, Nx2))
    ResE = np.zeros((Nx1, Nx2))
    ResB1 = np.zeros((Nx1 + 1, Nx2))
    ResB2 = np.zeros((Nx1, Nx2 + 1))
    ResB3 = np.zeros((Nx1, Nx2))
    
    fluxB21 = np.zeros_like((grid.fx1))
    fluxB12 = np.zeros_like((grid.fx2))
    #electric field
    Efld3 = np.zeros((Nx1 + 1, Nx2 + 1))
    
    #fluxes in 1-dimension 
    if (grid.Nx1 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 1-dim
        #here we reconstruct density, 3 components of velocity and pressure
        dens_rec_L, dens_rec_R = VarReconstruct(MHD.dens, grid, par.rec_type, 1)
        vel1_rec_L, vel1_rec_R = VarReconstruct(MHD.vel1, grid, par.rec_type, 1)
        vel2_rec_L, vel2_rec_R = VarReconstruct(MHD.vel2, grid, par.rec_type, 1)
        vel3_rec_L, vel3_rec_R = VarReconstruct(MHD.vel3, grid, par.rec_type, 1)
        pres_rec_L, pres_rec_R = VarReconstruct(MHD.pres, grid, par.rec_type, 1)
        bfi2_rec_L, bfi2_rec_R = VarReconstruct(MHD.bfi2, grid, par.rec_type, 1)
        bfi3_rec_L, bfi3_rec_R = VarReconstruct(MHD.bfi3, grid, par.rec_type, 1)
        
        #fluxes calculation with approximate Riemann solver (see flux_type) in 1-dim
        Fmass, Fmom1, Fmom2, Fmom3, Fetot, Fbfi1, fluxB21[Ngc:Nx1r+1,Ngc:-Ngc], Fbfi3 = \
            Riemann_flux_nr_MHD(dens_rec_L, dens_rec_R, vel1_rec_L, vel1_rec_R, \
            vel2_rec_L, vel2_rec_R, vel3_rec_L, vel3_rec_R, \
            pres_rec_L, pres_rec_R, MHD.fb1[:,:], MHD.fb1[:,:], \
            bfi2_rec_L, bfi2_rec_R, bfi3_rec_L, bfi3_rec_R, eos, par.flux_type, 1)
        
        #residuals calculation for mass, 3 components of momentum, 
        #total energy and 3 component of magnetic field in 1-dim
        ResM = ( Fmass[1:,:]*grid.fS1[1:,:] - Fmass[:-1,:]*grid.fS1[:-1,:] ) / grid.cVol[:,:]
        ResV1 = ( Fmom1[1:,:]*grid.fS1[1:,:] - Fmom1[:-1,:]*grid.fS1[:-1,:] ) / grid.cVol[:,:]
        ResV2 = ( Fmom2[1:,:]*grid.fS1[1:,:] - Fmom2[:-1,:]*grid.fS1[:-1,:] ) / grid.cVol[:,:]
        ResV3 = ( Fmom3[1:,:]*grid.fS1[1:,:] - Fmom3[:-1,:]*grid.fS1[:-1,:] ) / grid.cVol[:,:]
        ResE = ( Fetot[1:,:]*grid.fS1[1:,:] - Fetot[:-1,:]*grid.fS1[:-1,:] ) / grid.cVol[:,:]
        ResB3 = ( Fbfi3[1:,:]*grid.edg2[1:,:] - Fbfi3[:-1,:]*grid.edg2[:-1,:] ) / grid.fS3[:,:]
        
    #fluxes in 2-dimension
    if (grid.Nx2 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 2-dim
        #here we reconstruct density, 3 components of velocity and pressure
        dens_rec_L, dens_rec_R = VarReconstruct(MHD.dens, grid, par.rec_type, 2)
        pres_rec_L, pres_rec_R = VarReconstruct(MHD.pres, grid, par.rec_type, 2)
        vel1_rec_L, vel1_rec_R = VarReconstruct(MHD.vel1, grid, par.rec_type, 2)
        vel2_rec_L, vel2_rec_R = VarReconstruct(MHD.vel2, grid, par.rec_type, 2)
        vel3_rec_L, vel3_rec_R = VarReconstruct(MHD.vel3, grid, par.rec_type, 2)
        bfi1_rec_L, bfi1_rec_R = VarReconstruct(MHD.bfi1, grid, par.rec_type, 2)
        bfi3_rec_L, bfi3_rec_R = VarReconstruct(MHD.bfi3, grid, par.rec_type, 2)
     
        #fluxes calculation with approximate Riemann solver (see flux_type) in 2-dim
        Fmass, Fmom1, Fmom2, Fmom3, Fetot, fluxB12[Ngc:-Ngc,Ngc:Nx2r+1], Fbfi2, Fbfi3 = \
            Riemann_flux_nr_MHD(dens_rec_L, dens_rec_R, vel1_rec_L, vel1_rec_R, \
            vel2_rec_L, vel2_rec_R, vel3_rec_L, vel3_rec_R, \
            pres_rec_L, pres_rec_R, bfi1_rec_L, bfi1_rec_R, \
            MHD.fb2[:,:], MHD.fb2[:,:], bfi3_rec_L, bfi3_rec_R, eos, par.flux_type, 2)
        
        #residuals calculation for mass, 3 components of momentum, 
        #total energy and 3 components of magnetic field in 2-dim
        #here we add the fluxes differences to the residuals after 1-dim calculation
        ResM += ( Fmass[:,1:]*grid.fS2[:,1:] - Fmass[:,:-1]*grid.fS2[:,:-1] ) / grid.cVol[:,:]
        ResV1 += ( Fmom1[:,1:]*grid.fS2[:,1:] - Fmom1[:,:-1]*grid.fS2[:,:-1] ) / grid.cVol[:,:]
        ResV2 += ( Fmom2[:,1:]*grid.fS2[:,1:] - Fmom2[:,:-1]*grid.fS2[:,:-1] ) / grid.cVol[:,:]
        ResV3 += ( Fmom3[:,1:]*grid.fS2[:,1:] - Fmom3[:,:-1]*grid.fS2[:,:-1] ) / grid.cVol[:,:]
        ResE += ( Fetot[:,1:]*grid.fS2[:,1:] - Fetot[:,:-1]*grid.fS2[:,:-1] ) / grid.cVol[:,:]
        ResB3 += ( Fbfi3[:,1:]*grid.edg1[:,1:] - Fbfi3[:,:-1]*grid.edg1[:,:-1] ) / grid.fS3[:,:]
     
    #apply ghost cells for electric field 
    fluxB21, fluxB12 = boundCond_electric_field(grid, fluxB21, fluxB12, par.BC)
    
    #electric field on the edges
    Efld3 = -(fluxB21[Ngc:Nx1r+1,Ngc-1:Nx2r] + fluxB21[Ngc:Nx1r+1,Ngc:Nx2r+1])/4.0 + \
        (fluxB12[Ngc-1:Nx1r,Ngc:Nx2r+1] + fluxB12[Ngc:Nx1r+1,Ngc:Nx2r+1])/4.0
       
    #residual update 
    ResB1 = (Efld3[:,1:]*grid.edg3[:,1:] - Efld3[:,:-1]*grid.edg3[:,:-1])/(grid.fS1[:,:]+1e-30)
    ResB2 = -(Efld3[1:,:]*grid.edg3[1:,:] - Efld3[:-1,:]*grid.edg3[:-1,:])/(grid.fS2[:,:]+1e-30)
      
    #curvature source terms
    STv1, STv2, STv3 = MHD_curv_sources(grid, MHD)
      
    #finally, here we add the external force and curvature source terms
    #we add forces in momentum res, while in energy we add Power = Force*Vel 
    ResV1 += - MHD.dens[Ngc:-Ngc, Ngc:-Ngc] * MHD.F1 - STv1
    ResV2 += - MHD.dens[Ngc:-Ngc, Ngc:-Ngc] * MHD.F2 - STv2
    ResV3 += - STv3
    ResE += - MHD.dens[Ngc:-Ngc, Ngc:-Ngc] * \
        (MHD.F1 * MHD.vel1[Ngc:-Ngc, Ngc:-Ngc] + \
         MHD.F2 * MHD.vel2[Ngc:-Ngc, Ngc:-Ngc])
                
        
    #return the residuals for mass, 3 components of momentum, total energy and magnetic field
    return ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3




def MHD_curv_sources(g, MHD):
    """
    Compute geometric source terms for the MHD equations 
    in curvilinear coordinates (finite-volume formulation for CT MHD).

    In Cartesian coordinates, the Euler equations are source-free, but in 
    curvilinear geometries (e.g., cylindrical, spherical) additional terms 
    appear due to the divergence operator expressed in non-Cartesian bases.
    This function evaluates those terms for momentum and induction equations.

    Currently implemented for cylindrical geometry ('cyl')

    Parameters
    ----------
    g : object
        Grid object containing:
        - ``geom`` : str, geometry type ('cyl' supported).
        - ``cx1`` : ndarray, radial cell-center positions.
        - ``Ngc`` : int, number of ghost cells.
        - ``Nx1, Nx2`` : int, number of grid points.
    MHD : object
        Fluid state containing:
        - ``dens`` : ndarray, density field.
        - ``pres`` : ndarray, pressure field.
        - ``vel1, vel2, vel3`` : ndarray, velocity.
        - ``Bfi1, Bfi2, Bfi3`` : ndarray, magnetic field.

    Returns
    -------
    STv1 : ndarray
        Radial momentum source term.
    STv2 : ndarray
        Axial momentum source term (zero in cylindrical geometry).
    STv3 : ndarray
        Azimuthal momentum source term.

    Notes
    -----
    - Arrays are allocated with the full grid size (including ghost cells).
    - Source terms are nonzero only inside the physical domain
      (ghost zones excluded).
    - Extension to spherical coordinates would require additional terms.
    """
    Ngc = g.Ngc 
    STv1 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    STv2 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    STv3 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    
    if (g.geom == 'cyl'):
        
        STv1 = (MHD.pres[Ngc:-Ngc,Ngc:-Ngc] + (MHD.bfi1[Ngc:-Ngc,Ngc:-Ngc]**2 + \
               MHD.bfi2[Ngc:-Ngc,Ngc:-Ngc]**2 - MHD.bfi3[Ngc:-Ngc,Ngc:-Ngc]**2)/2.0 + \
               MHD.dens[Ngc:-Ngc,Ngc:-Ngc] * MHD.vel3[Ngc:-Ngc,Ngc:-Ngc]**2) / \
               g.cx1[Ngc:-Ngc,Ngc:-Ngc]

        STv3 = (MHD.bfi3[Ngc:-Ngc,Ngc:-Ngc] * MHD.bfi1[Ngc:-Ngc,Ngc:-Ngc] - \
               MHD.dens[Ngc:-Ngc,Ngc:-Ngc] * MHD.vel3[Ngc:-Ngc,Ngc:-Ngc] * \
               MHD.vel1[Ngc:-Ngc,Ngc:-Ngc]) / g.cx1[Ngc:-Ngc,Ngc:-Ngc]
            
    return STv1, STv2, STv3




def boundCond_electric_field(grid, Efld3x1, Efld3x2, BC):
    """
    Apply boundary conditions to face-centered third component of the electric field
    (for instance Efld3x1 corresponds to Ez at faces along X-coordinate,
     and Efld3x2 -- to Ez at faces along Y-coordinate in 2D MHD for Cartesian geometry)

    To obtain Efld3 at cell edges, we have to take into account boundary conditions for the 
    electric field (for E3x1 we have to fill "ghost faces" along x2, and 
    for E3x2 we have the fill them along x1)
 
    This function updates ghost cells for the Z electric field along faces
    in the x1- and x2-directions. Efld3x1 corresponds to faces along x1, 
    and Efld3x2 corresponds to faces along x2.

    Parameters
    ----------
    grid : object
        Grid object containing domain information: Nx1, Nx2, Ngc.
    Efld3x1 : np.ndarray
        Z-component of electric field on faces along x1, including ghost cells.
    Efld3x2 : np.ndarray
        Z-component of electric field on faces along x2, including ghost cells.
    BC : list of str
        Boundary types for each boundary: [inner_x1, inner_x2, outer_x1, outer_x2].
        Supported types:
            'free' : non-reflective (zero gradient) boundary,
            'wall' : reflective boundary (normal component flips sign),
            'peri' : periodic boundary.
            'axis' : axis boundary (for curvilinear coordinates)

    Returns
    -------
    Efld3x1, Efld3x2 : np.ndarray
        Electric field arrays with updated ghost cells along x1 and x2.
    """
    Nx1 = grid.Nx1
    Nx2 = grid.Nx2
    Ngc = grid.Ngc
    
    for i in range(Ngc):
        # inner boundary
        if BC[1] == 'free':
            Efld3x1[:, i] = Efld3x1[:, 2 * Ngc - 1 - i]
        elif BC[1] == 'wall':
            Efld3x1[:, i] = -Efld3x1[:, 2 * Ngc - 1 - i]
        elif BC[1] == 'peri':
            Efld3x1[:, i] = Efld3x1[:, Nx2 + i]
        
        # outer boundary
        if BC[3] == 'free':
            Efld3x1[:, Nx2 + Ngc + i] = Efld3x1[:, Nx2 + Ngc - 1 - i]
        elif BC[3] == 'wall':
            Efld3x1[:, Nx2 + Ngc + i] = Efld3x1[:, Nx2 + Ngc - 1 - i]
        elif BC[3] == 'peri':
            Efld3x1[:, Nx2 + Ngc + i] = Efld3x1[:, Ngc + i]
    
    for i in range(Ngc):
        # inner boundary
        if BC[0] == 'free':
            Efld3x2[i, :] = Efld3x2[2 * Ngc - 1 - i, :]
        elif BC[0] == 'wall':
            Efld3x2[i, :] = -Efld3x2[2 * Ngc - 1 - i, :]
        elif BC[0] == 'peri':
            Efld3x2[i, :] = Efld3x2[Nx1 + i, :]
        elif BC[0] == 'axis':
            Efld3x2[i, :] = -Efld3x2[2 * Ngc - 1 - i, :]
        
        # outer boundary
        if BC[2] == 'free':
            Efld3x2[Nx1 + Ngc + i, :] = Efld3x2[Nx1 + Ngc - 1 - i, :]
        elif BC[2] == 'wall':
            Efld3x2[Nx1 + Ngc + i, :] = -Efld3x2[Nx1 + Ngc - 1 - i, :]
        elif BC[2] == 'peri':
            Efld3x2[Nx1 + Ngc + i, :] = Efld3x2[Ngc + i, :]
    
    return Efld3x1, Efld3x2

