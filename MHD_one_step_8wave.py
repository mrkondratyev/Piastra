# -*- coding: utf-8 -*-
"""
===============================================================================
MHD_one_step_8wave.py
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
- ``MHD_curv_sources`` : evaluate curvature source terms (e.g., cylindrical).

Features
--------
- Spatial accuracy via piecewise constant, linear, PPM, or WENO-type reconstructions.
- Riemann solvers: Local Lax-Friedrichs (LLF), HLL, HLLD.
- Temporal accuracy: TVD Runge-Kutta methods (1st, 2nd, or 3rd order).
- Divergence of the magnetic field is treated via simple Powell's 8-wave approach.
- Curvilinear support (currently cylindrical geometry implemented).

@author:
    mrkondratyev
"""

from MHD_phys import *
from grid_misc import div_cell_vector 
from reconstruction import VarReconstruct 
import numpy as np 
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



def CFLcondition_MHD(g, MHD, eos, CFL):
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
    Ngc = g.Ngc
    
    #sound speed calculation for whole domain
    csound = eos.sound_speed(MHD.dens[Ngc:-Ngc, Ngc:-Ngc], MHD.pres[Ngc:-Ngc, Ngc:-Ngc])
    
    #fast magnetosonic speed calculation for whole domain 
    cfast = max_wavespeed_MHD(csound, \
        MHD.bfi1[Ngc:-Ngc, Ngc:-Ngc], MHD.bfi2[Ngc:-Ngc, Ngc:-Ngc], MHD.bfi3[Ngc:-Ngc, Ngc:-Ngc], MHD.dens[Ngc:-Ngc, Ngc:-Ngc])
    
    #FIRST APPROACH
    #dt1 = np.min( g.dx1[Ngc:-Ngc, Ngc:-Ngc] / (1e-14 + np.abs(MHD.vel1[Ngc:-Ngc, Ngc:-Ngc]) + cfast) )
    #dt2 = np.min( g.dx2[Ngc:-Ngc, Ngc:-Ngc] / (1e-14 + np.abs(MHD.vel2[Ngc:-Ngc, Ngc:-Ngc]) + cfast) )    
    #return  CFL * min(dt1, dt2)
    
    #SECOND APPROACH 
    dt_inv = np.max((np.abs(MHD.vel1[Ngc:-Ngc, Ngc:-Ngc]) + cfast)/g.dx1[Ngc:-Ngc, Ngc:-Ngc] + \
        (np.abs(MHD.vel2[Ngc:-Ngc, Ngc:-Ngc]) + cfast)/g.dx2[Ngc:-Ngc, Ngc:-Ngc])
    return CFL/dt_inv


"""

in the function "oneStep_MHD_RK" we call all the key ingredients of our MHD simulations.
the high order in space can be adjusted by rec_type = 'PCM' (1st order), 'PLM' (2nd order) or 'WENO' (3rd or 5th order CWENO or WENO5 methods, see reconstuction.py)
the Riemann problem approximate solution can be switched bewteen Local Lax-Friedrichs (Rusanov) flux ('flux_type = 'LLF'), HLL flux ('HLL') or HLLC flux ('HLLC') 
here we use multistage Total Variation Diminishing Runge-Kutta timestepping with "RK_order" = 'RK1', 'RK2' or 'RK3'. 
input: GRID class object (grid), FLUIDSTATE class object (fluid) at time t, EOSdata class object (eos),
timestep dt, "rec_type" -- type of reconstruction, "flux_type" -- Riemann problem solution approximation method and "RK_order" -- order of temporal integration
output: FLUIDSTATE class object (fluid) at time t + dt

For RK timestepping one can see (Shu and Osher (1988))

for a given timestep dt and a primitive fluid state, we calculate conservative state and the residuals for them 
if RK method is beyond the first order, we additionally introduce the intermediate conservative and primitive states
on the predictor stage, we update the initial fluid state to the intermediate one on each stage, 
and after the final stage, we update the fluid state itself, using the information from the intermediate stages 

"""

#this function provides one hydro time step
def oneStep_MHD_RK_8wave(g, MHD, eos, par, dt):
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
    MHD.mass, MHD.mom1, MHD.mom2, MHD.mom3, MHD.etot,  MHD.bcon1, MHD.bcon2, MHD.bcon3 = \
        prim2cons_nr_MHD(MHD.dens[Ngc:-Ngc,Ngc:-Ngc], 
        MHD.vel1[Ngc:-Ngc,Ngc:-Ngc], MHD.vel2[Ngc:-Ngc,Ngc:-Ngc],  
        MHD.vel3[Ngc:-Ngc,Ngc:-Ngc], MHD.pres[Ngc:-Ngc,Ngc:-Ngc], 
        MHD.bfi1[Ngc:-Ngc,Ngc:-Ngc], MHD.bfi2[Ngc:-Ngc,Ngc:-Ngc], 
        MHD.bfi3[Ngc:-Ngc,Ngc:-Ngc], eos)
    
    #residuals for conservative variables calculation
    #1st Runge-Kutta iteration - predictor stage
    ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = flux_calc_MHD_8wave(g, MHD, par, eos)
    
    # Conservative update - 1st RK stage (predictor)
    MHD_h.mass = MHD.mass - dt * ResM 
    MHD_h.mom1 = MHD.mom1 - dt * ResV1 
    MHD_h.mom2 = MHD.mom2 - dt * ResV2 
    MHD_h.mom3 = MHD.mom3 - dt * ResV3 
    MHD_h.etot = MHD.etot - dt * ResE 
    MHD_h.bcon1 = MHD.bcon1 - dt * ResB1
    MHD_h.bcon2 = MHD.bcon2 - dt * ResB2
    MHD_h.bcon3 = MHD.bcon3 - dt * ResB3
    
    #first order Runge-Kutta scheme
    if (par.RK_order == 'RK1'): 
        
        #simply rewrite the conservative state here for clarity
        MHD.mass = MHD_h.mass
        MHD.mom1 = MHD_h.mom1
        MHD.mom2 = MHD_h.mom2
        MHD.mom3 = MHD_h.mom3
        MHD.etot = MHD_h.etot
        MHD.bcon1 = MHD_h.bcon1
        MHD.bcon2 = MHD_h.bcon2
        MHD.bcon3 = MHD_h.bcon3
    
    
    #second-order Runge-Kutta scheme
    if (par.RK_order == 'RK2'):
        
        #Primitive variables recovery after predictor stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        MHD_h.dens[Ngc:-Ngc, Ngc:-Ngc], MHD_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            MHD_h.vel2[Ngc:-Ngc, Ngc:-Ngc], MHD_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            MHD_h.pres[Ngc:-Ngc, Ngc:-Ngc], MHD_h.bfi1[Ngc:-Ngc, Ngc:-Ngc],  \
            MHD_h.bfi2[Ngc:-Ngc, Ngc:-Ngc], MHD_h.bfi3[Ngc:-Ngc, Ngc:-Ngc] = \
            cons2prim_nr_MHD(MHD_h.mass, MHD_h.mom1, MHD_h.mom2, MHD_h.mom3, MHD_h.etot, \
            MHD_h.bcon1, MHD_h.bcon2, MHD_h.bcon3, eos) 
            
        #2nd Runge-Kutta stage - corrector
        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = flux_calc_MHD_8wave(g, MHD_h, par, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy
        MHD.mass = (MHD_h.mass + MHD.mass) / 2.0 - dt * ResM / 2.0
        MHD.mom1 = (MHD_h.mom1 + MHD.mom1) / 2.0 - dt * ResV1 / 2.0 
        MHD.mom2 = (MHD_h.mom2 + MHD.mom2) / 2.0 - dt * ResV2 / 2.0 
        MHD.mom3 = (MHD_h.mom3 + MHD.mom3) / 2.0 - dt * ResV3 / 2.0  
        MHD.etot = (MHD_h.etot + MHD.etot) / 2.0 - dt * ResE / 2.0 
        MHD.bcon1 = (MHD_h.bcon1 + MHD.bcon1) / 2.0 - dt * ResB1 / 2.0 
        MHD.bcon2 = (MHD_h.bcon2 + MHD.bcon2) / 2.0 - dt * ResB2 / 2.0 
        MHD.bcon3 = (MHD_h.bcon3 + MHD.bcon3) / 2.0 - dt * ResB3 / 2.0  
        
    if (par.RK_order == 'RK3'):
        
        #Primitive variables recovery after predictor stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        MHD_h.dens[Ngc:-Ngc, Ngc:-Ngc], MHD_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            MHD_h.vel2[Ngc:-Ngc, Ngc:-Ngc], MHD_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            MHD_h.pres[Ngc:-Ngc, Ngc:-Ngc], MHD_h.bfi1[Ngc:-Ngc, Ngc:-Ngc],  \
            MHD_h.bfi2[Ngc:-Ngc, Ngc:-Ngc], MHD_h.bfi3[Ngc:-Ngc, Ngc:-Ngc] = \
            cons2prim_nr_MHD(MHD_h.mass, MHD_h.mom1, MHD_h.mom2, MHD_h.mom3, MHD_h.etot, \
            MHD_h.bcon1, MHD_h.bcon2, MHD_h.bcon3, eos) 
            
        #2nd Runge-Kutta stage
        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = flux_calc_MHD_8wave(g, MHD_h, par, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy
        MHD_h.mass = (MHD_h.mass + 3.0 * MHD.mass) / 4.0 - dt * ResM / 4.0
        MHD_h.mom1 = (MHD_h.mom1 + 3.0 * MHD.mom1) / 4.0 - dt * ResV1 / 4.0 
        MHD_h.mom2 = (MHD_h.mom2 + 3.0 * MHD.mom2) / 4.0 - dt * ResV2 / 4.0 
        MHD_h.mom3 = (MHD_h.mom3 + 3.0 * MHD.mom3) / 4.0 - dt * ResV3 / 4.0  
        MHD_h.etot = (MHD_h.etot + 3.0 * MHD.etot) / 4.0 - dt * ResE / 4.0 
        MHD_h.bcon1 = (MHD_h.bcon1 + 3.0 * MHD.bcon1) / 4.0 - dt * ResB1 / 4.0 
        MHD_h.bcon2 = (MHD_h.bcon2 + 3.0 * MHD.bcon2) / 4.0 - dt * ResB2 / 4.0 
        MHD_h.bcon3 = (MHD_h.bcon3 + 3.0 * MHD.bcon3) / 4.0 - dt * ResB3 / 4.0  
        
        # Primitive variables recovery after the second stage
        #density, 3 components of velocity and pressure are evaluated 
        MHD_h.dens[Ngc:-Ngc, Ngc:-Ngc], MHD_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            MHD_h.vel2[Ngc:-Ngc, Ngc:-Ngc], MHD_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            MHD_h.pres[Ngc:-Ngc, Ngc:-Ngc], MHD_h.bfi1[Ngc:-Ngc, Ngc:-Ngc],  \
            MHD_h.bfi2[Ngc:-Ngc, Ngc:-Ngc], MHD_h.bfi3[Ngc:-Ngc, Ngc:-Ngc] = \
            cons2prim_nr_MHD(MHD_h.mass, MHD_h.mom1, MHD_h.mom2, MHD_h.mom3, MHD_h.etot, \
            MHD_h.bcon1, MHD_h.bcon2, MHD_h.bcon3, eos) 
        
        #3rd Runge-Kutta stage
        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = flux_calc_MHD_8wave(g, MHD_h, par, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, 3 components of momentum, total energy and 3 comps of magnetic field
        MHD.mass = (2.0 * MHD_h.mass + MHD.mass) / 3.0 - 2.0 * dt * ResM / 3.0
        MHD.mom1 = (2.0 * MHD_h.mom1 + MHD.mom1) / 3.0 - 2.0 * dt * ResV1 / 3.0 
        MHD.mom2 = (2.0 * MHD_h.mom2 + MHD.mom2) / 3.0 - 2.0 * dt * ResV2 / 3.0 
        MHD.mom3 = (2.0 * MHD_h.mom3 + MHD.mom3) / 3.0 - 2.0 * dt * ResV3 / 3.0  
        MHD.etot = (2.0 * MHD_h.etot + MHD.etot) / 3.0 - 2.0 * dt * ResE / 3.0 
        MHD.bcon1 = (2.0 * MHD_h.bcon1 + MHD.bcon1) / 3.0 - 2.0 * dt * ResB1 / 3.0 
        MHD.bcon2 = (2.0 * MHD_h.bcon2 + MHD.bcon2) / 3.0 - 2.0 * dt * ResB2 / 3.0 
        MHD.bcon3 = (2.0 * MHD_h.bcon3 + MHD.bcon3) / 3.0 - 2.0 * dt * ResB3 / 3.0  
        
    # Primitive variables recovery at the end of the timestep
    #density, 3 components of velocity and pressure are evaluated 
    MHD.dens[Ngc:-Ngc, Ngc:-Ngc], MHD.vel1[Ngc:-Ngc, Ngc:-Ngc], \
        MHD.vel2[Ngc:-Ngc, Ngc:-Ngc], MHD.vel3[Ngc:-Ngc, Ngc:-Ngc], \
        MHD.pres[Ngc:-Ngc, Ngc:-Ngc], MHD.bfi1[Ngc:-Ngc, Ngc:-Ngc],  \
        MHD.bfi2[Ngc:-Ngc, Ngc:-Ngc], MHD.bfi3[Ngc:-Ngc, Ngc:-Ngc] = \
        cons2prim_nr_MHD(MHD.mass, MHD.mom1, MHD.mom2, MHD.mom3, MHD.etot, \
        MHD.bcon1, MHD.bcon2, MHD.bcon3, eos) 
    
    MHD.divB = div_cell_vector(g, MHD.bfi1, MHD.bfi2)
    
    #return the updated class object of the fluid state on the next timestep 
    return MHD



def flux_calc_MHD_8wave(g, MHD, par, eos):
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
    MHD = boundCond_MHD(g, par.BC, MHD)
    
    #make copies of ghost cell and real cell numbers in each direction
    #to simplify indexing below 
    Ngc = g.Ngc 
    Nx1 = g.Nx1
    Nx2 = g.Nx2
    
    #nulifying the divergence of the magnetic field 
    MHD.divB[:,:] = 0.0
    
    #residuals initialization (only for real cells)
    ResM = np.zeros((Nx1, Nx2))
    ResV1 = np.zeros((Nx1, Nx2))
    ResV2 = np.zeros((Nx1, Nx2))
    ResV3 = np.zeros((Nx1, Nx2))
    ResE = np.zeros((Nx1, Nx2))
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
        Fmass, Fmom1, Fmom2, Fmom3, Fetot, Fbfi1, Fbfi2, Fbfi3 = \
            Riemann_flux_nr_MHD(dens_rec_L, dens_rec_R, vel1_rec_L, vel1_rec_R, \
            vel2_rec_L, vel2_rec_R, vel3_rec_L, vel3_rec_R, \
            pres_rec_L, pres_rec_R, bfi1_rec_L, bfi1_rec_R, \
            bfi2_rec_L, bfi2_rec_R, bfi3_rec_L, bfi3_rec_R, eos, par.flux_type, 1)
        
        #residuals calculation for mass, 3 components of momentum, 
        #total energy and 3 component of magnetic field in 1-dim
        ResM = ( Fmass[1:,:]*g.fS1[1:,:] - Fmass[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        ResV1 = ( Fmom1[1:,:]*g.fS1[1:,:] - Fmom1[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        ResV2 = ( Fmom2[1:,:]*g.fS1[1:,:] - Fmom2[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        ResV3 = ( Fmom3[1:,:]*g.fS1[1:,:] - Fmom3[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        ResE = ( Fetot[1:,:]*g.fS1[1:,:] - Fetot[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        #ResB1 = ( Fbfi1[1:,:]*g.fS1[1:,:] - Fbfi1[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
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
        Fmass, Fmom1, Fmom2, Fmom3, Fetot, Fbfi1, Fbfi2, Fbfi3 = \
            Riemann_flux_nr_MHD(dens_rec_L, dens_rec_R, vel1_rec_L, vel1_rec_R, \
            vel2_rec_L, vel2_rec_R, vel3_rec_L, vel3_rec_R, \
            pres_rec_L, pres_rec_R, bfi1_rec_L, bfi1_rec_R, \
            bfi2_rec_L, bfi2_rec_R, bfi3_rec_L, bfi3_rec_R, eos, par.flux_type, 2)
        
        #residuals calculation for mass, 3 components of momentum, 
        #total energy and 3 components of magnetic field in 2-dim
        #here we add the fluxes differences to the residuals after 1-dim calculation
        ResM += ( Fmass[:,1:]*g.fS2[:,1:] - Fmass[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResV1 += ( Fmom1[:,1:]*g.fS2[:,1:] - Fmom1[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResV2 += ( Fmom2[:,1:]*g.fS2[:,1:] - Fmom2[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResV3 += ( Fmom3[:,1:]*g.fS2[:,1:] - Fmom3[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResE += ( Fetot[:,1:]*g.fS2[:,1:] - Fetot[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResB1 += ( Fbfi1[:,1:]*g.fS2[:,1:] - Fbfi1[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        #ResB2 += ( Fbfi2[:,1:]*g.fS2[:,1:] - Fbfi2[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResB3 += ( Fbfi3[:,1:]*g.fS2[:,1:] - Fbfi3[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
         
        #calculation of magnetic field divergence for Powell (1999) 8-wave approach
        MHD.divB += ( (bfi2_rec_L[:,1:] + bfi2_rec_R[:,1:]) / 2.0 * g.fS2[:,1:] - \
            ( bfi2_rec_L[:,:-1] + bfi2_rec_R[:,:-1]) / 2.0 * g.fS2[:,:-1] ) / g.cVol[:,:]     
    
    #finally, here we add the external force terms
    #we add forces in momentum res, while in energy we add Power = Force*Vel 
    ResV1 += - MHD.dens[Ngc:-Ngc, Ngc:-Ngc] * MHD.F1 
    ResV2 += - MHD.dens[Ngc:-Ngc, Ngc:-Ngc] * MHD.F2 
    ResE += - MHD.dens[Ngc:-Ngc, Ngc:-Ngc] * (MHD.F1 * MHD.vel1[Ngc:-Ngc, Ngc:-Ngc] + \
        MHD.F2 * MHD.vel2[Ngc:-Ngc, Ngc:-Ngc])
    
    #curvature source terms 
    STv1, STv2, STv3, STm1, STm2, STm3 = MHD_curv_sources(g, MHD)
    
    #Powell 8-wave cleaning method + curvature sources
    ResV1 += MHD.bfi1[Ngc:-Ngc, Ngc:-Ngc] * MHD.divB - STv1
    ResV2 += MHD.bfi2[Ngc:-Ngc, Ngc:-Ngc] * MHD.divB - STv2
    ResV3 += MHD.bfi3[Ngc:-Ngc, Ngc:-Ngc] * MHD.divB - STv3
    ResE += MHD.divB * (MHD.vel1[Ngc:-Ngc, Ngc:-Ngc] * MHD.bfi1[Ngc:-Ngc, Ngc:-Ngc] + \
        MHD.vel2[Ngc:-Ngc, Ngc:-Ngc] * MHD.bfi2[Ngc:-Ngc, Ngc:-Ngc] + \
        MHD.vel3[Ngc:-Ngc, Ngc:-Ngc] * MHD.bfi3[Ngc:-Ngc, Ngc:-Ngc])
    ResB1 += MHD.vel1[Ngc:-Ngc, Ngc:-Ngc] * MHD.divB - STm1
    ResB2 += MHD.vel2[Ngc:-Ngc, Ngc:-Ngc] * MHD.divB - STm2
    ResB3 += MHD.vel3[Ngc:-Ngc, Ngc:-Ngc] * MHD.divB - STm3
    
    #return the residuals for mass, 3 components of momentum, total energy and magnetic field
    return ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3




def MHD_curv_sources(g, MHD):
    """
    Compute geometric source terms for the MHD equations 
    in curvilinear coordinates (finite-volume formulation).

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
    STm1 : ndarray
        Radial magnetic field source term.
    STm3 : ndarray
        Axial B-field source term (zero in cylindrical geometry).
    STm3 : ndarray
        Azimuthal B-field source term.

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
    STm1 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    STm2 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    STm3 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    
    if (g.geom == 'cyl'):
        
        STv1 = (MHD.pres[Ngc:-Ngc,Ngc:-Ngc] + (MHD.bfi1[Ngc:-Ngc,Ngc:-Ngc]**2 + \
               MHD.bfi2[Ngc:-Ngc,Ngc:-Ngc]**2 - MHD.bfi3[Ngc:-Ngc,Ngc:-Ngc]**2)/2.0 + \
               MHD.dens[Ngc:-Ngc,Ngc:-Ngc] * MHD.vel3[Ngc:-Ngc,Ngc:-Ngc]**2) / \
               g.cx1[Ngc:-Ngc,Ngc:-Ngc]

        STv3 = (MHD.bfi3[Ngc:-Ngc,Ngc:-Ngc] * MHD.bfi1[Ngc:-Ngc,Ngc:-Ngc] - \
               MHD.dens[Ngc:-Ngc,Ngc:-Ngc] * MHD.vel3[Ngc:-Ngc,Ngc:-Ngc] * \
               MHD.vel1[Ngc:-Ngc,Ngc:-Ngc]) / g.cx1[Ngc:-Ngc,Ngc:-Ngc]
        
        STm3 = (MHD.bfi3[Ngc:-Ngc,Ngc:-Ngc] * MHD.vel1[Ngc:-Ngc,Ngc:-Ngc] - \
               MHD.bfi1[Ngc:-Ngc,Ngc:-Ngc] * MHD.vel3[Ngc:-Ngc,Ngc:-Ngc]**2) / \
               g.cx1[Ngc:-Ngc,Ngc:-Ngc]
            
    return STv1, STv2, STv3, STm1, STm2, STm3


