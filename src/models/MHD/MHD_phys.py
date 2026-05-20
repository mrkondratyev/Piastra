"""
Non-relativistic Magnetohydrodynamics (MHD) Routines
====================================================

This module provides functions for evolving the equations of
non-relativistic ideal magnetohydrodynamics (MHD) using
finite-volume Godunov-type schemes. The routines handle conversion
between primitive and conservative variables, calculation of wave
speeds, approximate Riemann solvers (LLF, HLL, HLLD), and divergence
cleaning with the GLM method.

Implemented features
--------------------
- Conversion between primitive and conservative variables
- Calculation of maximum characteristic wave speeds (fast modes)
- Boundaries handling for cell-center variables
- Approximate Riemann solvers:
  * LLF (Local Lax-Friedrichs / Rusanov)
  * HLL (Harten–Lax–van Leer)
  * HLLD (Harten–Lax–van Leer with Discontinuities)
- Dedner GLM divergence cleaning subsystem (solver will be supported in future)

Assumptions
-----------
- Ideal gas equation of state with constant gamma.
- Non-relativistic limit (velocities << c).
- Conservative formulation with eight variables:
  mass density, three momentum components, total energy,
  and three magnetic field components.

References
----------
- Rusanov (1961), USSR J. Comp. Math. Phys.
- Harten, Lax & van Leer (1983), SIAM Rev.
- Davis (1988), SIAM J. Sci. Stat. Comput.
- Miyoshi & Kusano (2005), J. Comp. Phys.
- Dedner et al. (2002), J. Comp. Phys.

Author
------
mrkondratyev, 2024–2025
"""

import numpy as np
from src.common.boundaries import (
    apply_bc_scalar, 
    apply_bc_vector)
from src.models.MHD.riemann_approx import (
    LLF_flux,
    HLL_flux,
    HLLD_flux)


def prim2cons_nr_MHD(dens, vel1, vel2, vel3, pres, bfld1, bfld2, bfld3, eos):
    """
    Convert primitive variables to conservative variables (non-relativistic MHD).

    Parameters
    ----------
    dens : ndarray
       Mass density.
    vel1, vel2, vel3 : ndarray
       Velocity components in Cartesian directions.
    pres : ndarray
       Gas pressure.
    bfld1, bfld2, bfld3 : ndarray
       Magnetic field components (primitive state).
    eos : object
       Equation of state object with attribute `GAMMA`.

    Returns
    -------
    mass : ndarray
       Mass density (conserved).
    mom1, mom2, mom3 : ndarray
       Momentum density components.
    etot : ndarray
       Total energy density.
    bcon1, bcon2, bcon3 : ndarray
       Magnetic field components (unchanged in conservative form).
    """
    
    mass = dens
    mom1 = dens * vel1; mom2 = dens * vel2; mom3 = dens * vel3
    #kinetic energy 
    ekin = dens * (vel1**2 + vel2**2 + vel3**2) / 2.0 
    #internal energy
    eint = eos.eint(dens, pres)
    #magnetic energy 
    emag = (bfld1**2 + bfld2**2 + bfld3**2) / 2.0
    #total energy 
    etot = ekin + emag + eint 
    
    bcon1 = bfld1; bcon2 = bfld2; bcon3 = bfld3
    
    return mass, mom1, mom2, mom3, etot, bcon1, bcon2, bcon3


def cons2prim_nr_MHD(mass, mom1, mom2, mom3, etot, bcon1, bcon2, bcon3, eos):
    """
    Convert conservative variables to primitive variables (non-relativistic MHD).

    Parameters
    ----------
    mass : ndarray
        Mass density.
    mom1, mom2, mom3 : ndarray
        Momentum density components.
    etot : ndarray
        Total energy density (kinetic + thermal + magnetic).
    bcon1, bcon2, bcon3 : ndarray
        Magnetic field components (conservative state).
    eos : object
        Equation of state object with attribute `GAMMA`.

    Returns
    -------
    dens : ndarray
        Mass density.
    vel1, vel2, vel3 : ndarray
        Velocity components.
    pres : ndarray
        Gas pressure.
    bfld1, bfld2, bfld3 : ndarray
        Magnetic field components (identical to input).
    """
    dens = mass
    vel1 = mom1 / dens; vel2 = mom2 / dens; vel3 = mom3 / dens
    bfld1 = bcon1; bfld2 = bcon2; bfld3 = bcon3
    eint = etot - dens * (vel1**2 + vel2**2 + vel3**2) / 2.0 - (bfld1**2 + bfld2**2 + bfld3**2) / 2.0
    pres = eos.pres(dens, eint) 
    
    return dens, vel1, vel2, vel3, pres, bfld1, bfld2, bfld3


# ============================================================================
# Helper: call cons2prim_nr_MHD for a SimState object
# ============================================================================
def _prim_recovery(state, Ngc, eos):
    """
    Call cons2prim_nr_MHD and write results back into
    state.{dens,vel*,pres,bfi*}.

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
     state.pres[Ngc:-Ngc, Ngc:-Ngc],
     state.bfi1[Ngc:-Ngc, Ngc:-Ngc],
     state.bfi2[Ngc:-Ngc, Ngc:-Ngc],
     state.bfi3[Ngc:-Ngc, Ngc:-Ngc]) = \
        cons2prim_nr_MHD(
            state.mass, state.mom1, state.mom2, state.mom3, state.etot,
            state.bcon1, state.bcon2, state.bcon3, eos)


def max_wavespeed_MHD(csound, b1, b2, b3, dens):
    """
    Compute the maximum fast magnetosonic speed.

    Parameters
    ----------
    csound : ndarray
        Sound speed.
    b1, b2, b3 : ndarray
        Magnetic field components.
    dens : ndarray
        Mass density.

    Returns
    -------
    cfast : ndarray
        Maximum fast magnetosonic speed.
    """
    cfast = np.sqrt( csound**2 + (b1**2 + b2**2 + b3**2) / dens )

    return cfast


def boundCond_MHD(grid, BC, fluid):
    """
    Apply boundary conditions to MHD variables.

    Parameters
    ----------
    grid : object
        Grid object containing domain information (Nx1, Nx2, Ngc).
    BC : list of str
        Boundary types for each boundary [inner_x1, inner_x2, outer_x1, outer_x2].
        Supported: 'free', 'wall', 'peri', 'axis'.
    fluid : object
        MHD state object with attributes dens, pres, vel1, vel2, vel3, bfi1, bfi2, bfi3.

    Returns
    -------
    fluid : object
        MHD object with ghost cells updated according to BCs.
    """
    Ngc = grid.Ngc
    
    # Apply BCs for density and pressure
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[0], axis=1, side='inner')
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[1], axis=2, side='inner')
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[2], axis=1, side='outer')
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[3], axis=2, side='outer')
    
    fluid.pres = apply_bc_scalar(fluid.pres, Ngc, BC[0], axis=1, side='inner')
    fluid.pres = apply_bc_scalar(fluid.pres, Ngc, BC[1], axis=2, side='inner')
    fluid.pres = apply_bc_scalar(fluid.pres, Ngc, BC[2], axis=1, side='outer')
    fluid.pres = apply_bc_scalar(fluid.pres, Ngc, BC[3], axis=2, side='outer')
    
    # Apply BCs for velocity
    fluid.vel1, fluid.vel2, fluid.vel3 = \
        apply_bc_vector(fluid.vel1, fluid.vel2, fluid.vel3, Ngc, BC[0], axis=1, side='inner')
    fluid.vel1, fluid.vel2, fluid.vel3 = \
        apply_bc_vector(fluid.vel1, fluid.vel2, fluid.vel3, Ngc, BC[1], axis=2, side='inner')
    fluid.vel1, fluid.vel2, fluid.vel3 = \
        apply_bc_vector(fluid.vel1, fluid.vel2, fluid.vel3, Ngc, BC[2], axis=1, side='outer')
    fluid.vel1, fluid.vel2, fluid.vel3 = \
        apply_bc_vector(fluid.vel1, fluid.vel2, fluid.vel3, Ngc, BC[3], axis=2, side='outer')
    
    # Apply BCs for magnetic fields
    fluid.bfi1, fluid.bfi2, fluid.bfi3 = \
        apply_bc_vector(fluid.bfi1, fluid.bfi2, fluid.bfi3, Ngc, BC[0], axis=1, side='inner')
    fluid.bfi1, fluid.bfi2, fluid.bfi3 = \
        apply_bc_vector(fluid.bfi1, fluid.bfi2, fluid.bfi3, Ngc, BC[1], axis=2, side='inner')
    fluid.bfi1, fluid.bfi2, fluid.bfi3 = \
        apply_bc_vector(fluid.bfi1, fluid.bfi2, fluid.bfi3, Ngc, BC[2], axis=1, side='outer')
    fluid.bfi1, fluid.bfi2, fluid.bfi3 = \
        apply_bc_vector(fluid.bfi1, fluid.bfi2, fluid.bfi3, Ngc, BC[3], axis=2, side='outer')
    
    return fluid


def Riemann_flux_nr_MHD(rhol,rhor, vxl,vxr, vyl,vyr, vzl,vzr, pl,pr, bxl,bxr, byl,byr, bzl,bzr, eos, flux_type, dim):
    """
    Compute approximate Riemann fluxes for non-relativistic MHD.

    Supports LLF, HLL, and HLLD solvers.

    Parameters
    ----------
    rhol, rhor : ndarray
        Left and right densities.
    vxl, vxr, vyl, vyr, vzl, vzr : ndarray
        Left and right velocity components.
    pl, pr : ndarray
        Left and right pressures.
    bxl, bxr, byl, byr, bzl, bzr : ndarray
        Left and right magnetic field components.
    eos : object
        Equation of state object with attribute `GAMMA`.
    flux_type : {'LLF', 'HLL', 'HLLD'}
        Choice of Riemann solver.
    dim : int
        Normal direction (1 or 2). If `dim == 2`, system is rotated.

    Returns
    -------
    Fmass : ndarray
        Flux of mass density.
    Fmomx, Fmomy, Fmomz : ndarray
        Fluxes of momentum components.
    Fetot : ndarray
        Flux of total energy.
    Fbfix, Fbfiy, Fbfiz : ndarray
        Fluxes of magnetic field components.
    """
    
    #check in what direction we solve the problem
    if dim == 2: #2-direction -- rotate the coordinate system
        templ, tempr = vxl, vxr
        vxl, vxr = vyl, vyr
        vyl, vyr = -templ, -tempr
        
        templ, tempr = bxl, bxr
        bxl, bxr = byl, byr
        byl, byr = -templ, -tempr
    
    #here we calculate the flux using LLF approximate Riemann solver (Rusanov (1961))
    if flux_type == 'LLF':
        
        Fmass, Fmomx, Fmomy, Fmomz, Fetot, Fbfix, Fbfiy, Fbfiz = \
            LLF_flux(rhol,rhor, vxl,vxr, vyl,vyr, vzl,vzr, pl,pr, bxl,bxr, byl,byr, bzl,bzr, eos)
        
    #here we calculate the flux using HLL approximate Riemann solver (3 states between two fast shocks)
    #solution of Riemann problem according to Harten, Lax and van Leer, SIAM (1983)
    #see also Miyoshi and Kusano, JCP (2005)
    elif flux_type == 'HLL':  
        
        Fmass, Fmomx, Fmomy, Fmomz, Fetot, Fbfix, Fbfiy, Fbfiz = \
            HLL_flux(rhol,rhor, vxl,vxr, vyl,vyr, vzl,vzr, pl,pr, bxl,bxr, byl,byr, bzl,bzr, eos)
            
    #here we calculate the flux using HLLD approximate Riemann solver 
    #(6 states between two fast shocks, two Alfven discontinuities and contact surface)
    #solution of Riemann problem according to Miyoshi and Kusano, JCP (2005) 
    elif flux_type == 'HLLD':
        
        Fmass, Fmomx, Fmomy, Fmomz, Fetot, Fbfix, Fbfiy, Fbfiz = \
            HLLD_flux(rhol,rhor, vxl,vxr, vyl,vyr, vzl,vzr, pl,pr, bxl,bxr, byl,byr, bzl,bzr, eos)
        
    else:
        
        #flux_type is incorrect -> throw an error
        raise ValueError(f"Unknown flux_type: {flux_type}. Expected one of ['LLF', 'HLL', 'HLLD'].")
        
    #check in what direction we solve the problem
    if dim == 2: #2-direction -- rotate the coordinate system
        temp = Fmomx
        Fmomx = -Fmomy
        Fmomy = temp
        
        temp = Fbfix
        Fbfix = -Fbfiy
        Fbfiy = temp
        
    #return approximate Riemann flux for MHD -- 
    #8 fluxes for conservative variables 
    #(mass, three components of momentum, energy, three components of the B-field)
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot, Fbfix, Fbfiy, Fbfiz


def divB_clean_GLM_sol_MHD(c_h, bnl,bnr, psil,psir):
    """
    Solve GLM divergence cleaning subsystem.

    Implements mixed hyperbolic–parabolic divergence control
    according to Dedner et al., JCP (2002).

    Parameters
    ----------
    c_h : float
        Cleaning wave speed.
    bnl, bnr : ndarray
        Left and right normal magnetic field components.
    psil, psir : ndarray
        Left and right scalar potentials.

    Returns
    -------
    bnf : ndarray
        Flux for the normal magnetic field.
    psif : ndarray
        Flux for the scalar potential.
    """    
    bnf  = np.where(np.abs(c_h) > 1e-14, (bnr  + bnl )/2.0 - (psir - psil)/c_h/2.0, (bnr + bnl)/2.0)
    psif = np.where(np.abs(c_h) > 1e-14, (psir + psil)/2.0 - (bnr  - bnl )*c_h/2.0, 0.0            )
    
    return bnf, psif 
