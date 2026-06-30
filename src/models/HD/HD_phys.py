# -*- coding: utf-8 -*-
"""
HD_phys.py

Core routines for non-relativistic hydrodynamics solvers
========================================================

This module provides conversions between primitive and conservative variables,
boundaries handling, as well as approximate Riemann solvers for the Euler equations of ideal-gas
hydrodynamics. Implemented solvers include:

- LLF   : Local Lax-Friedrichs / Rusanov (1961)
- HLL   : Harten-Lax-van Leer (1983)
- HLLC  : HLL with contact restoration (Toro et al., 1994)
- Roe   : Linearized Roe solver (Roe, 1981)
- Exact : Exact Riemann solver / Godunov flux (Godunov, 1959; Toro, Ch. 4)

All solvers assume an ideal gas equation of state with a gamma-law closure.

References
----------
- Toro, E. F., "Riemann Solvers and Numerical Methods for Fluid Dynamics", Springer (2009)
- Toro, Spruce & Speares (1994), "Restoration of the contact surface in the HLL-Riemann solver"
- Roe, P. L. (1981), "Approximate Riemann solvers, parameter vectors, and difference schemes"
- Rusanov, V. V. (1961), "Calculation of interaction of non-steady shock waves with obstacles"
- Godunov, S. K. (1959), "A difference method for the numerical computation of discontinuous
  solutions of the equations of hydrodynamics"

Author
------
mrkondratyev
"""

import numpy as np
from src.models.HD.HD_riemann_exact import exact_riemann_godunov_state
from src.common.boundaries import (
    apply_bc_scalar, 
    apply_bc_vector,
    apply_bc_fixed)
from src.models.HD.HD_riemann_approx import (
    LLF_flux,
    HLL_flux,
    HLLC_flux,
    Roe_flux,
    cons_and_flux_HD)


def prim2cons_HD(dens, vel1, vel2, vel3, pres, eos):
    """
    Convert primitive variables to conservative variables for ideal hydrodynamics.

    Parameters
    ----------
    dens : ndarray
        Mass density.
    vel1, vel2, vel3 : ndarray
        Velocity components in x, y, z directions.
    pres : ndarray
        Pressure.
    eos : object
        Equation of state object with attribute `GAMMA`.

    Returns
    -------
    mass : ndarray
        Conserved mass density.
    mom1, mom2, mom3 : ndarray
        Conserved momentum densities.
    etot : ndarray
        Conserved total energy density.
    """
    mass = dens
    mom1 = dens * vel1; mom2 = dens * vel2; mom3 = dens * vel3
    #kinetic energy 
    ekin = dens * (vel1**2 + vel2**2 + vel3**2) / 2.0 
    #internal energy
    eint = eos.eint(dens, pres)
    
    etot = ekin + eint 
    
    return mass, mom1, mom2, mom3, etot



def cons2prim_HD(mass, mom1, mom2, mom3, etot, eos):
    """
    Convert conservative variables to primitive variables for ideal hydrodynamics.

    Parameters
    ----------
    mass : ndarray
        Conserved mass density.
    mom1, mom2, mom3 : ndarray
        Conserved momentum densities.
    etot : ndarray
        Conserved total energy density.
    eos : object
        Equation of state object with attribute `GAMMA`.

    Returns
    -------
    dens : ndarray
        Mass density.
    vel1, vel2, vel3 : ndarray
        Velocity components in x, y, z directions.
    pres : ndarray
        Pressure.
    """
    
    dens = mass
    #apply density floor for problematic flows 
    dens = np.maximum(dens, 1e-10)
    vel1 = mom1 / dens; vel2 = mom2 / dens; vel3 = mom3 / dens
    #internal energy
    eint = etot - dens * (vel1**2 + vel2**2 + vel3**2) / 2.0 
    #get pressure from internal energy 
    pres = eos.pres(dens, eint)
    #apply pressure floor for very cold/problematic flows 
    pres = np.maximum(pres, 1e-10)
    
    return dens, vel1, vel2, vel3, pres



def boundCond_HD(grid, BC, fluid, BC_fixed=None):
    """
    Apply boundary conditions to hydrodynamic variables.

    Parameters
    ----------
    grid : object
        Grid object containing domain information (Nx1, Nx2, Ngc).
    BC : list of str
        Boundary types for each boundary [inner_x1, inner_x2, outer_x1, outer_x2].
        Supported: 'free', 'wall', 'peri', 'axis'.
    fluid : object
        Fluid state object with attributes dens, pres, vel1, vel2, vel3.

    Returns
    -------
    fluid : object
        Fluid object with ghost cells updated according to BCs.
    """
    Ngc = grid.Ngc
    
    # Apply BCs for density
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[0], axis=1, side='inner')
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[1], axis=2, side='inner')
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[2], axis=1, side='outer')
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[3], axis=2, side='outer')
    
    # Apply BCs for pressure
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
    
    # --- fixed (Dirichlet) ghost-fill, applied LAST so it overrides the above ---
    if BC_fixed is not None:
        N1, N2 = fluid.dens.shape
        state_fields = {
            'dens': fluid.dens, 'pres': fluid.pres,
            'vel1': fluid.vel1, 'vel2': fluid.vel2, 'vel3': fluid.vel3,
        }
        for face in (0, 1, 2, 3):
            if BC_fixed.get(face):
                apply_bc_fixed(state_fields, Ngc, N1, N2, face, BC_fixed[face])
    
    return fluid






def Riemann_HD(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos, solver_type, dim):
    """
   Approximate Riemann solver for the Euler equations of gas dynamics.

   Parameters
   ----------
   rhol, rhor : ndarray
       Left and right densities.
   vxl, vxr, vyl, vyr, vzl, vzr : ndarray
       Velocity components (x, y, z) for left and right states.
   pl, pr : ndarray
       Left and right pressures.
   eos : object
       Equation of state object with attribute `GAMMA`.
   solver_type : str
       Type of flux solver: 'LLF', 'HLL', 'HLLC', 'Roe', 'Exact'.
   dim : int
       Coordinate direction (1 or 2). Other directions obtained by rotation.

   Returns
   -------
   Fmass : ndarray
       Flux of mass density.
   Fmomx, Fmomy, Fmomz : ndarray
       Fluxes of momentum density in x, y, z.
   Fetot : ndarray
       Flux of total energy density.
   """
    
    #check in what direction we solve the problem
    if dim == 2: #2-direction -- rotate the coordinate system
        templ, tempr = vxl, vxr
        vxl, vxr = vyl, vyr
        vyl, vyr = -templ, -tempr
        
    #here we calculate the flux using various Riemann solvers
    if solver_type == 'LLF':
        
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            LLF_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos)
        
    elif solver_type == 'HLL':  
        
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            HLL_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos)
               
    elif solver_type == 'HLLC':
        
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            HLLC_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos)
    
    elif solver_type == 'Roe':
        
        if (eos.ideal != 1):
            raise ValueError(
                f"Roe HD Riemann works only with ideal gamma-law EOS!" 
                f"Expected eos.ideal = {1} and eos.GAMMA > {1}.")
        
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            Roe_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos)

    elif solver_type == 'Exact':

        if (eos.ideal != 1):
            raise ValueError(
                f"Exact HD Riemann works only with ideal gamma-law EOS!" 
                f"Expected eos.ideal = {1} and eos.GAMMA > {1}.")
        
        #solve exact Riemann problem and sample Godunov state at x/t = 0
        dens0, vel0, pres0, ustar = exact_riemann_godunov_state(
            rhol, rhor, vxl, vxr, pl, pr, eos.GAMMA)

        #tangential velocities from the upwind side of the contact wave
        vy0 = np.where(ustar > 0.0, vyl, vyr)
        vz0 = np.where(ustar > 0.0, vzl, vzr)

        #compute Godunov fluxes and conservatives from the exact sampled state       
        mass, momx, momy, momz, etot, \
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            cons_and_flux_HD(dens0, vel0, vy0, vz0, pres0, eos)

    else:

        #solver_type is incorrect -> throw an error
        raise ValueError(
            f"Unknown HD solver_type: {solver_type}. " 
            f"Expected one of ['LLF', 'HLL', 'HLLC', 'Roe', 'Exact'].")
    
    #check in what direction we solve the problem
    if dim == 2: #2-direction -- rotate the coordinate system
        temp = Fmomx
        Fmomx = -Fmomy
        Fmomy = temp
        
    #return Riemann flux for gas dynamics -- 
    #5 fluxes for conservative variables (mass, three components of momentum and energy)
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot
