# -*- coding: utf-8 -*-
"""
adv_phys.py

Core routines for advection
========================================================

This module provides boundaries handling, as well as exact Riemann solver

References
----------
- Toro, E. F., "Riemann Solvers and Numerical Methods for Fluid Dynamics", Springer (2009)

Author
------
mrkondratyev
"""
import numpy as np
from src.common.boundaries import apply_bc_scalar, apply_bc_fixed



def boundCond_adv(grid, BC, adv, BC_fixed=None):
    """
    Apply boundary conditions to advection variable.

    Parameters
    ----------
    grid : object
        Grid object containing domain information (Nx1, Nx2, Ngc).
    BC : list of str
        Boundary types for each boundary [inner_x1, inner_x2, outer_x1, outer_x2].
        Supported: 'free', 'wall', 'peri', 'axis'.
    adv : object
        Fluid state object with attribute 'dens'.

    Returns
    -------
    adv : object
        Object with ghost cells updated according to BCs.
    """
    Ngc = grid.Ngc
    
    # Apply BCs for density
    adv.dens = apply_bc_scalar(adv.dens, Ngc, BC[0], axis=1, side='inner')
    adv.dens = apply_bc_scalar(adv.dens, Ngc, BC[1], axis=2, side='inner')
    adv.dens = apply_bc_scalar(adv.dens, Ngc, BC[2], axis=1, side='outer')
    adv.dens = apply_bc_scalar(adv.dens, Ngc, BC[3], axis=2, side='outer')
    
    # --- fixed (Dirichlet) ghost-fill, applied LAST so it overrides the above ---
    if BC_fixed is not None:
       N1, N2 = adv.dens.shape
       state_fields = {'dens': adv.dens}
       for face in (0, 1, 2, 3):
           if BC_fixed.get(face):
               apply_bc_fixed(state_fields, Ngc, N1, N2, face, BC_fixed[face])
    
    return adv



def Riemann_adv(rhol, rhor, vel):
    """
    Exact Riemann solver for the advection (it is super simple!).

    Parameters
    ----------
    rhol, rhor : ndarray
        Left and right densities.
    vel : float
        advection velocity.

    Returns
    -------
    Flux : ndarray
        Flux of advected quantity (e.g. mass density).
    """
    Flux = vel * (rhol + rhor) / 2.0 - np.abs(vel) * (rhor - rhol) / 2.0
   
    return Flux 

