# -*- coding: utf-8 -*-
"""
diff_phys.py

Some additional routines for diffusion solvers
========================================================

This module provides boundaries handling and 
a non-linear diffusion coefficient calculation.

Notes
-------

Boundary conditions are applied through ``apply_bc_scalar``, 
``apply_bc_fixed`` from ``boundaries.py``.  
The four-element array ``BC`` encodes:

    BC[0] : x1 inner (left / bottom-R)
    BC[1] : x2 inner (bottom / inner-Z)
    BC[2] : x1 outer (right / top-R)
    BC[3] : x2 outer (top / outer-Z)

Supported types: ``'free'`` (zero-gradient), ``'wall'`` (identical to
``'free'`` for scalars), ``'peri'`` (periodic).

``BC_fixed`` stores the information about Dirichlet boundaries 
(see ``boundaries.py``)

Author
------
mrkondratyev
"""
import numpy as np
from src.common.boundaries import apply_bc_scalar, apply_bc_fixed



def boundCond_diff(grid, BC, diff, BC_fixed=None):
    """
    Apply boundary conditions for diffusion.

    Parameters
    ----------
    grid : object
        Grid object containing domain information (Nx1, Nx2, Ngc).
    BC : list of str
        Boundary types for each boundary [inner_x1, inner_x2, outer_x1, outer_x2].
        Supported: 'free', 'wall', 'peri', 'axis'.
    diff : object
        Fluid state object with attribute 'T'.

    Returns
    -------
    diff : object
        Object with ghost cells updated according to BCs.
        
    """
    Ngc = grid.Ngc
    
    # Apply BCs for density
    diff.T = apply_bc_scalar(diff.T, Ngc, BC[0], axis=1, side='inner')
    diff.T = apply_bc_scalar(diff.T, Ngc, BC[1], axis=2, side='inner')
    diff.T = apply_bc_scalar(diff.T, Ngc, BC[2], axis=1, side='outer')
    diff.T = apply_bc_scalar(diff.T, Ngc, BC[3], axis=2, side='outer')
    
    # --- fixed (Dirichlet) ghost-fill, applied LAST so it overrides the above ---
    if BC_fixed is not None:
       N1, N2 = diff.T.shape
       state_fields = {'T': diff.T}
       for face in (0, 1, 2, 3):
           if BC_fixed.get(face):
               apply_bc_fixed(state_fields, Ngc, N1, N2, face, BC_fixed[face])
    
    return diff



def nonlinear_coef_diff(grid, diff):
    """
    Evaluate the diffusion coefficient (will be added in future).

    Parameters
    ----------
    grid : object
        Grid object containing domain information (Nx1, Nx2, Ngc).
    diff : object
        Fluid state object with the diffused variable 

    Returns
    -------
    diff : object
        Fluid object with updated diffusion coefficient.
    """
    
    # some function of x,y,T...
    diff.kappa[:, :] = 1.0
    
    return diff 
