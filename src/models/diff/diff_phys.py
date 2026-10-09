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
from ``boundaries.py`` ('BC_fixed' is treated internally here) 
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
from src.common.boundaries import apply_bc_scalar



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
    BC_fixed : list of Dirichlet ICs 
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
    # T_b sits ON THE FACE: the ghost cell is mirrored about it, T_g = 2 T_b - T_1
    # this leads to the second-order spatial convergence instead of first order 
    # see M. Zingale's "Tutorial on computational astrophysical hydrodynamics"
    if BC_fixed is not None:
        N1, N2 = diff.T.shape
        T = diff.T
        # ---- face 0: x1 inner ----
        for (start, end, values) in BC_fixed.get(0, []):
            if 'T' in values:
                j0 = Ngc + start; j1 = Ngc + end
                T[Ngc - 1, j0:j1] = 2.0 * values['T'] - T[Ngc, j0:j1]
        # ---- face 2: x1 outer ----
        for (start, end, values) in BC_fixed.get(2, []):
            if 'T' in values:
                j0 = Ngc + start; j1 = Ngc + end
                T[N1 - Ngc, j0:j1] = 2.0 * values['T'] - T[N1 - Ngc - 1, j0:j1]
        # ---- face 1: x2 inner ----
        for (start, end, values) in BC_fixed.get(1, []):
            if 'T' in values:
                i0 = Ngc + start; i1 = Ngc + end
                T[i0:i1, Ngc - 1] = 2.0 * values['T'] - T[i0:i1, Ngc]
        # ---- face 3: x2 outer ----
        for (start, end, values) in BC_fixed.get(3, []):
            if 'T' in values:
                i0 = Ngc + start; i1 = Ngc + end
                T[i0:i1, N2 - Ngc] = 2.0 * values['T'] - T[i0:i1, N2 - Ngc - 1]
    
    return diff



def nonlinear_coef_diff(grid, diff):
    """
    Evaluate a (currently constant, placeholder) diffusion coefficient.

    Not called by diff_step.py's solver loop -- diff.kappa stays whatever
    the IC set it to (a scalar by default, see SimState) for the whole
    run. This is a stub for a future T- or position-dependent kappa(x,
    T); call it manually (and turn diff.kappa into an array first, since
    this assigns into it elementwise) before stepping if you need that.

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
