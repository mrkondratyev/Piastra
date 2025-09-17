# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:00:11 2023

Boundary condition module for 2D hydrodynamics and MHD simulations.

This module provides functions to fill ghost cells for scalar and vector fields.
Supported boundary types:
    'free'  - non-reflective (zero-gradient) boundary
    'wall'  - reflective (normal component flips) boundary
    'peri'  - periodic boundary
    'axis'  - axis boundary

Functions:
- apply_bc_scalar(var, Ngc, BC_type, axis=1, side='inner'):
      Fill ghost cells for a scalar field along a given axis.
- apply_bc_vector(V1, V2, V3, Ngc, BC_type, axis=1, side='inner'):
      Fill ghost cells for a 3-component vector field along a given axis.
- boundCond_hydro(grid, BC, fluid):
      Apply BCs to hydrodynamic variables (density, pressure, velocities).
- boundCond_mhd(grid, BC, fluid):
      Apply BCs to MHD variables (density, pressure, velocities, magnetic fields).
- boundCond_electric_field:
      Fill ghost cells for face-centered electric field E3 along x1 and x2.
      (E3 = Ez for MHD in 2D XY coordinates, for instance)

Ghost Cell Implementation Note
------------------------------

This module provides routines to fill ghost cells for 2D hydrodynamic
and MHD simulations. The approach separates scalar, vector, and face-centered
fields for clarity and correctness:

1. apply_bc_scalar(var, ...)  
   - For scalar quantities (e.g., density, pressure).  
   - Fills ghost cells along the specified axis according to BC_type.

2. apply_bc_vector(V1, V2, V3, ...)  
   - For 3-component vector quantities (e.g., velocity, cell-centered magnetic field).  
   - Treats the normal component differently for reflective (wall) boundaries
     while leaving tangential components unchanged.

3. Face-centered third component of the electric field (Efld3 along x1/x2)  
   - needed for CT MHD 
   - Separate function boundCond_electric_field
     to handle this explicitly.

Author: mrkondratyev
"""

import numpy as np


def apply_bc_scalar(var, Ngc, BC_type, axis=1, side='inner'):
    """
    Fill ghost cells for a scalar field along a given axis.

    Parameters
    ----------
    var : np.ndarray
        Scalar field array including ghost cells.
    Ngc : int
        Number of ghost cells.
    BC_type : str
        Boundary type: 'free', 'wall', 'peri', 'axis.
    axis : int
        Axis along which to apply BC (1 for x1, 2 for x2).
    side : str
        'inner' or 'outer' boundary.

    Returns
    -------
    var : np.ndarray
        Field with ghost cells updated.
    """
    shape = var.shape
    N1, N2 = shape[0], shape[1]

    for i in range(Ngc):
        if axis == 1:  # x1-direction
            if side == 'inner':
                if BC_type == 'free':
                    var[i, :] = var[2*Ngc - 1 - i, :]
                elif BC_type == 'wall':
                    var[i, :] = var[2*Ngc - 1 - i, :]
                elif BC_type == 'peri':
                    var[i, :] = var[N1 - 2*Ngc + i, :]
                elif BC_type == 'axis':
                    var[i, :] = var[2*Ngc - 1 - i, :]
            elif side == 'outer':
                if BC_type == 'free':
                    var[N1 - Ngc + i, :] = var[N1 - Ngc - 1 - i, :]
                elif BC_type == 'wall':
                    var[N1 - Ngc + i, :] = var[N1 - Ngc - 1 - i, :]
                elif BC_type == 'peri':
                    var[N1 - Ngc + i, :] = var[Ngc + i, :]
        elif axis == 2:  # x2-direction
            if side == 'inner':
                if BC_type == 'free':
                    var[:, i] = var[:, 2*Ngc - 1 - i]
                elif BC_type == 'wall':
                    var[:, i] = var[:, 2*Ngc - 1 - i]
                elif BC_type == 'peri':
                    var[:, i] = var[:, N2 - 2*Ngc + i]
                elif BC_type == 'axis':
                    var[:, i] = var[:, 2*Ngc - 1 - i] #spherical axis
            elif side == 'outer':
                if BC_type == 'free':
                    var[:, N2 - Ngc + i] = var[:, N2 - Ngc - 1 - i]
                elif BC_type == 'wall':
                    var[:, N2 - Ngc + i] = var[:, N2 - Ngc - 1 - i]
                elif BC_type == 'peri':
                    var[:, N2 - Ngc + i] = var[:, Ngc + i]
                elif BC_type == 'axis':
                    var[:, N2 - Ngc + i] = var[:, N2 - Ngc - 1 - i] #spherical axis

    return var



def apply_bc_vector(V1, V2, V3, Ngc, BC_type, axis=1, side='inner'):
    """
    Fill ghost cells for a 3-component vector field along a given axis.

    Parameters
    ----------
    V1, V2, V3 : np.ndarray
        Vector field components including ghost cells.
    Ngc : int
        Number of ghost cells.
    BC_type : str
        Boundary type: 'free', 'wall', 'peri', 'axis'.
    axis : int
        Axis along which to apply BC (1 for x1, 2 for x2).
    side : str
        'inner' or 'outer' boundary.

    Returns
    -------
    V1, V2, V3 : np.ndarray
        Vector field components with ghost cells updated.
    """
    shape = V1.shape
    N1, N2 = shape[0], shape[1]
    
    for i in range(Ngc):
        if axis == 1:  # x1-direction
            if side == 'inner':
                if BC_type == 'free':
                    V1[i, :] = V1[2*Ngc - 1 - i, :]
                    V2[i, :] = V2[2*Ngc - 1 - i, :]
                    V3[i, :] = V3[2*Ngc - 1 - i, :]
                elif BC_type == 'wall':
                    V1[i, :] = -V1[2*Ngc - 1 - i, :]  # normal component flips
                    V2[i, :] = V2[2*Ngc - 1 - i, :]
                    V3[i, :] = V3[2*Ngc - 1 - i, :]
                elif BC_type == 'peri':
                    V1[i, :] = V1[N1 - 2*Ngc + i, :]
                    V2[i, :] = V2[N1 - 2*Ngc + i, :]
                    V3[i, :] = V3[N1 - 2*Ngc + i, :]
                elif BC_type == 'axis':
                    V1[i, :] = -V1[2*Ngc - 1 - i, :]  # normal component flips
                    V2[i, :] = V2[2*Ngc - 1 - i, :]
                    V3[i, :] = -V3[2*Ngc - 1 - i, :] # azimuthal component flips
            elif side == 'outer':
                if BC_type == 'free':
                    V1[N1 - Ngc + i, :] = V1[N1 - Ngc - 1 - i, :]
                    V2[N1 - Ngc + i, :] = V2[N1 - Ngc - 1 - i, :]
                    V3[N1 - Ngc + i, :] = V3[N1 - Ngc - 1 - i, :]
                elif BC_type == 'wall':
                    V1[N1 - Ngc + i, :] = -V1[N1 - Ngc - 1 - i, :]
                    V2[N1 - Ngc + i, :] = V2[N1 - Ngc - 1 - i, :]
                    V3[N1 - Ngc + i, :] = V3[N1 - Ngc - 1 - i, :]
                elif BC_type == 'peri':
                    V1[N1 - Ngc + i, :] = V1[Ngc + i, :]
                    V2[N1 - Ngc + i, :] = V2[Ngc + i, :]
                    V3[N1 - Ngc + i, :] = V3[Ngc + i, :]
        elif axis == 2:  # x2-direction
            if side == 'inner':
                if BC_type == 'free':
                    V1[:, i] = V1[:, 2*Ngc - 1 - i]
                    V2[:, i] = V2[:, 2*Ngc - 1 - i]
                    V3[:, i] = V3[:, 2*Ngc - 1 - i]
                elif BC_type == 'wall':
                    V1[:, i] = V1[:, 2*Ngc - 1 - i]
                    V2[:, i] = -V2[:, 2*Ngc - 1 - i]  # normal component flips
                    V3[:, i] = V3[:, 2*Ngc - 1 - i]
                elif BC_type == 'peri':
                    V1[:, i] = V1[:, N2 - 2*Ngc + i]
                    V2[:, i] = V2[:, N2 - 2*Ngc + i]
                    V3[:, i] = V3[:, N2 - 2*Ngc + i]
                elif BC_type == 'axis':
                    V1[:, i] = V1[:, 2*Ngc - 1 - i]
                    V2[:, i] = -V2[:, 2*Ngc - 1 - i]  # normal component flips
                    V3[:, i] = -V3[:, 2*Ngc - 1 - i]  # azimuthal component flips
            elif side == 'outer':
                if BC_type == 'free':
                    V1[:, N2 - Ngc + i] = V1[:, N2 - Ngc - 1 - i]
                    V2[:, N2 - Ngc + i] = V2[:, N2 - Ngc - 1 - i]
                    V3[:, N2 - Ngc + i] = V3[:, N2 - Ngc - 1 - i]
                elif BC_type == 'wall':
                    V1[:, N2 - Ngc + i] = V1[:, N2 - Ngc - 1 - i]
                    V2[:, N2 - Ngc + i] = -V2[:, N2 - Ngc - 1 - i]  # normal component flips
                    V3[:, N2 - Ngc + i] = V3[:, N2 - Ngc - 1 - i]
                elif BC_type == 'peri':
                    V1[:, N2 - Ngc + i] = V1[:, Ngc + i]
                    V2[:, N2 - Ngc + i] = V2[:, Ngc + i]
                    V3[:, N2 - Ngc + i] = V3[:, Ngc + i]
                elif BC_type == 'axis':
                    V1[:, N2 - Ngc + i] = V1[:, N2 - Ngc - 1 - i]
                    V2[:, N2 - Ngc + i] = -V2[:, N2 - Ngc - 1 - i]  # normal component flips
                    V3[:, N2 - Ngc + i] = -V3[:, N2 - Ngc - 1 - i]  # azimuthal component flips

    return V1, V2, V3





