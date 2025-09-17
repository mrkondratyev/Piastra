# -*- coding: utf-8 -*-
"""
===============================================================================
grid_misc.py
===============================================================================

@author: mrkondratyev

Utility functions for finite-volume hydrodynamics/MHD solvers
=============================================================

This module provides helper routines for working with structured 2D grids,
including interpolation between staggered (face-centered) and cell-centered
quantities, and divergence operators.

The routines assume the grid object `grid` contains:
    - grid.Ngc:   number of ghost cells
    - grid.Nx1, grid.Nx2: total number of cells in x1/x2 directions
    - grid.Nx1r, grid.Nx2r: indices of the last real (non-ghost) cells
    - grid.cx1, grid.cx2: cell-centered coordinates
    - grid.fx1, grid.fx2: face-centered coordinates
    - grid.dx1, grid.dx2: cell widths in x1/x2 directions
    - grid.fS1, grid.fS2: face areas in x1/x2 directions
    - grid.cVol:  cell volumes

All operations are consistent with a finite-volume discretization.
"""

import numpy as np



def interp_face_to_cell(grid, fV1, fV2):
    """
    Interpolate a staggered (face-centered) vector field to cell centers.

    Parameters
    ----------
    grid : object
        Grid object with geometry and metric information.
    fV1 : ndarray
        x1-component of the vector field, defined on x1-faces.
    fV2 : ndarray
        x2-component of the vector field, defined on x2-faces.

    Returns
    -------
    V1 : ndarray
        x1-component interpolated to cell centers.
    V2 : ndarray
        x2-component interpolated to cell centers.

    Notes
    -----
    - Uses linear interpolation weighted by distances between face and cell centers.
    - Assumes `fV1` and `fV2` include ghost zones consistent with `grid.Ngc`.
    """
    Ngc = grid.Ngc 

    V1 = (fV1[1:, :]  * (grid.cx1[Ngc:-Ngc, Ngc:-Ngc] - grid.fx1[Ngc:-Ngc-1, Ngc:-Ngc]) +
          fV1[:-1, :] * (grid.fx1[Ngc+1:-Ngc, Ngc:-Ngc] - grid.cx1[Ngc:-Ngc, Ngc:-Ngc])
         ) / grid.dx1[Ngc:-Ngc, Ngc:-Ngc]

    V2 = (fV2[:, 1:]  * (grid.cx2[Ngc:-Ngc, Ngc:-Ngc] - grid.fx2[Ngc:-Ngc, Ngc:-Ngc-1]) +
          fV2[:, :-1] * (grid.fx2[Ngc:-Ngc, Ngc+1:-Ngc] - grid.cx2[Ngc:-Ngc, Ngc:-Ngc])
         ) / grid.dx2[Ngc:-Ngc, Ngc:-Ngc]

    return V1, V2



def div_face_vector(grid, fV1, fV2):
    """
    Compute divergence of a face-centered vector field on a 2D grid.

    Parameters
    ----------
    grid : object
        Grid object with geometry and metric information.
    fV1 : ndarray
        x1-component of the vector field, defined on x1-faces.
    fV2 : ndarray
        x2-component of the vector field, defined on x2-faces.

    Returns
    -------
    divV : ndarray
        Divergence of the vector field, stored at cell centers.

    Notes
    -----
    - Discretization uses Gauss’s theorem: flux differences divided by cell volume.
    - Shape of `divV` is `(grid.Nx1, grid.Nx2)`.
    """
    Ngc = grid.Ngc 
    divV = np.zeros((grid.Nx1, grid.Nx2))
    
    if grid.Nx1 > 1:
        divV += (grid.fS1[1:, :] * fV1[1:, :] -
                 grid.fS1[:-1, :] * fV1[:-1, :]) / grid.cVol[:, :]
    if grid.Nx2 > 1:
        divV += (grid.fS2[:, 1:] * fV2[:, 1:] -
                 grid.fS2[:, :-1] * fV2[:, :-1]) / grid.cVol[:, :]
    
    return divV



def div_cell_vector(grid, V1, V2):
    """
    Compute divergence of a cell-centered vector field on a 2D grid.

    Parameters
    ----------
    grid : object
        Grid object with geometry and metric information.
    V1 : ndarray
        x1-component of the vector field at cell centers.
    V2 : ndarray
        x2-component of the vector field at cell centers.

    Returns
    -------
    divV : ndarray
        Divergence of the vector field, stored at cell centers.

    Notes
    -----
    - Fluxes at cell faces are approximated using arithmetic averages
      of adjacent cell-centered values.
    - Divergence is computed as the flux difference divided by cell volume.
    - Shape of `divV` is `(grid.Nx1, grid.Nx2)`.
    """
    Ngc = grid.Ngc 
    Nx1r = grid.Nx1r
    Nx2r = grid.Nx2r
    divV = np.zeros((grid.Nx1, grid.Nx2))
    
    if grid.Nx1 > 1:
        divV += 0.5 * (grid.fS1[1:, :]  * (V1[Ngc+1:Nx1r+1, Ngc:-Ngc] + V1[Ngc:Nx1r, Ngc:-Ngc]) -
                       grid.fS1[:-1, :] * (V1[Ngc-1:Nx1r-1, Ngc:-Ngc] + V1[Ngc:Nx1r, Ngc:-Ngc])
                      ) / grid.cVol[:, :]
            
    if grid.Nx2 > 1: 
        divV += 0.5 * (grid.fS2[:, 1:]  * (V2[Ngc:-Ngc, Ngc+1:Nx2r+1] + V2[Ngc:-Ngc, Ngc:Nx2r]) -
                       grid.fS2[:, :-1] * (V2[Ngc:-Ngc, Ngc-1:Nx2r-1] + V2[Ngc:-Ngc, Ngc:Nx2r])
                      ) / grid.cVol[:, :]
    
    return divV




def Ln_norm(grid, n, var_num, var_ref):
    """
    Compute L-n norm for the difference of two grid cell-centered arrays

    Parameters
    ----------
    grid : object
        Grid object with geometry and metric information.
    n : int
        number of desired order for the norm
    var_num : ndarray
        numerical variable.
    var_ref : ndarray
        reference variable.

    Returns
    -------
    norm : double
        norm value
    """
    Ngc = grid.Ngc 
    norm = 0.0
    
    norm = np.sum( grid.cVol[:,:]*(var_num[Ngc:-Ngc, Ngc:-Ngc] - var_ref[Ngc:-Ngc, Ngc:-Ngc])**n )
    
    return norm
    
    

def Ln_norm(grid, n, var_num, var_ref):
    """
    Compute L-n norm for the difference of two grid cell-centered arrays

    Parameters
    ----------
    grid : object
        Grid object with geometry and metric information.
    n : int
        number of desired order for the norm
    var_num : ndarray
        numerical variable.
    var_ref : ndarray
        reference variable.

    Returns
    -------
    norm : double
        norm value
    """
    Ngc = grid.Ngc 
    norm = 0.0
    
    norm = np.sum( grid.cVol[:,:]*np.abs(var_num[Ngc:-Ngc, Ngc:-Ngc] - var_ref[Ngc:-Ngc, Ngc:-Ngc])**n )
    
    return norm
    


def integral_over_grid(grid, var):
    """
    Compute integral over the 2D grid for cell-centered arrays

    Parameters
    ----------
    grid : object
        Grid object with geometry and metric information.
    var : ndarray
        numerical cell-centered variable.
    

    Returns
    -------
    double
        integral value
    """
    return np.sum( grid.cVol[:,:]*var[grid.Ngc:-grid.Ngc, grid.Ngc:-grid.Ngc] )
