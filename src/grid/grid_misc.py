# -*- coding: utf-8 -*-
"""
===============================================================================
grid_misc.py
===============================================================================

Utility functions for finite-volume grid solvers.

This module provides helper routines for working with structured 2D grids,
including:
  - Interpolation from staggered (face-centered) to cell-centered fields
  - Divergence operators for face-centered and cell-centered vector fields
  - Cell-centred gradient operator with geometry-aware metric factors
  - L-n norm and volume-integral helpers for convergence testing
  - Central finite-difference helpers (private: ``_ddx1``, ``_ddx2``)

The routines assume the grid object ``grid`` provides:
    - grid.Ngc          : number of ghost cells
    - grid.Nx1, Nx2     : number of real cells in x1 / x2 directions
    - grid.Nx1r, Nx2r   : last real-cell indices (= Nx + Ngc)
    - grid.cx1, cx2     : cell-centred coordinates
    - grid.fx1, fx2     : face-centred coordinates
    - grid.dx1, dx2     : cell widths
    - grid.fS1, fS2     : face areas perpendicular to x1 / x2
    - grid.cVol         : cell volumes
    - grid.geom         : geometry marker ('cart', 'cyl', 'pol')

All operations are consistent with a finite-volume discretization on
Cartesian, cylindrical (R,Z), or polar (R,φ) geometries.

Author: mrkondratyev
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

    if grid.Nx1 == 1:
        V1 = fV1[1:, :]
    else:  
        V1 = (fV1[1:, :]  * (grid.cx1[Ngc:-Ngc, Ngc:-Ngc] - grid.fx1[Ngc:-Ngc-1, Ngc:-Ngc]) +
          fV1[:-1, :] * (grid.fx1[Ngc+1:-Ngc, Ngc:-Ngc] - grid.cx1[Ngc:-Ngc, Ngc:-Ngc])
         ) / grid.dx1[Ngc:-Ngc, Ngc:-Ngc]
    
    if grid.Nx2 == 1:
        V2 = fV2[:, 1:]
    else:
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




# ============================================================================
# Helper: central finite differences on cell-centered data
# ============================================================================

def _ddx1(grid, f):
    """
    Central ∂f/∂x1 on real cells using ghost-zone data.

    Parameters
    ----------
    grid : Grid
    f : ndarray, shape grid.grid_shape
        Cell-centered scalar field (including ghost zones).

    Returns
    -------
    ndarray, shape (grid.Nx1, grid.Nx2)
    """
    Ngc, Nx1r, Nx2r = grid.Ngc, grid.Nx1r, grid.Nx2r
    return (f[Ngc+1:Nx1r+1, Ngc:Nx2r] - f[Ngc-1:Nx1r-1, Ngc:Nx2r]) / \
           (grid.cx1[Ngc+1:Nx1r+1, Ngc:Nx2r] - grid.cx1[Ngc-1:Nx1r-1, Ngc:Nx2r])


def _ddx2(grid, f):
    """
    Central ∂f/∂x2 on real cells using ghost-zone data.

    Parameters
    ----------
    grid : Grid
    f : ndarray, shape grid.grid_shape
        Cell-centered scalar field (including ghost zones).

    Returns
    -------
    ndarray, shape (grid.Nx1, grid.Nx2)
    """
    Ngc, Nx1r, Nx2r = grid.Ngc, grid.Nx1r, grid.Nx2r
    return (f[Ngc:Nx1r, Ngc+1:Nx2r+1] - f[Ngc:Nx1r, Ngc-1:Nx2r-1]) / \
           (grid.cx2[Ngc:Nx1r, Ngc+1:Nx2r+1] - grid.cx2[Ngc:Nx1r, Ngc-1:Nx2r-1])


# ============================================================================
# Gradient
# ============================================================================

def gradient(grid, f):
    """
    Compute the gradient of a cell-centered scalar field.

    Uses second-order central differences.  For curvilinear grids the
    metric scale factors are included so that the result represents the
    *physical* gradient components.

    Coordinate conventions
    ----------------------
    - Cartesian  (cart): ∇f = (∂f/∂x, ∂f/∂y)
    - Cylindrical (cyl): ∇f = (∂f/∂R, ∂f/∂Z)          [h_R=1, h_Z=1]
    - Polar       (pol): ∇f = (∂f/∂R, (1/R) ∂f/∂φ)     [h_R=1, h_φ=R]

    Parameters
    ----------
    grid : Grid
        Grid object (must have ``geom`` attribute set).
    f : ndarray, shape grid.grid_shape
        Cell-centered scalar field (including ghost zones).

    Returns
    -------
    g1 : ndarray, shape (grid.Nx1, grid.Nx2)
        x1-component of the gradient on real cells.
    g2 : ndarray, shape (grid.Nx1, grid.Nx2)
        x2-component of the gradient on real cells.
    """
    g1 = _ddx1(grid, f) if grid.Nx1 > 1 else np.zeros((grid.Nx1, grid.Nx2))
    g2 = _ddx2(grid, f) if grid.Nx2 > 1 else np.zeros((grid.Nx1, grid.Nx2))

    # Polar: x2 = φ, h2 = R  →  (∇f)_φ = (1/R) ∂f/∂φ
    if grid.geom == 'pol' and grid.Nx2 > 1:
        Ngc = grid.Ngc
        R = grid.cx1[Ngc:-Ngc, Ngc:-Ngc]
        g2 /= np.where(np.abs(R) > 1e-30, R, 1e-30)

    return g1, g2


