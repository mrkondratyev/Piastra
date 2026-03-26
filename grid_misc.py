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
quantities, divergence operators, gradient, curl (rotor), and Laplacian.

Supported geometries
--------------------
- Cartesian  (geom='cart'): x1 = x,  x2 = y,  x3 = z
- Cylindrical (geom='cyl'): x1 = R,  x2 = Z,  x3 = φ   (axisymmetric)
- Polar       (geom='pol'): x1 = R,  x2 = φ,  x3 = z

The routines assume the grid object `grid` contains:
    - grid.Ngc:   number of ghost cells
    - grid.Nx1, grid.Nx2: total number of cells in x1/x2 directions
    - grid.Nx1r, grid.Nx2r: indices of the last real (non-ghost) cells
    - grid.cx1, grid.cx2: cell-centered coordinates
    - grid.fx1, grid.fx2: face-centered coordinates
    - grid.dx1, grid.dx2: cell widths in x1/x2 directions
    - grid.fS1, grid.fS2: face areas in x1/x2 directions
    - grid.cVol:  cell volumes
    - grid.geom:  geometry marker ('cart', 'cyl', or 'pol')

All operations are consistent with a finite-volume discretization.
Differential operators return arrays of shape (grid.Nx1, grid.Nx2)
covering real (non-ghost) cells only.
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


# ============================================================================
# Curl (rotor)
# ============================================================================

def curl(grid, A1, A2, A3):
    """
    Compute the curl (rotor) of a cell-centered vector field.

    The input vector has three physical components (A1, A2, A3) even though
    the grid is 2D; the third component may carry an out-of-plane field.
    Second-order central differences are used for all partial derivatives.

    Coordinate conventions  (∂/∂x3 = 0 in 2D)
    -------------------------------------------
    **Cartesian** (x1=x, x2=y, x3=z):
        (∇×A)_x =  ∂A_z/∂y
        (∇×A)_y = −∂A_z/∂x
        (∇×A)_z =  ∂A_y/∂x − ∂A_x/∂y

    **Cylindrical** (x1=R, x2=Z, x3=φ; h1=1, h2=1, h3=R; axisymmetric):
        (∇×A)_R =  (1/R) ∂(R A_φ)/∂Z     = ∂A_φ/∂Z
        (∇×A)_Z = −(1/R) ∂(R A_φ)/∂R
        (∇×A)_φ =  ∂A_Z/∂R − ∂A_R/∂Z

    **Polar** (x1=R, x2=φ, x3=z; h1=1, h2=R, h3=1):
        (∇×A)_R =  (1/R) ∂A_z/∂φ
        (∇×A)_φ = −∂A_z/∂R
        (∇×A)_z =  (1/R) [∂(R A_φ)/∂R − ∂A_R/∂φ]

    Parameters
    ----------
    grid : Grid
        Grid object (must have ``geom`` attribute set).
    A1, A2, A3 : ndarray, shape grid.grid_shape
        Cell-centered vector-field components (including ghost zones).

    Returns
    -------
    c1, c2, c3 : ndarray, shape (grid.Nx1, grid.Nx2)
        Curl components on real cells.
    """
    Ngc, Nx1r, Nx2r = grid.Ngc, grid.Nx1r, grid.Nx2r
    Nx1, Nx2 = grid.Nx1, grid.Nx2

    c1 = np.zeros((Nx1, Nx2))
    c2 = np.zeros((Nx1, Nx2))
    c3 = np.zeros((Nx1, Nx2))

    # ------------------------------------------------------------------
    # Cartesian: (x, y, z)
    # ------------------------------------------------------------------
    if grid.geom == 'cart':
        if Nx2 > 1:
            dA3_dy = _ddx2(grid, A3)
            dA1_dy = _ddx2(grid, A1)
            c1 += dA3_dy           # ∂Az/∂y
            c3 -= dA1_dy           # −∂Ax/∂y
        if Nx1 > 1:
            dA3_dx = _ddx1(grid, A3)
            dA2_dx = _ddx1(grid, A2)
            c2 -= dA3_dx           # −∂Az/∂x
            c3 += dA2_dx           #  ∂Ay/∂x

    # ------------------------------------------------------------------
    # Cylindrical: (R, Z, φ)   h1=1, h2=1, h3=R
    # ------------------------------------------------------------------
    elif grid.geom == 'cyl':
        R  = grid.cx1[Ngc:Nx1r, Ngc:Nx2r]
        Ri = 1.0 / np.where(np.abs(R) > 1e-30, R, 1e-30)

        # (∇×A)_R = (1/R) ∂(R Aφ)/∂Z = ∂Aφ/∂Z  (R indep. of Z)
        if Nx2 > 1:
            c1 += _ddx2(grid, A3)

        # (∇×A)_Z = −(1/R) ∂(R Aφ)/∂R
        if Nx1 > 1:
            Rp = grid.cx1[Ngc+1:Nx1r+1, Ngc:Nx2r]
            Rm = grid.cx1[Ngc-1:Nx1r-1, Ngc:Nx2r]
            RA3p = Rp * A3[Ngc+1:Nx1r+1, Ngc:Nx2r]
            RA3m = Rm * A3[Ngc-1:Nx1r-1, Ngc:Nx2r]
            c2[:] = -Ri * (RA3p - RA3m) / (Rp - Rm)

        # (∇×A)_φ = ∂AZ/∂R − ∂AR/∂Z
        if Nx1 > 1:
            c3 += _ddx1(grid, A2)
        if Nx2 > 1:
            c3 -= _ddx2(grid, A1)

    # ------------------------------------------------------------------
    # Polar: (R, φ, z)   h1=1, h2=R, h3=1
    # ------------------------------------------------------------------
    elif grid.geom == 'pol':
        R  = grid.cx1[Ngc:Nx1r, Ngc:Nx2r]
        Ri = 1.0 / np.where(np.abs(R) > 1e-30, R, 1e-30)

        # (∇×A)_R = (1/R) ∂Az/∂φ
        if Nx2 > 1:
            c1 += Ri * _ddx2(grid, A3)

        # (∇×A)_φ = −∂Az/∂R
        if Nx1 > 1:
            c2 -= _ddx1(grid, A3)

        # (∇×A)_z = (1/R) [∂(R Aφ)/∂R − ∂AR/∂φ]
        if Nx1 > 1:
            Rp = grid.cx1[Ngc+1:Nx1r+1, Ngc:Nx2r]
            Rm = grid.cx1[Ngc-1:Nx1r-1, Ngc:Nx2r]
            RA2p = Rp * A2[Ngc+1:Nx1r+1, Ngc:Nx2r]
            RA2m = Rm * A2[Ngc-1:Nx1r-1, Ngc:Nx2r]
            c3 += Ri * (RA2p - RA2m) / (Rp - Rm)
        if Nx2 > 1:
            c3 -= Ri * _ddx2(grid, A1)

    else:
        raise ValueError(f"curl: unsupported geometry '{grid.geom}'")

    return c1, c2, c3


# ============================================================================
# Laplacian
# ============================================================================

def laplacian(grid, f):
    """
    Compute the scalar Laplacian of a cell-centered field.

    Uses the conservative finite-volume form ∇²f = (1/V) ∮ (∇f · dS)
    discretised with a compact 5-point stencil.  Face-centred radial
    coordinates from ``grid.fx1`` are used for the metric terms so that the
    scheme is exact for linear functions on any supported geometry.

    Coordinate conventions
    ----------------------
    - Cartesian  (cart): ∇²f = ∂²f/∂x² + ∂²f/∂y²
    - Cylindrical (cyl): ∇²f = (1/R) ∂(R ∂f/∂R)/∂R + ∂²f/∂Z²
    - Polar       (pol): ∇²f = (1/R) ∂(R ∂f/∂R)/∂R + (1/R²) ∂²f/∂φ²

    Parameters
    ----------
    grid : Grid
        Grid object (must have ``geom`` attribute set).
    f : ndarray, shape grid.grid_shape
        Cell-centered scalar field (including ghost zones).

    Returns
    -------
    lap : ndarray, shape (grid.Nx1, grid.Nx2)
        Laplacian on real cells.
    """
    Ngc, Nx1r, Nx2r = grid.Ngc, grid.Nx1r, grid.Nx2r
    lap = np.zeros((grid.Nx1, grid.Nx2))

    fc = f[Ngc:Nx1r, Ngc:Nx2r]

    # --- x1-direction contribution ---
    if grid.Nx1 > 1:
        fp1 = f[Ngc+1:Nx1r+1, Ngc:Nx2r]
        fm1 = f[Ngc-1:Nx1r-1, Ngc:Nx2r]
        dx1 = grid.dx1[Ngc:Nx1r, Ngc:Nx2r]

        if grid.geom == 'cart':
            lap += (fp1 - 2.0*fc + fm1) / dx1**2

        elif grid.geom in ('cyl', 'pol'):
            # Conservative form: (1/R) ∂(R ∂f/∂R)/∂R
            # Face radii from the grid
            Rf_r = grid.fx1[Ngc+1:Nx1r+1, Ngc:Nx2r]   # right face R
            Rf_l = grid.fx1[Ngc:Nx1r,     Ngc:Nx2r]    # left  face R
            R    = grid.cx1[Ngc:Nx1r,      Ngc:Nx2r]
            Ri   = 1.0 / np.where(np.abs(R) > 1e-30, R, 1e-30)
            lap += Ri * (Rf_r*(fp1 - fc)/dx1 - Rf_l*(fc - fm1)/dx1) / dx1

    # --- x2-direction contribution ---
    if grid.Nx2 > 1:
        fp2 = f[Ngc:Nx1r, Ngc+1:Nx2r+1]
        fm2 = f[Ngc:Nx1r, Ngc-1:Nx2r-1]
        dx2 = grid.dx2[Ngc:Nx1r, Ngc:Nx2r]

        if grid.geom in ('cart', 'cyl'):
            # Cartesian: ∂²f/∂y²   Cylindrical: ∂²f/∂Z²
            lap += (fp2 - 2.0*fc + fm2) / dx2**2

        elif grid.geom == 'pol':
            # (1/R²) ∂²f/∂φ²
            R  = grid.cx1[Ngc:Nx1r, Ngc:Nx2r]
            R2i = 1.0 / np.where(R**2 > 1e-30, R**2, 1e-30)
            lap += R2i * (fp2 - 2.0*fc + fm2) / dx2**2

    return lap

