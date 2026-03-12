# -*- coding: utf-8 -*-
"""
===============================================================================
diffusion_init_cond.py
===============================================================================

Initial condition functions for 2D thermal diffusion test problems.

Each function sets up a complete problem:
  - configures the grid (geometry and domain bounds)
  - initialises the temperature field diff.T
  - sets the thermal diffusivity diff.kappa
  - sets boundary conditions par.BC
  - sets simulation time par.timenow / par.timefin

Signature convention (identical to other Piastra IC modules):

    IC_diffusion_<name>(grid, diff, par)  ->  grid, diff, par

Available problems
------------------
``'user_defined'``   IC_diffusion_user_defined
``'gauss2D'``        IC_diffusion2D_gaussian   – single Gaussian pulse, 2D Cartesian
``'cross2D'``        IC_diffusion2D_cross      – two crossed Gaussian pulses, 2D Cartesian
``'ring2D'``         IC_diffusion2D_ring       – ring-shaped hot band, 2D Cartesian
``'gauss1D'``        IC_diffusion1D_gaussian   – 1D Gaussian (Nx2 = 1), Cartesian

Author: mrkondratyev
"""

import numpy as np


def IC_diffusion_user_defined(grid, diff, par):
    """
    Template for a user-defined diffusion problem.

    Parameters
    ----------
    grid : Grid
    diff : DiffState
    par  : Parameters

    Returns
    -------
    grid, diff, par
    """
    print("Thermal diffusion – user-defined problem")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 0.1

    # Boundary conditions: 'free' (zero gradient), 'wall' (same), 'peri' (periodic)
    par.BC[:] = 'free'

    # Constant diffusivity
    diff.kappa = 1.0

    # ----- Set your initial condition for T below -----
    diff.T[:, :] = 0.0

    raise ValueError(
        "User-defined diffusion problem – see 'diffusion_init_cond.py', "
        "set your IC and remove this line."
    )

    return grid, diff, par


def IC_diffusion2D_gaussian(grid, diff, par):
    """
    2D Cartesian diffusion of a single Gaussian temperature pulse.

    The exact solution of the diffusion equation for an initial Gaussian

        T(x, y, 0) = exp( -((x-x0)^2 + (y-y0)^2) / sigma0^2 )

    is a spreading Gaussian:

        T(x, y, t) = sigma0^2/(sigma0^2 + 4*kappa*t)
                     * exp( -((x-x0)^2 + (y-y0)^2) / (sigma0^2 + 4*kappa*t) )

    so the simulation can be validated against this analytical result.

    Parameters
    ----------
    grid : Grid
    diff : DiffState
    par  : Parameters

    Returns
    -------
    grid, diff, par
    """
    print("Thermal diffusion – 2D Gaussian pulse")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 0.05

    par.BC[:] = 'free'

    diff.kappa = 0.01

    # Gaussian centred in the domain
    x0 = 0.5 * (x1ini + x1fin)
    y0 = 0.5 * (x2ini + x2fin)
    sigma0 = 0.08

    diff.T[:, :] = np.exp(
        -((grid.cx1 - x0)**2 + (grid.cx2 - y0)**2) / sigma0**2
    )

    return grid, diff, par


def IC_diffusion2D_cross(grid, diff, par):
    """
    2D Cartesian diffusion of two orthogonal Gaussian hot bands (cross shape).

    The initial temperature field is the sum of:
      - a narrow Gaussian ridge along x1 = x0  (tall in x2)
      - a narrow Gaussian ridge along x2 = y0  (tall in x1)

    This tests that diffusion in x1 and x2 is handled symmetrically and
    that the operator is correct in both directions independently.

    Parameters
    ----------
    grid : Grid
    diff : DiffState
    par  : Parameters

    Returns
    -------
    grid, diff, par
    """
    print("Thermal diffusion – 2D crossed Gaussian ridges")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 0.05

    par.BC[:] = 'free'

    diff.kappa = 0.005

    x0 = 0.5 * (x1ini + x1fin)
    y0 = 0.5 * (x2ini + x2fin)
    sigma_narrow = 0.04   # narrow direction of each ridge
    sigma_wide   = 0.40   # wide direction of each ridge

    # ridge along x = x0  (narrow in x1, wide in x2)
    T_ridge1 = np.exp(
        -((grid.cx1 - x0)**2 / sigma_narrow**2 +
          (grid.cx2 - y0)**2 / sigma_wide**2)
    )

    # ridge along y = y0  (wide in x1, narrow in x2)
    T_ridge2 = np.exp(
        -((grid.cx1 - x0)**2 / sigma_wide**2 +
          (grid.cx2 - y0)**2 / sigma_narrow**2)
    )

    diff.T[:, :] = T_ridge1 + T_ridge2

    return grid, diff, par


def IC_diffusion2D_ring(grid, diff, par):
    """
    2D Cartesian diffusion of a hot annular ring.

    The initial temperature is a Gaussian function of the distance from
    a circle of radius r0, so the profile is a torus cross-section:

        T(x, y, 0) = exp( -(r - r0)^2 / sigma^2 )
        r = sqrt((x - x0)^2 + (y - y0)^2)

    As time advances the ring spreads inward and outward, eventually
    filling the interior with a smooth hill.

    Parameters
    ----------
    grid : Grid
    diff : DiffState
    par  : Parameters

    Returns
    -------
    grid, diff, par
    """
    print("Thermal diffusion – 2D annular ring")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 0.08

    par.BC[:] = 'free'

    diff.kappa = 0.005

    x0 = 0.5 * (x1ini + x1fin)
    y0 = 0.5 * (x2ini + x2fin)
    r0    = 0.25   # ring radius
    sigma = 0.04   # ring width

    r = np.sqrt((grid.cx1 - x0)**2 + (grid.cx2 - y0)**2)
    diff.T[:, :] = np.exp(-((r - r0) / sigma)**2)

    return grid, diff, par


def IC_diffusion1D_gaussian(grid, diff, par):
    """
    1D Cartesian diffusion of a Gaussian pulse (Nx2 = 1).

    Sets up a 1D problem by using a flat y-domain (x2ini = x2fin = 0.5)
    with Nx2 = 1.  The initial temperature is a Gaussian in x1.

    The exact solution is:

        T(x, t) = sigma0 / sqrt(sigma0^2 + 4*kappa*t)
                  * exp( -(x - x0)^2 / (sigma0^2 + 4*kappa*t) )

    Parameters
    ----------
    grid : Grid  (must be created with Nx2 = 1)
    diff : DiffState
    par  : Parameters

    Returns
    -------
    grid, diff, par
    """
    print("Thermal diffusion – 1D Gaussian pulse")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 0.05

    par.BC[:] = 'free'

    diff.kappa = 0.01

    x0     = 0.5 * (x1ini + x1fin)
    sigma0 = 0.08

    diff.T[:, :] = np.exp(-((grid.cx1 - x0) / sigma0)**2)

    return grid, diff, par
