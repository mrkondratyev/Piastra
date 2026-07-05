# -*- coding: utf-8 -*-
"""
===============================================================================
diff_init_cond.py
===============================================================================

Initial condition functions for 2D thermal diffusion test problems.

Each function sets up a complete problem:
  - configures the grid (geometry and domain bounds)
  - initialises the temperature field diff.T
  - sets the thermal diffusivity diff.kappa
  - sets boundary conditions par.BC
  - sets simulation time par.timenow / par.timefin

Signature convention (identical to other Piastra IC modules):

    IC_diff<name>(grid, diff, par)  ->  grid, diff, par

Available problems
------------------
``'user_defined'``   IC_diff_user_defined
``'gauss2D'``        IC_diff2D_gaussian   – single Gaussian pulse, 2D Cartesian
``'cross2D'``        IC_diff2D_cross      – two crossed Gaussian pulses, 2D Cartesian
``'ring2D'``         IC_diff2D_ring       – ring-shaped hot band, 2D Cartesian
``'gauss1D'``        IC_diff1D_gaussian   – 1D Gaussian (Nx2 = 1), Cartesian
``'step1D'``         IC_diff1D_step       – 1D step function (Nx2 = 1), Cartesian
``'sine1D'``         IC_diff1D_sine       – 1D sinusoidal mode decay (Nx2 = 1), Cartesian
``'cyl2D'``          IC_diff2D_cyl        – 2D cylindrically-symmetric ring, Cartesian grid

Author: mrkondratyev
"""

import numpy as np


def IC_diff_user_defined(grid, diff, par):
    """
    Template for a user-defined diffusion problem.

    Parameters
    ----------
    grid : Grid
    diff : SimState
    par  : Parameters

    Returns
    -------
    grid, diff, par
    """
    print("Thermal diffusion – user-defined problem")

    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.1
    
    diff.kappa = 1.0 # Constant diffusivity

    # ----- Set your initial condition for T below -----
    diff.T[:, :] = 1.0
    
    #source term
    diff.ST[:, :] = 0.0

    # Boundary conditions: 'free' (zero gradient), 'wall' (same), 'peri' (periodic), 'axis'
    par.BC[:] = 'free'

    raise ValueError(
        "User-defined diffusion problem – see 'diff_init_cond.py', "
        "set your ICs and remove this line."
    )

    return grid, diff, par



def IC_diff2D_gaussian(grid, diff, par):
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
    diff : SimState
    par  : Parameters

    Returns
    -------
    grid, diff, par
    """
    print("Thermal diffusion – 2D Gaussian pulse")

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.5

    #diffusion coefficient 
    diff.kappa = 0.01

    # Gaussian centred in the domain
    x0 = 0.5 * (x1ini + x1fin)
    y0 = 0.5 * (x2ini + x2fin)
    sigma0 = 0.08

    diff.T[:, :] = np.exp(-((grid.cx1 - x0)**2 + (grid.cx2 - y0)**2) / sigma0**2)
    
    par.BC[:] = 'free'
    
    return grid, diff, par



def IC_diff2D_cross(grid, diff, par):
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
    diff : SimState
    par  : Parameters

    Returns
    -------
    grid, diff, par
    """
    print("Thermal diffusion – 2D crossed Gaussian ridges")

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.5

    #diffusion coefficient 
    diff.kappa = 0.005

    #center of the cross 
    x0 = 0.5 * (x1ini + x1fin)
    y0 = 0.5 * (x2ini + x2fin)
    sigma_narrow = 0.04   # narrow direction of each ridge
    sigma_wide   = 0.40   # wide direction of each ridge

    # ridge along x = x0  (narrow in x1, wide in x2)
    T_ridge1 = np.exp(
        -((grid.cx1 - x0)**2 / sigma_narrow**2 +
          (grid.cx2 - y0)**2 / sigma_wide**2))

    # ridge along y = y0  (wide in x1, narrow in x2)
    T_ridge2 = np.exp(
        -((grid.cx1 - x0)**2 / sigma_wide**2 +
          (grid.cx2 - y0)**2 / sigma_narrow**2))

    diff.T[:, :] = T_ridge1 + T_ridge2
    
    par.BC[:] = 'free'
    
    return grid, diff, par


def IC_diff2D_ring(grid, diff, par):
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
    diff : SimState
    par  : Parameters

    Returns
    -------
    grid, diff, par
    """
    print("Thermal diffusion – 2D annular ring")

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.2

    #diffusion coefficient 
    diff.kappa = 0.005

    #center of the ring  
    x0 = 0.5 * (x1ini + x1fin)
    y0 = 0.5 * (x2ini + x2fin)
    r0    = 0.25   # ring radius
    sigma = 0.04   # ring width

    r = np.sqrt((grid.cx1 - x0)**2 + (grid.cx2 - y0)**2)
    diff.T[:, :] = np.exp(-((r - r0) / sigma)**2)

    par.BC[:] = 'free'

    return grid, diff, par


def IC_diff1D_gaussian(grid, diff, par):
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
    diff : SimState
    par  : Parameters

    Returns
    -------
    grid, diff, par
    """
    print("Thermal diffusion – 1D Gaussian pulse")

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.5

    #diffusion coefficient 
    diff.kappa = 0.01

    x0 = 0.5 * (x1ini + x1fin) #center of the gaussian 
    sigma0 = 0.08 #semi-width

    diff.T[:, :] = np.exp(-((grid.cx1 - x0) / sigma0)**2)

    par.BC[:] = 'free'

    return grid, diff, par



def IC_diff1D_step(grid, diff, par):
    """
    1D diffusion of a step-function initial condition (Nx2 = 1).

    The initial temperature is a Heaviside step at x = 0.5:

        T(x, 0) = 1 for x < 0.5, 0 for x > 0.5

    The exact solution involves the complementary error function:

        T(x, t) = 0.5 * erfc( (x - 0.5) / (2 * sqrt(kappa * t)) )

    This provides a simple but non-trivial analytical benchmark for
    validating the diffusion operator on a sharp discontinuity.

    Parameters
    ----------
    grid : Grid
    diff : SimState
    par  : Parameters

    Returns
    -------
    grid, diff, par
    """
    print("Thermal diffusion - 1D step function")

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.5

    #diffusion coefficient
    diff.kappa = 0.01

    x0 = 0.5 * (x1ini + x1fin) #step location

    diff.T[:, :] = np.where(grid.cx1 < x0, 1.0, 0.0)

    par.BC[0] = 'wall'; par.BC[1] = 'free'
    par.BC[2] = 'wall'; par.BC[3] = 'free'

    return grid, diff, par


def IC_diff2D_cyl(grid, diff, par):
    """
    2D diffusion with cylindrical symmetry on a Cartesian grid.

    The initial temperature is a ring-like Gaussian distribution
    centred at the origin, computed as a function of radial distance
    only. As time evolves, the solution should remain radially
    symmetric. This tests that the 2D Cartesian Laplacian operator
    correctly handles problems with inherent cylindrical symmetry.

    T(x, y, 0) = exp( -((r - r0) / sigma)^2 )
    r = sqrt((x - x0)^2 + (y - y0)^2)

    Domain: [0, 0.5] x [0, 0.5] (quadrant symmetry)

    Parameters
    ----------
    grid : Grid
    diff : SimState
    par  : Parameters

    Returns
    -------
    grid, diff, par
    """
    print("Thermal diffusion - 2D cylindrical symmetry test")

    x1ini, x1fin = 0.0, 0.5; x2ini, x2fin = 0.0, 0.5
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.3

    #diffusion coefficient 
    diff.kappa = 0.005

    r = np.sqrt(grid.cx1**2 + grid.cx2**2)
    r0 = 0.2
    sigma = 0.04
    diff.T[:, :] = np.exp(-((r - r0) / sigma)**2)

    par.BC[0] = 'wall'; par.BC[1] = 'wall'
    par.BC[2] = 'free'; par.BC[3] = 'free'

    return grid, diff, par



def IC_diff1D_sine(grid, diff, par):
    """
    1D diffusion of a sinusoidal initial condition (Nx2 = 1).

    T(x, 0) = sin(2*pi*x)

    The exact solution is:

        T(x, t) = sin(2*pi*x) * exp(-(2*pi)^2 * kappa * t)

    This provides an excellent convergence test since the exact
    solution is smooth and known for all times.

    Parameters
    ----------
    grid : Grid
    diff : SimState
    par  : Parameters

    Returns
    -------
    grid, diff, par
    """
    print("Thermal diffusion - 1D sinusoidal mode decay")
    
    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.5

    #diffusion coefficient 
    diff.kappa = 0.01

    diff.T[:, :] = np.sin(2.0 * np.pi * grid.cx1)

    par.BC[0] = 'peri'; par.BC[1] = 'free'
    par.BC[2] = 'peri'; par.BC[3] = 'free'

    return grid, diff, par
