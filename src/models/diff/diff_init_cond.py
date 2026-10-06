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
``'step1D'``         IC_diff1D_step       – 1D step function (Nx2 = 1), Cartesian
``'sine1D'``         IC_diff1D_sine       – 1D sinusoidal mode decay (Nx2 = 1), Cartesian
``'cyl2D'``          IC_diff2D_cyl        – 2D cylindrically-symmetric ring, Cartesian grid
``'gauss2Dpol'``     IC_diff2D_gauss_polar– 2D gaussian on the polar grid 
``'gauss2Dsph'``     IC_diff2D_gauss_sph  - 2D gaussian on the spherical polar grid 

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


def IC_diff2D_gauss_polar(grid, diff, par):
    """
    Spreading of an OFF-CENTRE 2D Gaussian on a polar (R, phi) grid.

    Exact solution of dT/dt = kappa lap(T) in the plane (not polar-symmetric
    about the grid origin, so both the R- and the phi-metric are exercised):
        T(x, t) = t0/(t0+t) * exp( -|x - x0|^2 / (4 kappa (t0+t)) ),
    x0 = (R=1, phi=0). Annular wedge R in [0.5, 1.5], phi in [-pi/4, pi/4];
    at t_fin the Gaussian is ~1e-7 at the boundaries ('free').
    """
    print("Thermal diffusion - off-centre 2D Gaussian on a polar grid")
    grid.PolarGrid(0.5, 1.5, -0.25 * np.pi, 0.25 * np.pi)
    diff.kappa = 0.01
    t0 = 0.125                               # initial width^2 = 4 kappa t0 = 0.005
    par.timenow = 0.0; par.timefin = 0.125

    x = grid.cx1 * np.cos(grid.cx2); y = grid.cx1 * np.sin(grid.cx2)
    diff.T[:, :] = np.exp(-((x - 1.0)**2 + y**2) / (4.0 * diff.kappa * t0))

    par.BC[:] = 'free'
    return grid, diff, par


def IC_diff2D_gauss_sph(grid, diff, par):
    """
    Spreading of a 3D Gaussian centred ON THE AXIS, spherical-polar (r, theta).

    Exact solution (axisymmetric about the polar axis, not about the origin):
        T(x, t) = (t0/(t0+t))^(3/2) * exp( -|x - x0|^2 / (4 kappa (t0+t)) ),
    x0 on the axis at r = 1 (theta = 0). Shell r in [0.5, 1.5], theta in
    [0, pi/2]; the axis face is exercised directly, the other faces see
    T ~ 1e-7 at t_fin ('free').
    """
    print("Thermal diffusion - on-axis 3D Gaussian on a spherical-polar grid")
    grid.SphericalPolarGrid(0.5, 1.5, 0.0, 0.5 * np.pi)
    diff.kappa = 0.01
    t0 = 0.125
    par.timenow = 0.0; par.timefin = 0.125

    r, th = grid.cx1, grid.cx2
    dist2 = r**2 + 1.0 - 2.0 * r * np.cos(th)          # |x - x0|^2, x0 = (r=1, theta=0)
    diff.T[:, :] = np.exp(-dist2 / (4.0 * diff.kappa * t0))

    par.BC[0] = 'free'; par.BC[1] = 'axis'
    par.BC[2] = 'free'; par.BC[3] = 'free'
    return grid, diff, par