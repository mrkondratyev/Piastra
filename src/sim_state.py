# -*- coding: utf-8 -*-
"""
sim_state.py
============

Unified state container for all Piastra simulation modes on a 2D grid.

The SimState class allocates arrays according to the selected mode:

- 'adv'  : scalar density field plus constant advection velocities.
- 'HD'   : compressible hydrodynamics — primitive variables, conservative
  variables, and external source terms.
- 'rHD'  : special-relativistic hydrodynamics — same layout as HD.
- 'MHD'  : HD arrays plus cell-centred (bfi) and face-centred (fb)
  magnetic fields, conservative magnetic fluxes (bcon), and a
  divergence monitor (divB).
- 'rMHD' : same layout as MHD (SRMHD conservative/primitive + B-fields).
- 'diff' : scalar temperature field T and scalar thermal diffusivity kappa.
- 'SWE'  : shallow water equations — fluid height h, velocities vel1/vel2,
  bathymetry b and its gradient b_x/b_y, Coriolis parameter f_c.

Ghost cells are included only for primitive variables (and magnetic
fields), to simplify boundary condition handling.  Conservative variables
and source terms are interior-only arrays of shape ``(Nx1, Nx2)``.

Notes
-----
- Magnetic field arrays (bfi*, fb*, bcon*, divB) are allocated for
  modes 'MHD' and 'rMHD'.
- SWE arrays (b, b_x, b_y, f_c) include ghost cells so that source
  term application in SWE_one_step.py can index them identically to
  h, vel1, vel2. They are set once by the IC function and never change.
- This module does not implement any solvers; it is pure storage.
  External solver routines access the arrays consistently regardless
  of the mode.

Author: mrkondratyev
"""

import numpy as np

class SimState:
    """
    Container for simulation state variables on a 2D computational grid.

    Allocates arrays depending on ``par.mode``.

    Parameters
    ----------
    grid : Grid
        Grid object providing geometry and array sizes.  Must define
        ``grid.grid_shape``, ``grid.Nx1``, and ``grid.Nx2``.
    par : Parameters
        Simulation parameters.  ``par.mode`` controls which arrays are
        allocated: 'adv', 'HD', 'rHD', 'MHD', 'rMHD', 'diff', or 'SWE'.

    Attributes
    ----------
    Advection mode ('adv')
    ~~~~~~~~~~~~~~~~~~~~~~
    dens : ndarray, shape grid.grid_shape
        Scalar advected field (with ghost cells).
    vel1, vel2 : float
        Constant advection velocities in x1 and x2 directions.

    Hydrodynamics / relativistic HD / MHD / rMHD modes
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Primitive variables (shape grid.grid_shape, include ghost cells):

    dens : ndarray
        Rest-mass (or mass) density.
    vel1, vel2, vel3 : ndarray
        Velocity components along x1, x2, x3.
    pres : ndarray
        Thermal pressure.

    Conservative variables (shape (Nx1, Nx2), interior only):

    mass : ndarray
        Conserved mass density (D = rho W for relativistic modes).
    mom1, mom2, mom3 : ndarray
        Conserved momentum components.
    etot : ndarray
        Total energy density.
    F1, F2 : ndarray
        External force source terms (e.g., gravity).

    Magnetic fields ('MHD', 'rMHD'):

    bfi1, bfi2, bfi3 : ndarray, shape grid.grid_shape
        Cell-centred magnetic field components (with ghost cells).
    fb1 : ndarray, shape (Nx1+1, Nx2)
        Face-centred B-field component normal to x1-faces (staggered).
    fb2 : ndarray, shape (Nx1, Nx2+1)
        Face-centred B-field component normal to x2-faces (staggered).
    bcon1, bcon2, bcon3 : ndarray, shape (Nx1, Nx2)
        Conservative magnetic flux densities (interior only).
    divB : ndarray, shape (Nx1, Nx2)
        Divergence of the magnetic field (diagnostic monitor).

    Diffusion mode ('diff')
    ~~~~~~~~~~~~~~~~~~~~~~~
    T : ndarray, shape grid.grid_shape
        Temperature field (with ghost cells).
    kappa : float
        Uniform thermal diffusivity coefficient.

    Shallow water mode ('SWE')
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    h : ndarray, shape grid.grid_shape
        Fluid column height (with ghost cells).
    vel1, vel2 : ndarray, shape grid.grid_shape
        Depth-averaged velocity components (with ghost cells).
    b : ndarray, shape grid.grid_shape
        Bed elevation / bathymetry (with ghost cells).
        Set by the IC function; zero for flat bottom.
    b_x, b_y : ndarray, shape grid.grid_shape
        Bathymetry gradient components (with ghost cells).
        Computed from b by the IC function using grid_misc.gradient().
    f_c : ndarray, shape grid.grid_shape
        Coriolis parameter (with ghost cells).
        May be spatially variable (beta-plane) or constant (zero).
    """

    def __init__(self, grid, par):

        if par.mode == 'adv':
            self.dens = np.zeros(grid.grid_shape, dtype=np.double)
            # Advection velocities (constant)
            self.vel1 = 0.0
            self.vel2 = 0.0

        if par.mode == 'diff':
            self.T     = np.zeros(grid.grid_shape, dtype=np.double)
            self.kappa = 1.0

        if par.mode == 'SWE':
            # Primitive variables (with ghost cells)
            self.h    = np.zeros(grid.grid_shape, dtype=np.double)
            self.vel1 = np.zeros(grid.grid_shape, dtype=np.double)
            self.vel2 = np.zeros(grid.grid_shape, dtype=np.double)
            # free fall acceleration (constant)
            self.g_ff = 1.0
            # Bathymetry and its gradient (with ghost cells, set by IC function)
            # b_x and b_y are computed via grid_misc.gradient() in the IC function
            # and stored here so the time integrator can access them without
            # recomputing every step.
            self.b   = np.zeros(grid.grid_shape, dtype=np.double)
            self.b_x = np.zeros(grid.grid_shape, dtype=np.double)
            self.b_y = np.zeros(grid.grid_shape, dtype=np.double)

            # Coriolis parameter (with ghost cells, set by IC function)
            # Spatially variable for beta-plane problems, zero by default.
            self.f_c = np.zeros(grid.grid_shape, dtype=np.double)

        if par.mode in ('HD', 'MHD', 'rHD', 'rMHD'):
            # Primitive variables (with ghost cells)
            self.dens = np.zeros(grid.grid_shape, dtype=np.double)
            self.vel1 = np.zeros(grid.grid_shape, dtype=np.double)
            self.vel2 = np.zeros(grid.grid_shape, dtype=np.double)
            self.vel3 = np.zeros(grid.grid_shape, dtype=np.double)
            self.pres = np.zeros(grid.grid_shape, dtype=np.double)

            # Conservative variables (interior only)
            shape = (grid.Nx1, grid.Nx2)
            self.mass = np.zeros(shape, dtype=np.double)
            self.mom1 = np.zeros(shape, dtype=np.double)
            self.mom2 = np.zeros(shape, dtype=np.double)
            self.mom3 = np.zeros(shape, dtype=np.double)
            self.etot = np.zeros(shape, dtype=np.double)
            # Source terms
            self.F1 = np.zeros(shape, dtype=np.double)
            self.F2 = np.zeros(shape, dtype=np.double)

        # Magnetic fields
        if par.mode in ('MHD', 'rMHD'):
            # Primitive variables (with ghost cells)
            self.bfi1 = np.zeros(grid.grid_shape, dtype=np.double)
            self.bfi2 = np.zeros(grid.grid_shape, dtype=np.double)
            self.bfi3 = np.zeros(grid.grid_shape, dtype=np.double)

            # Staggered fields
            self.fb1 = np.zeros((grid.Nx1 + 1, grid.Nx2), dtype=np.double)
            self.fb2 = np.zeros((grid.Nx1, grid.Nx2 + 1), dtype=np.double)

            # Conservative variables (interior only)
            self.bcon1 = np.zeros(shape, dtype=np.double)
            self.bcon2 = np.zeros(shape, dtype=np.double)
            self.bcon3 = np.zeros(shape, dtype=np.double)
            self.divB  = np.zeros(shape, dtype=np.double)
