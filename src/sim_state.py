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

Ghost cells are included only for primitive variables (and magnetic
fields), to simplify boundary condition handling.  Conservative variables
and source terms are interior-only arrays of shape ``(Nx1, Nx2)``.

Notes
-----
- Magnetic field arrays (bfi*, fb*, bcon*, divB) are allocated for
  modes 'MHD' and 'rMHD'. 
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
        allocated: 'adv', 'HD', 'rHD', 'MHD', 'rMHD', or 'diff'.

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

    Magnetic fields ('MHD', 'rMHD', and 'HD' modes):

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
    """


    def __init__(self, grid, par):

        if par.mode == 'adv':
            self.dens = np.zeros(grid.grid_shape, dtype=np.double)
            # Advection velocities (they are constant by now)
            self.vel1 = 0.0
            self.vel2 = 0.0

        if par.mode == 'diff':
            self.T     = np.zeros(grid.grid_shape, dtype=np.double)
            self.kappa = 1.0

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

            # conservative state needed only for code clarity and uniformity
            # Conservative variables (interior only)
            self.bcon1 = np.zeros(shape, dtype=np.double)
            self.bcon2 = np.zeros(shape, dtype=np.double)
            self.bcon3 = np.zeros(shape, dtype=np.double)
            self.divB = np.zeros(shape, dtype=np.double)

        
