# -*- coding: utf-8 -*-
"""
sim_state.py

This module defines a unified class for storing the state of advection,
compressible hydrodynamics (HD), and magnetohydrodynamics (MHD) on a 2D
computational grid. It is designed for frameworks that compare numerical
methods and solvers across different physics modules.

The SimState (simulated state) class allocates arrays according to the selected mode:

- 'adv' : scalar field(s) for linear advection, plus optional advection
  velocities.
- 'HD'  : compressible hydrodynamics with primitive and conservative
  variables, plus optional source terms.
- 'MHD' : HD arrays plus magnetic fields, staggered magnetic fields,
  divergence-cleaning arrays, and conservative magnetic fluxes.

Primitive variables include ghost cells to simplify boundary condition
handling, while conservative variables and source terms are defined only
in the interior physical domain. Magnetic fields are allocated only when
MHD mode is enabled.

This module does not implement solvers; it only provides storage for state
variables. External solver routines can access the arrays consistently
regardless of the mode.

Notes
-----
- Ghost cells exist only for primitive variables.
- For 'adv' mode, only the scalar field and advection velocities are
  allocated.
- HD and MHD arrays are fully allocated for solver testing and comparison.
- MHD-specific arrays include both cell-centered (bfi) and staggered (fb)
  fields, as well as divergence-cleaning (divB).

Author: mrkondratyev
"""

import numpy as np

class SimState:
    """
    Container for advection, compressible fluid, or MHD state variables
    on a 2D computational grid.

    The class allocates arrays depending on the selected mode:

    Parameters
    ----------
    grid : Grid
        Grid object providing geometry and array sizes. Must define
        `grid.grid_shape`, `grid.Nx1`, and `grid.Nx2`.
    par : parameters
        Simulation parameters object with attribute `mode` that can be
        'adv', 'HD', or 'MHD'. Determines which arrays are allocated.

    Attributes
    ----------
    # Mode flags
    dens : ndarray
        Scalar field for linear advection (only in 'adv' mode).
    vel1, vel2 : float or ndarray
        Advection velocities (float for 'adv', arrays for HD/MHD).

    # Primitive variables (including ghost cells, HD/MHD only)
    dens : ndarray
        Mass density field.
    vel1, vel2, vel3 : ndarray
        Velocity components along each coordinate direction.
    pres : ndarray
        Pressure field.
    
    # Magnetic fields (MHD only)
    bfi1, bfi2, bfi3 : ndarray
        Cell-centered magnetic field components.
    fb1, fb2 : ndarray
        Face-centered (staggered) magnetic field components.
    
    # Conservative variables (interior only, HD/MHD only)
    mass : ndarray
        Mass per unit volume.
    mom1, mom2, mom3 : ndarray
        Momentum components per unit volume.
    etot : ndarray
        Total energy per unit volume.
    bcon1, bcon2, bcon3 : ndarray
        Conserved magnetic fluxes (MHD only).

    # Auxiliary
    divB : ndarray
        Divergence of magnetic field (MHD only).
    F1, F2 : ndarray
        User-defined source terms (HD/MHD only, e.g., gravity).
    """


    def __init__(self, grid, par):
        
        if par.mode == 'adv':
            self.dens = np.zeros(grid.grid_shape, dtype=np.double)
            # Advection velocities (they are constant by now)
            self.vel1 = 0.0
            self.vel2 = 0.0
        
        if par.mode == 'HD' or par.mode == 'MHD': 
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
        if par.mode == 'HD' or par.mode == 'MHD':
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

        
