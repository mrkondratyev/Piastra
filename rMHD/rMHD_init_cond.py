# -*- coding: utf-8 -*-
"""
rMHD_init_cond.py
=================

Initial conditions for Special-Relativistic Magnetohydrodynamics (SRMHD) tests.

All functions follow the same interface:

    IC_rMHD_<name>(grid, state, par, eos)  →  (grid, state, par, eos)

where *state* is a SimState object with attributes:
    dens, vel1, vel2, vel3, pres   – primitive variables (with ghost cells)
    bfi1, bfi2, bfi3               – cell-centred magnetic field
    fb1, fb2                       – face-centred (staggered) magnetic field
    F1, F2                         – external force (gravity, etc.)

The function must also set:
    par.timefin   – final physical time
    par.BC        – boundary-condition array  [x1_in, x2_in, x1_out, x2_out]

Currently implemented
---------------------
  IC_rMHD1D_blast   : 1D relativistic MHD blast (Mignone & Bodo 2006 test)
  IC_rMHD2D_rotor   : 2D relativistic rotor (Del Zanna et al. 2003 test)
  IC_rMHD_user_defined : placeholder for custom ICs

Author
------
mrkondratyev
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================================
# Utility: set staggered face-centred B from cell-centred B (uniform case)
# ============================================================================

def _set_face_B_from_cell(grid, state):
    """
    Initialise staggered fields fb1, fb2 from cell-centred bfi1, bfi2.

    Uses simple linear interpolation.  For divergence-free initialisations
    the staggered field should be set directly from a vector potential.
    """
    Ngc = grid.Ngc
    # fb1[i, j] = average of bfi1[i-1, j] and bfi1[i, j]
    state.fb1[:, :] = 0.5 * (state.bfi1[Ngc-1:-Ngc, Ngc:-Ngc] +
                              state.bfi1[Ngc:  -Ngc+1 or None, Ngc:-Ngc])
    # fb2[i, j] = average of bfi2[i, j-1] and bfi2[i, j]
    state.fb2[:, :] = 0.5 * (state.bfi2[Ngc:-Ngc, Ngc-1:-Ngc] +
                              state.bfi2[Ngc:-Ngc, Ngc:  -Ngc+1 or None])


# ============================================================================
# 1D relativistic MHD blast wave
# ============================================================================

def IC_rMHD1D_blast(grid, state, par, eos):
    """
    1D relativistic MHD blast wave (Mignone & Bodo 2006, test 1).

    A high-pressure region [−0.1, 0.1] is surrounded by low-pressure gas.
    A uniform transverse magnetic field B_y = 0.5 is present.

    Domain: x ∈ [−0.5, 0.5], resolved along x1 (1D, Nx2 = 1 cell).

    Parameters
    ----------
    grid, state, par, eos : standard Piastra objects

    Returns
    -------
    grid, state, par, eos
    """
    par.timefin = 0.4
    par.BC = np.array(['free', 'free', 'free', 'free'], dtype=object)

    Ngc = grid.Ngc
    x   = grid.cx1   # shape (Nx1_tot, Nx2_tot)

    # Default background
    state.dens[:, :] = 1.0
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0
    state.pres[:, :] = 1.0e-3
    state.bfi1[:, :] = 0.0
    state.bfi2[:, :] = 0.5    # uniform B_y
    state.bfi3[:, :] = 0.0

    # High-pressure region
    mask = np.abs(x) < 0.1
    state.pres[mask] = 1.0

    # Initialise staggered B
    Nx1 = grid.Nx1
    Nx2 = grid.Nx2
    state.fb1[:, :] = 0.0
    state.fb2[:, :] = 0.5

    return grid, state, par, eos


# ============================================================================
# 2D relativistic rotor
# ============================================================================

def IC_rMHD2D_rotor(grid, state, par, eos):
    """
    2D relativistic rotor (Del Zanna et al. 2003 test).

    A uniformly rotating dense cylinder (r < 0.1) embedded in a uniform
    ambient medium and a uniform background magnetic field B_x = 1.

    Domain: [−0.5, 0.5] × [−0.5, 0.5].

    Parameters
    ----------
    grid, state, par, eos

    Returns
    -------
    grid, state, par, eos
    """
    par.timefin = 0.4
    par.BC = np.array(['free', 'free', 'free', 'free'], dtype=object)

    Ngc = grid.Ngc
    x   = grid.cx1   # (Nx1_tot, Nx2_tot)
    y   = grid.cx2

    r   = np.sqrt(x**2 + y**2)

    # Ambient medium
    state.dens[:, :] = 1.0
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0
    state.pres[:, :] = 1.0
    state.bfi1[:, :] = 1.0
    state.bfi2[:, :] = 0.0
    state.bfi3[:, :] = 0.0

    # Rotating cylinder
    r0    = 0.1
    omega = 9.95   # chosen so that v_max ≈ 0.995 at r = r0
    mask  = r < r0
    state.dens[mask] = 10.0
    state.vel1[mask] = -omega * y[mask]
    state.vel2[mask] =  omega * x[mask]

    # Clip velocity to avoid superluminal values
    v2   = state.vel1**2 + state.vel2**2 + state.vel3**2
    vmax = np.sqrt(np.max(v2))
    if vmax >= 1.0:
        fac = 0.995 / vmax
        state.vel1[mask] *= fac
        state.vel2[mask] *= fac

    # Initialise staggered B (uniform Bx)
    state.fb1[:, :] = 1.0
    state.fb2[:, :] = 0.0

    return grid, state, par, eos


# ============================================================================
# User-defined placeholder
# ============================================================================

def IC_rMHD_user_defined(grid, state, par, eos):
    """
    Placeholder for user-defined SRMHD initial conditions.

    Fill in custom values for:
        state.dens, state.vel1-3, state.pres
        state.bfi1-3, state.fb1, state.fb2
        par.timefin
        par.BC
    """
    raise NotImplementedError(
        "IC_rMHD_user_defined: please implement your custom initial condition here."
    )
