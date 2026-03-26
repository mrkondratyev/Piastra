# -*- coding: utf-8 -*-
"""
rMHD_init_cond.py
=================

Initial conditions for Special-Relativistic Magnetohydrodynamics (SRMHD) tests.

All functions follow the same interface as the other Piastra IC modules:

    IC_rMHD_<name>(grid, state, par)  ->  (grid, state, par, eos)

where *state* is a SimState object with attributes:
    dens, vel1, vel2, vel3, pres   -- primitive variables (with ghost cells)
    bfi1, bfi2, bfi3               -- cell-centred magnetic field
    fb1, fb2                       -- face-centred (staggered) magnetic field
    F1, F2                         -- external force (gravity, etc.)

The function sets:
    par.timefin   -- final physical time
    par.BC        -- boundary-condition array  [x1_in, x2_in, x1_out, x2_out]

and creates and returns an EOSdata object.

Currently implemented
---------------------
  IC_rMHD1D_blast     : 1D relativistic MHD blast (Mignone & Bodo 2006)
  IC_rMHD2D_rotor     : 2D relativistic rotor (Del Zanna et al. 2003)
  IC_rMHD_user_defined : placeholder for custom ICs

Author
------
mrkondratyev
"""

import numpy as np
from eos_setup import EOSdata


# ============================================================================
# 1D relativistic MHD blast wave
# ============================================================================

def IC_rMHD1D_blast(grid, state, par):
    """
    1D relativistic MHD blast wave (Mignone & Bodo 2006, test 1).

    A high-pressure region |x| < 0.1 is surrounded by low-pressure gas.
    A uniform transverse magnetic field B_y = 0.5 is present.
    Adiabatic index GAMMA = 4/3 (ultrarelativistic gas).

    Domain: x in [-0.5, 0.5], resolved along x1 (1D, Nx2 = 1 cell).

    Parameters
    ----------
    grid  : Grid
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos
    """
    print("rMHD 1D -- relativistic MHD blast wave (Mignone & Bodo 2006)")

    x1ini, x1fin = -0.5, 0.5
    x2ini, x2fin =  0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timefin = 0.4
    par.timenow = 0.0
    par.BC = np.array(['free', 'free', 'free', 'free'], dtype=object)

    eos = EOSdata(4.0 / 3.0)

    x = grid.cx1

    # Default background
    state.dens[:, :] = 1.0
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0
    state.pres[:, :] = 1.0e-3
    state.bfi1[:, :] = 0.0
    state.bfi2[:, :] = 0.5
    state.bfi3[:, :] = 0.0

    # High-pressure region
    mask = np.abs(x) < 0.1
    state.pres[mask] = 1.0

    # Initialise staggered B
    state.fb1[:, :] = 0.0
    state.fb2[:, :] = 0.5

    return grid, state, par, eos


# ============================================================================
# 2D relativistic rotor
# ============================================================================

def IC_rMHD2D_rotor(grid, state, par):
    """
    2D relativistic rotor (Del Zanna et al. 2003 test).

    A uniformly rotating dense cylinder (r < 0.1) embedded in a uniform
    ambient medium with background magnetic field B_x = 1.
    Adiabatic index GAMMA = 5/3.

    Domain: [-0.5, 0.5] x [-0.5, 0.5].

    Parameters
    ----------
    grid  : Grid
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos
    """
    print("rMHD 2D -- relativistic rotor (Del Zanna et al. 2003)")

    x1ini, x1fin = -0.5, 0.5
    x2ini, x2fin = -0.5, 0.5
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timefin = 0.4
    par.timenow = 0.0
    par.BC = np.array(['free', 'free', 'free', 'free'], dtype=object)

    eos = EOSdata(5.0 / 3.0)

    x = grid.cx1
    y = grid.cx2
    r = np.sqrt(x**2 + y**2)

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
    omega = 9.95   # chosen so that v_max ~ 0.995 at r = r0
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

def IC_rMHD_user_defined(grid, state, par):
    """
    Placeholder for user-defined SRMHD initial conditions.

    Fill in custom values for:
        state.dens, state.vel1-3, state.pres
        state.bfi1-3, state.fb1, state.fb2
        par.timefin
        par.BC

    Parameters
    ----------
    grid  : Grid
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos
    """
    print("rMHD -- user-defined problem")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timefin = 0.4
    par.timenow = 0.0
    par.BC[:] = 'free'

    eos = EOSdata(4.0 / 3.0)

    state.dens[:, :] = 1.0
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0
    state.pres[:, :] = 1.0
    state.bfi1[:, :] = 0.0
    state.bfi2[:, :] = 0.0
    state.bfi3[:, :] = 0.0
    state.fb1[:, :]  = 0.0
    state.fb2[:, :]  = 0.0

    raise ValueError(
        "IC_rMHD_user_defined: please implement your custom initial "
        "condition here and remove this line."
    )

    return grid, state, par, eos
