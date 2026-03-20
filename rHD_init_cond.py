# -*- coding: utf-8 -*-
"""
===============================================================================
rHD_init_cond.py
===============================================================================

Initial condition functions for special-relativistic hydrodynamics (rHD)
test problems.

Each function sets up a complete problem:
  - configures the grid (geometry and domain bounds)
  - initialises the primitive variables (dens, vel1, vel2, vel3, pres)
  - sets boundary conditions par.BC
  - sets simulation times par.timenow / par.timefin
  - returns the EOS object

Signature convention (identical to other Piastra IC modules):

    IC_rHD_<name>(grid, state, par)  ->  grid, state, par, eos

Available problems
------------------
``'user_defined'``  IC_rHD_user_defined
``'RP1'``           IC_rHD1D_RP1   – Mignone & Bodo (2005) RP1: moving fluid
``'RP3'``           IC_rHD1D_RP3   – RP3: strong shock
``'RP4'``           IC_rHD1D_RP4   – RP4: ultra-relativistic (p=1000)
``'RP5'``           IC_rHD1D_RP5   – RP5: tangential velocity test
``'RP2D'``          IC_rHD2D_RP    – 2D Riemann problem
``'RTI'``           IC_rHD2D_RTI   – relativistic Rayleigh-Taylor instability

References
----------
Mignone, A. & Bodo, G. (2005), MNRAS 364, 126

Author: mrkondratyev
"""

import numpy as np
from eos_setup import EOSdata


# ============================================================================
#   User-defined template
# ============================================================================

def IC_rHD_user_defined(grid, state, par):
    """
    Template for a user-defined rHD problem.

    Parameters
    ----------
    grid  : Grid
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos
    """
    print("rHD – user-defined problem")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 0.4

    par.BC[:] = 'free'

    eos = EOSdata(5.0 / 3.0)

    # ----- Set your initial condition below -----
    state.dens[:, :] = 1.0
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0
    state.pres[:, :] = 1.0

    raise ValueError(
        "User-defined rHD problem – see 'rHD_init_cond.py', "
        "set your IC and remove this line."
    )

    return grid, state, par, eos


# ============================================================================
#   1D Riemann problems from Mignone & Bodo (2005)
# ============================================================================

def IC_rHD1D_RP1(grid, state, par):
    """
    Mignone & Bodo (2005) Riemann Problem 1: moving fluid colliding with
    a static region.

    Left  state: ρ=1, v₁=0.9, p=1
    Right state: ρ=1, v₁=0,   p=10
    Γ = 5/3,  t_fin = 0.4

    Parameters
    ----------
    grid  : Grid  (Nx2 = 1 for 1D)
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos
    """
    print("rHD 1D – Mignone & Bodo (2005) Riemann Problem 1")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 0.4

    par.BC[:] = 'free'

    eos = EOSdata(5.0 / 3.0)

    xmid = 0.5 * (x1ini + x1fin)
    mask_L = grid.cx1 < xmid

    state.dens[:, :] = np.where(mask_L, 1.0,  1.0)
    state.vel1[:, :] = np.where(mask_L, 0.9,  0.0)
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0
    state.pres[:, :] = np.where(mask_L, 1.0, 10.0)

    return grid, state, par, eos


def IC_rHD1D_RP3(grid, state, par):
    """
    Mignone & Bodo (2005) Riemann Problem 3: strong relativistic shock.

    Left  state: ρ=10, v₁=0, p=40/3
    Right state: ρ=1,  v₁=0, p=2/3×10⁻⁶
    Γ = 5/3,  t_fin = 0.4

    Parameters
    ----------
    grid  : Grid  (Nx2 = 1 for 1D)
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos
    """
    print("rHD 1D – Mignone & Bodo (2005) Riemann Problem 3 (strong shock)")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 0.4

    par.BC[:] = 'free'

    eos = EOSdata(5.0 / 3.0)

    xmid = 0.5 * (x1ini + x1fin)
    mask_L = grid.cx1 < xmid

    state.dens[:, :] = np.where(mask_L, 10.0, 1.0)
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0
    state.pres[:, :] = np.where(mask_L, 40.0 / 3.0, 2.0 / 3.0 * 1e-6)

    return grid, state, par, eos


def IC_rHD1D_RP4(grid, state, par):
    """
    Mignone & Bodo (2005) Riemann Problem 4: ultra-relativistic blast wave.

    Left  state: ρ=1, v₁=0, p=1000
    Right state: ρ=1, v₁=0, p=0.01
    Γ = 5/3,  t_fin = 0.4

    Parameters
    ----------
    grid  : Grid  (Nx2 = 1 for 1D)
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos
    """
    print("rHD 1D – Mignone & Bodo (2005) Riemann Problem 4 (ultra-relativistic)")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 0.4

    par.BC[:] = 'free'

    eos = EOSdata(5.0 / 3.0)

    xmid = 0.5 * (x1ini + x1fin)
    mask_L = grid.cx1 < xmid

    state.dens[:, :] = 1.0
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0
    state.pres[:, :] = np.where(mask_L, 1000.0, 0.01)

    return grid, state, par, eos


def IC_rHD1D_RP5(grid, state, par):
    """
    Special-relativistic shock tube with tangential velocity.

    Left  state: ρ=1, v₁=0, v₂=0,    p=1000
    Right state: ρ=1, v₁=0, v₂=0.99, p=0.01
    Γ = 5/3,  t_fin = 0.4

    Tests that the transverse velocity is properly advected across
    a contact wave in the relativistic framework.

    Parameters
    ----------
    grid  : Grid  (Nx2 = 1 for 1D)
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos
    """
    print("rHD 1D – shock tube with tangential velocity (RP5)")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 0.4

    par.BC[:] = 'free'

    eos = EOSdata(5.0 / 3.0)

    xmid = 0.5 * (x1ini + x1fin)
    mask_L = grid.cx1 < xmid

    state.dens[:, :] = 1.0
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = np.where(mask_L, 0.0, 0.99)
    state.vel3[:, :] = 0.0
    state.pres[:, :] = np.where(mask_L, 1000.0, 0.01)

    return grid, state, par, eos


# ============================================================================
#   2D problems
# ============================================================================

def IC_rHD2D_RP(grid, state, par):
    """
    Special-relativistic 2D Riemann problem from Mignone & Bodo (2005).

    The domain [-1, 1] × [-1, 1] is divided into four quadrants:

    ┌──────────┬──────────┐
    │ ρ=0.1    │ ρ=0.1    │
    │ v₁=0.99  │ v₁=0     │
    │ v₂=0     │ v₂=0     │
    │ p=1      │ p=0.01   │
    │ (x<0,y>0)│ (x>0,y>0)│
    ├──────────┼──────────┤
    │ ρ=0.5    │ ρ=0.1    │
    │ v₁=0     │ v₁=0     │
    │ v₂=0     │ v₂=0.99  │
    │ p=1      │ p=1      │
    │ (x<0,y<0)│ (x>0,y<0)│
    └──────────┴──────────┘

    Parameters
    ----------
    grid  : Grid
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos
    """
    print("rHD 2D – Riemann problem (Mignone & Bodo 2005)")

    x1ini, x1fin = -1.0, 1.0
    x2ini, x2fin = -1.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 0.8

    par.BC[:] = 'free'

    eos = EOSdata(5.0 / 3.0)

    cx1 = grid.cx1
    cx2 = grid.cx2

    # Quadrant I: x > 0, y > 0
    m1 = (cx1 > 0.0) & (cx2 > 0.0)
    # Quadrant II: x < 0, y > 0
    m2 = (cx1 < 0.0) & (cx2 > 0.0)
    # Quadrant III: x < 0, y < 0
    m3 = (cx1 < 0.0) & (cx2 < 0.0)
    # Quadrant IV: x > 0, y < 0
    m4 = (cx1 > 0.0) & (cx2 < 0.0)

    state.dens[:, :] = (  0.1 * m1 + 0.1  * m2
                        + 0.5 * m3 + 0.1  * m4)
    state.vel1[:, :] = (  0.0 * m1 + 0.99 * m2
                        + 0.0 * m3 + 0.0  * m4)
    state.vel2[:, :] = (  0.0 * m1 + 0.0  * m2
                        + 0.0 * m3 + 0.99 * m4)
    state.vel3[:, :] = 0.0
    state.pres[:, :] = (  0.01 * m1 + 1.0 * m2
                        + 1.0  * m3 + 1.0 * m4)

    return grid, state, par, eos


def IC_rHD2D_RTI(grid, state, par):
    """
    Relativistic Rayleigh-Taylor instability in 2D.

    A heavier fluid (ρ_up = 2) sits on top of a lighter one (ρ_dn = 1)
    in a gravitational field pointing downward (F₁ = g_ff < 0).
    The interface at x₁ = 0 is perturbed sinusoidally.

    Domain: x₁ ∈ [-1, 1], x₂ ∈ [0, 1]
    Pressure set to satisfy the SR hydrostatic equilibrium at t = 0.
    Γ = 5/3,  t_fin = 10

    Boundary conditions:
    - x₁ (vertical): reflecting on both ends
    - x₂ (horizontal): periodic

    Parameters
    ----------
    grid  : Grid
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos
    """
    print("rHD 2D – relativistic Rayleigh-Taylor instability")

    x1ini, x1fin = -1.0, 1.0
    x2ini, x2fin =  0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 10.0

    # reflecting walls in x1, periodic in x2
    # BC order: [x1_inner, x2_inner, x1_outer, x2_outer]
    par.BC[0] = 'wall'
    par.BC[1] = 'peri'
    par.BC[2] = 'wall'
    par.BC[3] = 'peri'

    eos = EOSdata(5.0 / 3.0)

    rho_u = 2.0      # upper (heavy) fluid
    rho_d = 1.0      # lower (light) fluid
    g_ff  = -0.5     # gravitational acceleration

    P0 = 10.0 / 7.0 + 0.25   # pressure at bottom of heavy layer
    P1 = 10.0 / 7.0 - 0.25   # pressure at top of heavy layer

    # Interface perturbation
    h0    = 0.03
    kappa = 4.0 * np.pi

    cx1 = grid.cx1
    cx2 = grid.cx2

    interface = h0 * np.cos(cx2 * kappa)
    upper     = cx1 > interface

    state.dens[:, :] = np.where(upper, rho_u, rho_d)
    state.pres[:, :] = np.where(
        upper,
        P1 + cx1 * g_ff * rho_u,
        P0 + (cx1 + 1.0) * g_ff * rho_d
    )
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0

    # Gravitational source term
    Ngc = grid.Ngc
    state.F1[:, :] = g_ff
    state.F2[:, :] = 0.0

    return grid, state, par, eos
