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
``'jet2D'``         IC_rHD2D_jet   – relativistic jet propagation (Cartesian)
``'jet2Dcyl'``      IC_rHD2D_jet_cyl – axisymmetric relativistic jet (cylindrical)
``'sheat1D'``       IC_rHD1D_shock_heating – relativistic shock heating (Thompson 1986)
``'pshock1D'``      IC_rHD1D_perturbed_shock – SR Shu-Osher analogue

References
----------
Mignone, A. & Bodo, G. (2005), MNRAS 364, 126

Author: mrkondratyev
"""

import numpy as np
from src.common.eos_setup import EOSdata


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
        "set your ICs and remove this line."
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
    state.F1[:, :] = g_ff
    state.F2[:, :] = 0.0

    return grid, state, par, eos



def IC_rHD1D_perturbed_shock(grid, state, par):
    """
    Relativistic version of the perturbed shock test.

    A Mach ~3 relativistic shock runs into a small sinusoidal density
    perturbation. This is the SR analogue of the Shu-Osher problem
    and tests the scheme's ability to capture fine post-shock
    oscillations in the relativistic regime.

    Left  state (x < -4): rho=3.86, v=0.68, p=42.5
    Right state (x > -4): rho=1+0.2*sin(5x), v=0, p=1
    Gamma = 5/3, domain [-5, 5], t_fin = 1.8

    Parameters
    ----------
    grid  : Grid  (Nx2 = 1 for 1D)
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos
    """
    print("rHD 1D - relativistic perturbed shock (SR Shu-Osher analogue)")

    x1ini, x1fin = -5.0, 5.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 1.8
    par.BC[:] = 'free'
    eos = EOSdata(5.0 / 3.0)

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < -4.0:
                state.dens[i, j] = 3.86
                state.vel1[i, j] = 0.68
                state.vel2[i, j] = 0.0
                state.vel3[i, j] = 0.0
                state.pres[i, j] = 42.5
            else:
                state.dens[i, j] = 1.0 + 0.2 * np.sin(5.0 * grid.cx1[i, j])
                state.vel1[i, j] = 0.0
                state.vel2[i, j] = 0.0
                state.vel3[i, j] = 0.0
                state.pres[i, j] = 1.0

    return grid, state, par, eos


# ============================================================================
#   2D astrophysical problems
# ============================================================================

def IC_rHD2D_jet(grid, state, par):
    """
    Relativistic 2D jet propagation problem.

    A relativistic jet is injected from the left boundary into a
    uniform ambient medium. The jet develops a cocoon, bow shock,
    and internal shock structure characteristic of astrophysical
    relativistic jets (e.g. AGN jets, GRB afterglows).

    Domain: [0, 10] x [-2, 2]
    Ambient: rho=10, p=0.01, v=0
    Jet (injected at x1=0, |x2| < 0.5): rho=0.1, v=0.99, p=0.01
    Gamma = 5/3, t_fin = 15

    The jet is realized by setting the left boundary ghost cells
    in the jet region to the beam state (handled through the IC
    by placing beam values in the left portion of the domain and
    using 'free' BC at x1=0 inner boundary, with the jet nozzle
    implemented via initial conditions at the first active cells).

    Parameters
    ----------
    grid  : Grid
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos

    References
    ----------
    Marti, J. M. & Mueller, E. (2003), Living Rev. Relativ. 6, 7
    """
    print("rHD 2D - relativistic jet propagation")

    x1ini, x1fin = 0.0, 10.0
    x2ini, x2fin = -2.0, 2.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 15.0
    eos = EOSdata(5.0 / 3.0)

    # Ambient medium
    rho_amb = 10.0
    p_amb = 0.01

    # Jet parameters
    rho_jet = 0.1
    v_jet = 0.99
    r_jet = 0.5  # jet half-width

    state.dens[:, :] = rho_amb
    state.pres[:, :] = p_amb
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0

    # Set jet nozzle at x1 = 0 (first few active cells)
    for i in range(grid.Ngc, grid.Ngc + 3):
        for j in range(grid.Ngc, grid.Nx2r):
            if np.abs(grid.cx2[i, j]) < r_jet:
                state.dens[i, j] = rho_jet
                state.vel1[i, j] = v_jet
                state.pres[i, j] = p_amb

    # Find the interior x2-index range for the jet nozzle (|y| < r_jet)
    j_start = None
    j_end = 0
    for j in range(grid.Nx2):
        if np.abs(grid.cx2[grid.Ngc, j + grid.Ngc]) < r_jet:
            if j_start is None:
                j_start = j
            j_end = j + 1
    if j_start is None:
        j_start = 0

    # Register fixed BC for the jet nozzle on the x1-inner boundary (face 0)
    par.BC_fixed[0] = [
        (j_start, j_end, {'dens': rho_jet, 'vel1': v_jet, 'vel2': 0.0,
                           'vel3': 0.0, 'pres': p_amb})
    ]

    par.BC[0] = 'free'
    par.BC[1] = 'free'
    par.BC[2] = 'free'
    par.BC[3] = 'free'

    return grid, state, par, eos


def IC_rHD2D_jet_cyl(grid, state, par):
    """
    Axisymmetric relativistic jet in cylindrical (R, Z) coordinates.

    A light, ultra-relativistic jet (Lorentz factor W ~ 7) is injected
    along the symmetry axis into a denser ambient medium. The jet
    develops a bow shock, cocoon, Mach disk, and reconfinement shocks.
    This is the standard setup for modelling FR-II radio galaxy jets
    and gamma-ray burst afterglows.

    Coordinate system: cylindrical (R, Z)
    Domain: R in [0, 6], Z in [0, 25]
    Jet nozzle: R < 1 at Z = 0, rho=0.01, v_Z=0.99, p=0.01/3
    Ambient: rho=1, v=0, p=0.01/3
    Density ratio eta = rho_jet/rho_amb = 0.01
    Gamma = 5/3, t_fin = 30

    Parameters
    ----------
    grid  : Grid
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos

    References
    ----------
    Marti, J. M. & Mueller, E. (1997), J. Fluid Mech. 258, 317
    Mignone, A., Plewa, T. & Bodo, G. (2005), ApJS 160, 199
    """
    print("rHD 2D - axisymmetric relativistic jet (cylindrical)")

    R_in, R_out = 0.0, 6.0
    Z_in, Z_out = 0.0, 25.0
    grid.CylindricalGrid(R_in, R_out, Z_in, Z_out)

    par.timenow = 0.0
    par.timefin = 30.0
    eos = EOSdata(5.0 / 3.0)

    # Jet parameters
    rho_jet = 0.01
    v_jet = 0.99          # Lorentz factor ~ 7
    p_match = 0.01 / 3.0  # pressure-matched jet
    r_jet = 1.0            # jet radius

    # Ambient medium
    rho_amb = 1.0

    state.dens[:, :] = rho_amb
    state.pres[:, :] = p_match
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0

    # Jet nozzle at Z = 0: set first active cells to jet values
    for i in range(grid.Ngc, grid.Ngc + 3):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.cx1[i, j] < r_jet:
                state.dens[i, j] = rho_jet
                state.vel2[i, j] = v_jet  # v_Z in cylindrical
                state.pres[i, j] = p_match

    # Find the interior x1-index where R exceeds the jet radius
    idx_jet = 0
    for i in range(grid.Nx1):
        if grid.cx1[i + grid.Ngc, grid.Ngc] < r_jet:
            idx_jet = i + 1

    # Register fixed BC for the jet nozzle on the x2-inner boundary (face 1)
    par.BC_fixed[1] = [
        (0, idx_jet, {'dens': rho_jet, 'vel1': 0.0, 'vel2': v_jet,
                       'vel3': 0.0, 'pres': p_match})
    ]

    par.BC[0] = 'axis'
    par.BC[1] = 'wall'
    par.BC[2] = 'free'
    par.BC[3] = 'free'

    return grid, state, par, eos



def IC_rHD1D_shock_heating(grid, state, par):
    """
    Relativistic shock heating test (Thompson 1986).

    A cold fluid with high bulk Lorentz factor (W=10) runs into a
    wall. All kinetic energy is converted to thermal energy.
    The exact post-shock state can be computed analytically
    from the relativistic Rankine-Hugoniot conditions, making
    this an excellent validation test.

    Initial state: rho=1, v=sqrt(1-1/W^2), p=1e-6
    Wall at x=1 (reflecting BC)
    Gamma = 4/3 (ultra-relativistic EOS), t_fin = 0.4

    Parameters
    ----------
    grid  : Grid  (Nx2 = 1 for 1D)
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos

    References
    ----------
    Thompson, K. W. (1986), J. Fluid Mech. 171, 365
    """
    print("rHD 1D - relativistic shock heating (Thompson 1986)")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 0.4
    eos = EOSdata(4.0 / 3.0)

    W = 10.0  # Lorentz factor
    v_bulk = np.sqrt(1.0 - 1.0 / W**2)

    state.dens[:, :] = 1.0
    state.vel1[:, :] = v_bulk
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0
    state.pres[:, :] = 1.0e-6

    par.BC[0] = 'free'
    par.BC[1] = 'free'
    par.BC[2] = 'wall'
    par.BC[3] = 'free'

    return grid, state, par, eos
