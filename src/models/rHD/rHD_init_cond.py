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

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.4

    eos = EOSdata(5.0 / 3.0)

    # ----- Set your initial condition below -----
    state.dens[:, :] = 1.0
    state.vel1[:, :] = state.vel2[:, :] = state.vel3[:, :] = 0.0
    state.pres[:, :] = 1.0

    raise ValueError(
        "User-defined rHD problem – see 'rHD_init_cond.py', "
        "set your ICs and remove this line."
    )

    par.BC[:] = 'free'

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

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.4

    eos = EOSdata(5.0 / 3.0)

    xc = 0.5 * (x1ini + x1fin)
    left = grid.cx1 < xc

    state.dens[:, :] = np.where(left, 1.0,  1.0)
    state.vel1[:, :] = np.where(left, 0.9,  0.0)
    state.vel2[:, :] = 0.0; state.vel3[:, :] = 0.0
    state.pres[:, :] = np.where(left, 1.0, 10.0)

    par.BC[:] = 'free'

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

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.4

    eos = EOSdata(5.0 / 3.0)

    xc = 0.5 * (x1ini + x1fin)
    left = grid.cx1 < xc

    state.dens[:, :] = np.where(left, 10.0, 1.0)
    state.vel1[:, :] = state.vel2[:, :] = state.vel3[:, :] = 0.0
    state.pres[:, :] = np.where(left, 40.0 / 3.0, 2.0 / 3.0 * 1e-6)
    
    par.BC[:] = 'free'

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

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.4

    eos = EOSdata(5.0 / 3.0)

    xc = 0.5 * (x1ini + x1fin)
    left = grid.cx1 < xc

    state.dens[:, :] = 1.0
    state.vel1[:, :] = state.vel2[:, :] = state.vel3[:, :] = 0.0
    state.pres[:, :] = np.where(left, 1000.0, 0.01)

    par.BC[:] = 'free'
    
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

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.4

    eos = EOSdata(5.0 / 3.0)

    xc = 0.5 * (x1ini + x1fin)
    left = grid.cx1 < xc

    state.dens[:, :] = 1.0
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = np.where(left, 0.0, 0.99)
    state.vel3[:, :] = 0.0
    state.pres[:, :] = np.where(left, 1000.0, 0.01)

    par.BC[:] = 'free'

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

    x1ini, x1fin = -1.0, 1.0; x2ini, x2fin = -1.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.8
    
    eos = EOSdata(5.0 / 3.0)
    
    cx1 = grid.cx1; cx2 = grid.cx2

    m1 = (cx1 > 0.0) & (cx2 > 0.0) # Quadrant I: x > 0, y > 0
    m2 = (cx1 < 0.0) & (cx2 > 0.0) # Quadrant II: x < 0, y > 0
    m3 = (cx1 < 0.0) & (cx2 < 0.0) # Quadrant III: x < 0, y < 0
    m4 = (cx1 > 0.0) & (cx2 < 0.0) # Quadrant IV: x > 0, y < 0

    state.dens[:, :] = (0.1 * m1 + 0.1  * m2 + 0.5 * m3 + 0.1  * m4)
    state.vel1[:, :] = (0.0 * m1 + 0.99 * m2 + 0.0 * m3 + 0.0  * m4)
    state.vel2[:, :] = (0.0 * m1 + 0.0  * m2 + 0.0 * m3 + 0.99 * m4)
    state.vel3[:, :] = 0.0
    state.pres[:, :] = (0.01 * m1 + 1.0 * m2 + 1.0  * m3 + 1.0 * m4)
    
    par.BC[:] = 'free'

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

    #grid creation
    x1ini, x1fin = -1.0, 1.0; x2ini, x2fin =  0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 10.0

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
    state.vel1[:, :] = 0.0; state.vel2[:, :] = 0.0; state.vel3[:, :] = 0.0

    # Gravitational source term
    state.F1[:, :] = g_ff; state.F2[:, :] = 0.0
    
    # reflecting walls in x1, periodic in x2
    # BC order: [x1_inner, x2_inner, x1_outer, x2_outer]
    par.BC[0] = 'wall'; par.BC[1] = 'peri'
    par.BC[2] = 'wall'; par.BC[3] = 'peri'

    return grid, state, par, eos


# ============================================================================
#   2D astrophysical problems
# ============================================================================

def IC_rHD2D_jet_cart(grid, state, par):
    """
    Relativistic 2D jet propagation problem (Cartesian).

    A relativistic beam (v = 0.99, Lorentz factor ~ 7) is injected through a
    nozzle on the LEFT boundary (x1-inner, face 0) over |y| < r_jet, into a
    uniform, pressure-matched ambient medium. The jet develops a cocoon, bow
    shock and internal shocks characteristic of AGN / GRB jets.

    Coordinate system : Cartesian (x, y) = (x1, x2)
    Domain            : x in [0, 10], y in [-2, 2]
    Inlet (face 0)    : |y| < 0.5, rho=0.1, v_x=0.99, p=0.01
    Ambient           : rho=10, v=0, p=0.01

    The inlet is a fixed (Dirichlet) ghost-fill in par.BC_fixed[0]; the interior
    starts as pure ambient, so the jet is entirely a boundary condition (no
    internal seed, hence no initial discontinuity / start-up transient). Because
    the beam is relativistic (all characteristics inward at the inlet), the
    soft ghost-pin inlet is exact enough; the flux routine is untouched.

    Requires Parameters to define BC_fixed = {0:[],1:[],2:[],3:[]} and
    boundCond_rHD to apply apply_bc_fixed (after the standard fills).

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
    print("rHD 2D - relativistic jet propagation (Cartesian, inlet BC)")

    # --- grid + time ---
    x1ini, x1fin = 0.0, 10.0
    x2ini, x2fin = -4.0, 4.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    par.timenow = 0.0
    par.timefin = 15.0
    eos = EOSdata(5.0 / 3.0)

    # --- aliases ---
    Ngc  = grid.Ngc
    Nx2  = grid.Nx2; Nx2r = grid.Nx2r

    # --- jet / ambient parameters (pressure-matched) ---
    rho_amb = 10.0
    p_amb   = 0.01
    rho_jet = 0.1
    v_jet   = 0.99            # Lorentz factor W ~ 7.09
    r_jet   = 0.5            # jet half-width in y

    # --- uniform ambient everywhere (incl. ghosts) ---
    state.dens[:, :] = rho_amb
    state.pres[:, :] = p_amb
    state.vel1[:, :] = state.vel2[:, :] = state.vel3[:, :] = 0.0

    # --- nozzle extent along y (tangential to the left face) ---
    yc = grid.cx2[Ngc, Ngc:Nx2r]            # 1D interior y cell-centres
    in_jet = np.nonzero(np.abs(yc) < r_jet)[0]
    j_start = int(in_jet[0])
    j_end   = int(in_jet[-1]) + 1

    # --- fixed (Dirichlet) inlet on the left face (x1-inner = face 0) ---
    par.BC_fixed[0] = [
        (j_start, j_end, {'dens': rho_jet, 'pres': p_amb,
                          'vel1': v_jet, 'vel2': 0.0, 'vel3': 0.0})
    ]

    par.BC[0] = 'free'    # x1 inner (left)  -- nozzle via BC_fixed[0]
    par.BC[1] = 'free'    # x2 inner (bottom)
    par.BC[2] = 'free'    # x1 outer (right)
    par.BC[3] = 'free'    # x2 outer (top)

    return grid, state, par, eos


def IC_rHD2D_jet_cyl(grid, state, par):
    """
    Axisymmetric relativistic jet in cylindrical (R, Z) coordinates.

    A light, ultra-relativistic beam (v_Z = 0.99, Lorentz factor ~ 7) is
    injected along the symmetry axis through a nozzle on the BOTTOM boundary
    (x2-inner, face 1) over R < r_jet, into a denser, pressure-matched ambient
    medium. The jet develops a bow shock, cocoon, Mach disk and reconfinement
    shocks (FR-II radio-galaxy / GRB-afterglow morphology).

    Coordinate system : cylindrical (R, Z) = (x1, x2)
    Domain            : R in [0, 6], Z in [0, 25]
    Inlet (face 1)    : R < 1, rho=0.01, v_Z=0.99, p=0.01/3
    Ambient           : rho=1, v=0, p=0.01/3      (eta = rho_jet/rho_amb = 0.01)

    The inlet is a fixed (Dirichlet) ghost-fill in par.BC_fixed[1]; the interior
    starts as pure ambient (no internal seed). The beam is relativistic (all
    characteristics inward at the inlet), so the soft ghost-pin inlet is exact
    enough and the flux routine is untouched.

    Requires Parameters to define BC_fixed = {0:[],1:[],2:[],3:[]} and
    boundCond_rHD to apply apply_bc_fixed (after the standard fills).

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
    Marti, J. M. et al. (1997), ApJ 479, 151
    Mignone, A., Plewa, T. & Bodo, G. (2005), ApJS 160, 199
    """
    print("rHD 2D - axisymmetric relativistic jet (cylindrical, inlet BC)")

    # --- grid + time ---
    R_in, R_out = 0.0, 6.0
    Z_in, Z_out = 0.0, 25.0
    grid.CylindricalGrid(R_in, R_out, Z_in, Z_out)
    par.timenow = 0.0
    par.timefin = 30.0
    eos = EOSdata(5.0 / 3.0)

    # --- aliases ---
    Ngc  = grid.Ngc
    Nx1  = grid.Nx1
    Nx1r = grid.Nx1r

    # --- jet / ambient parameters (pressure-matched) ---
    rho_jet = 0.01
    v_jet   = 0.99            # v_Z; Lorentz factor W ~ 7.09
    p_match = 0.01 / 3.0
    r_jet   = 1.0            # jet radius
    rho_amb = 1.0            # eta = rho_jet/rho_amb = 0.01

    # --- uniform ambient everywhere (incl. ghosts) ---
    state.dens[:, :] = rho_amb
    state.pres[:, :] = p_match
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0

    # --- nozzle extent along R (tangential to the bottom face) ---
    Rc = grid.cx1[Ngc:Nx1r, Ngc]            # 1D interior R cell-centres
    in_jet = np.nonzero(Rc < r_jet)[0]      # contiguous from the axis
    start  = int(in_jet[0])                 # 0
    end    = int(in_jet[-1]) + 1

    # --- fixed (Dirichlet) inlet on the bottom face (x2-inner = face 1) ---
    par.BC_fixed[1] = [
        (start, end, {'dens': rho_jet, 'pres': p_match,
                      'vel1': 0.0, 'vel2': v_jet, 'vel3': 0.0})
    ]

    par.BC[0] = 'axis'    # x1 inner (R = 0)
    par.BC[1] = 'wall'    # x2 inner (Z = 0, nozzle via BC_fixed[1])
    par.BC[2] = 'free'    # x1 outer (R = 6)
    par.BC[3] = 'free'    # x2 outer (Z = 25)

    return grid, state, par, eos
