# -*- coding: utf-8 -*-
"""
===============================================================================
SWE_init_cond.py
===============================================================================

Initial condition functions for the 2D Shallow Water Equations.

Each function follows the standard Piastra IC signature:

    IC_SWE<name>(grid, state, par)  ->  grid, state, par, eos=None

The function is responsible for:
  - Setting the grid geometry (always CartesianGrid for SWE)
  - Setting state.g_ff, par.timenow, par.timefin, par.BC
  - Initialising state.h, state.vel1, state.vel2 in interior cells
  - Initialising state.b, state.b_x, state.b_y (bathymetry and its gradient)
  - Initialising state.f_c (Coriolis parameter)

Bathymetry gradient
-------------------
state.b_x and state.b_y are computed using the cell_gradient() routine
from grid_misc.py, which applies central differences and handles
geometry-aware metric factors. This ensures consistency with the rest of
the framework.

Available problems (the dispatch dictionary, mapping problem name to IC
function, lives in src/misc/helpers.py's initial_model)
------------------------------------------------------------------------
  'dam1D'      -- IC_SWE1D_dam      : 1D dam break along x1 (SWE analogue of Sod tube)
  'bump1D'     -- IC_SWE1D_bump     : 1D supercritical flow over a Gaussian bump
  'dam2D'      -- IC_SWE2D_dam      : 2D radial dam break (SWE analogue of Sedov)
  'bathtub2D'  -- IC_SWE2D_bathtub  : gravity waves in a closed basin
  'expl2D'     -- IC_SWE2D_expl     : cylindrical SWE blast wave
  'tsunami2D'  -- IC_SWE2D_tsunami  : 2D Gaussian SSH bump on deep ocean
  'ocean2D'    -- IC_SWE2D_ocean    : geostrophic ocean eddy with beta-plane Coriolis
  'atmo2D'     -- IC_SWE2D_atmo     : geostrophic atmospheric ridge with seeded noise
  'jet2D'      -- IC_SWE2D_bickley  : barotropic instability of a Bickley jet
  'KHI2D'      -- IC_SWE2D_KH       : Kelvin-Helmholtz instability (SWE analogue)
  'user_defined' -- IC_SWE_user_defined : blank template for custom problems

Author: mrkondratyev
"""

import numpy as np
from src.grid.grid_misc import cell_gradient


# ============================================================================
# User-defined template
# ============================================================================

def IC_SWE_user_defined(grid, state, par):
    """
    Blank template for a user-defined SWE problem.

    Fill in custom values below and remove the NotImplementedError; until
    then this deliberately stops the run (same pattern as the
    'user_defined' problem in every other Piastra mode).

    Parameters
    ----------
    grid : Grid
    state : SimState
    par : Parameters

    Returns
    -------
    grid, state, par, eos
        eos is always None for SWE.
    """
    print("SWE -- user-defined problem")

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 1.0

    state.g_ff = 1.0
    state.h[:, :] = 1.0
    state.vel1[:, :] = 0.0; state.vel2[:, :] = 0.0
    
    #center of the domain
    x0 = 0.5 * (x1ini + x1fin); y0 = 0.5 * (x2ini + x2fin)

    # Beta-plane Coriolis parameter can be added (usual beta-plane)
    f0 = 1.0e-4 # mid-latitude f₀ (rad/s)
    beta  = 1.6e-11 # df/dy (rad/m/s)
    state.f_c[:, :] = f0 + beta * (grid.cx2 - y0)

    # flat bathimetry -- can be adjusted here 
    state.b_x[:, :] = state.b_y[:, :] = 0.0
    
    par.BC[:] = 'free'

    raise NotImplementedError(
        "User-defined SWE problem -- set your ICs in SWE_init_cond.py "
        "and remove this line."
    )

    return grid, state, par, None



# ============================================================================
# Dam break
# ============================================================================

def IC_SWE1D_dam(grid, state, par):
    """
    1D dam break problem.

    The shallow water analogue of the Sod shock tube. A discontinuity
    in fluid height at x₁ = 0.5 generates a rarefaction wave propagating
    left and a shock propagating right. The exact solution is known.

    Left  state: h = 1.0, v₁ = 0, v₂ = 0
    Right state: h = 0.125, v₁ = 0, v₂ = 0
    g = 1, t_fin = 0.3, free boundaries
    """
    print("SWE -- dam break")

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    state.g_ff = 1.0
    par.timenow = 0.0; par.timefin = 0.3

    left = grid.cx1 < 0.5
    state.h[:, :] = np.where(left, 1.0, 0.125)
    state.vel1[:, :] = 0.0; state.vel2[:, :] = 0.0
    
    par.BC[:] = 'free'

    return grid, state, par, None


# ============================================================================
# Circular dam break (radially symmetric 2D Riemann problem)
# ============================================================================

def IC_SWE2D_dam(grid, state, par):
    """
    Circular dam break — radially symmetric 2D Riemann-like problem.

    A cylindrical column of water (h = h_in for r < r₀) collapses outward
    into a quiescent ambient (h = h_out), leading to: 
    
      • An outward-propagating circular shock wave (right-moving wave)
      • An inward-propagating circular rarefaction (left-moving wave)
        which converges at the centre and reflects, leaving a low
        depression there
        
    Parameters:
        h_in  = 2.5,   h_out = 0.5,   r₀ = 0.5,   g = 1
    Domain: [0, 2] × [0, 2], free outflow on all sides.
    Final time t = 0.25 captures the shock at r ≈ 0.8, well inside the
    domain so boundary effects are negligible.
    """
    print("SWE -- circular dam break (2D radial Riemann problem)")

    x1ini, x1fin = 0.0, 2.0; x2ini, x2fin = 0.0, 2.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.25
    
    state.g_ff = 1.0
    x_c = 0.5 * (x1ini + x1fin); y_c = 0.5 * (x2ini + x2fin)

    r0 = 0.5; h_in = 2.5; h_out = 0.5

    r = np.sqrt((grid.cx1 - x_c)**2 + (grid.cx2 - y_c)**2)

    state.h   [:, :] = np.where(r < r0, h_in, h_out)
    state.vel1[:, :] = 0.0; state.vel2[:, :] = 0.0
    
    par.BC[:] = 'free'

    return grid, state, par, None


# ============================================================================
# Bathtub (closed basin with gravity waves)
# ============================================================================

def IC_SWE2D_bathtub(grid, state, par):
    """
    Gravity wave propagation in a closed square basin.

    A smooth sinc-shaped height perturbation at the center generates
    outward-propagating gravity waves that reflect off the walls, producing
    a complex interference pattern.

    Domain: [0, 1] × [0, 1], wall boundaries on all sides.
    g = 9.81, t_fin = 1.5
    """
    print("SWE -- bathtub (gravity waves in closed basin)")

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    # ~6-7 wave traversals (c≈4.4, domain=1)
    par.timenow = 0.0; par.timefin = 1.5  

    # the bump
    x_c = 0.5 * (x1ini + x1fin); y_c = 0.5 * (x2ini + x2fin)
    r = np.sqrt((grid.cx1 - x_c)**2 + (grid.cx2 - y_c)**2)    
    sigma = 0.1; ampl = 0.1
    func = ampl * np.exp(-r**2/sigma**2)
  
    state.g_ff = 9.81
    state.h   [:, :] = 1.0 + func
    state.vel1[:, :] = 0.0; state.vel2[:, :] = 0.0
              
    par.BC[:]   = 'wall'

    return grid, state, par, None


# ============================================================================
# Rotating collapse of a water column
# ============================================================================
def IC_SWE2D_rotdam(grid, state, par):
    """
    Collapse of a raised water column on an f-plane: the circular dam break
    of dam2D, but with rotation.
 
    Without rotation the column spreads out completely (dam2D). With
    rotation potential vorticity q = (zeta + f)/h is conserved on fluid
    columns: as the column spreads, h decreases and zeta must become
    negative, zeta = f (h / h_in - 1). The spreading is arrested at a
    distance ~ L_R, leaving a balanced ANTICYCLONE (clockwise for f > 0)
    surrounded by an outgoing ring of inertia-gravity waves.

    Domain [-3, 3]^2, free outflow.
    """
    print("SWE -- rotating collapse of a water column")
 
    x1ini, x1fin = -3.0, 3.0; x2ini, x2fin = -3.0, 3.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
 
    par.timenow = 0.0; par.timefin = 10.0 # ~ 5 inertial periods 2 pi / f0
 
    g_phys = 1.0; H0 = 0.5; f0 = 3.0 # c = 0.707, L_R = c / f0 = 0.236
    r0   = 0.5  # column radius, r0 / L_R = 2.1
    eta0 = 0.025 # try 1.0 for the nonlinear regime
 
    state.g_ff = g_phys
    state.f_c[:, :] = f0
    state.b[:, :] = 0.0; state.b_x[:, :] = 0.0; state.b_y[:, :] = 0.0
 
    r = np.sqrt(grid.cx1**2 + grid.cx2**2)
    state.h[:, :] = np.where(r < r0, H0 + eta0, H0)
    state.vel1[:, :] = 0.0; state.vel2[:, :] = 0.0
 
    par.BC[:] = 'free'
 
    return grid, state, par, None



# ============================================================================
# Coastal Kelvin wave (replaces atmo2D)
# ============================================================================
def IC_SWE2D_kelvin(grid, state, par):
    """
    Coastal Kelvin wave along the southern wall (x2 = 0) of a periodic
    channel on an f-plane (Kelvin 1879).
 
    With v2 = 0 everywhere the linear equations split into a plain gravity
    wave along the coast and a geostrophic balance across it,
        f0 v1 = -g d(eta)/dx2,
    which gives the exact, NON-DISPERSIVE solution
        eta = eta0 * exp(-x2 / L_R) * F(x1 - c t),
        v1  = sqrt(g / H0) * eta,     v2 = 0,     L_R = c / f0.
    It travels with the coast on its RIGHT (eastward here, f0 > 0). Since
    v2 = 0, the wall conditions are satisfied exactly on BOTH walls, for
    any channel width.
 
    After one period t = Lx / c the wave is back at its initial position.
    Diagnostics:
      - amplitude / phase error of eta vs. the initial state
        (numerical dissipation / dispersion; convergence order);
      - max|v2| / max|v1| vs. time: any error in the Coriolis treatment
        breaks the cross-shore balance and generates v2 and Poincare waves.
    Try f0 -> -f0: the IC is no longer a solution and splits into a
    westward Kelvin wave plus Poincare waves.
    """
    print("SWE -- coastal Kelvin wave")
 
    Lx, Ly = 16.0, 6.0   # Ly = 6 L_R for a clear picture
    x1ini, x1fin = 0.0, Lx; x2ini, x2fin = 0.0, Ly
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
 
    g_phys = 1.0; H0 = 1.0; f0 = 1.0
    c   = np.sqrt(g_phys * H0)
    L_R = c / f0
 
    par.timenow = 0.0; par.timefin = Lx / c   # one full period
 
    state.g_ff = g_phys
    state.f_c[:, :] = f0
    state.b[:, :] = 0.0; state.b_x[:, :] = 0.0; state.b_y[:, :] = 0.0
 
    eta0 = 1.0e-2 * H0 # linear regime
    x0 = 0.5 * Lx # initial pulse position
    s  = 1.0 # pulse half-width along the coast
 
    eta = eta0 * np.exp(-grid.cx2 / L_R) * np.exp(-((grid.cx1 - x0) / s)**2)
 
    state.h[:, :]    = H0 + eta
    state.vel1[:, :] = np.sqrt(g_phys / H0) * eta
    state.vel2[:, :] = 0.0
 
    # periodic along the coast (x1), walls across (x2)
    par.BC[0] = 'peri'; par.BC[1] = 'wall'
    par.BC[2] = 'peri'; par.BC[3] = 'wall'
 
    return grid, state, par, None
 

# ============================================================================
# Barotropic instability of a Bickley jet
# ============================================================================
def IC_SWE2D_bickley(grid, state, par):
    """
    Barotropic (Rayleigh-Kuo) instability of a zonal jet.

    A Bickley jet has the sech²(y) velocity profile:

        v₁(x₂) = U₀ sech²((x₂ - y_c)/L)
        v₂(x₂) = 0

    The vorticity ω = -∂v₁/∂x₂ has an inflection point inside the jet,
    so by Rayleigh's inflection-point theorem the flow is linearly
    unstable. The Bickley jet is the canonical case studied analytically
    by Lipps (1962) and Drazin & Howard, with the most-unstable mode
    at kL ≈ 0.9 and growth rate σ ≈ 0.165 U₀/L.

    Height is set in geostrophic balance with the jet on an f-plane:
        f v₁ = -g ∂h/∂x₂
        → h = h₀ - (f U₀ L / g) tanh((x₂ - y_c)/L)

    A small sinusoidal perturbation (broad spectrum is unnecessary —
    the unstable wavenumber dominates within a few e-foldings) seeds
    the instability; the jet rolls up into a chain of vortices.

    Domain: [0, 8πL] × [0, 8L];, periodic in x₁, wall in x₂.
    Reference: Poulin & Flierl (2003).
    """
    print("SWE -- barotropic instability of a Bickley jet")

    # Domain in units of jet half-width L = 1.  Use a long zonal channel
    # so several wavelengths of the most-unstable mode fit.
    L_jet = 1.0
    Lx    = 8.0 * np.pi * L_jet     # ≈ 25.13, fits ~3.5 wavelengths of kL≈0.9
    Ly    = 8.0 * L_jet

    x1ini, x1fin = 0.0, Lx
    x2ini, x2fin = 0.0, Ly
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 80.0 # several e-foldings at σ ≈ 0.165

    y_c = 0.5 * (x2ini + x2fin)

    # Physical parameters
    U0 = 1.0 # jet peak velocity
    f0 = 1.0 # f-plane Coriolis (Rossby number U/(fL) = 1)
    g_phys = 9.81
    h0 = 10.0 # background depth → c = sqrt(gh) ~ 9.9 ≫ U, low Froude, well within SWE regime
    state.g_ff = g_phys
    state.f_c[:, :] = f0

    # Bickley jet velocity profile
    sech2 = 1.0 / np.cosh((grid.cx2 - y_c) / L_jet)**2
    v1_jet = U0 * sech2

    # Geostrophic height: h = h₀ - (f U₀ L / g) tanh((x₂-y_c)/L)
    state.h[:, :] = h0 - (f0 * U0 * L_jet / g_phys) * \
                          np.tanh((grid.cx2 - y_c) / L_jet)

    # Small sinusoidal perturbation in v₂ to break translational symmetry.
    # k_x = 2π·n/Lx with n = 3 puts the perturbation near the most-unstable
    # wavenumber kL ≈ 0.9 (since k = 2π·3/(8π) = 0.75/L).
    eps = 3.0e-2 * U0
    pert = eps * np.sin(2.0 * np.pi * 3.0 * grid.cx1 / Lx) * sech2

    state.vel1[:, :] = v1_jet
    state.vel2[:, :] = pert
    
    par.BC[0] = 'peri'; par.BC[1] = 'wall'
    par.BC[2] = 'peri'; par.BC[3] = 'wall'

    return grid, state, par, None


# ============================================================================
# Lake at rest over a seamount (well-balancing diagnostic)
# ============================================================================
def IC_SWE2D_lake(grid, state, par):
    """
    "Lake at rest" over a Gaussian seamount: h + b = H0, v = 0.
 
    The EXACT solution is that nothing moves. A scheme is called
    well-balanced if it keeps this state to round-off. Here the bed source
    -g grad(b) is a cell-centred central difference, applied separately
    from the face fluxes of g h^2/2, so the two do not cancel discretely:
    expect spurious currents and surface waves at the level of the
    truncation error, decreasing with resolution.
 
    Diagnostics: max|v| and max|h + b - H0| vs. time and vs. resolution.
    """
    print("SWE -- lake at rest over a seamount (well-balancing test)")
 
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
 
    par.timenow = 0.0; par.timefin = 1.0      # ~ 3 gravity-wave crossings
 
    g_phys = 9.81; H0 = 1.0
    state.g_ff = g_phys
    state.f_c[:, :] = 0.0
 
    x_c = 0.5 * (x1ini + x1fin); y_c = 0.5 * (x2ini + x2fin)
    b_max = 0.5; sigma_b = 0.1                # seamount: half of the depth
 
    state.b[:, :] = b_max * np.exp(
        -((grid.cx1 - x_c)**2 + (grid.cx2 - y_c)**2) / sigma_b**2)
    state.b_x[:, :], state.b_y[:, :] = _gradient_full(grid, state.b)
 
    state.h[:, :] = H0 - state.b              # flat free surface
    state.vel1[:, :] = 0.0; state.vel2[:, :] = 0.0
 
    par.BC[:] = 'wall'
 
    return grid, state, par, None


# ============================================================================
# Shallow-water Kelvin-Helmholtz instability
# ============================================================================
def IC_SWE2D_KHI(grid, state, par):
    """
    Shallow-water analogue of the Kelvin-Helmholtz instability.

    A tangential velocity jump across a horizontal interface is unstable
    to perturbations along the interface. We use a smoothed velocity
    profile (tanh) to give a well-defined linear growth rate, and h is
    set uniform so the instability is purely shear-driven (no Coriolis,
    no buoyancy).

        v₁(x₂) = U₀ tanh((x₂ - y_c)/δ)
        v₂(x₂) = small seed perturbation
        h = h₀ (constant)

    With h constant, gravity does not enter the linear instability
    problem, but the gravity-wave speed c = √(gh) must satisfy U₀ ≪ c
    (low Froude) so the flow is incompressible-like and rolls up into
    the canonical cat's-eye pattern. With c ≫ U₀, the instability is
    almost identical to the incompressible 2D KHI.

    The contact wave (which carries the shear) is exactly where the
    exact Riemann solver outperforms HLL

    Domain: [0, 1] × [0, 1], with periodic boundaries along X and walls along Y.
    """
    print("SWE -- 'Kelvin-Helmholtz' instability (shear layer)")

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 2.0

    y_c = 0.5 * (x2ini + x2fin)

    # Physical parameters
    U0    = 0.5 # half-jump in velocity
    delta = 0.025 # shear-layer half-thickness (≪ Lx)
    h0    = 1.0
    g_phys = 9.81  # → c = √(gh) ≈ 3.13, Froude ≈ 0.16

    state.g_ff = g_phys
    # No rotation
    state.f_c[:, :] = 0.0

    # Smooth shear layer
    state.vel1[:, :] = U0 * np.tanh((grid.cx2 - y_c) / delta)

    # Seed: two-mode perturbation in v₂ localised on the interface,
    # with two wavelengths fitting in the box.  
    eps = 1.0e-2 * U0
    envelope = np.exp(-((grid.cx2 - y_c) / (8.0 * delta))**2)
    pert = eps * envelope * (
              np.sin(2.0 * np.pi * 2.0 * grid.cx1)
            + 0.5 * np.sin(2.0 * np.pi * 4.0 * grid.cx1 + 0.7))
    state.vel2[:, :] = pert

    # Uniform height
    state.h[:, :] = h0
    
    par.BC[0] = 'peri'
    par.BC[1] = 'wall'
    par.BC[2] = 'peri'
    par.BC[3] = 'wall'

    return grid, state, par, None


# ============================================================================
# Internal helper: full-grid gradient (including ghost cells in output)
# ============================================================================
def _gradient_full(grid, var):
    """
    Compute the gradient of a full-grid array (including ghost cells)
    using cell_gradient() from grid_misc.py.

    cell_gradient() returns interior-only arrays of shape (Nx1, Nx2).
    This wrapper embeds the result back into full-grid arrays so that the
    source terms in the time integrator can be applied with ghost-cell
    indexing consistently.

    Parameters
    ----------
    grid : Grid
    var  : ndarray, shape grid.grid_shape

    Returns
    -------
    gx, gy : ndarray, shape grid.grid_shape
        Gradient components on the full grid. Interior cells are filled
        by gradient(); ghost cells remain zero (not needed for source terms).
    """
    Ngc = grid.Ngc
    gx_full = np.zeros(grid.grid_shape)
    gy_full = np.zeros(grid.grid_shape)

    g1, g2 = cell_gradient(grid, var)
    gx_full[Ngc:-Ngc, Ngc:-Ngc] = g1
    gy_full[Ngc:-Ngc, Ngc:-Ngc] = g2

    return gx_full, gy_full