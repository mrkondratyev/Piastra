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
    into a quiescent ambient (h = h_out). This is the SWE analogue of
    the Sedov-Taylor blast wave and is the canonical 2D test for
    radially-symmetric shock dynamics on a Cartesian grid:

      • An outward-propagating circular shock wave (right-moving wave)
      • An inward-propagating circular rarefaction (left-moving wave)
        which converges at the centre and reflects, leaving a low
        depression there ("Mexican hat" profile)
      • A contact-like circular ridge separating shock from rarefaction

    A clean reference solution can be obtained by running Piastra in
    cylindrical 1D mode with the same initial discontinuity at r₀; the
    1D radial profile must match the azimuthally-averaged 2D solution.
    Departures from circular symmetry in the 2D run reveal the grid
    imprint of the Riemann solver (Cartesian asymmetries are most
    visible along the diagonals and axes).

    Standard parameters (Toro 2009, §17.7.1; Liska & Wendroff 2003):
        h_in  = 2.5,   h_out = 0.5,   r₀ = 0.5,   g = 1
    Domain: [0, 2] × [0, 2], free outflow on all sides.
    Final time t = 0.25 captures the shock at r ≈ 1.0, well inside the
    domain so boundary effects are negligible.
    """
    print("SWE -- circular dam break (2D radial Riemann problem)")

    x1ini, x1fin = 0.0, 2.0; x2ini, x2fin = 0.0, 2.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.25

    x_c = 0.5 * (x1ini + x1fin); y_c = 0.5 * (x2ini + x2fin)

    r0 = 0.5
    h_in = 2.5; h_out = 0.5

    r = np.sqrt((grid.cx1 - x_c)**2 + (grid.cx2 - y_c)**2)

    state.g_ff = 1.0
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

    # center of the bump
    x_c = 0.5 * (x1ini + x1fin); y_c = 0.5 * (x2ini + x2fin)

    r = np.sqrt((grid.cx1 - x_c)**2 + (grid.cx2 - y_c)**2)
    func = np.sinc(r / np.pi)          # np.sinc includes the pi factor
    func[r > np.pi] = 0.0
    
    state.g_ff = 9.81
    state.h   [:, :] = 1.0 + func
    state.vel1[:, :] = 0.0; state.vel2[:, :] = 0.0
              
    par.BC[:]   = 'wall'

    return grid, state, par, None



# ============================================================================
# Cylindrical explosion (SWE blast wave)
# ============================================================================

def IC_SWE2D_expl(grid, state, par):
    """
    Cylindrical SWE explosion.

    The shallow water pressure is p = g h² / 2, equivalent to a
    barotropic EOS with gamma = 2. This test is the SWE analogue of
    the Sedov blast wave and tests the solver in 2D with strong
    height gradients.

    Domain: [0, 1] × [0, 1], wall boundaries.
    g = 1, t_fin = 2.5
    """
    print("SWE -- cylindrical explosion (SWE blast wave)")

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    # shock reaches wall at ~0.3 time units in
    par.timenow = 0.0; par.timefin = 0.30   

    #center of the explosion
    x_c = 0.5 * (x1ini + x1fin); y_c = 0.5 * (x2ini + x2fin)
    r   = np.sqrt((grid.cx1 - x_c)**2 + (grid.cx2 - y_c)**2)

    state.g_ff = 1.0
    state.h[:, :] = np.where(r < 0.1, 10.0, 0.1)
    state.vel1[:, :] = 0.0; state.vel2[:, :] = 0.0
           
    par.BC[:]   = 'wall'
    
    return grid, state, par, None



# ============================================================================
# Tsunami propagation
# ============================================================================

def IC_SWE2D_tsunami(grid, state, par):
    """
    Tsunami propagation over a deep ocean.

    A localised 2D Gaussian sea-surface displacement (initial elevation
    of order 1 m on top of 4000 m of water) propagates outward as a
    long gravity wave. With H = 4000 m the long-wave speed is
    c = √(gH) ≈ 198 m/s — fast enough to cross a 1000-km domain in
    ~85 minutes, which sets the natural timescale.

    The bathymetry is flat (b = 0); modify state.b here for a more
    interesting variable-depth test (continental shelf, seamount, etc.).

    Domain: [0, 1e6 m] × [0, 1e6 m] (SI units), free outflow.
    g = 9.81, t_fin = 1 hour.
    """
    print("SWE -- tsunami propagation over deep ocean")

    x1ini, x1fin = 0.0, 1.0e6; x2ini, x2fin = 0.0, 1.0e6
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    # 1 hour — wave crosses ~70% of domain
    par.timenow = 0.0; par.timefin = 3600.0          

    x_c = 0.5 * (x1ini + x1fin)
    y_c = 0.5 * (x2ini + x2fin)
    
    #free-fall acceleration
    state.g_ff      = 9.81
    
    # Coriolis irrelevant on tsunami timescales
    # (period 2π/f ≈ 17 hr ≫ t_fin = 1 hr)
    state.f_c[:, :] = 0.0         
    
    # Flat bathymetry (b = 0); customise here for variable bed.
    # state.b[:, :] = ...
    state.b_x[:, :], state.b_y[:, :] = _gradient_full(grid, state.b)

    # 2D Gaussian sea-surface displacement: 1 m amplitude on 4000 m of water
    H0    = 4000.0           # ocean depth
    eta0  = 1.0              # peak surface displacement
    sigma = 5.0e4            # 50 km half-width — typical earthquake source

    state.h[:, :] = H0 + eta0 * np.exp(
        -((grid.cx1 - x_c)**2 + (grid.cx2 - y_c)**2) / sigma**2)
    state.vel1[:, :] = 0.0; state.vel2[:, :] = 0.0
    
    # let the wave leave the domain
    par.BC[:] = 'free'
    
    return grid, state, par, None


# ============================================================================
# Geostrophic ocean flow
# ============================================================================

def IC_SWE2D_ocean(grid, state, par):
    """
    Geostrophic ocean eddy on a beta-plane.

    A localised circular height anomaly is initialised in geostrophic
    balance with the velocity field:
        v₁ = -g/f ∂h/∂x₂
        v₂ = +g/f ∂h/∂x₁

    A positive height anomaly (∂h/∂x₂ < 0 on the north side) generates
    eastward flow on the south side and westward flow on the north
    side, i.e. *anticyclonic* circulation in the Northern hemisphere
    (rotating clockwise as seen from above). The eddy size is set close
    to the Rossby radius L_R = √(gH)/f₀ ≈ 1000 km so the dynamics are
    on the cusp between geostrophic adjustment and Rossby-wave
    propagation.

    The Coriolis parameter varies linearly with x₂ (beta-plane):
        f_c(x₂) = f₀ + β (x₂ - y_c)

    On the β-plane an isolated geostrophic eddy westward-drifts at the
    long Rossby-wave speed c_R = β L_R² ≈ 16 m/s ≈ 1400 km/day, so the
    2-day run shows clear westward propagation across the basin.

    Domain: [0, 1e6 m] × [0, 1e6 m] (SI units).
    g = 9.81, t_fin = 2 days.
    """
    print("SWE -- geostrophic ocean eddy (beta-plane)")

    x1ini, x1fin = 0.0, 1.0e6
    x2ini, x2fin = 0.0, 1.0e6
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 2.0 * 86400.0      # 2 days
    par.BC[0]   = 'peri'
    par.BC[1]   = 'wall'
    par.BC[2]   = 'peri'
    par.BC[3]   = 'wall'

    x_c = 0.5 * (x1ini + x1fin)
    y_c = 0.5 * (x2ini + x2fin)

    state.g_ff = 9.81
    g_phys     = state.g_ff

    # Beta-plane Coriolis parameter
    f0    = 1.0e-4                   # mid-latitude f₀ (rad/s)
    beta  = 1.6e-11                  # df/dy (rad/m/s)
    state.f_c[:, :] = f0 + beta * (grid.cx2 - y_c)

    # 2D Gaussian SSH anomaly: 1 m bump on 1000 m background, ~150 km wide
    H0    = 1000.0
    A     = 1.0                      # 1 m sea-surface height anomaly
    sigma = 1.5e5                    # 150 km half-width
    state.h[:, :] = H0 + A * np.exp(
        -((grid.cx1 - x_c)**2 + (grid.cx2 - y_c)**2) / sigma**2)

    # Geostrophic balance: v = (g/f) ẑ × ∇h
    h_x, h_y = _gradient_full(grid, state.h)
    f_safe   = np.where(np.abs(state.f_c) > 1e-20, state.f_c, 1e-20)

    state.vel1[:, :] = -g_phys * h_y / f_safe
    state.vel2[:, :] =  g_phys * h_x / f_safe

    return grid, state, par, None


# ============================================================================
# Geostrophic atmospheric flow
# ============================================================================

def IC_SWE2D_atmo(grid, state, par):
    """
    Shallow-water atmospheric ridge with seeded instability.

    A broad Gaussian pressure (height) ridge is initialised in
    geostrophic balance:
        v₁ = -g/f ∂h/∂x₂,   v₂ = +g/f ∂h/∂x₁

    A small *velocity* perturbation is added to seed any instability
    (the height field itself remains a clean balanced state). This
    matters: adding noise to h would induce O(g·δh/(f·dx)) ~ 30 m/s
    grid-scale velocities through the geostrophic relation, swamping
    the balanced eddy with random jets at the start. Adding noise
    directly to v keeps the IC physical and the noise amplitude
    transparent.

    The β-plane Coriolis varies linearly with x₂:
        f_c(x₂) = f₀ + β (x₂ - y_c)

    Domain: [0, 1e6 m] × [0, 1e6 m] (SI units).
    g = 9.81, t_fin = 2 days.
    """
    print("SWE -- geostrophic atmospheric ridge (beta-plane)")

    x1ini, x1fin = 0.0, 1.0e6
    x2ini, x2fin = 0.0, 1.0e6
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0
    par.timefin = 2.0 * 86400.0
    par.BC[0]   = 'peri'
    par.BC[1]   = 'wall'
    par.BC[2]   = 'peri'
    par.BC[3]   = 'wall'

    x_c = 0.5 * (x1ini + x1fin)
    y_c = 0.5 * (x2ini + x2fin)

    state.g_ff = 9.81
    g_phys     = state.g_ff

    f0   = 1.5e-4
    beta = 1.6e-11
    state.f_c[:, :] = f0 + beta * (grid.cx2 - y_c)

    # 2D Gaussian height anomaly (no noise on h — see docstring).
    scale_x = (x1fin - x1ini) / 8.0
    scale_y = (x2fin - x2ini) / 8.0
    A = 40.0
    H0 = 2000.0

    state.h[:, :] = H0 + A * np.exp(
        -((grid.cx1 - x_c)**2 / scale_x**2
          + (grid.cx2 - y_c)**2 / scale_y**2))

    # Geostrophic balance from the smooth height field
    h_x, h_y = _gradient_full(grid, state.h)
    f_safe   = np.where(np.abs(state.f_c) > 1e-20, state.f_c, 1e-20)

    state.vel1[:, :] = -g_phys * h_y / f_safe
    state.vel2[:, :] =  g_phys * h_x / f_safe

    # Small velocity noise to break symmetry (1% of geostrophic peak).
    rng    = np.random.default_rng(42)
    v_peak = float(np.max(np.abs(state.vel2)))
    eps    = 0.01 * v_peak
    state.vel1 += eps * (rng.random(grid.grid_shape) - 0.5)
    state.vel2 += eps * (rng.random(grid.grid_shape) - 0.5)

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

    Domain: [0, 4 L_jet] × [0, 4 L_jet], periodic in x₁, wall in x₂.
    Reference: Poulin & Flierl, J. Fluid Mech. 481, 329 (2003).
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

    par.timenow = 0.0
    par.timefin = 80.0          # several e-foldings at σ ≈ 0.165
    par.BC[0]   = 'peri'        # x₁_min  (zonal)
    par.BC[1]   = 'peri'        # x₁_max
    par.BC[2]   = 'wall'        # x₂_min  (meridional)
    par.BC[3]   = 'wall'

    y_c = 0.5 * (x2ini + x2fin)

    # Physical parameters
    U0    = 1.0                 # jet peak velocity
    f0    = 1.0                 # f-plane Coriolis (Rossby number U/(fL) = 1)
    g_phys = 9.81
    h0    = 10.0                # background depth → c = sqrt(gh) ~ 9.9 ≫ U0
                                #  ⇒ low Froude, well within shallow-water regime
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
    eps = 1.0e-3 * U0
    pert = eps * np.sin(2.0 * np.pi * 3.0 * grid.cx1 / Lx) * sech2

    state.vel1[:, :] = v1_jet
    state.vel2[:, :] = pert

    return grid, state, par, None


# ============================================================================
# Shallow-water Kelvin-Helmholtz instability
# ============================================================================
def IC_SWE2D_KH(grid, state, par):
    """
    Shallow-water analogue of the Kelvin-Helmholtz instability.

    A tangential velocity jump across a horizontal interface is unstable
    to perturbations along the interface. We use a smoothed velocity
    profile (tanh) to give a well-defined linear growth rate, and h is
    set uniform so the instability is purely shear-driven (no Coriolis,
    no buoyancy).

        v₁(x₂) = U₀ tanh((x₂ - y_c)/δ)
        v₂(x₂) = small seed perturbation
        h     = h₀  (constant)

    With h constant, gravity does not enter the linear instability
    problem, but the gravity-wave speed c = √(gh) must satisfy U₀ ≪ c
    (low Froude) so the flow is incompressible-like and rolls up into
    the canonical cat's-eye pattern. With c ≫ U₀, the instability is
    almost identical to the incompressible 2D KHI.

    The contact wave (which carries the shear) is exactly where the
    exact Riemann solver outperforms HLL — running this with HLL
    gives a notably more diffuse roll-up than the exact solver.

    Domain: [0, 1] × [0, 1], periodic in both directions.
    Reference: layered version in Hesthaven & Warburton, Sec. 13.3.
    """
    print("SWE -- 'Kelvin-Helmholtz' instability (shear layer)")

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 2.0

    y_c = 0.5 * (x2ini + x2fin)

    # Physical parameters
    U0    = 0.5                 # half-jump in zonal velocity
    delta = 0.025               # shear-layer half-thickness (≪ Lx)
    h0    = 1.0
    g_phys = 9.81               # → c = √(gh) ≈ 3.13, Froude ≈ 0.16

    state.g_ff = g_phys
    # No rotation
    state.f_c[:, :] = 0.0

    # Smooth shear layer
    state.vel1[:, :] = U0 * np.tanh((grid.cx2 - y_c) / delta)

    # Seed: two-mode perturbation in v₂ localised on the interface,
    # with two wavelengths fitting in the box.  Two modes prevent the
    # roll-up from being a single huge eddy.
    eps = 1.0e-2 * U0
    envelope = np.exp(-((grid.cx2 - y_c) / (8.0 * delta))**2)
    pert = eps * envelope * (
              np.sin(2.0 * np.pi * 2.0 * grid.cx1)
            + 0.5 * np.sin(2.0 * np.pi * 4.0 * grid.cx1 + 0.7))
    state.vel2[:, :] = pert

    # Uniform height
    state.h[:, :] = h0
    
    par.BC[:]   = 'peri'

    return grid, state, par, None


# ============================================================================
# Flow over a bump (transcritical with hydraulic jump)
# ============================================================================
def IC_SWE1D_bump(grid, state, par):
    """
    Steady (asymptotic) supercritical flow over a smooth bump.

    A subcritical inflow on the left enters a channel containing a
    smooth Gaussian bump in the bed, accelerates over the bump to
    supercritical, and forms a stationary hydraulic jump downstream
    where the flow returns to subcritical. This is the SWE analogue
    of the de Laval nozzle / shock-in-divergent-section problem.

    Bathymetry:
        b(x₁, x₂) = b_max exp(-((x₁-x_b)/σ_b)²)            (uniform in x₂)

    Initial condition (will adjust to the steady transcritical state
    as boundary conditions force the solution):
        h + b = H₀     (lake at rest perturbed by bump)
        v₁    = q / h  (uniform discharge)
        v₂    = 0

    Choose Froude number well above 1 just downstream of the bump
    crest so a jump forms.  Specifically:
        H₀ = 2.0,   b_max = 0.2,   q = 4.42  (gives Fr ≈ 1 at crest)

    Note on well-balancing: this test will reveal whether the SWE
    solver preserves "lake at rest" (h + b = const, v = 0) over a
    non-trivial bathymetry. A naive non-well-balanced scheme produces
    spurious waves even when the analytical solution is stationary
    --- a useful warning to the user.

    Domain: [0, 25] × [0, 5], inflow on left (free), outflow on right.
    Reference: Vázquez-Cendón, J. Comput. Phys. 148, 497 (1999).
    """
    print("SWE -- transcritical flow over a bump")

    x1ini, x1fin = 0.0, 25.0; x2ini, x2fin = 0.0, 5.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 12.0     
    state.g_ff      = 9.81
    state.f_c[:, :] = 0.0

    # ─── Bathymetry: smooth Gaussian bump centred at x_b ──────────────
    x_b   = 10.0
    sig_b =  2.0
    b_max =  0.2

    state.b[:, :] = b_max * np.exp(-((grid.cx1 - x_b) / sig_b)**2)

    # ─── Bathymetry gradients (geometry-aware central differences) ────
    state.b_x[:, :], state.b_y[:, :] = _gradient_full(grid, state.b)

    # ─── Initial state: lake at rest perturbed by bump + uniform flow ─
    # Total free-surface elevation H₀ above z = 0
    H0 = 2.0
    q  = 4.42      # discharge per unit width (m²/s); chosen for Fr_crest ≈ 1

    state.h[:, :]    = np.maximum(H0 - state.b, 0.1)   # avoid dry cells
    state.vel1[:, :] = q / state.h
    state.vel2[:, :] = 0.0
    
    par.BC[:]   = 'free'

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