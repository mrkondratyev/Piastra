# -*- coding: utf-8 -*-
"""
gravity.py
===========
Additional gravitational source terms for Piastra.

This module provides the body-force sources that feed the momentum and
energy residuals of every HD/MHD-family solver, through ``state.F1`` and
``state.F2``:

  1. planet_gravity_polar(...)       -- star + orbiting planet in POLAR
                                        (r, phi), LAB frame
  1b. corotating_planet_disk(...)    -- star + planet in the frame CO-ROTATING
                                        with the planet: static potential plus
                                        centrifugal and Coriolis forces. The
                                        standard disk-planet (gap-opening) setup
  2. selfgravity_monopole_spherical  -- spherically-averaged (monopole) self
                                        gravity in SPHERICAL-POLAR (r, theta)
  3. selfgravity_poisson(...)        -- general self-gravity, any density
                                        field, any geometry, via the finite-
                                        volume Poisson solver (poisson_solver.py)
  4. *_hook(...) factories           -- wrap any of the above into the
                                        per-stage callable described next

------------------------------------------------------------------------------
STATIC vs. PER-STAGE FORCES
------------------------------------------------------------------------------
A force that never changes -- a fixed central point mass -- can simply be
written into state.F1/F2 once by the initial-condition function: the arrays
persist, and the RK integrators' deep copies carry them along.

A force that depends on the EVOLVING SOLUTION (self-gravity, where rho
changes every stage; Coriolis, where v does) or explicitly on TIME (an
orbiting perturber) must instead be recomputed every Runge-Kutta stage.
The solvers support that through the optional ``state.body_force`` hook,
called as body_force(grid, state, par) at the top of each stage's residual
evaluation, on that stage's own state. Use the ``*_hook`` factories at the
bottom of this module to build one:

    state.body_force = selfgravity_poisson_hook(G=1.0, BC=['peri'] * 4)

------------------------------------------------------------------------------
SIGN / SOURCE CONVENTION
------------------------------------------------------------------------------
Every solver in this project (HD, MHD -- all three divB treatments --, rHD,
rMHD) adds the body force to the momentum/energy residual the same way (see
e.g. HD_step.flux_calc_HD):

    Res1 += -dens * F1          # momentum residual, before the dt * (-Res)
    Res2 += -dens * F2          #   update that advances the conserved state
    ResE += -dens * (F1*vel1 + F2*vel2)

and that residual enters the update as  U_{n+1} = U_n - dt * Res  (see the
"U_t + RES = 0" note in HD_step.oneStep_HD_RK), so the net momentum source
is  +dt * dens * F.  For that to be the physical force per unit mass,
``state.F1, state.F2`` MUST hold the ACCELERATION ITSELF (i.e. F = a =
-grad(Phi) for a potential Phi), not its negative -- F1 < 0 means "pulls
toward smaller x1", exactly like a physical inward gravitational pull.
All three functions below follow this convention.

------------------------------------------------------------------------------
GRID-ATTRIBUTE FLAGS
------------------------------------------------------------------------------
  * state.F1, state.F2  : interior body-force/acceleration arrays (Nx1, Nx2).
  * state.dens          : density (full grid, with ghosts).
  * grid.cx1, grid.cx2  : cell-centre coordinates, full 2D arrays with ghosts.
                          POLAR    : cx1 = r,  cx2 = phi.
                          SPHERICAL: cx1 = r,  cx2 = theta.  cx1 varies along
                          axis 0 (radius) and is constant along axis 1 (angle).
  * grid.cVol           : cell volume, full 2D array with ghosts (Nx1+2Ngc,
                          Nx2+2Ngc).  Used only for the volume-weighted angular
                          average of density in the monopole solver.
  * par.timenow         : current time (drives the planet's orbital phase).
"""

import numpy as np

from src.grid.grid_misc import cell_gradient
from src.common.poisson_solver import solve_poisson


def _mass_ramp(t, t_ramp):
    """
    Smooth switch-on factor for a perturber mass, sin^2(pi t / (2 t_ramp)).

    Rises from 0 at t=0 to 1 at t=t_ramp with zero derivative at both ends,
    so the disk is not shocked by a planet that appears instantaneously --
    the standard taper of the de Val-Borro et al. (2006) comparison problem.

    Parameters
    ----------
    t : float        current physical time.
    t_ramp : float   ramp duration; <= 0 disables the ramp (returns 1.0).

    Returns
    -------
    float in [0, 1].
    """
    if t_ramp <= 0.0:
        return 1.0
    if t >= t_ramp:
        return 1.0
    return np.sin(0.5 * np.pi * t / t_ramp) ** 2


# ===========================================================================
# 1) Orbiting planet in polar geometry  (central star + planet, lab frame)
# ===========================================================================
def planet_gravity_polar(grid, state, par,
                         M_star=1.0, M_planet=1.0e-3, r_planet=1.0,
                         phi0=0.0, soft=0.03, G=1.0, indirect=True,
                         t_ramp=0.0):
    """Star + orbiting-planet gravitational acceleration in polar (r, phi).

    LAB-FRAME treatment: the planet is on a fixed circular orbit at radius
    ``r_planet`` with Keplerian angular frequency
    Omega_p = sqrt(G (M_star + M_planet) / r_p^3), and its azimuth advances as
    phi_p(t) = phi0 + Omega_p * par.timenow.  The potential is therefore
    genuinely time-dependent, so this MUST be re-evaluated every step to mean
    anything -- install it as a body-force hook rather than calling it once
    from an initial condition (see ``planet_gravity_polar_hook``).  Called
    once and left alone, it describes a perturber frozen at its t=0 azimuth,
    which is not a physical configuration.

    For a disk-planet run the co-rotating frame (``corotating_planet_disk``)
    is usually the better choice: there the planet potential is static and
    only the frame forces move, which removes the large azimuthal advection
    velocity and its associated timestep and diffusion penalty.

    Parameters
    ----------
    grid : Grid
        Polar (r, phi) grid.
    state : SimState
        Must expose `F1`, `F2` (interior acceleration arrays).
    par : Parameters
        Read for `par.timenow`, which sets the planet's orbital phase.
    M_star, M_planet : float
        Masses of the central star and the planet, in code units.
    r_planet : float
        Planet orbital radius.
    phi0 : float
        Planet azimuth at t = 0.
    soft : float
        Gravitational softening length: smooths the planet potential inside
        ~soft, so the point mass does not produce an unresolved singularity.
        Pick a fraction of the local disk scale height (0.6 H is the usual
        choice) or of the Hill radius -- never smaller than a cell.
    G : float
        Gravitational constant in code units.
    indirect : bool
        Add the indirect term accounting for the acceleration of the
        star-centred (non-inertial) frame by the planet. Standard in
        planet-disk codes.
    t_ramp : float
        If > 0, ramp the planet mass up as sin^2(pi t / (2 t_ramp)) over
        the first `t_ramp` of physical time, so the disk is not shocked by
        a perturber switched on instantaneously. 0 disables the ramp.

    Returns
    -------
    state : SimState
        With F1, F2 set to the total (star + planet) acceleration on
        interior cells.
    """
    # grid indexes for slicing
    Ngc  = grid.Ngc; Nx1  = grid.Nx1; Nx2  = grid.Nx2
    Nx1r = Ngc + Nx1; Nx2r = Ngc + Nx2
    sl = np.s_[Ngc:Nx1r, Ngc:Nx2r]

    #cell centers
    r = grid.cx1[sl]; phi = grid.cx2[sl]

    # --- planet orbital phase and (optionally ramped) mass ---
    Omega_p = np.sqrt(G * (M_star + M_planet) / r_planet**3)
    phi_p   = phi0 + Omega_p * par.timenow
    Mp      = M_planet * _mass_ramp(par.timenow, t_ramp)

    # --- planet potential gradient (softened point mass) ---
    # Phi_p = -G mp / sqrt(r^2 + r_p^2 - 2 r r_p cos(phi-phi_p) + soft^2)
    dphi = phi - phi_p
    cosd = np.cos(dphi); sind = np.sin(dphi)
    Dsq  = r * r + r_planet**2 - 2.0 * r * r_planet * cosd + soft * soft
    Dm32 = Dsq**(-1.5)

    a_r1 = -G * Mp * (r - r_planet * cosd) * Dm32          # a_r   from planet
    a_p1 = -G * Mp * (r_planet * sind)     * Dm32          # a_phi from planet

    # --- indirect term (star-centred frame acceleration) ---
    if indirect:
        a_r1 += -G * Mp * cosd / r_planet**2
        a_p1 +=  G * Mp * sind / r_planet**2

    # --- central star monopole is accounted here ---
    a_r1 += -G * M_star / (r * r)

    # --- store as F = a  (see SIGN / SOURCE CONVENTION at top) ---
    state.F1[:, :] = a_r1; state.F2[:, :] = a_p1

    return state


# ===========================================================================
# 1b) Star + planet in the CO-ROTATING frame (polar), with frame forces
# ===========================================================================
def corotating_planet_disk(grid, state, par,
                            M_star=1.0, M_planet=1.0e-3, r_planet=1.0,
                            phi_planet=np.pi, soft=0.03, G=1.0,
                            indirect=True, t_ramp=0.0):
    """
    Star + planet gravity plus the frame forces, in the frame CO-ROTATING
    with the planet's circular orbit -- the standard setup for disk-planet
    (gap-opening) simulations in polar (r, phi) geometry.

    In this frame the planet sits still at (r_planet, phi_planet), so its
    potential is static and only the frame forces depend on the solution.
    The rotating frame adds a centrifugal and a Coriolis acceleration:

        a_r   = Omega_p^2 r + 2 Omega_p v_phi     (centrifugal + Coriolis)
        a_phi =              -2 Omega_p v_r       (Coriolis)

    with Omega_p = sqrt(G (M_star + M_planet) / r_planet^3) and (v_r, v_phi)
    the ROTATING-frame velocities the solver actually stores (state.vel1,
    state.vel2 in polar geometry).  Because the Coriolis term depends on the
    current velocity, this routine MUST be re-evaluated every Runge-Kutta
    stage -- install it as ``state.body_force`` via
    ``corotating_planet_disk_hook``, never call it once from an IC.

    Why the co-rotating frame is preferred here: in the lab frame the whole
    disk streams past the grid at the local Keplerian speed, which both
    shrinks the timestep and smears the (slowly-growing) gap by numerical
    advection diffusion.  Co-rotating removes the bulk azimuthal motion near
    the planet's orbit, so the gap is resolved for the same cost.

    Set up the initial condition in the SAME frame: the rotating-frame
    azimuthal velocity is  v_phi_rot = v_phi_inertial - Omega_p * r.

    Parameters
    ----------
    grid : Grid
        Polar (r, phi) grid.
    state : SimState
        Read for `dens` is not needed, but `vel1`, `vel2` ARE (Coriolis);
        `F1`, `F2` are overwritten.
    par : Parameters
        Read for `par.timenow` (only to drive the mass ramp -- the potential
        itself is static in this frame).
    M_star, M_planet : float
        Masses of the central star and the planet, in code units.
    r_planet : float
        Planet orbital radius (also sets Omega_p, hence the frame rotation).
    phi_planet : float
        Fixed planet azimuth in the co-rotating frame. Default pi puts it at
        the middle of a [0, 2pi] domain, furthest from the periodic seam.
    soft : float
        Gravitational softening length for the planet potential (see
        ``planet_gravity_polar``).
    G : float
        Gravitational constant in code units.
    indirect : bool
        Add the indirect term (acceleration of the star-centred frame by the
        planet). Standard in planet-disk codes.
    t_ramp : float
        Planet-mass ramp duration (see ``_mass_ramp``); 0 disables it.

    Returns
    -------
    state : SimState
        With F1, F2 set to gravity + centrifugal + Coriolis on interior cells.

    References
    ----------
    de Val-Borro, M. et al. (2006), MNRAS 370, 529
    """
    Ngc  = grid.Ngc; Nx1  = grid.Nx1; Nx2  = grid.Nx2
    sl = np.s_[Ngc:Ngc + Nx1, Ngc:Ngc + Nx2]

    r = grid.cx1[sl]; phi = grid.cx2[sl]
    v_r   = state.vel1[sl]
    v_phi = state.vel2[sl]

    Omega_p = np.sqrt(G * (M_star + M_planet) / r_planet**3)
    Mp      = M_planet * _mass_ramp(par.timenow, t_ramp)

    # --- central star monopole ---
    a_r   = -G * M_star / (r * r)
    a_phi = np.zeros_like(a_r)

    # --- softened planet potential, STATIC in this frame ---
    dphi = phi - phi_planet
    cosd = np.cos(dphi); sind = np.sin(dphi)
    Dsq  = r * r + r_planet**2 - 2.0 * r * r_planet * cosd + soft * soft
    Dm32 = Dsq**(-1.5)
    a_r   += -G * Mp * (r - r_planet * cosd) * Dm32
    a_phi += -G * Mp * (r_planet * sind)     * Dm32

    # --- indirect term (star-centred frame is accelerated by the planet) ---
    if indirect:
        a_r   += -G * Mp * cosd / r_planet**2
        a_phi +=  G * Mp * sind / r_planet**2

    # --- frame forces: centrifugal (static) + Coriolis (velocity-dependent) ---
    a_r   += Omega_p**2 * r + 2.0 * Omega_p * v_phi
    a_phi += -2.0 * Omega_p * v_r

    # --- store as F = a  (see SIGN / SOURCE CONVENTION at top) ---
    state.F1[:, :] = a_r; state.F2[:, :] = a_phi

    return state


# ===========================================================================
# 2) Spherically-averaged (monopole) self-gravity in spherical-polar geometry
# ===========================================================================
def selfgravity_monopole_spherical(grid, state, par):
    """1D monopole self-gravity for spherical-polar (r, theta) grids.

    The angular structure of the density is collapsed into a spherically
    averaged profile rho_bar(r); the enclosed mass M(<r) is integrated
    radially and the radial acceleration is

        g_r(r) = -G * M(<r) / r^2 ,        a_theta = 0 .

    This is the standard l = 0 (monopole) approximation used in core-collapse
    codes: cheap, robust, and exact for a spherically symmetric mass
    distribution.  It IGNORES the gravity of non-radial density variations
    (the l >= 1 multipoles), so it is appropriate for nearly-spherical
    configurations (e.g. a proto-neutron star) and approximate otherwise.
    <-- MONOPOLE-ONLY CAVEAT

    Parameters
    ----------
    grid : Grid
    state : SimState
        Must expose `dens` (density, with ghosts) and `F1`, `F2` (interior
        acceleration arrays) -- any HD/MHD-family mode.
    par : Parameters
        Unused directly; kept so this function has the same call signature
        as the other gravity routines in this module.

    Returns
    -------
    state : SimState
        With F1 set to the radial self-gravity acceleration (F2 = 0) on
        interior cells.

    Method
    ------
    rho_bar(r) is the volume-weighted angular mean of the density (uses
    grid.cVol), so it is independent of the grid's angular coverage (a theta
    wedge gives the same rho_bar as full pole-to-pole).  The full-sphere shell
    mass is then rho_bar * (4/3) pi (r_out^3 - r_in^3), summed to M(<r) at the
    radial faces, evaluated as -G M / r^2 there, and averaged to cell centres.
    """

    #central point mass
    M_central = 0.0
    #grav acceleration
    G = 1.0

    #slicing of density and volume -- grid.cVol is interior-only (no ghost
    #cells), so it must NOT be re-sliced with the ghost-offset index range
    #(that range is only valid for ghost-inclusive arrays like state.dens)
    Ngc = grid.Ngc; Nx1 = grid.Nx1; Nx2 = grid.Nx2
    sl  = np.s_[Ngc:Ngc + Nx1, Ngc:Ngc + Nx2]
    rho = state.dens[sl]; vol = grid.cVol

    # --- volume-weighted angular mean density per radial shell ---
    wsum = vol.sum(axis=1) # (Nx1,)  shell volume (grid)
    rho_bar = (rho * vol).sum(axis=1) / np.maximum(wsum, 1e-30)  # (Nx1,)

    # faces, mass centers and resolution
    rf = grid.fx1[Ngc:Nx1+1+Ngc,Ngc]
    rc = grid.ax1[Ngc:Nx1+Ngc,Ngc]
    dr = grid.dx1[Ngc:Nx1+Ngc,Ngc]

    # --- enclosed mass at radial faces ---
    shell_mass = (4.0 / 3.0) * np.pi * rho_bar * (rf[1:]**3 - rf[:-1]**3)  # (Nx1,)
    Menc = np.zeros(Nx1 + 1)
    Menc[1:] = np.cumsum(shell_mass)
    Menc += M_central

    # --- acceleration at faces, then average to cell centres ---
    g_f = -G * Menc / (rf**2 + 1e-30)
    g_r = g_f[:-1]*(rf[1:] - rc)/dr + g_f[1:]*(rc - rf[:-1])/dr

    # --- store as F = a  (see SIGN / SOURCE CONVENTION at top) ---
    state.F1[:, :] = g_r[:, None]
    state.F2[:, :] = 0.0

    return state


# ===========================================================================
# 3) General self-gravity via the finite-volume Poisson solver
# ===========================================================================
def selfgravity_poisson(grid, state, par, G=1.0, BC=None, BC_value=None,
                         tol=1e-10, maxiter=None):
    """
    General self-gravity acceleration, for any density field and any of the
    four grid geometries, by solving the Poisson equation

        div( grad(Phi) ) = 4 * pi * G * rho

    with the finite-volume CG solver (poisson_solver.solve_poisson) and
    setting state.F1, state.F2 to the resulting acceleration -grad(Phi) on
    interior cells -- no ghost cells, matching state.F1/F2's existing
    (Nx1, Nx2) shape used by every solver's momentum source term (see the
    SIGN / SOURCE CONVENTION note at the top of this module).

    Unlike planet_gravity_polar / selfgravity_monopole_spherical (geometry-
    specific approximations: a fixed point-mass orbit, a spherically
    averaged monopole), this is the general case -- an arbitrary,
    not-necessarily-symmetric density distribution -- at the cost of an
    actual elliptic solve every call.

    Parameters
    ----------
    grid : Grid
    state : SimState
        Must expose `dens` (density, with ghosts) and `F1`, `F2` (interior
        acceleration arrays) -- any HD/MHD-family mode.
    par : Parameters
        Unused directly; kept so this function has the same call signature
        as the other gravity routines in this module.
    G : float, optional
        Gravitational constant in code units. Default 1.0 (matching the
        other two routines in this module).
    BC : sequence of 4 str, optional
        Poisson boundary conditions for the potential, forwarded to
        solve_poisson: 'peri', 'free', or 'dirichlet' per face (see
        poisson_solver.py / boundaries.apply_bc_scalar_Ngc1). Defaults to
        ['free', 'free', 'free', 'free'] -- safe even though it leaves
        Phi's absolute normalisation undetermined (a pure-Neumann Poisson
        problem is only fixed up to an additive constant), because only
        Phi's GRADIENT is used here and a constant offset has zero
        gradient. Pass 'peri' for a periodic box, or 'dirichlet' with
        BC_value set from an analytic/multipole exterior potential, for a
        genuinely isolated (non-periodic, non-zero-gradient) boundary.
    BC_value : dict, optional
        Dirichlet boundary values, forwarded to solve_poisson.
    tol, maxiter : optional
        Forwarded to solve_poisson's CG iteration.

    Returns
    -------
    state : SimState
        With F1, F2 set to the self-gravity acceleration on interior cells.
    """
    Ngc = grid.Ngc
    rho = state.dens[Ngc:-Ngc, Ngc:-Ngc]
    rhs = 4.0 * np.pi * G * rho

    if BC is None:
        BC = ['free', 'free', 'free', 'free']

    phi, info = solve_poisson(grid, rhs, BC, BC_value=BC_value,
                               tol=tol, maxiter=maxiter)
    if not info['converged']:
        print(f"[gravity] selfgravity_poisson: CG did not converge "
              f"(niter={info['niter']}, residual={info['residual']:.3e})")

    # grad(Phi) on interior cells, geometry-aware (Cartesian/cylindrical/
    # polar/spherical-polar all handled by cell_gradient's metric factors)
    g1, g2 = cell_gradient(grid, phi)

    # --- store as F = a = -grad(Phi)  (see SIGN / SOURCE CONVENTION at top) ---
    state.F1[:, :] = -g1
    state.F2[:, :] = -g2

    return state


# ===========================================================================
# 4) Body-force HOOK factories  (for state.body_force)
# ===========================================================================
#
# Every routine above writes state.F1/F2 once, for the state it is handed.
# A force that never changes (a fixed point mass) can therefore just be
# called once from the initial condition and left alone.  A force that
# depends on the EVOLVING state (self-gravity: rho changes; Coriolis: v
# changes) or on TIME (an orbiting perturber) must instead be re-evaluated
# every Runge-Kutta stage.
#
# The solvers do that through ``state.body_force``: an optional callable
# invoked as body_force(grid, state, par) at the top of each stage's
# residual evaluation, on THAT stage's state.  The factories below wrap the
# routines above into exactly such a callable, so an IC only has to write
#
#     state.body_force = selfgravity_poisson_hook(G=1.0, BC=['peri']*4)
#
# Note: the hook is a plain closure, which copy.deepcopy treats as atomic --
# so the per-stage deep copy the RK integrators make of the state shares the
# hook rather than duplicating whatever it captured.  It is NOT stored by
# io_utils.save_data (only arrays and scalars are), so a run resumed with
# restart_simulation must re-install its hook; see save_data's warning.
# ===========================================================================

def selfgravity_poisson_hook(G=1.0, BC=None, BC_value=None,
                              tol=1e-10, maxiter=None):
    """
    Build a ``state.body_force`` hook that re-solves the Poisson equation
    for the CURRENT density every Runge-Kutta stage.

    This is the only correct way to run self-gravity: the potential is a
    functional of rho, which the hydro update changes every stage, so a
    potential computed once in the initial condition is stale immediately.

    Parameters
    ----------
    G, BC, BC_value, tol, maxiter
        Forwarded unchanged to ``selfgravity_poisson`` on every call.

    Returns
    -------
    callable
        hook(grid, state, par) -> None, suitable for ``state.body_force``.

    Examples
    --------
    >>> state.body_force = selfgravity_poisson_hook(G=1.0, BC=['peri'] * 4)
    """
    def _hook(grid, state, par):
        selfgravity_poisson(grid, state, par, G=G, BC=BC, BC_value=BC_value,
                             tol=tol, maxiter=maxiter)
    return _hook


def selfgravity_monopole_hook():
    """
    Build a ``state.body_force`` hook applying the spherically-averaged
    (monopole) self-gravity of ``selfgravity_monopole_spherical`` every
    Runge-Kutta stage.

    Much cheaper than the Poisson hook (a radial cumulative sum instead of
    an elliptic solve) and exact for a spherically symmetric configuration,
    but blind to every l >= 1 multipole -- see that function's caveat.

    Returns
    -------
    callable
        hook(grid, state, par) -> None, suitable for ``state.body_force``.
    """
    def _hook(grid, state, par):
        selfgravity_monopole_spherical(grid, state, par)
    return _hook


def planet_gravity_polar_hook(**kwargs):
    """
    Build a ``state.body_force`` hook for a LAB-FRAME orbiting planet.

    Parameters
    ----------
    **kwargs
        Forwarded unchanged to ``planet_gravity_polar`` (M_star, M_planet,
        r_planet, phi0, soft, G, indirect, t_ramp).

    Returns
    -------
    callable
        hook(grid, state, par) -> None, suitable for ``state.body_force``.

    Notes
    -----
    The planet's azimuth is read from ``par.timenow``, which the integrators
    only advance BETWEEN steps, not between stages.  The potential is
    therefore frozen at the step-start time within a step, making this source
    term first-order accurate in time even under RK2/RK3.  That is the usual
    accuracy of an explicitly time-dependent external potential in a
    Godunov code, and is another reason to prefer
    ``corotating_planet_disk_hook``, whose potential is static.
    """
    def _hook(grid, state, par):
        planet_gravity_polar(grid, state, par, **kwargs)
    return _hook


def corotating_planet_disk_hook(**kwargs):
    """
    Build a ``state.body_force`` hook for a disk-planet run in the frame
    co-rotating with the planet (gravity + centrifugal + Coriolis).

    Parameters
    ----------
    **kwargs
        Forwarded unchanged to ``corotating_planet_disk`` (M_star, M_planet,
        r_planet, phi_planet, soft, G, indirect, t_ramp).

    Returns
    -------
    callable
        hook(grid, state, par) -> None, suitable for ``state.body_force``.

    Notes
    -----
    Unlike the lab-frame hook, only the mass ramp here depends on time; the
    potential itself is static and the Coriolis term is evaluated on the
    current stage's velocity, so this source term keeps the integrator's
    full temporal order once the ramp is over.
    """
    def _hook(grid, state, par):
        corotating_planet_disk(grid, state, par, **kwargs)
    return _hook


# ===========================================================================
# 5) Timestep limit from the body force
# ===========================================================================
def body_force_dt(grid, state, CFL):
    """
    Timestep limit imposed by the body force: no cell may be accelerated
    across its own width in a single step.

        (1/2) |a| dt^2  <=  CFL * dx        =>    dt <= sqrt(2 CFL dx / |a|)

    evaluated per cell, minimised over the grid.

    WHY THIS IS NOT OPTIONAL FOR SELF-GRAVITY
    -----------------------------------------
    An ordinary CFL condition limits dt by the SIGNAL speed, |v| + c_s.  In a
    cold, self-gravitating flow released from rest both terms are ~ 0, so the
    hydrodynamic CFL condition imposes essentially NO limit -- a pressureless
    collapse starting at v = 0 will happily take one enormous step and
    integrate the whole run in a single forward Euler update, producing a
    smooth, plausible-looking, completely wrong answer.  (This is exactly what
    the dust-collapse problem did before this limit existed: 1 step for the
    entire run, giving a uniform sphere with v ~ r -- homologous, as it should
    be, but at 0.58 t_ff worth of collapse when 0.80 was requested.)

    Gravity has to supply its own constraint, because it accelerates the gas
    without any wave carrying information about it.

    Parameters
    ----------
    grid : Grid
    state : SimState
        Read for the interior body-force arrays F1, F2 (accelerations).
    CFL : float
        Courant number; kept in the expression so the user's CFL knob scales
        this limit along with the hydrodynamic one.

    Returns
    -------
    float
        Maximum stable timestep from the body force, or ``np.inf`` when the
        force is identically zero (no constraint).

    References
    ----------
    Truelove, J. K. et al. (1997), ApJ 489, L179
    """
    F1 = getattr(state, 'F1', None)
    if F1 is None:
        return np.inf

    Ngc = grid.Ngc
    acc = np.zeros_like(F1)

    if grid.Nx1 > 1:
        dx1 = grid.dx1[Ngc:-Ngc, Ngc:-Ngc]
        acc = np.maximum(acc, np.abs(F1) / dx1)
    if grid.Nx2 > 1:
        # physical width along x2 includes the metric factor (r dphi, r dtheta)
        dx2 = (grid.dx2[Ngc:-Ngc, Ngc:-Ngc] * grid.hx2[Ngc:-Ngc, Ngc:-Ngc])
        acc = np.maximum(acc, np.abs(state.F2) / dx2)

    amax = float(np.max(acc))
    if not np.isfinite(amax) or amax <= 0.0:
        return np.inf
    return float(np.sqrt(2.0 * CFL / amax))
