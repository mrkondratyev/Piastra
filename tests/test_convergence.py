# -*- coding: utf-8 -*-
"""
===============================================================================
test_convergence.py
===============================================================================

Convergence-order checks: refine the grid and confirm the error against a
known reference shrinks at close to the reconstruction's design order. Three
standard, dependency-free ways to get a reference without an external exact
solution are used here:

  * exact analytic solution -- diffusion of a sine mode has a closed form,
    T(x,t) = sin(2*pi*x) * exp(-(2*pi)^2 * kappa * t) (see
    diff_init_cond.IC_diff1D_sine), with no Riemann solver/limiter involved
    at all, the cleanest possible convergence target.
  * exact return-to-initial-state -- linear advection of a smooth profile on
    a periodic domain, run for exactly one period (vel * t_fin == domain
    length, set by the IC itself): whatever the scheme does in between, the
    exact solution at t_fin is identically the initial condition. The
    circularly polarised Alfven wave (Toth 2000) is the same trick applied
    to a genuinely NONLINEAR exact solution of the ideal MHD equations: at
    unit background field and density the Alfven speed is 1, so the wave
    also returns to its initial condition after each unit of time.
  * preserved steady state -- the Gresho vortex is an exact, time-independent
    solution of the Euler equations; comparing the numerical state at t > 0
    back to its own initial condition is the standard vortex-preservation
    convergence test (Gresho & Chan 1990; e.g. used this way in Miczek,
    Roepke & Edelmann 2015).

Order thresholds below are deliberately well under the nominal design order
(2nd order for PLM/RKL2, ~2nd+ for PPM) -- calibrated against the orders this
testbed actually measures (typically PLM ~1.4-1.7, PPM/diffusion ~2.0-2.5,
Gresho ~1.2-1.4; see the module docstrings of the schemes for why a limited
2nd-order scheme commonly falls short of 2.0 on a smooth-but-not-flat
profile: TVD limiters must still clip at smooth local extrema). The point is
to catch a REGRESSION (order collapsing toward 0, or growing error), not to
pin down the exact constant.

Author: mrkondratyev
"""

import numpy as np

from tests.testbed_common import build_case, run_to_tfin, l2_error, observed_order


def _mean_order(errors):
    orders = observed_order(errors)
    finite = [o for o in orders if np.isfinite(o)]
    return float(np.mean(finite)) if finite else float("inf")


def _assert_converging(errors, min_order, label):
    for e_coarse, e_fine in zip(errors[:-1], errors[1:]):
        assert e_fine <= e_coarse * 1.05, (
            f"{label}: error did not shrink under refinement "
            f"({errors})")
    order = _mean_order(errors)
    assert order > min_order, (
        f"{label}: mean observed order {order:.2f} below expected "
        f"minimum {min_order} (errors={errors})")


# ============================================================================
#   Linear advection: exact return to the initial profile after one period
# ============================================================================

def test_advection_1d_smooth_convergence_plm():
    """1D periodic advection, one full period, PLM -- expect close to 2nd order."""
    errors = []
    for Nx in (32, 64, 128, 256):
        grid, state, par, eos, solver = build_case(
            "adv", "smooth1D", Nx, 1, rec_type="PLM", RK_order="RK3", CFL=0.4)
        Ngc = grid.Ngc
        dens0 = state.dens[Ngc:-Ngc, Ngc:-Ngc].copy()
        state, _ = run_to_tfin(solver, par)
        errors.append(l2_error(grid, state.dens[Ngc:-Ngc, Ngc:-Ngc], dens0))
    _assert_converging(errors, min_order=1.2, label="adv1D_smooth/PLM")


def test_advection_1d_smooth_convergence_ppm():
    """Same setup with PPM -- expect closer to (or above) 2nd order than PLM."""
    errors = []
    for Nx in (32, 64, 128, 256):
        grid, state, par, eos, solver = build_case(
            "adv", "smooth1D", Nx, 1, rec_type="PPM", RK_order="RK3", CFL=0.4)
        Ngc = grid.Ngc
        dens0 = state.dens[Ngc:-Ngc, Ngc:-Ngc].copy()
        state, _ = run_to_tfin(solver, par)
        errors.append(l2_error(grid, state.dens[Ngc:-Ngc, Ngc:-Ngc], dens0))
    _assert_converging(errors, min_order=1.7, label="adv1D_smooth/PPM")


def test_advection_2d_smooth_convergence():
    """2D periodic advection of a radial Gaussian, one full period, PLM."""
    errors = []
    for Nx in (16, 32, 64):
        grid, state, par, eos, solver = build_case(
            "adv", "smooth2D", Nx, Nx, rec_type="PLM", RK_order="RK3", CFL=0.4)
        Ngc = grid.Ngc
        dens0 = state.dens[Ngc:-Ngc, Ngc:-Ngc].copy()
        state, _ = run_to_tfin(solver, par)
        errors.append(l2_error(grid, state.dens[Ngc:-Ngc, Ngc:-Ngc], dens0))
    _assert_converging(errors, min_order=0.7, label="adv2D_smooth/PLM")


# ============================================================================
#   Diffusion: exact analytic decaying-sine-mode solution
# ============================================================================

def test_diffusion_1d_sine_convergence():
    """1D diffusion of sin(2*pi*x) against its closed-form decay solution."""
    errors = []
    for Nx in (32, 64, 128, 256):
        grid, state, par, eos, solver = build_case(
            "diff", "sine1D", Nx, 1, solver_type="rkl2", rkl2_stages=8)
        Ngc = grid.Ngc
        kappa = state.kappa
        state, _ = run_to_tfin(solver, par)
        x = grid.cx1[Ngc:-Ngc, Ngc]
        exact = (np.sin(2.0 * np.pi * x)
                 * np.exp(-(2.0 * np.pi) ** 2 * kappa * par.timenow))[:, None]
        errors.append(l2_error(grid, state.T[Ngc:-Ngc, Ngc:-Ngc], exact))
    _assert_converging(errors, min_order=1.7, label="diff/sine1D")


# ============================================================================
#   HD: Gresho vortex preservation (exact steady state of the Euler eqs)
# ============================================================================

def test_hd_gresho_vortex_preservation_convergence():
    """
    The Gresho vortex is torque-free and pressure-balanced, so the exact
    solution at any t > 0 is identical to the initial condition; the
    departure from it is pure truncation error.
    """
    errors = []
    for Nx in (16, 32, 64):
        grid, state, par, eos, solver = build_case(
            "HD", "gresho2D", Nx, Nx, rec_type="PLM", RK_order="RK2", CFL=0.6)
        Ngc = grid.Ngc
        vel1_0 = state.vel1[Ngc:-Ngc, Ngc:-Ngc].copy()
        vel2_0 = state.vel2[Ngc:-Ngc, Ngc:-Ngc].copy()
        par.timefin = 0.3   # well short of the usual full run -- only need
                             # the truncation-error trend, not the long-time behaviour
        state, _ = run_to_tfin(solver, par)
        e1 = l2_error(grid, state.vel1[Ngc:-Ngc, Ngc:-Ngc], vel1_0)
        e2 = l2_error(grid, state.vel2[Ngc:-Ngc, Ngc:-Ngc], vel2_0)
        errors.append(float(np.hypot(e1, e2)))
    _assert_converging(errors, min_order=0.8, label="HD/gresho2D")


# ============================================================================
#   MHD: circularly polarised Alfven wave (exact nonlinear travelling wave)
# ============================================================================
#
# NOTE on the order thresholds below: unlike every other test in this file,
# these were NOT empirically calibrated against an actual run (per request,
# this pair was implemented and reviewed -- IC field names/shapes, BC,
# periodicity-implies-exact-return reasoning, solver/divb_tr validity --
# without being executed). The thresholds are therefore set conservatively
# low relative to what PLM measures elsewhere in this file (1.2-1.7) rather
# than tuned to an observed value; tighten them once a real run's numbers
# are in hand.

def test_mhd_alfven_1d_convergence():
    """
    1D circularly polarised Alfven wave (Toth 2000, JCP 161, 605): an
    exact NONLINEAR travelling-wave solution of the ideal MHD equations
    (B1 = B_par = 1 uniform, B2/B3 a circularly polarised perpendicular
    component tied to v2/v3 by the Alfven relation). With rho0 = 1,
    B_par = 1 the Alfven speed is 1, and the IC's own t_fin = 1.0 is
    exactly one period over the domain length 1 (see
    MHD_init_cond.IC_MHD1D_Alfven) -- so, like the linear-advection tests
    above, the exact solution at t_fin is identically the initial
    condition; the transverse field B2, B3 is what actually carries the
    wave, so that is what is compared. Uses CT (the standard choice for
    this benchmark in the literature, Toth 2000; Gardiner & Stone 2005)
    so the comparison isn't contaminated by GLM/8wave's approximate
    divergence handling.
    """
    errors = []
    for Nx in (32, 64, 128, 256):
        grid, state, par, eos, solver = build_case(
            "MHD", "alfven1D", Nx, 1, rec_type="PLM", RK_order="RK2",
            solver_type="HLLD", divb_tr="CT", CFL=0.5)
        Ngc = grid.Ngc
        bfi2_0 = state.bfi2[Ngc:-Ngc, Ngc:-Ngc].copy()
        bfi3_0 = state.bfi3[Ngc:-Ngc, Ngc:-Ngc].copy()
        state, _ = run_to_tfin(solver, par)
        e2 = l2_error(grid, state.bfi2[Ngc:-Ngc, Ngc:-Ngc], bfi2_0)
        e3 = l2_error(grid, state.bfi3[Ngc:-Ngc, Ngc:-Ngc], bfi3_0)
        errors.append(float(np.hypot(e2, e3)))
    _assert_converging(errors, min_order=1.0, label="MHD/alfven1D")


def test_mhd_alfven_2d_convergence():
    """
    2D circularly polarised Alfven wave propagating along the grid
    diagonal (Toth 2000; Gardiner & Stone 2005, JCP 205, 509): the same
    exact nonlinear travelling wave as the 1D case, rotated by theta =
    pi/4 with the domain/IC engineered (see MHD_init_cond.IC_MHD2D_Alfven)
    so the propagation coordinate is exactly periodic with period 1 --
    again an exact return to the initial condition after every unit of
    time. The out-of-plane field B3 = Bz directly carries the wave (set
    as a clean function of the propagation coordinate in the IC) and is
    used for the comparison; the in-plane field is built from a
    divergence-free corner vector potential, also exactly reproduced at
    t=0. par.timefin is overridden to one period (1.0) rather than the
    IC's default of five (5.0, set for a long-time divB-control
    demonstration) purely to keep the resolution sweep affordable; one
    period is exactly as valid a comparison as five.
    """
    errors = []
    for Nx in (16, 32, 64):
        grid, state, par, eos, solver = build_case(
            "MHD", "alfven2D", Nx, Nx, rec_type="PLM", RK_order="RK2",
            solver_type="HLLD", divb_tr="CT", CFL=0.4)
        par.timefin = 1.0   # one period instead of the IC's default 5
        Ngc = grid.Ngc
        bfi3_0 = state.bfi3[Ngc:-Ngc, Ngc:-Ngc].copy()
        state, _ = run_to_tfin(solver, par)
        e3 = l2_error(grid, state.bfi3[Ngc:-Ngc, Ngc:-Ngc], bfi3_0)
        errors.append(e3)
    _assert_converging(errors, min_order=1.0, label="MHD/alfven2D")


# ============================================================================
#   Self-gravity: growth rate and free-fall against exact solutions
# ============================================================================

def test_jeans_growth_rate():
    """
    2D Jeans instability against the EXACT linear dispersion relation

        sigma = sqrt( 4 pi G rho0 - c_s^2 k^2 ) ,

    which is a far sharper statement than a convergence order: the seeded
    mode's amplitude must grow at a specific, analytically known rate.

    This exercises the whole self-gravity chain end to end -- the periodic
    Poisson solve (whose automatic mean subtraction IS the Jeans swindle),
    the cell-centred gradient, the body force entering the momentum and
    energy residuals, and the per-stage ``body_force`` hook that re-solves
    the potential as rho evolves. A sign error, a stale potential (hook not
    firing), or a mis-scaled 4 pi G would all show up here as a wrong rate,
    not merely a wrong order.

    The amplitude is measured by projecting drho/rho0 onto cos(kx), so a
    contaminating decaying mode or a phase drift cannot inflate it, and
    the run stops at t = 1 (about 2.5 e-foldings) while still linear.
    """
    G, rho0, cs = 1.0, 1.0, 0.4
    k = 2.0 * np.pi
    sigma_exact = np.sqrt(4.0 * np.pi * G * rho0 - cs**2 * k**2)

    errors = []
    for Nx in (32, 64):
        grid, state, par, eos, solver = build_case(
            "HD", "jeans2D", Nx, Nx // 2, rec_type="PLM", RK_order="RK2",
            solver_type="HLLC", CFL=0.4)
        par.timefin = 1.0
        Ngc = grid.Ngc
        x = grid.cx1[Ngc:-Ngc, Ngc:-Ngc]

        def modal_amp(s):
            drho = s.dens[Ngc:-Ngc, Ngc:-Ngc] / rho0 - 1.0
            return 2.0 * float(np.mean(drho * np.cos(k * x)))

        a0, t0 = modal_amp(state), par.timenow
        state, _ = run_to_tfin(solver, par)
        a1 = modal_amp(state)

        assert a1 > a0 > 0.0, (
            f"jeans2D at Nx={Nx}: the mode did not grow "
            f"(a0={a0:.3e} -> a1={a1:.3e}); self-gravity may be absent or "
            f"have the wrong sign")
        sigma = np.log(a1 / a0) / (par.timenow - t0)
        errors.append(abs(sigma - sigma_exact))

    # tightest resolution must land close to the analytic rate
    rel = errors[-1] / sigma_exact
    assert rel < 0.01, (
        f"jeans2D: measured growth rate is off by {rel*100:.2f}% "
        f"(exact sigma = {sigma_exact:.6f}); expected < 1%")
    _assert_converging(errors, min_order=1.0, label="HD/jeans2D growth rate")


def test_dust_collapse_freefall():
    """
    1D spherical pressureless collapse against the EXACT free-fall (cycloid)
    solution.

    A uniform sphere released from rest collapses homologously, staying
    uniform, so its half-mass radius is exactly R(t) * 2^(-1/3) with R(t)
    given by the parametric free-fall solution.  Comparing against that is a
    nonlinear check on self-gravity, complementing the linear-regime Jeans
    test above.

    It is also the test that pins the GRAVITATIONAL TIMESTEP LIMIT
    (gravity.body_force_dt) in place: a cold flow released from rest has
    |v| + c_s ~ 0, so without that limit the hydrodynamic CFL condition
    permits an essentially unbounded step and the entire run collapses into
    one forward-Euler update -- which still produces a smooth, uniform,
    v ~ r profile (homologous, as it should be) at completely the wrong
    time.  Convergence of the half-mass radius is what catches that.

    Only first order is demanded: the sphere's surface is a contact
    discontinuity, and a Godunov scheme smears it at first order, which
    limits how fast an integral measure of its position can converge.
    """
    G, rho0, R0 = 1.0, 1.0, 1.0
    t_ff = np.sqrt(3.0 * np.pi / (32.0 * G * rho0))

    def exact_radius(frac):
        """Invert t/t_ff = (2/pi)(xi + sin xi cos xi) for R/R0 = cos^2 xi."""
        lo, hi = 0.0, 0.5 * np.pi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if (2.0 / np.pi) * (mid + np.sin(mid) * np.cos(mid)) < frac:
                lo = mid
            else:
                hi = mid
        return np.cos(0.5 * (lo + hi))**2

    M_sphere = (4.0 / 3.0) * np.pi * R0**3 * rho0
    errors = []
    for Nx in (64, 128):
        grid, state, par, eos, solver = build_case(
            "HD", "collapse1D", Nx, 1, rec_type="PLM", RK_order="RK2",
            solver_type="HLLC", CFL=0.4)
        Ngc = grid.Ngc
        r = grid.cx1[Ngc:-Ngc, Ngc]
        vol = grid.cVol[:, 0]
        state, nsteps = run_to_tfin(solver, par)

        assert nsteps > 5, (
            f"collapse1D at Nx={Nx}: only {nsteps} step(s) for the whole run. "
            f"The gravitational timestep limit (gravity.body_force_dt) is not "
            f"constraining dt -- a cold collapse has no hydrodynamic CFL limit.")

        d = state.dens[Ngc:-Ngc, Ngc]
        r_half = float(np.interp(0.5 * M_sphere, np.cumsum(vol * d), r))
        r_half_exact = exact_radius(par.timenow / t_ff) * 2.0**(-1.0 / 3.0)
        errors.append(abs(r_half - r_half_exact) / r_half_exact)

    assert errors[-1] < 0.03, (
        f"collapse1D: half-mass radius is off the exact free-fall solution "
        f"by {errors[-1]*100:.2f}% at the finest resolution; expected < 3%")
    _assert_converging(errors, min_order=0.7, label="HD/collapse1D free-fall")
