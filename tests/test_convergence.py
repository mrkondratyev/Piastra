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
    exact solution at t_fin is identically the initial condition.
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
