# -*- coding: utf-8 -*-
"""
===============================================================================
testbed_common.py
===============================================================================

Shared building blocks for the Piastra testbed (tests/).

The testbed checks the framework along the three axes standard for a
computational (astrophysical) fluid-dynamics code:

  * sanity (test_sanity.py) -- every (mode, problem) in the catalogue
    builds and steps without crashing, NaNs, or unphysical (negative
    density/pressure/height) values.
  * conservation (test_conservation.py) -- mass / momentum / energy stay
    constant to round-off on closed (periodic or zero-flux) domains, the
    discrete signature of a flux-difference finite-volume update.
  * convergence (test_convergence.py) -- measured error against a known
    exact solution shrinks at (close to) the reconstruction's design order
    as resolution is refined.
  * robustness (test_robustness.py) -- every solver_type / rec_type /
    RK_order combination stays finite and positivity-preserving on strong
    shock/blast problems (Sod, Woodward-Colella, Brio-Wu, dam-break, ...).

This module holds only the plumbing shared by all four: building a
(grid, state, par, eos, solver) tuple for one case, stepping it, and a
handful of small numerical diagnostics (finiteness/positivity, conserved
volume integrals, an L2 error, observed convergence order).

Solver construction reuses ``main.py``'s ``SOLVER_DISPATCH`` (the same
mode -> solver-class mapping `main()` uses) rather than duplicating it.

Author: mrkondratyev
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # the testbed never opens a plot window

from src.parameters import Parameters
from src.grid.grid_setup import Grid
from src.sim_state import SimState
from src.misc.helpers import initial_model
from main import SOLVER_DISPATCH


# ============================================================================
#   Case construction / time-stepping
# ============================================================================

def build_case(mode, problem, Nx1, Nx2, **par_kwargs):
    """
    Build one runnable case: grid, state, par, eos and its solver.

    Parameters
    ----------
    mode : str
    problem : str
    Nx1, Nx2 : int
        Grid resolution. Deliberately allowed to differ (Nx1 != Nx2) by every
        caller in this testbed: a square grid hides axis-swap bugs (this is
        exactly how the cVol mis-slicing in the Sedov ICs went unnoticed).
    par_kwargs : dict
        Forwarded to ``Parameters`` (rec_type, RK_order, solver_type, CFL,
        divb_tr, rkl2_stages, ...).

    Returns
    -------
    grid, state, par, eos, solver
    """
    par = Parameters(mode=mode, problem=problem, Nx1=Nx1, Nx2=Nx2, **par_kwargs)
    grid = Grid(par.Nx1, par.Nx2, par.Ngc)
    state = SimState(grid, par)
    grid, state, par, eos = initial_model(grid, state, par)
    solver = SOLVER_DISPATCH[mode](grid, state, eos, par)
    return grid, state, par, eos, solver


def run_steps(solver, par, nsteps, on_step=None):
    """
    Advance `solver` by at most `nsteps` steps (stops early at par.timefin).

    Parameters
    ----------
    solver : object exposing step_RK()
    par : Parameters
    nsteps : int
    on_step : callable(state) or None
        If given, called after every step (e.g. a positivity check) so a
        transient failure mid-run cannot hide behind a healthy final state.

    Returns
    -------
    state : object
        Whatever the last step_RK() call returned (None if nsteps == 0 or
        par.timenow already reached par.timefin).
    nsteps_taken : int
    """
    state = None
    taken = 0
    for _ in range(nsteps):
        if par.timenow >= par.timefin:
            break
        state = solver.step_RK()
        taken += 1
        if on_step is not None:
            on_step(state)
    return state, taken


def run_to_tfin(solver, par, on_step=None, max_steps=200_000):
    """
    Advance `solver` until par.timenow reaches par.timefin.

    Parameters
    ----------
    solver : object exposing step_RK()
    par : Parameters
    on_step : callable(state) or None
    max_steps : int
        Safety cap. Needing more than this for a catalogue test's (short,
        by design) timefin is itself a robustness red flag (e.g. a CFL
        condition that collapses to ~0), so it raises rather than hanging.

    Returns
    -------
    state : object
    nsteps_taken : int
    """
    state = None
    taken = 0
    while par.timenow < par.timefin:
        if taken >= max_steps:
            raise AssertionError(
                f"run_to_tfin: exceeded max_steps={max_steps} without "
                f"reaching timefin={par.timefin} (stuck at t={par.timenow}, "
                f"dt collapsed?)")
        state = solver.step_RK()
        taken += 1
        if on_step is not None:
            on_step(state)
    return state, taken


# ============================================================================
#   Finiteness / positivity diagnostics
# ============================================================================

# Per-mode field roles, used by both the sanity and robustness suites.
#   'positive' fields must stay > 0 (density, pressure, SWE height, ...)
#   'finite'   fields just must never become NaN/Inf (velocities, B, ...)
FIELD_ROLES = {
    "adv":  {"finite": ["dens"]},
    "HD":   {"positive": ["dens", "pres"], "finite": ["vel1", "vel2", "vel3"]},
    "rHD":  {"positive": ["dens", "pres"], "finite": ["vel1", "vel2", "vel3"]},
    "MHD":  {"positive": ["dens", "pres"],
             "finite": ["vel1", "vel2", "vel3", "bfi1", "bfi2", "bfi3"]},
    "rMHD": {"positive": ["dens", "pres"],
             "finite": ["vel1", "vel2", "vel3", "bfi1", "bfi2", "bfi3"]},
    "diff": {"finite": ["T"]},
    "SWE":  {"positive": ["h"], "finite": ["vel1", "vel2"]},
}


def check_state_sane(grid, mode, state, label=""):
    """
    Assert every field in ``FIELD_ROLES[mode]`` is finite (and, where
    required, strictly positive) on the INTERIOR cells.

    Deliberately restricted to the interior: ghost cells are auxiliary
    stencil support, not part of the physical solution, and IC functions
    routinely only set the interior slice (e.g. IC_HD2D_KHI), relying on
    the solver's own boundary-condition call -- run once per step, not at
    construction time -- to populate the ghosts before the first flux
    evaluation. Checking ghosts here would therefore fail on a perfectly
    healthy, untouched initial condition.

    Parameters
    ----------
    grid : Grid
    mode : str
    state : SimState
    label : str
        Prepended to the assertion message for a useful failure report.

    Raises
    ------
    AssertionError
        On the first violated field.
    """
    Ngc = grid.Ngc
    roles = FIELD_ROLES[mode]
    for name in roles.get("finite", []) + roles.get("positive", []):
        val = getattr(state, name, None)
        if val is None or np.isscalar(val):
            continue                       # e.g. adv's constant vel1/vel2
        interior = val[Ngc:-Ngc, Ngc:-Ngc] if val.shape == grid.grid_shape else val
        assert np.all(np.isfinite(interior)), (
            f"{label}: field '{name}' has non-finite values "
            f"(NaN/Inf count={np.sum(~np.isfinite(interior))})")
    for name in roles.get("positive", []):
        val = getattr(state, name, None)
        if val is None or np.isscalar(val):
            continue
        interior = val[Ngc:-Ngc, Ngc:-Ngc] if val.shape == grid.grid_shape else val
        assert np.all(interior > 0.0), (
            f"{label}: field '{name}' has non-positive values "
            f"(min={np.min(interior):.6e})")


# ============================================================================
#   Conserved volume integrals
# ============================================================================

def interior_integral(grid, var_interior):
    """Volume integral of an interior-only (Nx1, Nx2) array (conservative
    variables: mass, mom1/2/3, etot, bcon1/2/3, ... -- no ghost cells)."""
    return float(np.sum(grid.cVol * var_interior))


def ghost_integral(grid, var_full):
    """Volume integral of a ghost-padded (grid.grid_shape) array (primitive
    fields: dens, T, h, ...)."""
    Ngc = grid.Ngc
    return float(np.sum(grid.cVol * var_full[Ngc:-Ngc, Ngc:-Ngc]))


def rel_drift(new, old, scale=None):
    """
    Relative drift of a conserved quantity, |new - old| / scale.

    `scale` defaults to |old|; pass an explicit physical scale (e.g. total
    mass times a characteristic velocity) for quantities that are ~0 by
    symmetry (net momentum of a centred vortex, ...), where dividing by
    |old| itself would be meaningless.
    """
    denom = abs(old) if scale is None else scale
    denom = denom if denom > 1e-300 else 1.0
    return abs(new - old) / denom


# ============================================================================
#   Convergence-order diagnostics
# ============================================================================

def l2_error(grid, a, b):
    """
    Volume-weighted RMS error between two interior-cell arrays of the same
    shape, sqrt( sum(cVol*(a-b)^2) / sum(cVol) ) -- a per-cell RMS that is
    directly comparable across resolutions (unlike a raw, unnormalized
    sum-of-squares), which is what a clean log2-ratio convergence-order
    estimate needs.
    """
    return float(np.sqrt(np.sum(grid.cVol * (a - b) ** 2) / np.sum(grid.cVol)))


def observed_order(errors, refinement=2.0):
    """
    Observed convergence order between successive entries of `errors`
    (assumed measured at successive resolutions related by `refinement`,
    e.g. Nx, 2*Nx, 4*Nx, ...).

    Returns
    -------
    orders : list of float, length len(errors) - 1
        orders[i] = log(errors[i] / errors[i+1]) / log(refinement)
    """
    orders = []
    for e_coarse, e_fine in zip(errors[:-1], errors[1:]):
        if e_fine <= 0.0 or e_coarse <= 0.0:
            orders.append(float("inf"))    # exact to round-off: as good as it gets
        else:
            orders.append(np.log(e_coarse / e_fine) / np.log(refinement))
    return orders
