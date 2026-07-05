# -*- coding: utf-8 -*-
"""
===============================================================================
test_conservation.py
===============================================================================

Conservation checks: the textbook validation of a flux-difference,
finite-volume update. At any interior face the flux added to one cell is
subtracted from its neighbour, so the only way the volume integral of a
conserved quantity can change is through flux leaving across a domain
boundary. On a closed domain -- periodic (every face is "interior", just
wrapped around) or wall (the boundary flux is exactly zero by
construction, see boundaries.py) -- mass, momentum and total energy must
therefore be constant to round-off, for as many steps as the disturbance
hasn't reached a non-periodic boundary.

Mass/momentum/energy are read from the CONSERVATIVE arrays (state.mass,
mom1/2, etot), which only get populated from the primitive initial
condition inside the solver's first prim2cons call -- so every check below
takes its "before" snapshot after one warm-up step, not at t=0 (state.mass
etc. are still all zero straight out of build_case(), since nothing has
gone through prim2cons yet).

Author: mrkondratyev
"""

import numpy as np

from tests.testbed_common import (
    build_case, run_steps, interior_integral, ghost_integral, rel_drift)

TOL = 1e-8           # relative drift; round-off in practice is ~1e-14..1e-16,
                      # so this leaves ~6 orders of margin before a real bug


# ============================================================================
#   HD
# ============================================================================

def test_hd_periodic_conservation():
    """Mass, momentum and total energy on the fully-periodic Gresho vortex."""
    grid, state, par, eos, solver = build_case(
        "HD", "gresho2D", 24, 22, rec_type="PLM", RK_order="RK2", CFL=0.6)

    state, _ = run_steps(solver, par, 1)               # warm-up: fill mass/mom/etot
    mass0  = interior_integral(grid, state.mass)
    mom1_0 = interior_integral(grid, state.mom1)
    mom2_0 = interior_integral(grid, state.mom2)
    etot0  = interior_integral(grid, state.etot)
    mom_scale = mass0 * 1.0     # Gresho's net momentum is ~0 by symmetry, so
                                 # judge its drift against mass*(O(1) velocity)

    state, _ = run_steps(solver, par, 40)

    assert rel_drift(interior_integral(grid, state.mass), mass0) < TOL
    assert rel_drift(interior_integral(grid, state.mom1), mom1_0, mom_scale) < TOL
    assert rel_drift(interior_integral(grid, state.mom2), mom2_0, mom_scale) < TOL
    assert rel_drift(interior_integral(grid, state.etot), etot0) < TOL


def test_hd_wall_conservation():
    """
    Mass and total energy on a wall+free-bounded blast (Sedov), run only
    long enough that the blast wave is still well inside the domain (an
    ideal reflecting wall passes zero mass/energy flux; a free/zero-gradient
    face passes zero flux only while the state there is still undisturbed
    -- momentum is NOT checked here, since a wall legitimately exerts a net
    force on the fluid).
    """
    grid, state, par, eos, solver = build_case(
        "HD", "sedov2Dcart", 24, 22, rec_type="PLM", RK_order="RK2", CFL=0.6)

    state, _ = run_steps(solver, par, 1)
    mass0 = interior_integral(grid, state.mass)
    etot0 = interior_integral(grid, state.etot)

    state, _ = run_steps(solver, par, 15)
    assert par.timenow < 0.1 * par.timefin, (
        "blast reached a substantial fraction of timefin -- it may have hit "
        "the free boundary, invalidating the zero-flux assumption")

    assert rel_drift(interior_integral(grid, state.mass), mass0) < TOL
    assert rel_drift(interior_integral(grid, state.etot), etot0) < TOL


# ============================================================================
#   MHD
# ============================================================================

def test_mhd_periodic_conservation():
    """Mass, momentum and total energy on the fully-periodic Orszag-Tang vortex."""
    grid, state, par, eos, solver = build_case(
        "MHD", "OT2D", 24, 22, rec_type="PLM", RK_order="RK2",
        divb_tr="CT", CFL=0.5)

    state, _ = run_steps(solver, par, 1)
    mass0  = interior_integral(grid, state.mass)
    mom1_0 = interior_integral(grid, state.mom1)
    mom2_0 = interior_integral(grid, state.mom2)
    etot0  = interior_integral(grid, state.etot)
    mom_scale = mass0 * 1.0

    state, _ = run_steps(solver, par, 40)

    assert rel_drift(interior_integral(grid, state.mass), mass0) < TOL
    assert rel_drift(interior_integral(grid, state.mom1), mom1_0, mom_scale) < TOL
    assert rel_drift(interior_integral(grid, state.mom2), mom2_0, mom_scale) < TOL
    assert rel_drift(interior_integral(grid, state.etot), etot0) < TOL


def test_mhd_ct_divb_preservation():
    """
    Constrained Transport keeps the discrete div(B) at the cell centres at
    round-off, by construction (it updates the staggered field through an
    EMF that is exactly curl-like). A drift away from ~1e-13 is a CT-update
    regression, not a discretisation-error issue.
    """
    grid, state, par, eos, solver = build_case(
        "MHD", "OT2D", 24, 22, rec_type="PLM", RK_order="RK2",
        divb_tr="CT", CFL=0.5)

    state, _ = run_steps(solver, par, 40)

    # characteristic |B| to turn the raw divB into a dimensionless number
    Ngc = grid.Ngc
    bscale = max(np.max(np.abs(state.bfi1[Ngc:-Ngc, Ngc:-Ngc])),
                 np.max(np.abs(state.bfi2[Ngc:-Ngc, Ngc:-Ngc])), 1e-300)
    dx = min(grid.dx1uc, grid.dx2uc)

    assert np.max(np.abs(state.divB)) * dx / bscale < 1e-10


# ============================================================================
#   Diffusion
# ============================================================================

def test_diff_zero_flux_conservation():
    """
    Total heat content integral(T dV) is conserved under zero-gradient
    (free) boundaries -- the diffusive flux at a mirrored ghost cell is
    identically zero, so there is nothing to telescope away at the edge.
    """
    grid, state, par, eos, solver = build_case(
        "diff", "gauss2D", 24, 22, solver_type="rkl2", rkl2_stages=6)

    T0 = ghost_integral(grid, state.T)
    state, _ = run_steps(solver, par, 20)
    T1 = ghost_integral(grid, state.T)

    assert rel_drift(T1, T0) < TOL


# ============================================================================
#   SWE
# ============================================================================

def test_swe_wall_conservation():
    """Water volume and momentum in a closed (all-wall) basin."""
    grid, state, par, eos, solver = build_case("SWE", "bathtub2D", 24, 22)

    h0    = ghost_integral(grid, state.h)
    mom1_0 = ghost_integral(grid, state.h * state.vel1)
    mom2_0 = ghost_integral(grid, state.h * state.vel2)

    state, _ = run_steps(solver, par, 30)

    assert rel_drift(ghost_integral(grid, state.h), h0) < TOL
    assert rel_drift(ghost_integral(grid, state.h * state.vel1), mom1_0, h0) < TOL
    assert rel_drift(ghost_integral(grid, state.h * state.vel2), mom2_0, h0) < TOL
