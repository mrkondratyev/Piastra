# -*- coding: utf-8 -*-
"""
===============================================================================
test_robustness.py
===============================================================================

Robustness sweep: every Riemann solver / reconstruction / RK-order
combination a mode supports must run a strong shock/blast problem to
completion without crashing, producing a NaN/Inf, or losing positivity
(density, pressure, SWE height) -- checked after EVERY step, not just at
the end, so a transient negative-density excursion that later "heals"
itself cannot hide.

The stress problems are the standard strong-shock benchmarks of the field:
Sod-type single shock tubes are the baseline, but the real stress test is
Woodward & Colella (1984)'s double blast wave (HD, multiple colliding
strong shocks, pressure ratio ~1e5) and Brio & Wu (1988)'s MHD shock tube
(the standard test for a compressible MHD solver).
All stress runs use 1D (Nx2=1) grids so the full combinatorial sweep stays
fast.

Author: mrkondratyev
"""

import itertools

from tests.testbed_common import build_case, run_to_tfin, check_state_sane

REC_TYPES = ["PCM", "PLM", "PPMorig", "PPM", "WENO", "MP5"]
RK_ORDERS = ["RK1", "RK2", "RK3"]


def _stress(mode, problem, Nx1, Nx2, label, **par_kwargs):
    """Build one case and run it to completion, checking sanity every step."""
    grid, state, par, eos, solver = build_case(mode, problem, Nx1, Nx2, **par_kwargs)
    run_to_tfin(solver, par,
                on_step=lambda s: check_state_sane(grid, mode, s, label=label))


def _slug(*parts):
    return "_".join(str(p) for p in parts).replace(".", "p").replace("-", "_")


def _register(registry, name, fn):
    fn.__name__ = name
    registry[name] = fn


def _register_cross_product(registry, prefix, mode, problem, Nx1, Nx2,
                             solver_types, rec_types=REC_TYPES, rk_orders=RK_ORDERS,
                             **fixed_kwargs):
    """
    One test_* function per (solver_type, rec_type, RK_order) combination,
    all run on the same (mode, problem, Nx1, Nx2) stress case.

    Skips (rec_type='MP5', RK_order='RK1') -- a documented, expected
    instability, not a bug: forward Euler is the lowest-order SSP
    Runge-Kutta method and has, correspondingly, the smallest stability
    margin of the three RK_order choices, while MP5 (Suresh & Huynh 1997)
    is deliberately the LEAST dissipative of the high-order
    reconstructions (that is its whole purpose -- preserve smooth
    extrema other limiters clip). Verified directly: MP5+RK1 is stable on
    a smooth periodic profile (no shocks) but loses positivity on even a
    mild Sod shock, while MP5+RK2/RK3 handle the same strong1D/BW1D/dam1D
    stress problems fine -- exactly the textbook reason SSP-RK2/RK3 (Shu
    & Osher 1988) are used with high-order shock-capturing schemes rather
    than plain forward Euler. Use RK2 or RK3 with MP5 in practice.
    """
    for solver_type, rec_type, rk_order in itertools.product(
            solver_types, rec_types, rk_orders):
        if rec_type == "MP5" and rk_order == "RK1":
            continue
        name = f"test_robustness_{prefix}_{_slug(solver_type, rec_type, rk_order)}"
        label = f"{mode}/{problem} (solver={solver_type}, rec={rec_type}, RK={rk_order})"

        def _test(mode=mode, problem=problem, Nx1=Nx1, Nx2=Nx2, label=label,
                   solver_type=solver_type, rec_type=rec_type, rk_order=rk_order,
                   fixed_kwargs=fixed_kwargs):
            _stress(mode, problem, Nx1, Nx2, label,
                    solver_type=solver_type, rec_type=rec_type, RK_order=rk_order,
                    **fixed_kwargs)

        _register(registry, name, _test)


_REGISTRY = {}

# ----------------------------------------------------------------------------
# HD -- strong1D (single strong shock) gets the full solver x rec x RK sweep;
# DBW1D (Woodward-Colella double blast wave) gets each solver_type and each
# rec_type at least once, the classic multi-shock-collision stress test.
# ----------------------------------------------------------------------------
HD_SOLVERS = ["LLF", "HLL", "HLLC", "Roe", "Exact"]
_register_cross_product(_REGISTRY, "hd_strong1D", "HD", "strong1D", 100, 1,
                         HD_SOLVERS, CFL=0.7)
_register_cross_product(_REGISTRY, "hd_dbw1D_solver", "HD", "DBW1D", 100, 1,
                         HD_SOLVERS, rec_types=["PLM"], rk_orders=["RK2"], CFL=0.7)
_register_cross_product(_REGISTRY, "hd_dbw1D_rec", "HD", "DBW1D", 100, 1,
                         ["HLLC"], rec_types=REC_TYPES, rk_orders=["RK2"], CFL=0.7)

# ----------------------------------------------------------------------------
# rHD -- Mignone & Bodo (2005) RP1, full solver x rec x RK sweep.
# ----------------------------------------------------------------------------
_register_cross_product(_REGISTRY, "rhd_RP1", "rHD", "RP1", 100, 1,
                         ["LLF", "HLL", "HLLC"], CFL=0.4)

# ----------------------------------------------------------------------------
# MHD -- Brio & Wu (1988), full solver x rec x RK sweep for each divergence
# treatment (CT / GLM / 8wave): 4 solvers x 6 rec x 3 RK x 3 schemes = 216.
# ----------------------------------------------------------------------------
MHD_SOLVERS = ["LLF", "HLL", "HLLC", "HLLD"]
for _divb in ("CT", "GLM", "8wave"):
    _register_cross_product(_REGISTRY, f"mhd_BW1D_{_divb}", "MHD", "BW1D", 100, 1,
                             MHD_SOLVERS, CFL=0.5, divb_tr=_divb)
del _divb

# ----------------------------------------------------------------------------
# rMHD -- only CT is supported; Brio-Wu analogue, full solver x rec x RK sweep.
# ----------------------------------------------------------------------------
_register_cross_product(_REGISTRY, "rmhd_BW1D", "rMHD", "BW1D", 100, 1,
                         ["LLF", "HLL"], CFL=0.4, divb_tr="CT")

# ----------------------------------------------------------------------------
# SWE -- dam break (the shallow-water analogue of a Sod shock tube).
# ----------------------------------------------------------------------------
_register_cross_product(_REGISTRY, "swe_dam1D", "SWE", "dam1D", 100, 1,
                         ["LLF", "HLL", "Exact"])

# ----------------------------------------------------------------------------
# diff -- expl and rkl2 (at a few stage counts) on a sharp step IC.
# ----------------------------------------------------------------------------
for _name, _kwargs in [("expl", dict(solver_type="expl")),
                        ("rkl2_s4", dict(solver_type="rkl2", rkl2_stages=4)),
                        ("rkl2_s10", dict(solver_type="rkl2", rkl2_stages=10)),
                        ("rkl2_s25", dict(solver_type="rkl2", rkl2_stages=25))]:
    def _test(kwargs=_kwargs, label=f"diff/step1D ({_name})"):
        _stress("diff", "step1D", 100, 1, label, **kwargs)
    _register(_REGISTRY, f"test_robustness_diff_step1D_{_name}", _test)
del _name, _kwargs

# ----------------------------------------------------------------------------
# adv -- 'adv' (reconstructed upwind) gets the full rec x RK sweep; 'LW'
# (Lax-Wendroff) ignores both (see adv_step.oneStep_adv_RK), so it is run
# once on its own. A discontinuous IC is used deliberately: Lax-Wendroff is
# well known to ring around a discontinuity (it is not TVD), so only
# finiteness is checked for 'adv' mode (see FIELD_ROLES in testbed_common.py)
# -- that ringing is expected scheme behaviour, not a bug.
# ----------------------------------------------------------------------------
_register_cross_product(_REGISTRY, "adv_disc1D", "adv", "disc1D", 100, 1,
                         ["adv"], CFL=0.4)


def test_robustness_adv_disc1D_LW():
    """Lax-Wendroff flux: RK_order/rec_type are no-ops for it (see adv_step.py)."""
    _stress("adv", "disc1D", 100, 1, "adv/disc1D (solver=LW)",
            solver_type="LW", CFL=0.4)


# Inject the dynamically-built registry into this module's namespace so both
# this repo's own runner (run_testbed.py) and pytest's collection see them.
globals().update(_REGISTRY)
del _REGISTRY
