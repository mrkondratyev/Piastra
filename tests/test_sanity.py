# -*- coding: utf-8 -*-
"""
===============================================================================
test_sanity.py
===============================================================================

Breadth smoke test: every (mode, problem) pair in Piastra's test-problem
catalogue (the same one documented in README.md / main.py) must build and
step a few times without crashing, producing a NaN/Inf, or going unphysical
(negative density, pressure, or SWE height).

Deliberately run on a NON-SQUARE grid (Nx1 != Nx2): a square grid silently
hides axis-swap / mis-slicing bugs -- this is exactly how the grid.cVol
ghost-offset bug in the Sedov ICs (and gravity.py's monopole self-gravity)
went unnoticed before being found and fixed.

'user_defined' is excluded from every mode's list: it is a deliberate
template stub that raises ValueError by design (see e.g.
HD_init_cond.IC_HD_user_defined).

Author: mrkondratyev
"""

from tests.testbed_common import build_case, run_steps, check_state_sane

# Mirrors the catalogue documented in README.md and main.py's module docstring.
CATALOGUE = {
    "adv":  ["smooth1D", "disc1D", "smooth2D", "disc2D"],
    "HD":   ["sod1Dcart", "sod1Dcyl", "sod1Dsph", "strong1D", "DBW1D",
             "shuosher1D", "einfeldt1D", "sod2Dcart", "sod2Dsph", "sod2Dpol",
             "sedov2Dcart", "sedov2Dcyl", "RP2D", "gresho2D", "KHI2D",
             "RTI2D", "shock-cloud", "gap-opening", "jet2Dcyl"],
    "rHD":  ["RP1", "RP3", "RP4", "RP5", "RP2D", "RTI", "jet2Dcart",
             "jet2Dcyl"],
    "MHD":  ["BW1D", "toth1D", "RJ1D", "alfven1D", "alfven2D",
             "blast2Dcart", "blast2Dcyl", "blast2Dsph", "rotor2D", "OT2D",
             "current-sheet", "field-loop", "disk2D", "jet2Dcyl",
             "shock-cloud"],
    "rMHD": ["BW1D", "RP2", "RP3", "RP4", "blast2D", "rotor2D"],
    "SWE":  ["dam1D", "bump1D", "bathtub2D", "expl2D", "tsunami2D",
             "ocean2D", "atmo2D", "dam2D", "jet2D", "KHI2D"],
    "diff": ["gauss1D", "gauss2D", "step1D", "sine1D", "cross2D", "ring2D",
             "cyl2D"],
}

NX1, NX2 = 22, 18     # small + deliberately rectangular (see module docstring)
NSTEPS = 5


def _run_one(mode, problem):
    grid, state, par, eos, solver = build_case(mode, problem, NX1, NX2)
    check_state_sane(grid, mode, state, label=f"{mode}/{problem} (initial condition)")

    state, taken = run_steps(
        solver, par, NSTEPS,
        on_step=lambda s: check_state_sane(grid, mode, s, label=f"{mode}/{problem} (mid-run)"))

    assert taken > 0, f"{mode}/{problem}: solver took zero steps"
    assert par.timenow > 0.0, f"{mode}/{problem}: timenow did not advance (dt collapsed to 0?)"


def _make_test(mode, problem):
    def _test():
        _run_one(mode, problem)
    _test.__name__ = f"test_sanity_{mode}_{problem}".replace("-", "_")
    _test.__doc__ = (f"Smoke-test {mode}/{problem}: builds, runs {NSTEPS} "
                      f"steps, stays finite and physical throughout.")
    return _test


# Dynamically register one test_* function per (mode, problem) pair, so each
# case is individually visible to both this repo's own runner
# (run_testbed.py) and, if installed, pytest's collection.
for _mode, _problems in CATALOGUE.items():
    for _problem in _problems:
        _name = f"test_sanity_{_mode}_{_problem}".replace("-", "_")
        globals()[_name] = _make_test(_mode, _problem)
del _mode, _problems, _problem, _name
