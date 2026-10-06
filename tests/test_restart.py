# -*- coding: utf-8 -*-
"""
===============================================================================
test_restart.py
===============================================================================

Restart suite: a run interrupted and continued from a snapshot must give the
same result as the uninterrupted run, bit for bit.

For every case:
  run A : t = 0 -> T in one go, through run_simulation with dt_output = T/2,
          writing snapshots at T/2 and T with save_data;
  run B : restart_simulation(snapshot at T/2) -> T, same dt_output.

Checks:
  * the snapshot lies exactly at t = T/2 (run_simulation shortens the step
    that would jump over an output time);
  * restart_simulation restores par.before_step whenever the problem has one;
  * every floating-point array of the final state of B equals that of A
    exactly (max |B - A| = 0).

Bitwise equality is expected, not just round-off agreement: the restarted
run starts from the same bits and takes the same sequence of time steps.
Any difference means some piece of state (a field, a parameter, a BC, the
gravity hook) was not stored or not restored.

The cases cover every mode and every MHD divergence treatment, both 1D and
2D, all four geometries, and all problems with per-step physics (gravity).
Grids are small and non-square (a square grid hides axis-swap bugs).

Author: mrkondratyev
"""

import glob
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from tests.testbed_common import build_case
from src.misc.helpers import run_simulation
from src.misc.io_utils import save_data, load_data, restart_simulation
from main import SOLVER_DISPATCH


#   (name, mode, problem, Nx1, Nx2, fraction of the IC's timefin, Parameters kwargs)
CASES = [
    ("adv_smooth1D",     "adv",  "smooth1D",     64, 1,  0.2,  {}),
    ("adv_disc2D_LW",    "adv",  "disc2D",       24, 20, 0.2,  {"solver_type": "LW"}),
    ("hd_sod1Dcart",     "HD",   "sod1Dcart",    64, 1,  1.0,  {}),
    ("hd_sod2Dpol_ppm",  "HD",   "sod2Dpol",     24, 20, 0.3,  {"rec_type": "PPM", "RK_order": "RK3"}),
    ("hd_gap_opening",   "HD",   "gap-opening",  16, 32, 0.02, {}),
    ("hd_jeans1D",       "HD",   "jeans1D",      64, 1,  0.1,  {}),
    ("hd_collapse_sph",  "HD",   "collapse_sph", 32, 1,  0.1,  {}),
    ("hd_merger2D",      "HD",   "merger2D",     24, 20, 0.05, {}),
    ("rhd_RP1",          "rHD",  "RP1",          64, 1,  0.5,  {}),
    ("rhd_RP2D",         "rHD",  "RP2D",         24, 20, 0.2,  {}),
    ("mhd_BW1D",         "MHD",  "BW1D",         64, 1,  0.5,  {}),
    ("mhd_OT2D_CT",      "MHD",  "OT2D",         24, 20, 0.1,  {"divb_tr": "CT"}),
    ("mhd_OT2D_GLM",     "MHD",  "OT2D",         24, 20, 0.1,  {"divb_tr": "GLM"}),
    ("mhd_OT2D_8wave",   "MHD",  "OT2D",         24, 20, 0.1,  {"divb_tr": "8wave"}),
    ("mhd_disk2D",       "MHD",  "disk2D",       24, 20, 0.02, {"divb_tr": "GLM"}),
    ("rmhd_BW1D",        "rMHD", "BW1D",         64, 1,  0.3,  {}),
    ("rmhd_rotor2D",     "rMHD", "rotor2D",      24, 20, 0.1,  {}),
    ("swe_dam1D",        "SWE",  "dam1D",        64, 1,  0.5,  {}),
    ("swe_bathtub2D",    "SWE",  "bathtub2D",    24, 20, 0.1,  {}),
    ("diff_gauss2D_rkl2","diff", "gauss2D",      24, 20, 0.2,  {"solver_type": "rkl2", "rkl2_stages": 8}),
    ("diff_step1D_expl", "diff", "step1D",       64, 1,  0.2,  {"solver_type": "expl"}),
]


def _plot_var(state):
    """Any ghost-inclusive field: run_simulation needs one to set up its plot."""
    for name in ("dens", "h", "T"):
        if hasattr(state, name):
            return lambda: getattr(state, name)
    raise AttributeError("no plottable field on state")


def _max_rel_diff(state_a, state_b):
    """max |B - A| / max |A| over all floating-point arrays; returns (value, field)."""
    worst, worst_name = 0.0, ""
    for name, a in vars(state_a).items():
        if not (isinstance(a, np.ndarray) and a.dtype.kind == "f"):
            continue
        b = getattr(state_b, name, None)
        if b is None or b.shape != a.shape:
            continue
        diff = np.max(np.abs(b - a)) / (np.max(np.abs(a)) + 1e-300)
        if diff > worst:
            worst, worst_name = diff, name
    return worst, worst_name


def _run_and_restart(mode, problem, Nx1, Nx2, frac, kwargs, restore_hooks=True):
    """
    Run A (0 -> T, snapshots at T/2 and T) and run B (restart at T/2 -> T).

    Returns
    -------
    par_a, state_a, par_b, state_b, t_snap : the two runs and the time
    stored in the T/2 snapshot.
    """
    outdir = tempfile.mkdtemp(prefix="piastra_restart_")

    # --- run A: uninterrupted, with snapshots ---
    grid, state, par, eos, solver = build_case(mode, problem, Nx1, Nx2, **kwargs)
    T = frac * par.timefin
    par.timefin = T
    dt_out = 0.5 * T
    save = lambda st, p, k: save_data(os.path.join(outdir, f"snap_{k:04d}"),
                                      grid, st, p, eos)
    state, _ = run_simulation(grid, state, par, solver, _plot_var(state), 10**9,
                              dt_output=dt_out, on_output=save)

    snap = sorted(glob.glob(os.path.join(outdir, "snap_*.npz")))[0]
    t_snap = load_data(snap)["timenow"]

    # --- run B: restart from the snapshot at T/2 ---
    grid_b, state_b, par_b, eos_b = restart_simulation(snap, restore_hooks=restore_hooks)
    solver_b = SOLVER_DISPATCH[mode](grid_b, state_b, eos_b, par_b)
    state_b, _ = run_simulation(grid_b, state_b, par_b, solver_b, _plot_var(state_b),
                                10**9, dt_output=dt_out, on_output=lambda st, p, k: None)
    plt.close("all")             # run_simulation opens a figure per call
    return par, state, par_b, state_b, t_snap


def _make_test(name, mode, problem, Nx1, Nx2, frac, kwargs):
    def _test():
        par_a, st_a, par_b, st_b, t_snap = _run_and_restart(
            mode, problem, Nx1, Nx2, frac, kwargs)
        T = par_a.timefin
        assert abs(t_snap - 0.5 * T) <= 1e-12 * T, \
            f"snapshot at t = {t_snap!r}, expected exactly T/2 = {0.5 * T!r}"
        assert (par_a.before_step is None) == (par_b.before_step is None), \
            "before_step was not restored on restart"
        assert par_b.timenow == par_a.timenow, \
            f"final time differs: {par_b.timenow!r} vs {par_a.timenow!r}"
        diff, field = _max_rel_diff(st_a, st_b)
        assert diff == 0.0, f"restarted run differs: max rel diff {diff:.3e} in '{field}'"
    _test.__name__ = f"test_restart_{name}"
    return _test


for _case in CASES:
    _fn = _make_test(*_case)
    globals()[_fn.__name__] = _fn
del _case, _fn


def test_restart_without_hooks_differs():
    """
    Control: with restore_hooks=False a self-gravitating run (jeans1D)
    continues without gravity and must NOT reproduce the uninterrupted run.
    Guards against the suite passing for a trivial reason.
    """
    par_a, st_a, par_b, st_b, _ = _run_and_restart(
        "HD", "jeans1D", 64, 1, 0.1, {}, restore_hooks=False)
    assert par_b.before_step is None
    diff, field = _max_rel_diff(st_a, st_b)
    assert diff > 1e-6, f"run without gravity matched the run with it (diff {diff:.1e})"
