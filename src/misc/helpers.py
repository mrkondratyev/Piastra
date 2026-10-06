# -*- coding: utf-8 -*-
"""
helpers.py

Helper routines for the Piastra simulation framework.

Provides:
  - run_simulation : main time-stepping loop with periodic visualisation
                     and (optional) periodic output via a callback
  - PROBLEMS       : the problem catalogue, {mode: {problem name: IC function}}
  - initial_model  : looks up (mode, problem) in PROBLEMS and calls the
                     initial-condition function

Supported modes: 'adv', 'HD', 'rHD', 'MHD', 'rMHD', 'diff', 'SWE'.

Author: mrkondratyev
"""

import time
import numpy as np
from src.misc.io_visual import plot_setup, plotting


def run_simulation(grid, state, par, solver, var_to_plot, n_plot,
                   dt_output=None, on_output=None):
    """
    Advance the numerical simulation in time.

    Runs the main time-integration loop, calling ``solver.step_RK()``
    each timestep.  Produces periodic plots of the selected variable
    and reports timing information.

    Parameters
    ----------
    grid : Grid
        Grid object with geometry and resolution information.
    state : SimState
        State container for the physical variables.
    par : Parameters
        Simulation parameters.  Must include timenow, timefin, mode,
        and (for all modes except diffusion) rec_type and RK_order.
    solver : object
        Numerical solver providing the method ``step_RK()``.
    var_to_plot : ndarray or callable
        Variable to visualise (2-D, with ghost cells). Either an array that
        the solver updates in place (e.g. state.dens), or a function with no
        arguments returning the array, e.g. ``lambda: state.h + state.b``.
        A function is re-evaluated before every plot, so it also works for
        derived quantities and for solvers that rebind the state arrays.
    n_plot : int
        Interval (in timesteps) between visualisation updates.
    dt_output : float, optional
        Physical time between outputs.  Output number k is due at
        t_k = k * dt_output (k counted from t = 0, so the numbering continues
        correctly after a restart).  The time step is shortened so that the
        run lands exactly on every t_k (this adds at most one short step per
        output).
    on_output : callable, optional
        Called as ``on_output(state, par, k)`` whenever output k is due, e.g.
        to write a snapshot with ``save_data``.  Both dt_output and on_output
        must be given to enable output.

    Returns
    -------
    state : SimState
        Updated simulation state at final time.
    timenow : float
        Final physical time reached by the simulation.
    """
    # Print solver configuration
    print("numerical model = ", par.mode)
    print("solver type  = ", par.solver_type)
    print("grid resolution = ", grid.Nx1, grid.Nx2)
    
    if par.mode == "diff":
        if par.solver_type == "rkl2":
            print("RKL2 stages     = ", par.rkl2_stages)
    else:
        print("reconstruction type  = ", par.rec_type)
        print("temporal integration = ", par.RK_order)

    print("final phys time = ", par.timefin)

    # Plot setup
    field = var_to_plot() if callable(var_to_plot) else var_to_plot
    line, ax, fig, im = plot_setup(grid, field, par.timenow)

    print("START OF SIMULATION")

    start_time1 = time.time()
    i_time = 0

    # --- periodic output ---------------------------------------------------
    # Output k is due at t_k = k * dt_output.  k_out is the first k with
    # t_k > timenow (the 1e-12 tolerance skips the snapshot we restarted from).
    #
    # To land exactly on t_k, the solvers' own clipping of the last step,
    #     dt = min(dt_CFL, par.timefin - par.timenow),
    # is reused: during each step par.timefin temporarily holds
    # t_stop = min(t_end, t_k), and the true final time t_end is put back
    # right after the step, so files and the caller always see t_end.
    do_output = (dt_output is not None) and (on_output is not None)
    t_end = par.timefin
    if do_output:
        k_out = int(np.floor(par.timenow / dt_output * (1.0 + 1e-12))) + 1

    while par.timenow < t_end:

        i_time += 1

        if do_output:
            par.timefin = min(t_end, k_out * dt_output)
        state = solver.step_RK()
        par.timefin = t_end

        if (i_time % n_plot == 0) or (t_end - par.timenow) < 1e-12:
            print("phys time = ", par.timenow)
            print('num of timesteps = ', i_time)
            field = var_to_plot() if callable(var_to_plot) else var_to_plot
            plotting(grid, field, par.timenow, line, ax, fig, im)

        if do_output and par.timenow >= k_out * dt_output * (1.0 - 1e-12):
            on_output(state, par, k_out)
            k_out += 1

    print("final phys time = ", par.timenow)
    print("END OF SIMULATION")
    end_time1 = time.time()
    print("elapsed time = ", end_time1 - start_time1, " secs")

    return state, par.timenow


# ── IC imports ────────────────────────────────────────────────────────────────

from src.models.adv.adv_init_cond import (
    IC_adv1D_smooth,
    IC_adv1D_disc,
    IC_adv2D_smooth,
    IC_adv2D_disc,
    IC_adv_user_defined,
)
from src.models.diff.diff_init_cond import (
    IC_diff2D_gaussian,
    IC_diff1D_step,
    IC_diff2D_cyl,
    IC_diff1D_sine,
    IC_diff2D_gauss_polar,
    IC_diff2D_gauss_sph,
    IC_diff_user_defined,
)
from src.models.HD.HD_init_cond import (
    IC_HD1D_Sod_cart,
    IC_HD1D_Sod_cyl,
    IC_HD1D_Sod_sph,
    IC_HD1D_strong_shock,
    IC_HD1D_DBW,
    IC_HD1D_ShuOsher,
    IC_HD1D_Einfeldt,
    IC_HD2D_KHI,
    IC_HD2D_RTI,
    IC_HD2D_Sod,
    IC_HD2D_Sod_sph,
    IC_HD2D_Sod_polar,
    IC_HD2D_Sedov_cart,
    IC_HD2D_Sedov_cyl,
    IC_HD2D_RP2D,
    IC_HD2D_Gresho,
    IC_HD2D_vortex,
    IC_HD2D_shock_cloud,
    IC_HD2D_gap_opening,
    IC_HD2D_jet_cyl,
    IC_HD1D_Jeans,
    IC_HD2D_collapse_sph,
    IC_HD2D_merger,
    IC_HD_user_defined,
)
from src.models.MHD.MHD_init_cond import (
    IC_MHD1D_BW,
    IC_MHD1D_Toth,
    IC_MHD1D_RJ,
    IC_MHD1D_Alfven,
    IC_MHD2D_blast_cart,
    IC_MHD2D_blast_cyl,
    IC_MHD2D_blast_sph, 
    IC_MHD2D_OT,
    IC_MHD2D_rotor,
    IC_MHD2D_Alfven,
    IC_MHD2D_current_sheet,
    IC_MHD2D_field_loop,
    IC_MHD2D_disk,
    IC_MHD2D_jet_cyl,
    IC_MHD2D_shock_cloud,
    IC_MHD_user_defined,
)
from src.models.rHD.rHD_init_cond import (
    IC_rHD1D_RP1,
    IC_rHD1D_RP3,
    IC_rHD1D_RP4,
    IC_rHD1D_RP5,
    IC_rHD2D_RP,
    IC_rHD2D_RTI,
    IC_rHD2D_jet_cart,
    IC_rHD2D_jet_cyl,
    IC_rHD_user_defined,
)
from src.models.rMHD.rMHD_init_cond import (
    IC_rMHD1D_BW,
    IC_rMHD1D_RP2,
    IC_rMHD1D_RP3,
    IC_rMHD1D_RP4,
    IC_rMHD2D_blast,
    IC_rMHD2D_rotor,
    IC_rMHD2D_OT,
    IC_rMHD_user_defined,
)
from src.models.SWE.SWE_init_cond import (
    IC_SWE1D_dam,
    IC_SWE2D_bathtub,
    IC_SWE2D_rotdam,
    IC_SWE2D_kelvin,
    IC_SWE2D_dam,
    IC_SWE2D_bickley,
    IC_SWE2D_KHI,
    IC_SWE2D_lake,
    IC_SWE_user_defined,
)


# ── Problem catalogue ─────────────────────────────────────────────────────────
# One dictionary per mode: problem name -> initial-condition function.
# This is the single source of truth for the available problems; the
# test suite (tests/test_sanity.py) builds its catalogue from PROBLEMS.

diff_dispatch = {
    "gauss2D":      IC_diff2D_gaussian,
    "step1D":       IC_diff1D_step,
    "sine1D":       IC_diff1D_sine,
    "cyl2D":        IC_diff2D_cyl,
    "gauss2Dpol":   IC_diff2D_gauss_polar,
    "gauss2Dsph":   IC_diff2D_gauss_sph,
    "user_defined": IC_diff_user_defined,
}

adv_dispatch = {
    "smooth1D":     IC_adv1D_smooth,
    "disc1D":       IC_adv1D_disc,
    "smooth2D":     IC_adv2D_smooth,
    "disc2D":       IC_adv2D_disc,
    "user_defined": IC_adv_user_defined,
}

hd_dispatch = {
    "sod1Dcart":    IC_HD1D_Sod_cart,
    "sod1Dcyl":     IC_HD1D_Sod_cyl,
    "sod1Dsph":     IC_HD1D_Sod_sph,
    "strong1D":     IC_HD1D_strong_shock,
    "DBW1D":        IC_HD1D_DBW,
    "shuosher1D":   IC_HD1D_ShuOsher,
    "einfeldt1D":   IC_HD1D_Einfeldt,
    "sod2Dsph":     IC_HD2D_Sod_sph,
    "sod2Dpol":     IC_HD2D_Sod_polar,
    "KHI2D":        IC_HD2D_KHI,
    "RTI2D":        IC_HD2D_RTI,
    "sod2Dcart":    IC_HD2D_Sod,
    "sedov2Dcart":  IC_HD2D_Sedov_cart,
    "sedov2Dcyl":   IC_HD2D_Sedov_cyl,
    "RP2D":         IC_HD2D_RP2D,
    "gresho2D":     IC_HD2D_Gresho,
    "vortex2D":     IC_HD2D_vortex,
    "shock-cloud":  IC_HD2D_shock_cloud,
    "gap-opening":  IC_HD2D_gap_opening,
    "jet2Dcyl":     IC_HD2D_jet_cyl,
    "jeans1D":      IC_HD1D_Jeans,
    "collapse_sph": IC_HD2D_collapse_sph,
    "merger2D":     IC_HD2D_merger,
    "user_defined": IC_HD_user_defined,
}

mhd_dispatch = {
    "BW1D":          IC_MHD1D_BW,
    "toth1D":        IC_MHD1D_Toth,
    "RJ1D":          IC_MHD1D_RJ,
    "alfven1D":      IC_MHD1D_Alfven,
    "alfven2D":      IC_MHD2D_Alfven,
    "blast2Dcart":   IC_MHD2D_blast_cart,
    "blast2Dcyl":    IC_MHD2D_blast_cyl,
    "blast2Dsph":    IC_MHD2D_blast_sph, 
    "rotor2D":       IC_MHD2D_rotor,
    "OT2D":          IC_MHD2D_OT,
    "current-sheet": IC_MHD2D_current_sheet,
    "field-loop":    IC_MHD2D_field_loop,
    "disk2D":        IC_MHD2D_disk,
    "jet2Dcyl":      IC_MHD2D_jet_cyl,
    "shock-cloud":   IC_MHD2D_shock_cloud,
    "user_defined":  IC_MHD_user_defined,
}

rhd_dispatch = {
    "RP1":          IC_rHD1D_RP1,
    "RP3":          IC_rHD1D_RP3,
    "RP4":          IC_rHD1D_RP4,
    "RP5":          IC_rHD1D_RP5,
    "RP2D":         IC_rHD2D_RP,
    "RTI":          IC_rHD2D_RTI,
    "jet2Dcart":    IC_rHD2D_jet_cart,
    "jet2Dcyl":     IC_rHD2D_jet_cyl,
    "user_defined": IC_rHD_user_defined,
}

rmhd_dispatch = {
    "BW1D":         IC_rMHD1D_BW,    
    "RP2":          IC_rMHD1D_RP2,
    "RP3":          IC_rMHD1D_RP3,
    "RP4":          IC_rMHD1D_RP4,
    "blast2D":      IC_rMHD2D_blast,
    "rotor2D":      IC_rMHD2D_rotor,
    "OT2D":         IC_rMHD2D_OT,
    "user_defined": IC_rMHD_user_defined,
}

swe_dispatch = {
    "dam1D":        IC_SWE1D_dam,
    "bathtub2D":    IC_SWE2D_bathtub,
    "rotdam2D":     IC_SWE2D_rotdam,
    "kelvin2D":     IC_SWE2D_kelvin,
    "dam2D":        IC_SWE2D_dam,
    "jet2D":        IC_SWE2D_bickley,
    "KHI2D":        IC_SWE2D_KHI,
    "lake2D":       IC_SWE2D_lake,
    "user_defined": IC_SWE_user_defined,
}

PROBLEMS = {
    "adv":  adv_dispatch,
    "diff": diff_dispatch,
    "HD":   hd_dispatch,
    "rHD":  rhd_dispatch,
    "MHD":  mhd_dispatch,
    "rMHD": rmhd_dispatch,
    "SWE":  swe_dispatch,
}


def initial_model(grid, state, par):
    """
    Initialize the chosen test problem based on simulation mode and problem name.

    Dispatches to the appropriate IC function which sets up grid geometry,
    primitive variables, boundary conditions, final time, and EOS.

    Parameters
    ----------
    grid : Grid
        Grid object containing mesh geometry and metric information.
    state : SimState
        Simulation state container.
    par : Parameters
        Parameters object containing simulation settings, including
        mode and problem name.

    Returns
    -------
    grid : Grid
        Grid with geometry initialised.
    state : SimState
        State with primitive variables set.
    par : Parameters
        Parameters with timefin, BC, etc. configured.
    eos : EOSdata or None
        Equation of state (None for advection, diffusion, and SWE modes).
    """

    if par.mode not in PROBLEMS:
        raise ValueError(
            f"Invalid simulation mode '{par.mode}'. "
            f"Expected one of {list(PROBLEMS.keys())}.")

    dispatch = PROBLEMS[par.mode]
    if par.problem not in dispatch:
        raise ValueError(
            f"Invalid {par.mode} problem '{par.problem}'. "
            f"Available: {list(dispatch.keys())}")
    IC_function = dispatch[par.problem]

    # advection and diffusion ICs return no EOS; SWE ICs return eos = None
    if par.mode in ("adv", "diff"):
        grid, state, par = IC_function(grid, state, par)
        eos = None
    else:
        grid, state, par, eos = IC_function(grid, state, par)

    return grid, state, par, eos
