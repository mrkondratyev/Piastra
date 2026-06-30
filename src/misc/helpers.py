# -*- coding: utf-8 -*-
"""
helpers.py

Helper routines for the Piastra simulation framework.

Provides:
  - run_simulation : main time-stepping loop with periodic visualisation
  - initial_model  : dispatcher that selects the correct initial-condition
                     function based on (mode, problem)

Supported modes: 'adv', 'HD', 'rHD', 'MHD', 'rMHD', 'diff', 'SWE'.

Author: mrkondratyev
"""

import time
from src.misc.io_visual import plot_setup, plotting


def run_simulation(grid, state, par, solver, var_to_plot, n_plot):
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
        and (for non-diffusion, non-SWE modes) rec_type and RK_order.
    solver : object
        Numerical solver providing the method ``step_RK()``.
    var_to_plot : ndarray
        2-D array (with ghost cells) of the variable to visualise.
    n_plot : int
        Interval (in timesteps) between visualisation updates.

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
    line, ax, fig, im = plot_setup(grid, var_to_plot, par.timenow)

    print("START OF SIMULATION")

    start_time1 = time.time()
    i_time = 0

    while par.timenow < par.timefin:

        i_time += 1

        state = solver.step_RK()

        if (i_time % n_plot == 0) or (par.timefin - par.timenow) < 1e-12:
            print("phys time = ", par.timenow)
            print('num of timesteps = ', i_time)
            plotting(grid, var_to_plot, par.timenow, line, ax, fig, im)

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
    IC_diff2D_cross,
    IC_diff2D_ring,
    IC_diff1D_gaussian,
    IC_diff1D_step,
    IC_diff2D_cyl,
    IC_diff1D_sine,
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
    IC_HD2D_shock_cloud,
    IC_HD2D_gap_opening,
    IC_HD2D_jet_cyl,
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
    IC_rMHD_user_defined,
)
from src.models.SWE.SWE_init_cond import (
    IC_SWE1D_dam,
    IC_SWE2D_bathtub,
    IC_SWE2D_expl,
    IC_SWE2D_tsunami,
    IC_SWE2D_ocean,
    IC_SWE2D_atmo,
    IC_SWE2D_dam,
    IC_SWE2D_bickley,
    IC_SWE2D_KH,
    IC_SWE1D_bump,
    IC_SWE_user_defined,
)


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

    # ── Dispatch dictionaries ─────────────────────────────────────────────

    diff_dispatch = {
        "gauss2D":      IC_diff2D_gaussian,
        "cross2D":      IC_diff2D_cross,
        "ring2D":       IC_diff2D_ring,
        "gauss1D":      IC_diff1D_gaussian,
        "step1D":       IC_diff1D_step,
        "sine1D":       IC_diff1D_sine,
        "cyl2D":        IC_diff2D_cyl,
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
        "shock-cloud":  IC_HD2D_shock_cloud,
        "gap-opening":  IC_HD2D_gap_opening,
        "jet2Dcyl":     IC_HD2D_jet_cyl,
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
        "user_defined": IC_rMHD_user_defined,
    }

    swe_dispatch = {
        "dam1D":        IC_SWE1D_dam,
        "bump1D":       IC_SWE1D_bump,
        "bathtub2D":    IC_SWE2D_bathtub,
        "expl2D":       IC_SWE2D_expl,
        "tsunami2D":    IC_SWE2D_tsunami,
        "ocean2D":      IC_SWE2D_ocean,
        "atmo2D":       IC_SWE2D_atmo,
        "dam2D":        IC_SWE2D_dam,
        "jet2D":        IC_SWE2D_bickley,
        "KHI2D":        IC_SWE2D_KH,
        "user_defined": IC_SWE_user_defined,
    }

    # ── Mode selection ────────────────────────────────────────────────────

    if par.mode == "diff":
        try:
            grid, state, par = diff_dispatch[par.problem](grid, state, par)
            eos = None
        except KeyError:
            raise ValueError(
                f"Invalid diffusion problem '{par.problem}'. "
                f"Available: {list(diff_dispatch.keys())}")

    elif par.mode == "adv":
        try:
            grid, state, par = adv_dispatch[par.problem](grid, state, par)
            eos = None
        except KeyError:
            raise ValueError(
                f"Invalid advection problem '{par.problem}'. "
                f"Available: {list(adv_dispatch.keys())}")

    elif par.mode == "HD":
        try:
            grid, state, par, eos = hd_dispatch[par.problem](grid, state, par)
        except KeyError:
            raise ValueError(
                f"Invalid HD problem '{par.problem}'. "
                f"Available: {list(hd_dispatch.keys())}")

    elif par.mode == "MHD":
        try:
            grid, state, par, eos = mhd_dispatch[par.problem](grid, state, par)
        except KeyError:
            raise ValueError(
                f"Invalid MHD problem '{par.problem}'. "
                f"Available: {list(mhd_dispatch.keys())}")

    elif par.mode == "rHD":
        try:
            grid, state, par, eos = rhd_dispatch[par.problem](grid, state, par)
        except KeyError:
            raise ValueError(
                f"Invalid rHD problem '{par.problem}'. "
                f"Available: {list(rhd_dispatch.keys())}")

    elif par.mode == "rMHD":
        try:
            grid, state, par, eos = rmhd_dispatch[par.problem](grid, state, par)
        except KeyError:
            raise ValueError(
                f"Invalid rMHD problem '{par.problem}'. "
                f"Available: {list(rmhd_dispatch.keys())}")

    elif par.mode == "SWE":
        try:
            grid, state, par, eos = swe_dispatch[par.problem](grid, state, par)
            eos = None 
        except KeyError:
            raise ValueError(
                f"Invalid SWE problem '{par.problem}'. "
                f"Available: {list(swe_dispatch.keys())}")

    else:
        raise ValueError(
            f"Invalid simulation mode '{par.mode}'. "
            f"Expected one of ['adv', 'HD', 'rHD', 'MHD', 'rMHD', 'diff', 'SWE'].")

    return grid, state, par, eos
