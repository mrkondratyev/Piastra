# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 14:18:25 2025

@author: mrkondratyev
"""

import time
from visualization import plot_setup, plotting


def run_simulation(grid, state, par, solver, var_to_plot, n_plot):
    """
    Advance the numerical simulation in time.

    This function runs the main time-integration loop. It uses the provided
    solver object to advance the system, produces periodic plots of the
    selected variable, and reports timing information.

    Parameters
    ----------
    grid : object
        Grid object containing geometry and resolution information.
    state : object
        State container for the physical variables.
    par : object
        Parameters object, must include:
        - mode      : problem type ('adv', 'HD', 'MHD')
        - rec_type  : reconstruction type
        - RK_order  : Runge–Kutta order
        - timenow   : current simulation time
        - timefin   : final simulation time
    solver : object
        Numerical solver providing the method `step_RK()`.
    var_to_plot : str
        Variable name to visualize during the run (e.g. "density").
    n_plot : int
        Interval (in timesteps) between visualization updates.

    Returns
    -------
    state : object
        Updated simulation state at final time.
    timenow : float
        Final physical time reached by the simulation.
    """
    #print solver parameters
    print("numerical model = ", par.mode)
    print("grid resolution = ", grid.Nx1, grid.Nx2)
    print("reconstruction type = ", par.rec_type)
    print("Temporal integration = ", par.RK_order)
    print("final phys time = ", par.timefin)    
    
    #plot setup
    line, ax, fig, im = plot_setup(grid, var_to_plot, par.timenow)
    
    print("START OF SIMULATION")
    
    #set the start timer to check the elapsed time 
    start_time1 = time.time() 
    #cycle over time 
    i_time = 0
    while par.timenow < par.timefin:
           
        #current timestep
        i_time = i_time + 1
        
        #advected variable update 
        state = solver.step_RK() 
        
        #"real time" output (animated)
        if (i_time % n_plot == 0) or (par.timefin - par.timenow) < 1e-12:
            
            print("phys time = ", par.timenow)
            print('num of timesteps = ', i_time)
    
            plotting(grid, var_to_plot, par.timenow, line, ax, fig, im)
     
    #print final physical time
    print("final phys time = ", par.timenow)    
    
    print("END OF SIMULATION")
    ##calculate and the elapsed time of the simulation
    end_time1 = time.time()
    print("elapsed time = ", end_time1 - start_time1, " secs")
    
    return state, par.timenow




from advection_init_cond import (
    IC_advection1D_smooth,
    IC_advection1D_disc,
    IC_advection2D_smooth,
    IC_advection2D_disc,
    IC_advection_user_defined,
)
from hydro_init_cond import (
    IC_hydro1D_Sod,
    IC_hydro1D_strong_shock,
    IC_hydro1D_DBW,
    IC_hydro2D_KHI,
    IC_hydro2D_RTI,
    IC_hydro2D_Sod,
    IC_hydro2D_Sedov_cart,
    IC_hydro2D_Sedov_cyl,
    IC_hydro_user_defined,
)
from MHD_init_cond import (
    IC_MHD1D_BW,
    IC_MHD1D_Toth,
    IC_MHD2D_blast_cart,
    IC_MHD2D_blast_cyl,
    IC_MHD2D_OT,
    IC_MHD_user_defined,
    
)


def initial_model(grid, state, par):
    """
    Initialize the chosen test problem based on simulation mode and problem name.

    Parameters
    ----------
    grid : object
        Grid object containing mesh geometry and metric information.
    state : object
        Simulation state object (e.g., Advection, Fluid2D, MHD2D).
    par : object
        Parameters object containing simulation settings, including
        mode ('adv', 'HD', 'MHD') and problem name.

    Returns
    -------
    tuple
        Depending on the mode:
        - (grid, state, par) for Advection
        - (grid, state, par, eos) for HD and MHD
    """

    # --- dispatch dictionaries ---
    adv_dispatch = {
        "smooth1D": IC_advection1D_smooth,
        "disc1D": IC_advection1D_disc,
        "smooth2D": IC_advection2D_smooth,
        "disc2D": IC_advection2D_disc,
        "user_defined": IC_advection_user_defined,
    }

    hd_dispatch = {
        "sod1Dcart": lambda g, s, p: IC_hydro1D_Sod(g, s, p, "cart"),
        "sod1Dcyl": lambda g, s, p: IC_hydro1D_Sod(g, s, p, "cyl"),
        "sod1Dpol": lambda g, s, p: IC_hydro1D_Sod(g, s, p, "pol"),
        "strong1D": IC_hydro1D_strong_shock,
        "DBW1D": IC_hydro1D_DBW,
        "KHI": IC_hydro2D_KHI,
        "RTI": IC_hydro2D_RTI,
        "sod2Dcart": IC_hydro2D_Sod,
        "sedov2Dcart": IC_hydro2D_Sedov_cart,
        "sedov2Dcyl": IC_hydro2D_Sedov_cyl,
        "user_defined": IC_hydro_user_defined,
    }

    mhd_dispatch = {
        "BW1D": IC_MHD1D_BW,
        "toth1D": IC_MHD1D_Toth,
        "blast-cart": IC_MHD2D_blast_cart,
        "blast-cyl": IC_MHD2D_blast_cyl,
        "OT2D": IC_MHD2D_OT,
        "user_defined": IC_MHD_user_defined,
    }

    # --- mode selection ---
    if par.mode == "adv":
        try:
            grid, state, par = adv_dispatch[par.problem](grid, state, par)
            eos = None 
        except KeyError:
            raise ValueError(
                f"Invalid advection problem '{par.problem}'. "
                f"Available: {list(adv_dispatch.keys())}"
            )

    elif par.mode == "HD":
        try:
            grid, state, par, eos = hd_dispatch[par.problem](grid, state, par)
        except KeyError:
            raise ValueError(
                f"Invalid hydro problem '{par.problem}'. "
                f"Available: {list(hd_dispatch.keys())}"
            )

    elif par.mode == "MHD":
        try:
            grid, state, par, eos = mhd_dispatch[par.problem](grid, state, par)
        except KeyError:
            raise ValueError(
                f"Invalid MHD problem '{par.problem}'. "
                f"Available: {list(mhd_dispatch.keys())}"
            )
    else:
        raise ValueError(
            f"Invalid simulation mode '{par.mode}'. Expected one of ['adv', 'HD', 'MHD']."
        )
        
    return grid, state, par, eos
