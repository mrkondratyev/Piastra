# -*- coding: utf-8 -*-
"""
main.py

Main driver for advection/fluid/MHD simulations.

This script handles:
- Grid construction
- Initial condition setup
- Solver selection
- Simulation control loop
- Optional visualization

Available modes:
---------------
- 'adv' : Linear advection problems
- 'HD'  : Hydrodynamics problems
- 'MHD' : Magnetohydrodynamics problems

Available problems (examples):
------------------------------
Advection (see advection_init_cond.py):
    - "smooth1D": IC_advection1D_smooth,
    - "disc1D": IC_advection1D_disc,
    - "smooth2D": IC_advection2D_smooth,
    - "disc2D": IC_advection2D_disc,
    - "user_defined": IC_advection_user_defined,
Hydrodynamics (see hydro_init_cond.py):
    - "sod1Dcart": lambda g, s, p: IC_hydro1D_Sod(g, s, p, "cart"),
    - "sod1Dcyl": lambda g, s, p: IC_hydro1D_Sod(g, s, p, "cyl"),
    - "sod1Dpol": lambda g, s, p: IC_hydro1D_Sod(g, s, p, "pol"),
    - "strong1D": IC_hydro1D_strong_shock,
    - "DBW1D": IC_hydro1D_DBW,
    - "KHI": IC_hydro2D_KHI,
    - "RTI": IC_hydro2D_RTI,
    - "sod2Dcart": IC_hydro2D_Sod,
    - "sedov2Dcart": IC_hydro2D_Sedov_cart,
    - "sedov2Dcyl": IC_hydro2D_Sedov_cyl,
    - "user_defined": IC_hydro_user_defined. 
Magnetohydrodynamics (see MHD_init_cond.py):
    - "BW1D": IC_MHD1D_BW,
    - "toth1D": IC_MHD1D_Toth,
    - "blast-cart": IC_MHD2D_blast_cart,
    - "blast-cyl": IC_MHD2D_blast_cyl,
    - "OT2D": IC_MHD2D_OT,
    - "user_defined": IC_MHD_user_defined,

Parameters (in Parameters class):
--------------------------------
- mode       : str   -- Simulation type ('adv', 'HD', 'MHD')
- problem    : str   -- Problem name (depends on mode)
- Nx1, Nx2   : int   -- Grid resolution
- flux_type  : str   -- Flux solver type ('adv', 'HLLC', 'HLLD', etc.)
- divb_tr    : str   -- Divergence cleaning method ('CT', '8wave' -- MHD ONLY)
- rec_type   : str   -- Reconstruction method ('PLM', 'PPM', 'WENO', etc.)
- RK_order   : str   -- Runge-Kutta integration order ('RK1', 'RK2', 'RK3')
- CFL        : float -- CFL stability number
- timefin    : float -- Final physical time
- timenow    : float -- Current physical time

available parameters:
--------------------------------

all modes :
    required :
        mode = str
        problem = str
        Nx1, Nx2 = integers
    optional :
        CFL = double < 1
        rec_type = 'PLM', 'PPM', 'PCM', 'PPMorig', 'WENO'
        RK_order = 'RK1', 'RK2', 'RK3'
    
'adv' : 
    flux_type = 'adv', 'LW'
    
'HD' : 
    flux_type = 'LLF', 'HLL', 'HLLC', 'Roe'
    
'MHD' : 
    flux_type = 'LLF', 'HLL', 'HLLD'
    divb_tr = 'CT', '8wave'
    CFL = integer < 1
    
Author: mrkondratyev
"""

import matplotlib.pyplot as plt
import numpy as np

from grid_setup import Grid
from sim_state import SimState
from parameters import Parameters
from MHD_one_step_CT import MHD2D_CT
from MHD_one_step_8wave import MHD2D_8wave
from hydro_one_step import Hydro2D
from advection_one_step import Advection2D
from helpers import run_simulation, initial_model
from visualization import plot_setup


# --- Solver dispatch dictionary ---
SOLVER_DISPATCH = {
    "adv": lambda grid, state, eos, par: Advection2D(grid, state, par),
    "HD":  lambda grid, state, eos, par: Hydro2D(grid, state, eos, par),
    "MHD": lambda grid, state, eos, par: (
        MHD2D_CT(grid, state, eos, par)
        if par.divb_tr == "CT" else
        MHD2D_8wave(grid, state, eos, par)
    ),
}


def main():
    """Main driver function for the simulation."""
    
    # --- Define main simulation parameters ---
    par = Parameters(
        mode="adv",
        problem="smooth2D",
        Nx1=128,
        Nx2=128,
        flux_type="adv",
        divb_tr="CT",
    )

    # --- Initialize grid and state ---
    grid = Grid(par.Nx1, par.Nx2, par.Ngc)
    print(par)  # show setup
    simstate = SimState(grid, par)
    grid, simstate, par, eos = initial_model(grid, simstate, par)

    # --- Select solver ---
    solver = SOLVER_DISPATCH[par.mode](grid, simstate, eos, par)

    # --- Run simulation ---
    nsteps_visual = 10
    simstate, par.timenow = run_simulation(
        grid, simstate, par, solver, simstate.dens, nsteps_visual
    )

    # --- Visualization (optional) ---
    if par.mode == "MHD":
        line, ax, fig, im = plot_setup(grid, simstate.divB, par.timenow)
        plt.show()


if __name__ == "__main__":
    main()
