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
- 'adv'  : Linear advection problems
- 'HD'   : Hydrodynamics problems
- 'MHD'  : Magnetohydrodynamics problems
- 'diff' : 2D thermal diffusion (explicit or RKL2 super time-stepping)

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
Diffusion (see diffusion_init_cond.py):
    - "gauss2D":      IC_diffusion2D_gaussian  (2D Gaussian pulse, Cartesian)
    - "cross2D":      IC_diffusion2D_cross      (crossed Gaussian ridges, Cartesian)
    - "ring2D":       IC_diffusion2D_ring       (annular hot ring, Cartesian)
    - "gauss1D":      IC_diffusion1D_gaussian   (1D Gaussian, Nx2=1, Cartesian)
    - "user_defined": IC_diffusion_user_defined

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

'diff' :
    diff_solver = 'expl', 'rkl2'
    rkl2_stages = integer >= 2   (only for rkl2)
    CFL = float < 1

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
from rHD_one_step import rHD2D
from advection_one_step import Advection2D
from diffusion_one_step import Diffusion2D
from helpers import run_simulation, initial_model
from visualization import plot_setup


# --- Solver dispatch dictionary ---
SOLVER_DISPATCH = {
    "adv":  lambda grid, state, eos, par: Advection2D(grid, state, par),
    "HD":   lambda grid, state, eos, par: Hydro2D(grid, state, eos, par),
    "rHD":  lambda grid, state, eos, par: rHD2D(grid, state, eos, par),
    "MHD":  lambda grid, state, eos, par: (
        MHD2D_CT(grid, state, eos, par)
        if par.divb_tr == "CT" else
        MHD2D_8wave(grid, state, eos, par)
    ),
    "diff": lambda grid, state, eos, par: Diffusion2D(
        grid, state, par,
        solver=par.diff_solver,
        rkl2_stages=par.rkl2_stages,
    ),
}


def main():
    """Main driver function for the simulation."""

    # --- Define main simulation parameters ---
    par = Parameters(
        mode="diff",
        problem="gauss2D",
        Nx1=128,
        Nx2=128,
        diff_solver="rkl2",
        rkl2_stages=20,
        CFL=0.9,
    )

    # --- Initialize grid and state ---
    grid = Grid(par.Nx1, par.Nx2, par.Ngc)
    print(par)  # show setup

    # State object (unified SimState for all modes)
    state = SimState(grid, par)

    grid, state, par, eos = initial_model(grid, state, par)

    # --- Select solver ---
    solver = SOLVER_DISPATCH[par.mode](grid, state, eos, par)

    # --- Variable to visualise ---
    var_to_plot = state.T if par.mode == "diff" else state.dens



    # --- Run simulation ---
    nsteps_visual = 10
    state, par.timenow = run_simulation(
        grid, state, par, solver, var_to_plot, nsteps_visual
    )

    # --- Final visualization (optional) ---
    if par.mode == "MHD":
        line, ax, fig, im = plot_setup(grid, state.divB, par.timenow)
        plt.show()


if __name__ == "__main__":
    main()
