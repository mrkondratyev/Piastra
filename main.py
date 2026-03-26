# -*- coding: utf-8 -*-
"""
main.py

Main driver for advection/fluid/MHD/rHD/diffusion simulations.

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
- 'rHD'  : Special-relativistic hydrodynamics problems
- 'MHD'  : Magnetohydrodynamics problems
- 'rMHD' : Special-relativistic magnetohydrodynamics problems
- 'diff' : 2D thermal diffusion (explicit or RKL2 super time-stepping)

Available problems (examples):
------------------------------
Advection (see advection_init_cond.py):
    - "smooth1D", "disc1D", "smooth2D", "disc2D", "user_defined"
Hydrodynamics (see hydro_init_cond.py):
    - "sod1Dcart", "sod1Dcyl", "sod1Dpol", "strong1D", "DBW1D",
    - "KHI", "RTI", "sod2Dcart", "sedov2Dcart", "sedov2Dcyl",
    - "user_defined"
Relativistic HD (see rHD_init_cond.py):
    - "RP1"  : Mignone & Bodo (2005) RP1 (moving fluid, v=0.9)
    - "RP3"  : strong relativistic shock
    - "RP4"  : ultra-relativistic blast wave (p=1000)
    - "RP5"  : tangential velocity test
    - "RP2D" : 2D relativistic Riemann problem
    - "RTI"  : relativistic Rayleigh-Taylor instability
    - "user_defined"
Magnetohydrodynamics (see MHD_init_cond.py):
    - "BW1D", "toth1D", "blast-cart", "blast-cyl", "OT2D", "user_defined"
Relativistic MHD (see rMHD_init_cond.py):
    - "blast1D" : 1D relativistic MHD blast wave (Mignone & Bodo 2006)
    - "rotor2D" : 2D relativistic rotor (Del Zanna et al. 2003)
    - "user_defined"
Diffusion (see diffusion_init_cond.py):
    - "gauss2D", "cross2D", "ring2D", "gauss1D", "user_defined"

Parameters (in Parameters class):
--------------------------------
- mode       : str   -- Simulation type ('adv', 'HD', 'rHD', 'MHD', 'rMHD', 'diff')
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

'rMHD' :
    flux_type = 'LLF', 'HLL'
    divb_tr = 'CT' (only CT is supported)
    CFL = float < 1

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
from rMHD_one_step import rMHD2D_CT
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
    "rMHD": lambda grid, state, eos, par: rMHD2D_CT(grid, state, eos, par),
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
