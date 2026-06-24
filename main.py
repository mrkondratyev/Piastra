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
- 'SWE'  : Shallow water problems  
- 'HD'   : Hydrodynamics problems
- 'rHD'  : Special-relativistic hydrodynamics problems
- 'MHD'  : Magnetohydrodynamics problems
- 'rMHD' : Special-relativistic magnetohydrodynamics problems
- 'diff' : 2D thermal diffusion

Available problems (examples, see models and src/common/helper.py for more):
------------------------------
Advection (see adv_init_cond.py):
    - "smooth1D", "disc1D", "smooth2D", "disc2D", "user_defined"
Shallow water (see SWE_init_cond.py):
    - "BW1D", "toth1D", "blast-cart", "blast-cyl", "OT2D", "rotor2D"
    - "user_defined"
Hydrodynamics (see HD_init_cond.py):
    - "sod1Dcart", "sod1Dcyl", "sod1Dpol", "strong1D", "DBW1D",
    - "KHI", "RTI", "sod2Dcart", "sedov2Dcart", "sedov2Dcyl"
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
    - "BW1D", "toth1D", "blast-cart", "blast-cyl", "OT2D", "rotor2D"
    - "user_defined"
Relativistic MHD (see rMHD_init_cond.py):
    - "BW1D", "RP2", "RP3", "RP4", "blast2D", "rotor2D" (Mignone & Bodo 2006)
    - "user_defined"
Diffusion (see diff_init_cond.py):
    - "gauss2D", "cross2D", "ring2D", "gauss1D"
    - "user_defined"

Parameters (in Parameters class):
--------------------------------
- mode       : str   -- Simulation type ('adv', 'HD', 'rHD', 'MHD', 'rMHD', 'diff')
- problem    : str   -- Problem name (depends on mode)
- Nx1, Nx2   : int   -- Grid resolution
- solver_type: str   -- Solver type ('adv', 'HLLC', 'HLLD', 'rkl2', etc.)
- divb_tr    : str   -- Divergence cleaning method ('CT', '8wave', 'GLM' -- MHD ONLY)
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
        rec_type = 'PLM', 'PPM', 'PCM', 'PPMorig', 'WENO', 'MP5'
        RK_order = 'RK1', 'RK2', 'RK3'
    
'adv' : 
    solver_type = 'adv', 'LW'
    
'HD' : 
    solver_type = 'LLF', 'HLL', 'HLLC', 'Roe', 'Exact'
    
'SWE' : 
    solver_type = 'LLF', 'HLL', 'Exact'
    
'MHD' :
    solver_type = 'LLF', 'HLL', 'HLLC', 'HLLD'
    divb_tr = 'CT', GLM', '8wave'

'rMHD' :
    solver_type = 'LLF', 'HLL'
    divb_tr = 'CT' (only CT is supported)

'diff' :
    solver_type = 'expl', 'rkl2'
    rkl2_stages = integer >= 2   (only for rkl2)

Author: mrkondratyev
"""

import matplotlib.pyplot as plt
import numpy as np

from src.grid.grid_setup import Grid
from src.sim_state import SimState
from src.parameters import Parameters
from src.models.MHD.MHD_step_CT import MHD2D_CT
from src.models.MHD.MHD_step_8wave import MHD2D_8wave
from src.models.MHD.MHD_step_GLM import MHD2D_GLM
from src.models.HD.HD_step import HD2D
from src.models.rHD.rHD_step import rHD2D
from src.models.rMHD.rMHD_step import rMHD2D_CT
from src.models.adv.adv_step import Adv2D
from src.models.SWE.SWE_step import SWE2D
from src.models.diff.diff_step import Diff2D
from src.misc.helpers import run_simulation, initial_model
from src.misc.io_visual import plot_setup, plotting 


# --- Solver dispatch dictionary ---
SOLVER_DISPATCH = {
    "adv":  lambda grid, state, eos, par: Adv2D(grid, state, par),
    "SWE":  lambda grid, state, eos, par: SWE2D(grid, state, par),
    "HD":   lambda grid, state, eos, par: HD2D(grid, state, eos, par),
    "rHD":  lambda grid, state, eos, par: rHD2D(grid, state, eos, par),
    "MHD":  lambda grid, state, eos, par: (
        MHD2D_CT(grid, state, eos, par) if par.divb_tr == "CT" else
        MHD2D_GLM(grid, state, eos, par) if par.divb_tr == "GLM" else
        MHD2D_8wave(grid, state, eos, par)),
    "rMHD": lambda grid, state, eos, par: rMHD2D_CT(grid, state, eos, par),
    "diff": lambda grid, state, eos, par: Diff2D(grid, state, par),
}


def main():
    """Main driver function for the simulation."""

    # --- Define main simulation parameters ---
    par = Parameters(
        mode="HD",
        Nx1=64,
        Nx2=64, 
        problem= "KHI2D",
        solver_type = 'HLLC',
        #timestep 
        CFL=0.7,
        rec_type = 'PPM',
        RK_order = 'RK3',
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
    if par.mode == "diff":
        var_to_plot = state.T 
    elif par.mode == "SWE":
        var_to_plot = state.h
    else:
        var_to_plot = state.dens

    # --- Run simulation ---
    nsteps_visual = 4
    state, par.timenow = run_simulation(
        grid, state, par, solver, var_to_plot, nsteps_visual
    )

    # --- Final visualization of B-divergence for MHD (optional) ---
    if (par.mode == "MHD" or par.mode == "rMHD") & (par.Nx1 > 1) & (par.Nx2 > 1):
        divB = np.zeros(grid.grid_shape, dtype=np.double)
        divB[grid.Ngc:grid.Nx1r, grid.Ngc:grid.Nx2r] = state.divB
        line, ax, fig, im = plot_setup(grid, divB, par.timenow)
        #plotting(grid, divB, par.timenow, line, ax, fig, im)

if __name__ == "__main__":
    main()
