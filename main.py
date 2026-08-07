# -*- coding: utf-8 -*-
"""
main.py

Main driver for Piastra simulations.

This script handles, end to end:
    - parameter setup and validation (Parameters)
    - grid construction (Grid)
    - initial-condition / test-problem selection (initial_model)
    - solver selection (SOLVER_DISPATCH)
    - the time-integration loop with live visualization (run_simulation)

To run a different case, edit the Parameters block inside main() and execute
`python main.py`. The notebook main.ipynb mirrors this file cell by cell.

Available modes
---------------
    'adv'  : linear scalar advection
    'HD'   : compressible (Euler) hydrodynamics
    'rHD'  : special-relativistic hydrodynamics
    'MHD'  : ideal magnetohydrodynamics
    'rMHD' : special-relativistic magnetohydrodynamics
    'SWE'  : shallow-water equations
    'diff' : 2D thermal diffusion

Available test problems (pass as `problem`; 'user_defined' exists for every mode)
--------------------------------------------------------------------------------
The authoritative mapping from a problem name to its initial-condition function
lives in the dispatch dictionaries of src/misc/helpers.py (initial_model).

    adv  (see src/models/adv/adv_init_cond.py):
        smooth1D, disc1D, smooth2D, disc2D

    HD   (see src/models/HD/HD_init_cond.py):
        sod1Dcart, sod1Dcyl, sod1Dsph, strong1D, DBW1D, shuosher1D, einfeldt1D,
        sod2Dcart, sod2Dsph, sod2Dpol, sedov2Dcart, sedov2Dcyl, RP2D, gresho2D,
        KHI2D, RTI2D, shock-cloud, gap-opening, jet2Dcyl,
        collapse1D, jeans2D, collapse2D          (self-gravitating)

    rHD  (see src/models/rHD/rHD_init_cond.py):
        RP1, RP3, RP4, RP5, RP2D, RTI, jet2Dcart, jet2Dcyl

    MHD  (see src/models/MHD/MHD_init_cond.py):
        BW1D, toth1D, RJ1D, alfven1D, blast2Dcart, blast2Dcyl, blast2Dsph,
        rotor2D, OT2D, current-sheet, field-loop, disk2D, shock-cloud

    rMHD (see src/models/rMHD/rMHD_init_cond.py):
        BW1D, RP2, RP3, RP4, blast2D, rotor2D

    SWE  (see src/models/SWE/SWE_init_cond.py):
        dam1D, bump1D, bathtub2D, expl2D, tsunami2D, ocean2D, atmo2D, dam2D,
        jet2D, KHI2D

    diff (see src/models/diff/diff_init_cond.py):
        gauss1D, gauss2D, step1D, sine1D, cross2D, ring2D, cyl2D

Parameters (Parameters class; required vs optional)
---------------------------------------------------
all modes:
    required:
        mode    = str    -- one of the modes listed above
        problem = str    -- one of the problem names listed above
        Nx1, Nx2 = int   -- grid resolution
    optional:
        CFL      = float < 1     (default 0.7; auto-capped at 0.4 for rHD/rMHD)
        rec_type = 'PCM', 'PLM', 'PPMorig', 'PPM', 'WENO', 'MP5'  (default 'PLM')
        RK_order = 'RK1', 'RK2', 'RK3'                            (default 'RK2')

per-mode solver options:
    'adv'  : solver_type = 'adv', 'LW'
    'HD'   : solver_type = 'LLF', 'HLL', 'HLLC', 'Roe', 'Exact'
    'rHD'  : solver_type = 'LLF', 'HLL', 'HLLC'
    'MHD'  : solver_type = 'LLF', 'HLL', 'HLLC', 'HLLD'
             divb_tr     = 'CT', 'GLM', '8wave'
    'rMHD' : solver_type = 'LLF', 'HLL'
             divb_tr     = 'CT' (only CT is supported)
    'SWE'  : solver_type = 'LLF', 'HLL', 'Exact'
    'diff' : solver_type = 'expl', 'rkl2'
             rkl2_stages = int >= 2   (only used by 'rkl2')

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
# Maps a mode string to a callable that builds the corresponding solver object.
# For MHD the choice of divergence-control scheme is resolved from par.divb_tr.
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
    """
    Configure and run a single Piastra simulation.

    Steps performed:
        1. build a Parameters object (edit this block to change the run);
        2. construct the Grid and allocate the SimState;
        3. load the chosen test problem via initial_model, which sets the grid
           geometry, primitive variables, boundary conditions, final time, and
           equation of state;
        4. instantiate the matching solver through SOLVER_DISPATCH;
        5. advance in time with run_simulation, plotting every nsteps_visual
           steps;
        6. for MHD/rMHD in 2D, optionally display the final magnetic-field
           divergence as a diagnostic.

    Returns
    -------
    None
    """
    # --- Define main simulation parameters ---
    par = Parameters(
        mode="HD",
        Nx1=64,
        Nx2=64,
        problem="KHI2D",
        solver_type='HLLC',
        # timestep
        CFL=0.7,
        rec_type='PPM',
        RK_order='RK3',
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
    if (par.mode == "MHD" or par.mode == "rMHD") and (par.Nx1 > 1) and (par.Nx2 > 1):
        divB = np.zeros(grid.grid_shape, dtype=np.double)
        divB[grid.Ngc:grid.Nx1r, grid.Ngc:grid.Nx2r] = state.divB
        line, ax, fig, im = plot_setup(grid, divB, par.timenow)
        # plotting(grid, divB, par.timenow, line, ax, fig, im)


if __name__ == "__main__":
    main()
