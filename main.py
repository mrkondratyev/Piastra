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
    - data output (snapshots) and restart from a snapshot (io_utils)

To run a different case, edit the RUN CONTROL block and the Parameters block
inside main() and execute `python main.py`. The notebook main.ipynb mirrors
this file cell by cell.

Output and restart
------------------
Snapshots are compressed NumPy archives (.npz) written by io_utils.save_data.
One file holds everything: the full state (primitive, conservative, staggered
CT and GLM fields), the grid description, the scheme settings, the boundary
conditions, the EOS and the current time. The same file serves for restart
and for analysis.

    OUTPUT_DIR    folder for the files; None switches all output off.
    DT_OUTPUT     physical time between snapshots: snapshot k is written at
                  t = k * DT_OUTPUT exactly (run_simulation shortens the step
                  that would jump over it). None: only the initial and the
                  final snapshot.
    SAVE_ASCII_1D for 1D runs, also write each snapshot as a text table (.dat).

File names:  <OUTPUT_DIR>/<mode>_<problem>_<kkkk>.npz,  k = 0, 1, 2, ...
             <OUTPUT_DIR>/<mode>_<problem>_final.npz    (end of the run)
The numbering is based on t / DT_OUTPUT, so it continues after a restart.

To restart, set RESTART = True and RESTART_FILE to one of these files. The
Parameters block is then ignored: grid, state, scheme, BCs, EOS and time all
come from the file. par.before_step (gravity) is a Python function and cannot
be stored; restart_simulation rebuilds it by re-running the problem's initial
condition on scratch objects. To continue a finished run further, set
TIMEFIN_NEW. Resolution and reconstruction cannot be changed on restart
(the number of ghost cells depends on rec_type).

Reading a snapshot for analysis:

    from src.misc.io_utils import load_data
    d = load_data("output/HD_sod1Dcart_final.npz")
    d['timenow'], d['dens'], d['cx1']      # time, field (with ghosts), centres

Field arrays include ghost cells; d['cx1'], d['cx2'] are the interior cell
centres, so the interior of a field is d['dens'][Ngc:-Ngc, Ngc:-Ngc] with
Ngc = d['Ngc'].

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
is PROBLEMS in src/misc/helpers.py.

    adv  (see src/models/adv/adv_init_cond.py):
        smooth1D, disc1D, smooth2D, disc2D

    HD   (see src/models/HD/HD_init_cond.py):
        sod1Dcart, sod1Dcyl, sod1Dsph, strong1D, DBW1D, shuosher1D, einfeldt1D,
        sod2Dcart, sod2Dsph, sod2Dpol, sedov2Dcart, sedov2Dcyl, RP2D, gresho2D,
        KHI2D, RTI2D, shock-cloud, gap-opening, jet2Dcyl, vortex2D, jeans1D, collapse_sph, merger2D

    rHD  (see src/models/rHD/rHD_init_cond.py):
        RP1, RP3, RP4, RP5, RP2D, RTI, jet2Dcart, jet2Dcyl

    MHD  (see src/models/MHD/MHD_init_cond.py):
        BW1D, toth1D, RJ1D, alfven1D, blast2Dcart, blast2Dcyl, blast2Dsph,
        rotor2D, OT2D, current-sheet, field-loop, disk2D, shock-cloud, alfven2D, jet2Dcyl

    rMHD (see src/models/rMHD/rMHD_init_cond.py):
        BW1D, RP2, RP3, RP4, blast2D, rotor2D, OT2D

    SWE  (see src/models/SWE/SWE_init_cond.py):
        dam1D, dam2D, rotdam2D, bathtub2D, kelvin2D, jet2D, KHI2D, lake2D

    diff (see src/models/diff/diff_init_cond.py):
        gauss2D, step1D, sine1D, cyl2D, gauss2Dpol, gauss2Dsph

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
    'adv'  : solver_type = 'adv', 'LW' (default 'adv')
    'HD'   : solver_type = 'LLF', 'HLL', 'HLLC', 'Roe', 'Exact' (default 'HLLC')
    'rHD'  : solver_type = 'LLF', 'HLL', 'HLLC' (default 'HLLC')
    'MHD'  : solver_type = 'LLF', 'HLL', 'HLLC', 'HLLD' (default 'HLLD')
             divb_tr     = 'CT', 'GLM', '8wave' (default 'GLM')
    'rMHD' : solver_type = 'LLF', 'HLL' (default 'HLL')
             divb_tr     = 'CT' (only CT is supported)
    'SWE'  : solver_type = 'LLF', 'HLL', 'Exact' (default 'HLL')
    'diff' : solver_type = 'expl', 'rkl2' (default 'rkl2')
             rkl2_stages = int >= 2   (only used by 'rkl2')

Author: mrkondratyev
"""

import os

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
from src.misc.io_utils import save_data, load_data, restart_simulation, save_1d_ascii


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


def snapshot_path(output_dir, par, tag):
    """
    File name (without extension) of a snapshot: <dir>/<mode>_<problem>_<tag>.
    `tag` is an int (snapshot number, printed as 0000, 0001, ...) or a str.
    """
    tag = f"{tag:04d}" if isinstance(tag, (int, np.integer)) else str(tag)
    return os.path.join(output_dir, f"{par.mode}_{par.problem}_{tag}")


def write_snapshot(output_dir, tag, grid, state, par, eos, ascii_1d=False):
    """
    Write one snapshot: always the .npz archive (restart + analysis); for a
    1D grid and ascii_1d=True also a text table of the profile (.dat).
    Returns the .npz path.
    """
    path = snapshot_path(output_dir, par, tag)
    npz = save_data(path, grid, state, par, eos)
    if ascii_1d and (grid.Nx1 == 1 or grid.Nx2 == 1):
        save_1d_ascii(path, grid, state, par, eos)
    return npz


def main():
    """
    Configure and run a single Piastra simulation.

    Steps performed:
        1. new run: build a Parameters object, the Grid and the SimState, and
           load the chosen test problem via initial_model (grid geometry,
           primitive variables, boundary conditions, final time, EOS);
           restart: rebuild all of this from a snapshot file;
        2. instantiate the matching solver through SOLVER_DISPATCH;
        3. advance in time with run_simulation, plotting every nsteps_visual
           steps and writing a snapshot every DT_OUTPUT of physical time;
        4. write the final snapshot and read it back (analysis example);
        5. for MHD/rMHD in 2D, optionally display the final magnetic-field
           divergence as a diagnostic.

    Returns
    -------
    None
    """
    # =====================================================================
    #   RUN CONTROL: output and restart
    # =====================================================================
    OUTPUT_DIR    = "output"   # folder for snapshots; None = write nothing
    DT_OUTPUT     = None       # physical time between snapshots;
                               # None = only initial + final snapshot
    SAVE_ASCII_1D = True       # 1D runs: also write each snapshot as .dat text

    RESTART       = False      # True = continue from RESTART_FILE
    RESTART_FILE  = "output/MHD_disk2D_final.npz"
    TIMEFIN_NEW   = None       # restart only: new final time (None = keep)

    if RESTART:
        # --- Restart: everything comes from the file; the Parameters block
        #     below is not used. before_step (gravity) is restored too. ---
        grid, state, par, eos = restart_simulation(RESTART_FILE)
        if TIMEFIN_NEW is not None:
            par.timefin = TIMEFIN_NEW
        # par.CFL = 0.5            # the CFL number may be changed here
        print(par)

    else:
        # --- Define main simulation parameters ---
        par = Parameters(
            mode="MHD",
            Nx1=100,
            Nx2=100,
            problem="disk2D",
            solver_type='HLLD',
            # timestep
            CFL=0.9,
            rkl2_stages = 8,
            rec_type='PLM',
            RK_order='RK2',
        )

        # --- Initialize grid and state ---
        grid = Grid(par.Nx1, par.Nx2, par.Ngc)
        print(par)  # show setup

        # State object (unified SimState for all modes)
        state = SimState(grid, par)

        grid, state, par, eos = initial_model(grid, state, par)

        # snapshot 0000 = the initial condition
        if OUTPUT_DIR is not None:
            write_snapshot(OUTPUT_DIR, 0, grid, state, par, eos, SAVE_ASCII_1D)

    # --- Select solver ---
    solver = SOLVER_DISPATCH[par.mode](grid, state, eos, par)

    # --- Variable to visualise ---
    # A function (re-evaluated before every plot), so derived quantities
    # such as the SWE free surface h + b work, and so does any solver that
    # rebinds the state arrays instead of updating them in place.
    if par.mode == "diff":
        var_to_plot = lambda: state.T
    elif par.mode == "SWE":
        var_to_plot = lambda: state.h + state.b     # free surface eta = h + b
    else:
        var_to_plot = lambda: state.dens

    # --- Periodic output: called by run_simulation as on_output(state, par, k) ---
    if OUTPUT_DIR is not None and DT_OUTPUT is not None:
        on_output = lambda st, p, k: write_snapshot(OUTPUT_DIR, k, grid, st, p,
                                                    eos, SAVE_ASCII_1D)
    else:
        on_output = None

    # --- Run simulation ---
    nsteps_visual = 15
    state, par.timenow = run_simulation(
        grid, state, par, solver, var_to_plot, nsteps_visual,
        dt_output=DT_OUTPUT, on_output=on_output
    )

    # --- Final snapshot (the natural restart point for extending the run) ---
    if OUTPUT_DIR is not None:
        final_file = write_snapshot(OUTPUT_DIR, "final", grid, state, par,
                                    eos, SAVE_ASCII_1D)

        # --- Analysis example: read the snapshot back ---
        d = load_data(final_file)
        name = "T" if par.mode == "diff" else "h" if par.mode == "SWE" else "dens"
        Ngc = d["Ngc"]
        f = d[name][Ngc:-Ngc, Ngc:-Ngc]          # interior cells only
        print(f"t = {d['timenow']:.4e}:  {name} in [{f.min():.4e}, {f.max():.4e}]")

    # --- Final visualization of B-divergence for MHD (optional) ---
    if (par.mode == "MHD" or par.mode == "rMHD") and (par.Nx1 > 1) and (par.Nx2 > 1):
        divB = np.zeros(grid.grid_shape, dtype=np.double)
        divB[grid.Ngc:grid.Nx1r, grid.Ngc:grid.Nx2r] = state.divB
        line, ax, fig, im = plot_setup(grid, divB, par.timenow)
        # plotting(grid, divB, par.timenow, line, ax, fig, im)


if __name__ == "__main__":
    main()
