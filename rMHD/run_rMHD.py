#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rMHD.py
===========

Standalone driver for the Special-Relativistic MHD (SRMHD) solver.

Usage
-----
    python run_rMHD.py

This script is intentionally self-contained so the rMHD/ folder can be
run in isolation from the rest of the Piastra tree (the parent directory
is added to sys.path automatically by the module imports).

Workflow
--------
  1.  Choose a problem by setting `PROBLEM` below.
  2.  Set numerical parameters (resolution, flux type, reconstruction, …).
  3.  Run the script; results are plotted at the end.

Available problems (rMHD_init_cond.py)
---------------------------------------
  'blast1D' : 1D relativistic MHD blast wave  (Mignone & Bodo 2006)
  'rotor2D' : 2D relativistic rotor           (Del Zanna et al. 2003)
  'user'    : user-defined initial conditions

Notes
-----
  cons2prim_sr_MHD in rMHD_phys.py is currently a stub.  The simulation
  will raise NotImplementedError at run-time until you implement the
  primitive-variable recovery (Newton-Raphson inversion).  See the
  docstring of that function for the recipe.

Author
------
mrkondratyev
"""

import os
import sys

# Add the parent Piastra directory to the path so we can import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from grid_setup import Grid
from sim_state import SimState
from eos_setup import EOSdata
from parameters import Parameters
from rMHD_one_step import rMHD2D_CT
from rMHD_init_cond import IC_rMHD1D_blast, IC_rMHD2D_rotor, IC_rMHD_user_defined

# ============================================================================
# USER SETTINGS
# ============================================================================

PROBLEM   = 'blast1D'   # 'blast1D', 'rotor2D', or 'user'

NX1       = 256         # grid cells in x1 direction
NX2       = 1           # grid cells in x2 direction (1 for 1D problems)
REC_TYPE  = 'PLM'       # 'PCM', 'PLM', 'PPM', 'WENO'
RK_ORDER  = 'RK2'       # 'RK1', 'RK2', 'RK3'
FLUX_TYPE = 'HLL'       # 'LLF' or 'HLL'  (HLLD not yet implemented)
CFL       = 0.4         # Courant number
GAMMA     = 4.0 / 3.0  # adiabatic index (relativistic gas: 4/3)
N_STEPS   = 1000        # maximum number of timesteps (safety limit)

# ============================================================================
# SETUP
# ============================================================================

# Build a minimal Parameters object (rMHD mode not registered in parameters.py,
# so we reuse 'MHD' settings but pass flux options manually).
# We set mode='MHD' only to get the correct ghost-cell count and BC array;
# the actual solver used is rMHD2D_CT.
par = Parameters(
    mode='MHD',
    problem=PROBLEM,
    Nx1=NX1,
    Nx2=NX2,
    rec_type=REC_TYPE,
    RK_order=RK_ORDER,
    flux_type=FLUX_TYPE,
    CFL=CFL,
    divb_tr='CT',
)
par.timefin = 0.0   # will be overwritten by IC

eos   = EOSdata(GAMMA)
grid  = Grid(NX1, NX2, par.Ngc)
state = SimState(grid, par)

# ---- initial conditions ---------------------------------------------------

_IC_MAP = {
    'blast1D': IC_rMHD1D_blast,
    'rotor2D': IC_rMHD2D_rotor,
    'user':    IC_rMHD_user_defined,
}

if PROBLEM not in _IC_MAP:
    raise ValueError(f"Unknown problem '{PROBLEM}'. Choose from {list(_IC_MAP)}.")

grid, state, par, eos = _IC_MAP[PROBLEM](grid, state, par, eos)

# ---- solver ---------------------------------------------------------------

solver = rMHD2D_CT(grid, state, eos, par)

print(f"rMHD solver: {PROBLEM}")
print(f"  Grid     : {NX1} x {NX2}  (Ngc = {par.Ngc})")
print(f"  Rec/RK   : {REC_TYPE} / {RK_ORDER}")
print(f"  Flux     : {FLUX_TYPE}")
print(f"  t_fin    : {par.timefin}")
print()

# ============================================================================
# TIME LOOP
# ============================================================================

step = 0
while par.timenow < par.timefin and step < N_STEPS:
    state = solver.step_RK()
    step += 1
    if step % 50 == 0:
        print(f"  step {step:5d}   t = {par.timenow:.4f}")

print(f"\nDone: {step} steps,  t_final = {par.timenow:.6f}")

# ============================================================================
# VISUALISATION
# ============================================================================

Ngc = grid.Ngc

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle(f"rMHD — {PROBLEM}   t = {par.timenow:.3f}")

fields = [
    (state.dens[Ngc:-Ngc, Ngc:-Ngc], r"Density $\rho$"),
    (state.pres[Ngc:-Ngc, Ngc:-Ngc], r"Pressure $p$"),
    (state.vel1[Ngc:-Ngc, Ngc:-Ngc], r"Velocity $v_x$"),
    (state.bfi2[Ngc:-Ngc, Ngc:-Ngc], r"Magnetic field $B_y$"),
]

for ax, (field, title) in zip(axes.flat, fields):
    if NX2 == 1:
        x = grid.cx1[Ngc:-Ngc, Ngc]
        ax.plot(x, field[:, 0])
    else:
        im = ax.imshow(
            field.T, origin='lower',
            extent=[grid.cx1[Ngc, Ngc], grid.cx1[-Ngc-1, Ngc],
                    grid.cx2[Ngc, Ngc], grid.cx2[Ngc, -Ngc-1]],
            aspect='auto',
        )
        plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(r"$x_1$")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "rMHD_result.png"),
            dpi=120, bbox_inches='tight')
plt.show()
print("Plot saved to rMHD/rMHD_result.png")
