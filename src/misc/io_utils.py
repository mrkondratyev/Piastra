# -*- coding: utf-8 -*-
"""
io_utils.py

Simple I/O utilities for the Piastra simulation framework.

Provides functions for saving and loading simulation snapshots using NumPy's
compressed .npz format.  A snapshot contains all primitive variables, grid
coordinates, simulation time, and mode-specific fields (magnetic fields,
temperature, etc.).

Supported modes: 'adv', 'HD', 'rHD', 'MHD', 'rMHD', 'diff'.

Usage
-----
Saving a snapshot after a simulation run::

    from io_utils import save_snapshot
    save_snapshot("output/blast_t04.npz", grid, state, par)

Loading a snapshot for post-processing::

    from io_utils import load_snapshot
    data = load_snapshot("output/blast_t04.npz")
    print(data["timenow"], data["mode"])
    rho = data["dens"]

Author: mrkondratyev
"""

import os
import numpy as np


def save_snapshot(filepath, grid, state, par, eos=None):
    """
    Save the current simulation state to a compressed NumPy archive (.npz).

    The archive stores grid coordinates, simulation parameters, and all
    primitive variables relevant to the current mode.

    Parameters
    ----------
    filepath : str
        Output file path (should end with '.npz').
    grid : Grid
        Grid object with cell-centre coordinates and resolution info.
    state : SimState
        State container with primitive (and optionally conservative) variables.
    par : Parameters
        Simulation parameters (mode, problem, timenow, timefin, etc.).
    eos : EOSdata, optional
        Equation of state.  If provided, GAMMA is stored in the archive.
    """
    Ngc = grid.Ngc

    # Grid coordinates (interior only)
    data = {
        "mode":    par.mode,
        "problem": par.problem,
        "timenow": par.timenow,
        "timefin": par.timefin,
        "Nx1":     grid.Nx1,
        "Nx2":     grid.Nx2,
        "Ngc":     Ngc,
        "cx1":     grid.cx1[Ngc:-Ngc, Ngc:-Ngc],
        "cx2":     grid.cx2[Ngc:-Ngc, Ngc:-Ngc],
    }

    if eos is not None:
        data["GAMMA"] = eos.GAMMA

    # --- Mode-specific fields ---

    if par.mode == "adv":
        data["dens"] = state.dens[Ngc:-Ngc, Ngc:-Ngc]

    elif par.mode in ("HD", "rHD"):
        data["dens"] = state.dens[Ngc:-Ngc, Ngc:-Ngc]
        data["vel1"] = state.vel1[Ngc:-Ngc, Ngc:-Ngc]
        data["vel2"] = state.vel2[Ngc:-Ngc, Ngc:-Ngc]
        data["vel3"] = state.vel3[Ngc:-Ngc, Ngc:-Ngc]
        data["pres"] = state.pres[Ngc:-Ngc, Ngc:-Ngc]

    elif par.mode in ("MHD", "rMHD"):
        data["dens"] = state.dens[Ngc:-Ngc, Ngc:-Ngc]
        data["vel1"] = state.vel1[Ngc:-Ngc, Ngc:-Ngc]
        data["vel2"] = state.vel2[Ngc:-Ngc, Ngc:-Ngc]
        data["vel3"] = state.vel3[Ngc:-Ngc, Ngc:-Ngc]
        data["pres"] = state.pres[Ngc:-Ngc, Ngc:-Ngc]
        data["bfi1"] = state.bfi1[Ngc:-Ngc, Ngc:-Ngc]
        data["bfi2"] = state.bfi2[Ngc:-Ngc, Ngc:-Ngc]
        data["bfi3"] = state.bfi3[Ngc:-Ngc, Ngc:-Ngc]
        data["fb1"]  = state.fb1
        data["fb2"]  = state.fb2
        data["divB"] = state.divB

    elif par.mode == "diff":
        data["T"]     = state.T[Ngc:-Ngc, Ngc:-Ngc]
        data["kappa"] = state.kappa

    # Create output directory if needed
    outdir = os.path.dirname(filepath)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    np.savez_compressed(filepath, **data)
    print(f"[io] snapshot saved -> {filepath}")


def load_snapshot(filepath):
    """
    Load a simulation snapshot from a NumPy archive (.npz).

    Parameters
    ----------
    filepath : str
        Path to the .npz file written by ``save_snapshot``.

    Returns
    -------
    data : dict
        Dictionary with all stored arrays and scalars.
        Scalar values (mode, timenow, GAMMA, ...) are extracted from their
        0-d arrays for convenience.
    """
    raw = np.load(filepath, allow_pickle=True)

    data = {}
    for key in raw.files:
        val = raw[key]
        # Convert 0-d arrays to plain Python scalars/strings
        if val.ndim == 0:
            data[key] = val.item()
        else:
            data[key] = val

    print(f"[io] snapshot loaded <- {filepath}  "
          f"(mode={data.get('mode')}, t={data.get('timenow')})")
    return data


def save_1d_ascii(filepath, grid, state, par, eos=None):
    """
    Save a 1D simulation profile to a plain-text ASCII file.

    Each row corresponds to one cell.  Columns depend on the mode:

    - adv  : x, dens
    - HD/rHD : x, dens, vel1, vel2, vel3, pres
    - MHD/rMHD : x, dens, vel1, vel2, vel3, pres, bfi1, bfi2, bfi3
    - diff : x, T

    Parameters
    ----------
    filepath : str
        Output file path (e.g. 'output/profile.dat').
    grid : Grid
    state : SimState
    par : Parameters
    eos : EOSdata, optional
    """
    Ngc = grid.Ngc

    if grid.Nx2 == 1:
        x = grid.cx1[Ngc:-Ngc, Ngc]
    elif grid.Nx1 == 1:
        x = grid.cx2[Ngc, Ngc:-Ngc]
    else:
        raise ValueError("save_1d_ascii: grid is 2D (Nx1 > 1 and Nx2 > 1). "
                         "Use save_snapshot for 2D data.")

    columns = [x]
    header_parts = ["x1"]

    if par.mode == "adv":
        columns.append(state.dens[Ngc:-Ngc, Ngc] if grid.Nx2 == 1
                       else state.dens[Ngc, Ngc:-Ngc])
        header_parts.append("dens")

    elif par.mode in ("HD", "rHD"):
        sl = (slice(Ngc, -Ngc), Ngc) if grid.Nx2 == 1 else (Ngc, slice(Ngc, -Ngc))
        for name in ("dens", "vel1", "vel2", "vel3", "pres"):
            columns.append(getattr(state, name)[sl])
            header_parts.append(name)

    elif par.mode in ("MHD", "rMHD"):
        sl = (slice(Ngc, -Ngc), Ngc) if grid.Nx2 == 1 else (Ngc, slice(Ngc, -Ngc))
        for name in ("dens", "vel1", "vel2", "vel3", "pres",
                      "bfi1", "bfi2", "bfi3"):
            columns.append(getattr(state, name)[sl])
            header_parts.append(name)

    elif par.mode == "diff":
        columns.append(state.T[Ngc:-Ngc, Ngc] if grid.Nx2 == 1
                       else state.T[Ngc, Ngc:-Ngc])
        header_parts.append("T")

    outdir = os.path.dirname(filepath)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    header = (f"Piastra {par.mode} | problem={par.problem} | "
              f"t={par.timenow:.8e}\n" + "  ".join(header_parts))
    np.savetxt(filepath, np.column_stack(columns), header=header, fmt="%.12e")
    print(f"[io] 1D ASCII saved -> {filepath}")
