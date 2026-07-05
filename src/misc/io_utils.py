# -*- coding: utf-8 -*-
"""
io_utils.py

Unified I/O for the Piastra framework: ONE writer and ONE reader that serve
both **restart** and **analysis**.

A single compressed NumPy archive (.npz) stores everything needed to continue a
run exactly *and* to post-process it offline:

  * run metadata        -- mode, problem, time, EOS, numerical scheme, BCs;
  * grid construction    -- geometry + domain bounds + resolution (the grid is
    rebuilt by re-running its deterministic constructor, so no metric arrays
    need to be stored; cell-centre coordinates are saved as a convenience for
    analysis);
  * the full simulation state -- every ndarray on the SimState object (primitive
    AND conservative AND staggered/GLM fields).  Dumping the whole state keeps
    the routine mode-agnostic and guarantees an exact restart, including the
    staggered CT fields (fb1/fb2) and the GLM scalar (bglm) that cannot be
    recovered from cell-centred primitives alone.

Public API
----------
save_data(filepath, grid, state, par, eos=None)
    Write one archive.  (Alias: ``save_snapshot``.)
load_data(filepath) -> dict
    Read one archive into a flat dict (scalars unwrapped).  Use directly for
    analysis: ``d = load_data(f); rho = d['dens']``.  (Alias: ``load_snapshot``.)
restart_simulation(filepath) -> (grid, state, par, eos)
    Rebuild live objects from an archive, ready to hand to a solver.
save_1d_ascii(filepath, grid, state, par, eos=None)
    Optional plain-text dump of a 1-D profile.

Restart uses ``reconstruct_grid`` from grid_setup.py to rebuild the grid.

Author: mrkondratyev
"""

import os
import numpy as np


# Keys that are run/grid metadata (everything else in the archive that matches
# a freshly-allocated SimState attribute is treated as a state field on reload).
_META_KEYS = {
    "mode", "problem", "geom", "Nx1", "Nx2", "Ngc",
    "x1ini", "x1fin", "x2ini", "x2fin",
    "timenow", "timefin", "CFL", "rec_type", "RK_order", "solver_type",
    "divb_tr", "rkl2_stages", "GAMMA", "BC", "BCm", "BC_fixed", "cx1", "cx2",
}

_NONE = "__none__"          # sentinel so Optional[str] params survive a round-trip


def _enc(x):
    """Encode a possibly-None string parameter for storage."""
    return _NONE if x is None else str(x)


def _dec(x):
    """Decode a stored string parameter back to str or None."""
    return None if (x is None or str(x) == _NONE) else str(x)


# ===========================================================================
#   Writer
# ===========================================================================
def save_data(filepath, grid, state, par, eos=None):
    """
    Write the complete simulation state to a compressed .npz archive.

    The archive is sufficient for both an exact restart and offline analysis.

    Parameters
    ----------
    filepath : str
        Output path (``.npz`` appended if missing).
    grid : Grid
        Must expose geom, Nx1, Nx2, Ngc, fx1, fx2, cx1, cx2 (and, if present,
        x1ini..x2fin -- otherwise the bounds are read off the face arrays).
    state : SimState
        Any mode; all ndarray / scalar attributes are stored.
    par : Parameters
    eos : EOSdata, optional
        If given, GAMMA is stored (and restored).
    """
    Ngc = int(grid.Ngc)

    # --- domain bounds (prefer stored attrs; fall back to the face arrays) ---
    x1ini = float(getattr(grid, "x1ini", grid.fx1[Ngc, Ngc]))
    x1fin = float(getattr(grid, "x1fin", grid.fx1[grid.Nx1r, Ngc]))
    x2ini = float(getattr(grid, "x2ini", grid.fx2[Ngc, Ngc]))
    x2fin = float(getattr(grid, "x2fin", grid.fx2[Ngc, grid.Nx2r]))

    archive = {
        # run metadata
        "mode":        str(par.mode),
        "problem":     str(par.problem),
        "timenow":     float(par.timenow),
        "timefin":     float(par.timefin),
        "CFL":         float(par.CFL),
        "rec_type":    _enc(getattr(par, "rec_type", None)),
        "RK_order":    _enc(getattr(par, "RK_order", None)),
        "solver_type": _enc(getattr(par, "solver_type", None)),
        "divb_tr":     _enc(getattr(par, "divb_tr", None)),
        "rkl2_stages": int(getattr(par, "rkl2_stages", None) or -1),
        # grid construction (enough to rebuild the grid exactly)
        "geom":  str(grid.geom),
        "Nx1":   int(grid.Nx1),
        "Nx2":   int(grid.Nx2),
        "Ngc":   Ngc,
        "x1ini": x1ini, "x1fin": x1fin,
        "x2ini": x2ini, "x2fin": x2fin,
        # cell-centre coordinates (interior) -- convenience for analysis
        "cx1": grid.cx1[Ngc:-Ngc, Ngc:-Ngc],
        "cx2": grid.cx2[Ngc:-Ngc, Ngc:-Ngc],
        # boundary conditions
        "BC": np.asarray(par.BC, dtype=str),
        "BC_fixed": np.array(getattr(par, "BC_fixed", {0: [], 1: [], 2: [], 3: []}),
                             dtype=object),
    }
    if getattr(par, "BCm", None) is not None:
        archive["BCm"] = np.asarray(par.BCm, dtype=str)
    if eos is not None:
        archive["GAMMA"] = float(eos.GAMMA)

    # --- the whole state, mode-agnostic ---
    for name, val in vars(state).items():
        if name in _META_KEYS:
            continue                         # never let a field shadow metadata
        if isinstance(val, np.ndarray):
            archive[name] = val
        elif isinstance(val, (int, float, np.integer, np.floating)):
            archive[name] = val              # e.g. adv's constant vel1/vel2

    if not filepath.endswith(".npz"):
        filepath += ".npz"
    outdir = os.path.dirname(filepath)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    np.savez_compressed(filepath, **archive)
    print(f"[io] state saved -> {filepath}  (mode={par.mode}, t={par.timenow:.6e})")
    return filepath


# ===========================================================================
#   Reader  (analysis)
# ===========================================================================
def load_data(filepath):
    """
    Read an archive written by :func:`save_data` into a flat dict.

    Scalars and strings are unwrapped from their 0-d arrays; object entries
    (e.g. ``BC_fixed``) are returned as the original Python object.  Field
    arrays are returned under their plain names (``'dens'``, ``'bfi1'``, ...),
    so analysis is simply ``d = load_data(f); rho = d['dens']``.

    Mirrors the ``.npz`` auto-append done by :func:`save_data`: a bare path
    with no extension is tried as-is first, then with ``.npz`` appended, so
    ``load_data(p)`` works with the same `p` that was passed to
    ``save_data(p, ...)``.
    """
    if not os.path.exists(filepath) and not filepath.endswith(".npz") \
            and os.path.exists(filepath + ".npz"):
        filepath += ".npz"
    raw = np.load(filepath, allow_pickle=True)
    data = {}
    for key in raw.files:
        val = raw[key]
        if val.dtype == object:
            data[key] = val.item()
        elif val.ndim == 0:
            data[key] = val.item()
        else:
            data[key] = val
    print(f"[io] state loaded <- {filepath}  "
          f"(mode={data.get('mode')}, t={data.get('timenow')})")
    return data


# ===========================================================================
#   Restart  (rebuild live objects)
# ===========================================================================
def restart_simulation(filepath):
    """
    Rebuild ``(grid, state, par, eos)`` from an archive, ready for a solver.

    The grid is regenerated by re-running its constructor (via
    ``grid_setup.reconstruct_grid``); parameters are rebuilt through the normal
    ``Parameters`` constructor so derived fields (e.g. Ngc) stay consistent;
    the state is allocated with ``SimState`` and then overwritten field-by-field.
    """
    # local imports keep analysis use of this module dependency-free
    from src.grid.grid_setup import reconstruct_grid
    from src.sim_state import SimState
    from src.parameters import Parameters
    from src.common.eos_setup import EOSdata

    d = load_data(filepath)

    grid = reconstruct_grid(int(d["Nx1"]), int(d["Nx2"]), int(d["Ngc"]),
                            str(d["geom"]),
                            float(d["x1ini"]), float(d["x1fin"]),
                            float(d["x2ini"]), float(d["x2fin"]))

    divb_tr = _dec(d.get("divb_tr")) or "GLM"      # ignored by non-MHD modes
    stages  = int(d.get("rkl2_stages", -1))
    par = Parameters(
        mode=str(d["mode"]), problem=str(d["problem"]),
        Nx1=int(d["Nx1"]), Nx2=int(d["Nx2"]),
        rec_type=_dec(d.get("rec_type")),
        RK_order=_dec(d.get("RK_order")),
        solver_type=_dec(d.get("solver_type")),
        CFL=float(d["CFL"]),
        divb_tr=divb_tr,
        rkl2_stages=(10 if stages < 0 else stages),
    )
    par.timenow = float(d["timenow"])
    par.timefin = float(d["timefin"])
    par.BC = np.asarray(d["BC"], dtype=str)
    if "BCm" in d and getattr(par, "BCm", None) is not None:
        par.BCm = np.asarray(d["BCm"], dtype=str)
    if "BC_fixed" in d:
        par.BC_fixed = d["BC_fixed"]

    eos = EOSdata(float(d["GAMMA"])) if "GAMMA" in d else None

    # allocate the right arrays for this mode, then overwrite every stored field.
    # Iterate the ARCHIVE's field keys (not the fresh state's attributes) so that
    # IC-set values which __init__ may not pre-allocate -- e.g. adv's constant
    # vel1/vel2 -- are still restored.
    state = SimState(grid, par)
    for name in d:
        if name in _META_KEYS:
            continue
        val = d[name]
        cur = getattr(state, name, None)
        if isinstance(val, np.ndarray) and isinstance(cur, np.ndarray):
            setattr(state, name, np.asarray(val, dtype=cur.dtype))
        else:
            setattr(state, name, val)

    print(f"[io] restart ready: mode={par.mode}, t={par.timenow:.6e} -> {par.timefin:.6e}")
    return grid, state, par, eos


# ===========================================================================
#   Optional: 1-D ASCII profile (analysis convenience)
# ===========================================================================
def save_1d_ascii(filepath, grid, state, par, eos=None):
    """
    Dump a 1-D profile (Nx2==1 or Nx1==1) to a plain-text file, one row per cell.
    Columns depend on the mode (x, then the primitive variables).
    """
    Ngc = grid.Ngc
    if grid.Nx2 == 1:
        x = grid.cx1[Ngc:-Ngc, Ngc]
        sl = (slice(Ngc, -Ngc), Ngc)
    elif grid.Nx1 == 1:
        x = grid.cx2[Ngc, Ngc:-Ngc]
        sl = (Ngc, slice(Ngc, -Ngc))
    else:
        raise ValueError("save_1d_ascii: grid is 2-D; use save_data instead.")

    fields = {
        "adv":  ["dens"],
        "HD":   ["dens", "vel1", "vel2", "vel3", "pres"],
        "rHD":  ["dens", "vel1", "vel2", "vel3", "pres"],
        "MHD":  ["dens", "vel1", "vel2", "vel3", "pres", "bfi1", "bfi2", "bfi3"],
        "rMHD": ["dens", "vel1", "vel2", "vel3", "pres", "bfi1", "bfi2", "bfi3"],
        "diff": ["T"],
        "SWE":  ["h", "vel1", "vel2"],
    }[par.mode]

    columns = [x] + [getattr(state, name)[sl] for name in fields]
    header = (f"Piastra {par.mode} | problem={par.problem} | t={par.timenow:.8e}\n"
              + "  ".join(["x1"] + fields))

    if not filepath.endswith((".dat", ".txt")):
        filepath += ".dat"
    outdir = os.path.dirname(filepath)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    np.savetxt(filepath, np.column_stack(columns), header=header, fmt="%.12e")
    print(f"[io] 1D ASCII saved -> {filepath}")
    return filepath


# Backward-compatible aliases
save_snapshot = save_data
load_snapshot = load_data
