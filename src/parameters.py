# -*- coding: utf-8 -*-
"""
===============================================================================
parameters.py
===============================================================================

Central module for storing simulation parameters for different
fluid dynamics solvers: advection, hydrodynamics (HD), special-relativistic
hydrodynamics (rHD), magnetohydrodynamics (MHD), special-relativistic
magnetohydrodynamics (rMHD), thermal diffusion, and shallow water (SWE).

The Parameters class:
- Defines defaults for numerical schemes
- Stores boundary conditions, CFL, and timing info
- Provides validation for mode and scheme selection
- Supports modes: 'adv', 'HD', 'rHD', 'MHD', 'rMHD', 'diff', 'SWE'

Author: mrkondratyev
"""
import numpy as np
from typing import Optional


class Parameters:
    """
    Container for simulation setup parameters.

    This class stores all auxiliary data required to configure a simulation,
    including numerical methods, boundary conditions, and timing information.

    Notes
    -----
    - CFL condition:
      In 1D: CFL ≤ 1
      In 2D: CFL ≤ 0.5
      Here we use a slightly different definition:
      dt = CFL * ( max( Σ λ_i / Δx_i ) )^(-1),
      so the same CFL works for 1D and 2D.
    - Ghost cells:
      Default is 2, but for PPM/WENO reconstructions 3 is required.

    Parameters
    ----------
    mode : str
        Simulation module ('adv', 'HD', 'rHD', 'MHD', 'rMHD', 'diff', 'SWE').
    problem : str
        Name of the initial problem (used by initial condition setup).
    Nx1 : int, optional
        Number of grid cells in the first dimension.
    Nx2 : int, optional
        Number of grid cells in the second dimension.
    rec_type : str, optional
        Reconstruction method. Default is 'PLM'.
        Options: 'PCM', 'PLM', 'PPMorig', 'PPM', 'WENO', 'MP5'.
        Not used for 'diff' mode.
    RK_order : str, optional
        Runge-Kutta temporal integration order. Default is 'RK3'.
        Options: 'RK1', 'RK2', 'RK3'.
        Not used for 'diff' mode.
    solver_type : str, optional
        Numerical solver type. If not provided, assigned from defaults.
    CFL : float, optional
        Courant-Friedrichs-Lewy number. Default is 0.7.
    divb_tr : str, optional
        Divergence of magnetic field treatment (MHD only): 'CT' , 'GLM', or '8wave'.
    rkl2_stages : int, optional
        Number of RKL2 stages s ≥ 2 (mode='diff', solver_type='rkl2' only).
        Default is 10.

    Attributes
    ----------
    BC : np.ndarray of str
        Boundary conditions for each face, default is 'free' on all sides
        (overwritten by the IC function for the chosen problem).
    BCm : np.ndarray of str or None
        Boundary conditions for the magnetic field (MHD, rMHD only);
        None for all other modes.
    timenow : float
        Current simulation time.
    timefin : float
        Final physical time (set by the initial condition function).
    Ngc : int
        Number of ghost cells (depends on reconstruction method).
    """

    # Default solver mapping per mode
    _default_solver = {
        "adv":  "adv",
        "SWE":  "HLL",
        "HD":   "HLLC",
        "rHD":  "HLLC",
        "MHD":  "HLLD",
        "rMHD": "HLL",
        "diff": "rkl2",
    }

    def __init__(self,
                 mode: str,
                 problem: str,
                 Nx1: Optional[int] = None,
                 Nx2: Optional[int] = None,
                 rec_type: str = "PLM",
                 RK_order: str = "RK2",
                 solver_type: Optional[str] = None,
                 CFL: float = 0.7,
                 divb_tr: str = 'GLM',
                 rkl2_stages: int = 10):

        # Simulation mode
        valid_modes = ["adv", "HD", "rHD", "MHD", "rMHD", "diff", "SWE"]
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown mode: '{mode}'. Expected one of {valid_modes}.")
        self.mode    = mode
        self.problem = problem

        # Grid resolution
        self.Nx1 = Nx1; self.Nx2 = Nx2

        # Physical time
        self.timenow = 0.0; self.timefin = 0.0

        # Boundary conditions (set by IC function)
        self.BC = np.array(["free", "free", "free", "free"], dtype=str)
        # Fixed (Dirichlet) ghost-fill patches, keyed by face index 0..3.
        # Each entry: list of (start, end, {field: value}) tuples giving an
        # interior index range along that boundary and the prescribed state.
        self.BC_fixed = {0: [], 1: [], 2: [], 3: []}

        # CFL number
        self.CFL = CFL
            
        # solver type
        self.solver_type = (solver_type if solver_type is not None
                          else self._default_solver[mode])

        # ── Diffusion mode ────────────────────────────────────────────────
        if mode == "diff":
            self.rkl2_stages  = int(rkl2_stages)                
            self.Ngc          = 1
            self.rec_type     = None
            self.RK_order     = None
            self.BCm          = None
            self.divb_tr      = None
            return

        # ── All remaining modes: adv / HD / rHD / MHD / rMHD / SWE ────────

        # Reconstruction method
        self.rec_type = rec_type
        self.Ngc = 2 if rec_type in ["PCM", "PLM"] else 3

        # Time integration
        self.RK_order = RK_order

        # Parameters unused by these modes
        self.rkl2_stages = None
        
        # -- relativistic flows should use lower CFL -----------------------
        if (mode == "rMHD" or mode == "rHD") & (self.CFL > 0.4):
            print("adjust CFL parameter to 0.4 for relativistic flows")
            self.CFL = 0.4
          
        # ── rMHD ──────────────────────────────────────────────────────────
        if mode == "rMHD":
            self.BCm = np.array(["free", "free", "free", "free"], dtype=str)
            if divb_tr not in ["CT"]:
                print("CT scheme is only available for rMHD")
            self.divb_tr = "CT"

        # ── MHD ───────────────────────────────────────────────────────────
        elif mode == "MHD":
            self.BCm = np.array(["free", "free", "free", "free"], dtype=str)
            if divb_tr not in ["CT", "8wave", "GLM"]:
                raise ValueError(
                    f"Invalid divb_tr: '{divb_tr}'. "
                    f"Expected one of ['CT', '8wave', 'GLM'].")
            self.divb_tr = divb_tr

        # ── adv / HD / rHD / SWE ──────────────────────────────────────────
        else:
            self.BCm     = None
            self.divb_tr = None



    def __str__(self):
        lines = [
            f"Simulation mode   : {self.mode}",
            f"Problem           : {self.problem}",
            f"Resolution        : Nx1={self.Nx1}, Nx2={self.Nx2}, Ngc={self.Ngc}",
        ]

        if self.mode == "diff":
            lines += [
                f"Time integrator   : {self.solver_type}",
                f"RKL2 stages       : {self.rkl2_stages}",
                f"CFL               : {self.CFL}",
            ]

        else:
            lines += [
                f"Reconstruction    : {self.rec_type}",
                f"RK Order          : {self.RK_order}",
                f"Solver Type       : {self.solver_type}",
                f"CFL               : {self.CFL}",
            ]
            if self.mode == "MHD":
                lines.append(f"divB treatment    : {self.divb_tr}")
            if self.mode == "rMHD":
                lines.append(f"divB treatment    : {self.divb_tr}")
                lines.append("Relativistic      : yes")
            if self.mode == "rHD":
                lines.append("Relativistic      : yes")

        return "\n".join(lines)
