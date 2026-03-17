# -*- coding: utf-8 -*-
"""
===============================================================================
parameters.py
===============================================================================

Central module for storing simulation parameters for different
fluid dynamics solvers: advection, hydrodynamics (HD), special-relativistic
hydrodynamics (rHD), magnetohydrodynamics (MHD), and thermal diffusion.

The Parameters class:
- Defines defaults for numerical schemes
- Stores boundary conditions, CFL, and timing info
- Provides validation for mode and scheme selection
- Supports modes: 'adv', 'HD', 'rHD', 'MHD', 'diff'

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
      Default is 2, but for PPM/WENO reconstructions 3 are required.

    Parameters
    ----------
    mode : str
        Simulation module ('adv', 'HD', 'rHD', 'MHD', 'diff').
    problem : str
        Name of the initial problem (used by initial condition setup).
    Nx1 : int, optional
        Number of grid cells in the first dimension (for grid-based models).
    Nx2 : int, optional
        Number of grid cells in the second dimension.
    rec_type : str, optional
        Reconstruction method. Default is 'PLM'.
        Options: 'PCM', 'PLM', 'PPMorig', 'PPM', 'WENO'.
    RK_order : str, optional
        Runge-Kutta temporal integration order. Default is 'RK3'.
        Options: 'RK1', 'RK2', 'RK3'.
    flux_type : str, optional
        Numerical flux type. If not provided, assigned from defaults:
        - adv: 'adv'
        - HD : 'HLLC'
        - MHD: 'HLLD'
    CFL : float, optional
        Courant–Friedrichs–Lewy number. Default is 0.7.
    divb_tr : str, optional
        divergence of magnetic field treatment (MHD only).
        - CT
        - 8wave
    diff_solver : str, optional
        Time-integration method for diffusion (mode='diff' only).
        - 'expl'  explicit forward Euler (default)
        - 'rkl2'  RKL2 super time-stepping
    rkl2_stages : int, optional
        Number of RKL2 stages s ≥ 2 (mode='diff', diff_solver='rkl2' only).
        Default is 10.

    Attributes
    ----------
    BC : np.ndarray of str
        Boundary conditions for each face, default is 'wall' on all sides.
    BCm : np.ndarray of str, only for MHD
        Boundary conditions for magnetic variables.
    timenow : float
        Current simulation time.
    timefin : float
        Final physical time (must be set by initial condition).
    Ngc : int
        Number of ghost cells (depends on reconstruction).
    """

    # Default flux mapping per module
    _default_flux = {
        "adv": "adv",
        "HD":  "HLLC",
        "rHD": "HLLC",
        "MHD": "HLLD",
    }

    def __init__(self,
                 mode: str,
                 problem: str,
                 Nx1: Optional[int] = None,
                 Nx2: Optional[int] = None,
                 rec_type: str = "PLM",
                 RK_order: str = "RK3",
                 flux_type: Optional[str] = None,
                 CFL: float = 0.7,
                 divb_tr: str = '8wave',
                 diff_solver: str = 'expl',
                 rkl2_stages: int = 10):

        # Simulation mode
        if mode not in ["adv", "HD", "rHD", "MHD", "diff"]:
            raise ValueError(f"Unknown mode: {mode}. Expected one of ['adv', 'HD', 'rHD', 'MHD', 'diff'].")
        self.mode = mode
        self.problem = problem

        # Grid resolution
        self.Nx1 = Nx1
        self.Nx2 = Nx2

        # Physical time
        self.timenow = 0.0
        self.timefin = 0.0

        # Boundary conditions
        self.BC = np.array(["wall", "wall", "wall", "wall"], dtype=str)

        # CFL condition
        self.CFL = CFL

        # --- Diffusion-specific parameters ---
        if mode == "diff":
            if diff_solver not in ["expl", "rkl2"]:
                raise ValueError(f"Invalid diff_solver: '{diff_solver}'. Expected 'expl' or 'rkl2'.")
            self.diff_solver = diff_solver
            self.rkl2_stages = int(rkl2_stages)
            # diffusion needs only one ghost-cell layer for the 2nd-order stencil
            self.Ngc = 1
            self.flux_type = None
            self.rec_type  = None
            self.RK_order  = None
            self.BCm       = None
            self.divb_tr   = None
            return

        # --- Parameters for adv / HD / MHD ---

        # Reconstruction method
        self.rec_type = rec_type
        self.Ngc = 2 if rec_type in ["PCM", "PLM"] else 3

        # Time integration
        if RK_order not in ["RK1", "RK2", "RK3"]:
            raise ValueError(f"Invalid RK_order: {RK_order}. Expected one of ['RK1', 'RK2', 'RK3'].")
        self.RK_order = RK_order

        # Flux type
        self.flux_type = flux_type if flux_type is not None else self._default_flux[mode]

        # Diffusion parameters (not used for these modes)
        self.diff_solver = None
        self.rkl2_stages = None

        # Magnetic boundary conditions and divB treatment (MHD only)
        if mode == "rHD":
            # rHD shares the same flux interface as HD, valid options differ
            valid_rHD = ["LLF", "HLL", "HLLC"]
            if self.flux_type not in valid_rHD:
                raise ValueError(
                    f"Invalid flux_type '{self.flux_type}' for rHD. "
                    f"Expected one of {valid_rHD}.")
            self.BCm     = None
            self.divb_tr = None

        if mode == "MHD":
            self.BCm = np.array(["wall", "wall", "wall", "wall"], dtype=str)
            if divb_tr not in ["CT", "8wave"]:
                raise ValueError(f"Invalid divb_tr: {divb_tr}. Expected one of ['CT', '8wave'].")
            self.divb_tr = divb_tr
        else:
            self.BCm = None
            self.divb_tr = None

    def __str__(self):
        lines = [
            f"Simulation mode   : {self.mode}",
            f"Problem           : {self.problem}",
            f"Resolution        : Nx1={self.Nx1}, Nx2={self.Nx2}, Ngc={self.Ngc}",
        ]
        if self.mode == "diff":
            lines += [
                f"Time integrator   : {self.diff_solver}",
                f"RKL2 stages       : {self.rkl2_stages}",
                f"CFL               : {self.CFL}",
            ]
        else:
            lines += [
                f"Reconstruction    : {self.rec_type}",
                f"RK Order          : {self.RK_order}",
                f"Flux Type         : {self.flux_type}",
                f"CFL               : {self.CFL}",
            ]
            if self.mode == "MHD":
                lines.append(f"divB treatment    : {self.divb_tr}")
            if self.mode == "rHD":
                lines.append("Relativistic      : yes")
        return "\n".join(lines)
