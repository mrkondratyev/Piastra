# -*- coding: utf-8 -*-
"""
===============================================================================
parameters.py
===============================================================================

Central module for storing simulation parameters for different
fluid dynamics solvers: advection, hydrodynamics (HD), and 
magnetohydrodynamics (MHD).

The Parameters class:
- Defines defaults for numerical schemes
- Stores boundary conditions, CFL, and timing info
- Provides validation for mode and scheme selection
- Can be extended for future models (e.g., SPH)

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
        Simulation module ('adv', 'HD', 'MHD').
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
        "HD": "HLLC",
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
                 divb_tr: str = '8wave'):

        # Simulation mode
        if mode not in ["adv", "HD", "MHD"]:
            raise ValueError(f"Unknown mode: {mode}. Expected one of ['adv', 'HD', 'MHD'].")
        self.mode = mode
        self.problem = problem

        # Grid resolution
        self.Nx1 = Nx1
        self.Nx2 = Nx2

        # Reconstruction method
        self.rec_type = rec_type
        self.Ngc = 2 if rec_type in ["PCM", "PLM"] else 3

        # Time integration
        if RK_order not in ["RK1", "RK2", "RK3"]:
            raise ValueError(f"Invalid RK_order: {RK_order}. Expected one of ['RK1', 'RK2', 'RK3'].")
        self.RK_order = RK_order

        # Flux type
        self.flux_type = flux_type if flux_type is not None else self._default_flux[mode]

        # Physical time
        self.timenow = 0.0
        self.timefin = 0.0

        # Boundary conditions
        self.BC = np.array(["wall", "wall", "wall", "wall"], dtype=str)

        # Magnetic boundary conditions for MHD and divB treatment for MHD 
        if mode == "MHD":
            self.BCm = np.array(["wall", "wall", "wall", "wall"], dtype=str)
            if divb_tr not in ["CT", "8wave"]:
                raise ValueError(f"Invalid divb_tr: {divb_tr}. Expected one of ['CT', '8wave'].")
            self.divb_tr = divb_tr
        else:
            self.BCm = None
            self.divb_tr = None
            
        # CFL condition
        self.CFL = CFL

    def __str__(self):
        lines = [
            f"Simulation mode   : {self.mode}",
            f"Problem           : {self.problem}",
            f"Resolution        : Nx1={self.Nx1}, Nx2={self.Nx2}, Ngc={self.Ngc}",
            f"Reconstruction    : {self.rec_type}",
            f"RK Order          : {self.RK_order}",
            f"Flux Type         : {self.flux_type}",
            f"CFL               : {self.CFL}",
        ]
        if self.mode == "MHD":
            lines.append(f"divB treatment    : {self.divb_tr}")
        return "\n".join(lines)
