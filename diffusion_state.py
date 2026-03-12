# -*- coding: utf-8 -*-
"""
===============================================================================
diffusion_state.py
===============================================================================

State container for 2D thermal diffusion simulations.

The DiffState class allocates and stores the temperature field T on the
computational grid (including ghost cells) and the thermal diffusivity
kappa, which may be a global scalar or a spatially varying cell-centred
array.

This module is the diffusion counterpart of sim_state.py, following the
same ghost-cell convention used by the rest of the Piastra framework.

Notes
-----
- T has shape grid.grid_shape = (Nx1 + 2*Ngc, Nx2 + 2*Ngc).
  Ghost cells are filled by the boundary-condition routines in
  diffusion_one_step.py.
- kappa may be a float (uniform diffusivity) or an ndarray of shape
  grid.grid_shape (spatially variable diffusivity).  When it is an array,
  face-centred values are computed as arithmetic means in the solver.

Author: mrkondratyev
"""

import numpy as np


class DiffState:
    """
    Container for 2D thermal-diffusion state variables.

    Parameters
    ----------
    grid : Grid
        Grid object providing grid_shape (Nx1+2*Ngc, Nx2+2*Ngc).
    kappa : float or ndarray, optional
        Thermal diffusivity.  Scalar for a uniform medium; ndarray of
        shape grid.grid_shape for a spatially variable medium.
        Default is 1.0.

    Attributes
    ----------
    T : ndarray, shape (Nx1+2*Ngc, Nx2+2*Ngc)
        Temperature field including ghost cells.
    kappa : float or ndarray
        Thermal diffusivity (uniform or cell-centred).
    """

    def __init__(self, grid, kappa=1.0):
        self.T     = np.zeros(grid.grid_shape, dtype=np.double)
        self.kappa = kappa

    def __str__(self):
        kappa_str = (
            f"scalar={self.kappa}"
            if np.isscalar(self.kappa)
            else f"array, max={np.max(self.kappa):.4g}, min={np.min(self.kappa):.4g}"
        )
        return (
            f"DiffState:\n"
            f"  T shape : {self.T.shape}\n"
            f"  T range : [{self.T.min():.4g}, {self.T.max():.4g}]\n"
            f"  kappa   : {kappa_str}"
        )
