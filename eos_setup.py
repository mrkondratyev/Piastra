# -*- coding: utf-8 -*-
"""
eos_setup.py

Equation of state module for ideal-gas hydrodynamics and MHD solvers.

Provides the EOSdata class, which stores the adiabatic index and offers
a convenience method for computing the adiabatic sound speed.

Author: mrkondratyev
"""

import numpy as np


class EOSdata:
    """
    Equation of state container for an ideal gas with constant adiabatic index.

    Parameters
    ----------
    GAMMA : float
        Adiabatic index (ratio of specific heats).
        Typical values: 5/3 (monatomic), 7/5 (diatomic), 4/3 (relativistic).

    Attributes
    ----------
    GAMMA : float
        Adiabatic index.
    """

    def __init__(self, GAMMA):
        self.GAMMA = GAMMA

    def sound_speed(self, dens, pres):
        """
        Adiabatic sound speed for a non-relativistic ideal gas.

            cs = sqrt(GAMMA * p / rho)

        Parameters
        ----------
        dens : ndarray
            Mass density.
        pres : ndarray
            Gas pressure.

        Returns
        -------
        cs : ndarray
            Sound speed.
        """
        return np.sqrt(self.GAMMA * pres / (dens + 1e-30))

    def __repr__(self):
        return f"EOSdata(GAMMA={self.GAMMA})"
