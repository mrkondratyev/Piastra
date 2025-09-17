# -*- coding: utf-8 -*-
"""
Equation of State (EOS) Routines
================================

Provides a simple gamma-law equation of state (ideal gas) for use in
non-relativistic hydrodynamics and magnetohydrodynamics.

Currently supports:
- Ideal gas EOS with constant gamma.

Functions
---------
- Sound speed calculation
- Specific internal energy
- Specific entropy

Author
------
mrkondratyev, 2024–2025
"""

import numpy as np


class EOSdata:
    """
    Equation of State (EOS) class for gamma-law (ideal gas).

    Parameters
    ----------
    gamma : float
        Adiabatic index (ratio of specific heats).

    Attributes
    ----------
    GAMMA : float
        Adiabatic index.
    eos_flag : str
        EOS type identifier, set to 'ideal'.
    """

    def __init__(self, gamma):
        """
        Initialize the EOS with a constant gamma-law index.

        Parameters
        ----------
        gamma : float
            Adiabatic index (ratio of specific heats).
        """
        self.GAMMA = np.float64(gamma)
        self.eos_flag = 'ideal'




    def sound_speed(self, dens, pres):
        """
        Compute adiabatic sound speed.

        Parameters
        ----------
        dens : float or ndarray
            Mass density.
        pres : float or ndarray
            Gas pressure.

        Returns
        -------
        cs : float or ndarray
            Adiabatic sound speed.
        """
        return np.sqrt(self.GAMMA * pres / dens)




    def specific_internal_energy(self, dens, pres):
        """
        Compute specific internal energy (per unit mass).

        Parameters
        ----------
        dens : float or ndarray
            Mass density.
        pres : float or ndarray
            Gas pressure.

        Returns
        -------
        eps : float or ndarray
            Specific internal energy.
        """
        return (self.GAMMA - 1.0) * pres / dens




    def specific_entropy(self, dens, pres):
        """
        Compute specific entropy (up to an additive constant).

        Parameters
        ----------
        dens : float or ndarray
            Mass density.
        pres : float or ndarray
            Gas pressure.

        Returns
        -------
        s : float or ndarray
            Specific entropy (nondimensional).
        """
        return pres / dens**self.GAMMA
