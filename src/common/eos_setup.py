# -*- coding: utf-8 -*-
"""
eos_setup.py

Equation of state module for ideal-gas hydrodynamics and MHD solvers.

Provides the EOSdata class, which stores the adiabatic index and offers
a convenience method for computing the adiabatic sound speed, 
primitive <-> conservative inversions, and energy fluxes calculations.

In future releases, it is planned to update this class to handle non-ideal EOS 

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
    ideal : integer
        Ideal EOS flag.
    """

    def __init__(self, GAMMA):
        self.GAMMA = GAMMA
        # flag to turn off some solvers (like Roe RS for HD), 
        # since they are implemented only for ideal gamma-law EOS  
        self.ideal = 1 


    # --- Non-relativistic sound speed ---
    def sound_speed_nr(self, dens, pres):
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
        return np.sqrt(self.GAMMA * pres / dens)
    
    
    # --- Internal-energy density from pressure ---
    def eint(self, dens, pres):
        """
        Internal-energy density: ε = p / (Γ - 1).

        Parameters
        ----------
        dens : ndarray
            Unused (kept for a uniform (dens, X) calling convention with
            the other EOSdata methods); eps depends only on pressure for
            an ideal gas.
        pres : ndarray
            Pressure.

        Returns
        -------
        eint : ndarray
            Internal-energy density.
        """
        return pres / (self.GAMMA - 1.0)


    # --- Pressure from internal-energy density ---
    def pres(self, dens, eint):
        """
        Pressure: p = (Γ - 1) ε.

        Parameters
        ----------
        dens : ndarray
            Unused (kept for a uniform (dens, X) calling convention with
            the other EOSdata methods); p depends only on eint for an
            ideal gas.
        eint : ndarray
            Internal-energy density.

        Returns
        -------
        pres : ndarray
            Pressure.
        """
        return (self.GAMMA - 1.0) * eint
    
    
    # --- Special-relativistic sound speed ---
    def sound_speed_sr(self, dens, pres):
        """
        Adiabatic sound speed for an ideal relativistic gas.

            cs² = Γ p / (ρ h)

        where h = 1 + Γ p / (ρ (Γ−1)) is the specific enthalpy.

        Parameters
        ----------
        dens, pres : ndarray

        Returns
        -------
        cs : ndarray   –  sound speed  (0 < cs < 1)
        """
        enth = 1.0 + pres / dens * self.GAMMA / (self.GAMMA - 1.0)
        cs2  = self.GAMMA * pres / (dens * enth)
        return np.sqrt(np.clip(cs2, 0.0, 1.0 - 1e-12))
    
    
    # --- Special-relativistic enthalpy ---
    def enthalpy_sr(self, dens, pres):
        """
        specific enthalpy for an ideal relativistic gas.

            h = 1 + Γ p / (ρ (Γ−1))
            
        Parameters
        ----------
        dens, pres : ndarray

        Returns
        -------
        h : ndarray   –  specific enthalpy
        """
        enth = 1.0 + pres / dens * self.GAMMA / (self.GAMMA - 1.0)
        return enth 
    
    
    def __repr__(self):
        return f"EOSdata(GAMMA={self.GAMMA})"
