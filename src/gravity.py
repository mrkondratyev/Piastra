# -*- coding: utf-8 -*-
"""
add_phys.py
===========
Additional gravitational source terms for Piastra.

This module provides two body-force sources that are meant to be called from
the time stepper, before the momentum update, once per step (or, for higher
accuracy, once per Runge-Kutta stage):

  1. planet_gravity_polar(...)       -- star + orbiting planet in POLAR (r, phi)
  2. selfgravity_monopole_spherical  -- spherically-averaged (monopole) self
                                        gravity in SPHERICAL-POLAR (r, theta)

------------------------------------------------------------------------------
GRID-ATTRIBUTE FLAGS
------------------------------------------------------------------------------
  * state.F1, state.F2  : interior body-force/acceleration arrays (Nx1, Nx2).
  * state.dens          : density (full grid, with ghosts).
  * grid.cx1, grid.cx2  : cell-centre coordinates, full 2D arrays with ghosts.
                          POLAR    : cx1 = r,  cx2 = phi.
                          SPHERICAL: cx1 = r,  cx2 = theta.  cx1 varies along
                          axis 0 (radius) and is constant along axis 1 (angle).
  * grid.cVol           : cell volume, full 2D array with ghosts (Nx1+2Ngc,
                          Nx2+2Ngc).  Used only for the volume-weighted angular
                          average of density in the monopole solver.
  * par.timenow         : current time (drives the planet's orbital phase).
"""

import numpy as np


# ===========================================================================
# 1) Orbiting planet in polar geometry  (central star + planet, lab frame)
# ===========================================================================
def planet_gravity_polar(grid, state, par,
                         M_star, M_planet, r_planet,
                         phi0=0.0, indirect=True):
    """Star + orbiting-planet gravitational acceleration in polar (r, phi).

    The planet is on a fixed circular orbit at radius ``r_planet`` with
    Keplerian angular frequency Omega_p = sqrt(G (M_star + M_planet) / r_p^3).
    Its azimuth advances as  phi_p(t) = phi0 + Omega_p * par.timenow, so calling
    this routine every step makes the potential genuinely time-dependent (the
    lab-frame treatment of an orbiting perturber).

    Parameters
    ----------
    M_star, M_planet : float   masses of the central star and the planet.
    indirect         : bool    add the indirect term that accounts for the
                               acceleration of the star-centred (non-inertial)
                               frame by the planet. Standard in planet-disk codes.

    Notes
    -----
    Planet parameters are included inside the function, they should be equal to 
    the ones from IC routines 
    
    """
    G = 1.0 # gravitational constant in code units
    
    phi0 = 0.0 # planet azimuth at time = 0
    r_planet = 1.0 # planet orbital radius
    
    soft = 0.05 #gravitational softening length 
    # (smooths the planet potential inside ~soft; pick a fraction
    # of the local cell size or the Hill radius)
    
    # grid indexes for slicing 
    Ngc  = grid.Ngc; Nx1  = grid.Nx1; Nx2  = grid.Nx2
    Nx1r = Ngc + Nx1; Nx2r = Ngc + Nx2
    sl = np.s_[Ngc:Nx1r, Ngc:Nx2r]
    
    #cell centers 
    r = grid.cx1[sl]; phi = grid.cx2[sl]

    # --- planet orbital phase and (optionally ramped) mass ---
    Omega_p = np.sqrt(G * (M_star + M_planet) / r_planet**3)
    phi_p   = phi0 + Omega_p * par.timenow

    # --- planet potential gradient (softened point mass) ---
    # Phi_p = -G mp / sqrt(r^2 + r_p^2 - 2 r r_p cos(phi-phi_p) + soft^2)
    dphi = phi - phi_p
    cosd = np.cos(dphi); sind = np.sin(dphi)
    Dsq  = r * r + r_planet**2 - 2.0 * r * r_planet * cosd + soft * soft
    Dm32 = Dsq**(-1.5)

    a_r1 = -G * M_planet * (r - r_planet * cosd) * Dm32          # a_r   from planet
    a_p1 = -G * M_planet * (r_planet * sind)     * Dm32          # a_phi from planet

    # --- indirect term (star-centred frame acceleration) ---
    if indirect:
        a_r1 += -G * M_planet * cosd / r_planet**2
        a_p1 +=  G * M_planet * sind / r_planet**2

    # --- central star monopole is accounted here ---
    a_r1 += -G * M_star / (r * r)

    # --- store as F = -a  (see SIGN/SOURCE CONVENTION at top) ---
    state.F1[:, :] = -a_r1; state.F2[:, :] = -a_p1
    
    return state


# ===========================================================================
# 2) Spherically-averaged (monopole) self-gravity in spherical-polar geometry
# ===========================================================================
def selfgravity_monopole_spherical(grid, state, par):
    """1D monopole self-gravity for spherical-polar (r, theta) grids.

    The angular structure of the density is collapsed into a spherically
    averaged profile rho_bar(r); the enclosed mass M(<r) is integrated
    radially and the radial acceleration is

        g_r(r) = -G * M(<r) / r^2 ,        a_theta = 0 .

    This is the standard l = 0 (monopole) approximation used in core-collapse
    codes: cheap, robust, and exact for a spherically symmetric mass
    distribution.  It IGNORES the gravity of non-radial density variations
    (the l >= 1 multipoles), so it is appropriate for nearly-spherical
    configurations (e.g. a proto-neutron star) and approximate otherwise.
    <-- MONOPOLE-ONLY CAVEAT

    Parameters
    ----------
    

    Method
    ------
    rho_bar(r) is the volume-weighted angular mean of the density (uses
    grid.cVol), so it is independent of the grid's angular coverage (a theta
    wedge gives the same rho_bar as full pole-to-pole).  The full-sphere shell
    mass is then rho_bar * (4/3) pi (r_out^3 - r_in^3), summed to M(<r) at the
    radial faces, evaluated as -G M / r^2 there, and averaged to cell centres.
    """
    
    #central point mass 
    M_central = 0.0
    #grav acceleration 
    G = 1.0 
    
    #slicing of density and volume 
    Ngc = grid.Ngc; Nx1 = grid.Nx1; Nx2 = grid.Nx2
    sl  = np.s_[Ngc:Ngc + Nx1, Ngc:Ngc + Nx2]
    rho = state.dens[sl]; vol = grid.cVol[sl]      

    # --- volume-weighted angular mean density per radial shell ---
    wsum = vol.sum(axis=1) # (Nx1,)  shell volume (grid)
    rho_bar = (rho * vol).sum(axis=1) / np.maximum(wsum, 1e-30)  # (Nx1,)

    # faces, mass centers and resolution
    rf = grid.fx1[Ngc:Nx1+1+Ngc,Ngc]
    rc = grid.ax1[Ngc:Nx1+Ngc,Ngc]
    dr = grid.dx1[Ngc:Nx1+Ngc,Ngc]

    # --- enclosed mass at radial faces ---
    shell_mass = (4.0 / 3.0) * np.pi * rho_bar * (rf[1:]**3 - rf[:-1]**3)  # (Nx1,)
    Menc = np.zeros(Nx1 + 1)
    Menc[1:] = np.cumsum(shell_mass)
    Menc += M_central 
    
    # --- acceleration at faces, then average to cell centres ---    
    g_f = -G * Menc / (rf**2 + 1e-30)
    g_r = g_f[:-1]*(rf[1:] - rc)/dr + g_f[1:]*(rc - rf[:-1])/dr

    #assign calculated values to the field potentials 
    state.F1[:, :] = -g_r[:, None]
    state.F2[:, :] = 0.0 
    
    return state