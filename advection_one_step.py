# -*- coding: utf-8 -*-
"""
Advection2D - Container class for 2D linear advection routines.

Hybrid approach:
- Functions remain modular and pedagogically simple
- Lightweight container class provides a clean interface for timestepping
  and groups related routines for 2D linear advection

This class handles:
- CFL-limited timestep calculation for 2D linear advection
- Single-step Runge-Kutta updates (RK1, RK2, RK3)
- Flux evaluation using either upwind or Lax-Wendroff schemes
- Primitive variable reconstruction to cell faces for higher-order accuracy
- Periodic boundary condition handling

The underlying methods are suitable for explicit, finite-volume advection
simulations of scalar or vector fields.

Attributes
----------
grid : object
    Grid object containing domain size, spacing, face areas, cell volumes, and ghost cells.
adv : object
    Advected state object containing:
        - adv : 2D array of advected scalar field
        - vel1, vel2 : velocity components in x1 and x2 directions
par : object
    Simulation parameters including:
        - CFL : Courant number
        - RK_order : 'RK1', 'RK2', or 'RK3'
        - flux_type : 'adv' (upwind) or 'LW' (Lax-Wendroff)
        - rec_type : reconstruction type
        - timenow : current simulation time
        - timefin : final simulation time

Example usage
-------------
>>> advector = Advection2D(grid, adv, par)
>>> adv = advector.step_RK()  # advances the solution by one RK timestep
"""


import numpy as np
import copy
from reconstruction import VarReconstruct


class Advection2D:
    """
    Container class for 2D linear advection routines.

    This class provides a clean interface for performing a single timestep of
    linear advection in 2D using finite-volume methods, supporting different
    Runge-Kutta orders and flux schemes (upwind, Lax-Wendroff).

    Attributes
    ----------
    g : object
        Grid object, expected to have attributes Nx1, Nx2, Ngc, dx1, dx2, fS1, fS2, cVol.
    adv : object
        Advected state object, expected to have attributes adv (2D array), vel1, vel2.
    par : object
        Simulation parameters object, expected to have attributes CFL, RK_order,
        flux_type ('adv' or 'LW'), rec_type, timenow, timefin.
    """

    def __init__(self, g, adv, par):
        """
        Initialize the advection container.

        Parameters
        ----------
        g : object
            Grid object containing domain size, spacing, volumes, and face areas.
        adv : object
            Advected state object containing conservative variable array and velocities.
        par : object
            Simulation parameters including CFL number, flux type, RK order, and time.
        """
        self.g = g
        self.adv = adv
        self.par = par

    def step_RK(self):
        """
        Perform a single Runge-Kutta timestep for 2D linear advection.

        Calculates the timestep using the CFL condition, applies one RK step
        (predictor-corrector) according to the chosen order, and updates
        the advected state in place.

        Returns
        -------
        adv : object
            Updated advected state after the timestep.
        """
        dt = min(CFLcondition_adv(self.g, self.adv, self.par.CFL),
                 self.par.timefin - self.par.timenow)
        self.par.timenow += dt

        self.adv = oneStep_advection_RK(self.g, self.adv, self.par, dt)
        
        return self.adv


# -------------------------
# Function definitions
# -------------------------
def oneStep_advection_RK(g, adv, par, dt):
    """
    Perform one Runge-Kutta timestep for 2D linear advection.

    This function implements first-, second-, and third-order Runge-Kutta
    timestepping for finite-volume linear advection. It also supports the
    Lax-Wendroff flux as a special case.

    Parameters
    ----------
    g : object
        Grid object containing cell sizes, face areas, and ghost cell count.
    adv : object
        Advected state object containing 2D array of conservative variables and velocities.
    par : object
        Simulation parameters including flux type ('adv' or 'LW') and RK order ('RK1', 'RK2', 'RK3').
    dt : float
        Timestep to use for this RK iteration.

    Returns
    -------
    adv : object
        Updated advected state object after one RK step.

    Notes
    -----
    - Predictor-corrector logic is used for RK2 and RK3.
    - Lax-Wendroff flux is treated as a special case without RK iterations.
    """
    Ngc = g.Ngc
    adv_h = copy.deepcopy(adv)

    # Predictor stage
    Res = flux_adv(g, adv, par, dt)
    adv_h.dens[Ngc:-Ngc, Ngc:-Ngc] = adv.dens[Ngc:-Ngc, Ngc:-Ngc] - dt * Res

    #Lax-Wendroff scheme
    if par.flux_type == 'LW':
        adv.dens = adv_h.dens
        return adv
    
    #Runge-Kutta multistage approach
    if par.RK_order == 'RK1':
        adv.dens = adv_h.dens
    elif par.RK_order == 'RK2':
        Res = flux_adv(g, adv_h, par, dt)
        adv.dens[Ngc:-Ngc, Ngc:-Ngc] = (adv_h.dens[Ngc:-Ngc, Ngc:-Ngc] + adv.dens[Ngc:-Ngc, Ngc:-Ngc]) / 2.0 - dt * Res / 2.0
    elif par.RK_order == 'RK3':
        Res = flux_adv(g, adv_h, par, dt)
        adv_h.dens[Ngc:-Ngc, Ngc:-Ngc] = (adv_h.dens[Ngc:-Ngc, Ngc:-Ngc] + 3.0 * adv.dens[Ngc:-Ngc, Ngc:-Ngc]) / 4.0 - dt * Res / 4.0
        Res = flux_adv(g, adv_h, par, dt)
        adv.dens[Ngc:-Ngc, Ngc:-Ngc] = (2.0 * adv_h.dens[Ngc:-Ngc, Ngc:-Ngc] + adv.dens[Ngc:-Ngc, Ngc:-Ngc]) / 3.0 - 2.0 * dt * Res / 3.0
    else:
        raise ValueError("Wrong RK_order: choose 'RK1', 'RK2', or 'RK3'.")

    return adv




def CFLcondition_adv(g, adv, CFL):
    """
    Compute the timestep according to the CFL stability condition for 2D advection.

    Parameters
    ----------
    g : object
        Grid object with cell sizes and ghost cell count.
    adv : object
        Advected state object with velocities vel1 and vel2.
    CFL : float
        Courant-Friedrichs-Lewy number to scale the timestep.

    Returns
    -------
    dt : float
        Maximum stable timestep according to CFL condition.

    Notes
    -----
    The CFL condition ensures that the fastest wave in the system does not
    propagate more than one cell per timestep.
    """
    Ngc = g.Ngc
    
    #FIRST APPROACH
    #dt1 = np.min(g.dx1[Ngc:-Ngc, Ngc:-Ngc] / (1e-14 + np.abs(adv.vel1)))
    #dt2 = np.min(g.dx2[Ngc:-Ngc, Ngc:-Ngc] / (1e-14 + np.abs(adv.vel2)))
    #return CFL * min(dt1, dt2)
    
    #SECOND APPROACH 
    inv_dt = np.max(np.abs(adv.vel1)/g.dx1[Ngc:-Ngc, Ngc:-Ngc] + \
        np.abs(adv.vel2)/g.dx2[Ngc:-Ngc, Ngc:-Ngc])
    
    return CFL / inv_dt 




def flux_adv(g, adv, par, dt):
    """
    Compute residuals for finite-volume linear advection in 2D.

    This function computes the fluxes and residuals for a 2D advected
    quantity using either simple upwind flux or Lax-Wendroff flux, and
    handles periodic boundary conditions in both directions.

    Parameters
    ----------
    g : object
        Grid object with cell sizes, volumes, face areas, and ghost cells.
    adv : object
        Advected state object with 2D array of conservative variables and velocities.
    par : object
        Simulation parameters including flux_type ('adv' or 'LW') and reconstruction type.
    dt : float
        Timestep used for Lax-Wendroff flux.

    Returns
    -------
    Res : np.ndarray
        Residual array of the same shape as the real domain (Nx1 x Nx2) representing
        the rate of change of the advected variable.

    Notes
    -----
    - Upwind flux uses linear reconstruction via VarReconstruct.
    - Lax-Wendroff flux includes a multi-dimensional antidiffusion correction.
    - Periodic boundary conditions are applied automatically for ghost cells.
    """
    Ngc = g.Ngc
    Nx1r = g.Nx1 + Ngc
    Nx2r = g.Nx2 + Ngc

    # Apply periodic boundary conditions
    for i in range(Ngc):
        adv.dens[i, :] = adv.dens[Nx1r - Ngc + i, :]
        adv.dens[Nx1r + i, :] = adv.dens[Ngc + i, :]
        adv.dens[:, i] = adv.dens[:, Nx2r - Ngc + i]
        adv.dens[:, Nx2r + i] = adv.dens[:, Ngc + i]

    Res = np.zeros((g.Nx1, g.Nx2), dtype=np.double)

    if par.flux_type == 'adv':
        if g.Nx1 > 1:
            L, R = VarReconstruct(adv.dens, g, par.rec_type, 1)
            flux = adv.vel1 * (L + R) / 2.0 - np.abs(adv.vel1) * (R - L) / 2.0
            Res = (flux[1:, :]*g.fS1[1:, :] - flux[:-1, :]*g.fS1[:-1, :]) / g.cVol[:, :]

        if g.Nx2 > 1:
            L, R = VarReconstruct(adv.dens, g, par.rec_type, 2)
            flux = adv.vel2 * (L + R) / 2.0 - np.abs(adv.vel2) * (R - L) / 2.0
            Res += (flux[:, 1:]*g.fS2[:, 1:] - flux[:, :-1]*g.fS2[:, :-1]) / g.cVol[:, :]

    elif par.flux_type == 'LW':
        # Lax-Wendroff flux in x1
        if g.Nx1 > 1:
            flux = adv.vel1 * (adv.dens[Ngc - 1:Nx1r, Ngc:-Ngc] + adv.dens[Ngc:Nx1r + 1, Ngc:-Ngc]) / 2.0 \
                   + adv.vel1 * (adv.vel1 * dt / g.dx1[Ngc:-Ngc, Ngc:-Ngc]) * \
                   (adv.dens[Ngc - 1:Nx1r, Ngc:-Ngc] - adv.dens[Ngc:Nx1r + 1, Ngc:-Ngc]) / 2.0
            Res = (flux[1:, :] * g.fS1[1:, :] - flux[:-1, :] * g.fS1[:-1, :]) / g.cVol[:, :]

        # Lax-Wendroff flux in x2
        if g.Nx2 > 1:
            flux = adv.vel2 * (adv.dens[Ngc:-Ngc, Ngc - 1:Nx2r] + adv.dens[Ngc:-Ngc, Ngc:Nx2r + 1]) / 2.0 \
                   + adv.vel2 * (adv.vel2 * dt / g.dx2[Ngc:-Ngc, Ngc:-Ngc]) * \
                   (adv.dens[Ngc:-Ngc, Ngc - 1:Nx2r] - adv.dens[Ngc:-Ngc, Ngc:Nx2r + 1]) / 2.0
            Res += (flux[:, 1:] * g.fS2[:, 1:] - flux[:, :-1] * g.fS2[:, :-1]) / g.cVol[:, :]

        # Multi-dimensional antidiffusion correction
        Res -= dt * adv.vel1 * adv.vel2 * (
                adv.dens[Ngc - 1:Nx1r - 1, Ngc - 1:Nx2r - 1] - adv.dens[Ngc - 1:Nx1r - 1, Ngc + 1:Nx2r + 1]
                - adv.dens[Ngc + 1:Nx1r + 1, Ngc - 1:Nx2r - 1] + adv.dens[Ngc + 1:Nx1r + 1, Ngc + 1:Nx2r + 1]
        ) / 4.0 / g.dx1[Ngc:-Ngc, Ngc:-Ngc] / g.dx2[Ngc:-Ngc, Ngc:-Ngc]

    return Res
