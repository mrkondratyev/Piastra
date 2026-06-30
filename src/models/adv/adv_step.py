# -*- coding: utf-8 -*-
"""
Adv2D - Container class for 2D linear advection routines.

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
g : object
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
from src.common.high_order_rec import VarReconstruct
from src.models.adv.adv_phys import ( 
    boundCond_adv,
    Riemann_adv)


class Adv2D:
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

        self.adv = oneStep_adv_RK(self.g, self.adv, self.par, dt)
        
        return self.adv


# -------------------------
# Function definitions
# -------------------------
def oneStep_adv_RK(g, adv, par, dt):
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
    Res = flux_calc_adv(g, adv, par, dt)        
    # Conservative update - 1st RK iteration (predictor stage)
    _rk_stage(adv_h, adv, adv, Res, dt, 1.0, 0.0, -1.0, Ngc)

    #Lax-Wendroff scheme
    if par.solver_type == 'LW':
        adv.dens = adv_h.dens
        return adv
    
    #Runge-Kutta multistage approach
    if par.RK_order == 'RK1':
        
        #simply rewrite the advected variable here for clarity
        adv.dens = adv_h.dens
        
    elif par.RK_order == 'RK2':
        
        # Conservative update - 2nd RK iteration (corrector stage)
        Res = flux_calc_adv(g, adv_h, par, dt)
        _rk_stage(adv, adv, adv_h, Res, dt, 1.0/2.0, 1.0/2.0, -1.0/2.0, Ngc)

    elif par.RK_order == 'RK3':
        
        # Conservative update - 2nd RK iteration
        Res = flux_calc_adv(g, adv_h, par, dt)
        _rk_stage(adv_h, adv_h, adv, Res, dt, 1.0/4.0, 3.0/4.0, -1.0/4.0, Ngc)
        
        # Conservative update - 3rd RK iteration
        Res = flux_calc_adv(g, adv_h, par, dt)
        _rk_stage(adv, adv_h, adv, Res, dt, 2.0/3.0, 1.0/3.0, -2.0/3.0, Ngc)
    
    else:
        
        raise ValueError(
            f"Invalid RK_order: '{par.RK_order}'. "
            f"Expected one of ['RK1', 'RK2', 'RK3'].")

    return adv


# -------------------------
# Small helper: one RK stage applied to update the advected variable 
# -------------------------
def _rk_stage(adv_out, adv_a, adv_b, Res, dt, a, b, c, Ngc):
    """
    Set adv_out.* = a * adv_a.* + b * adv_b.* + c * dt * Res
 
    For SSP-RK, the standard combinations are:
      Stage 1 (predictor): a=1,    b=0,    c=-1     -> adv_h = adv - dt*R(HD)
      RK2 corrector:       a=0.5,  b=0.5,  c=-0.5
      RK3 stage 2:         a=0.75, b=0.25, c=-0.25
      RK3 stage 3 (final): a=1/3,  b=2/3,  c=-2/3
    """
    adv_out.dens[Ngc:-Ngc, Ngc:-Ngc] = \
        a * adv_a.dens[Ngc:-Ngc, Ngc:-Ngc] + \
        b * adv_b.dens[Ngc:-Ngc, Ngc:-Ngc] + c * dt * Res


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
    inv_dt = np.max(np.abs(adv.vel1) / g.dx1[Ngc:-Ngc, Ngc:-Ngc] + \
        np.abs(adv.vel2) / g.dx2[Ngc:-Ngc, Ngc:-Ngc])
    
    return CFL / inv_dt 



def flux_calc_adv(g, adv, par, dt):
    """
    Compute residuals for finite-volume linear advection in 2D.

    This function computes the fluxes and residuals for a 2D advected
    quantity using either simple upwind flux or Lax-Wendroff flux, and
    handles boundary conditions in both directions.

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
    """
    Ngc = g.Ngc
    Nx1r = g.Nx1 + Ngc
    Nx2r = g.Nx2 + Ngc
    
    # Apply boundary conditions
    adv = boundCond_adv(g, par.BC, adv, par.BC_fixed)    

    #nulify the residual
    Res = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    
    #piecewise polynomial limited reconstruction with RK timestepping
    if par.solver_type == 'adv':
        if g.Nx1 > 1:
            #piecewise polynomial reconstruction
            dens_L, dens_R = VarReconstruct(adv.dens, g, par.rec_type, 1)
            #exact solution of the Riemann problem for advection
            flux = Riemann_adv(dens_L, dens_R, adv.vel1)
            #conservative residual update
            Res = (flux[1:, :]*g.fS1[1:, :] - flux[:-1, :]*g.fS1[:-1, :]) / g.cVol[:, :]

        if g.Nx2 > 1:
            #piecewise polynomial reconstruction
            dens_L, dens_R = VarReconstruct(adv.dens, g, par.rec_type, 2)
            #exact solution of the Riemann problem for advection
            flux = Riemann_adv(dens_L, dens_R, adv.vel2)
            Res += (flux[:, 1:]*g.fS2[:, 1:] - flux[:, :-1]*g.fS2[:, :-1]) / g.cVol[:, :]
            
    # second-order unlimited Lax-Wendroff scheme
    elif par.solver_type == 'LW':
        
        if (g.geom != 'cart'): raise ValueError("'LW' for advection works only for CARTESIAN grids!") 
        
        # Lax-Wendroff flux in x1 -- dx1uc (not the cell-shaped g.dx1 array,
        # which doesn't broadcast against the Nx1+1-wide face quantity below)
        # is the right spacing here: LW is restricted to Cartesian grids,
        # i.e. already uniform, just above.
        if g.Nx1 > 1:
            flux = adv.vel1 * (adv.dens[Ngc-1:Nx1r, Ngc:-Ngc] + adv.dens[Ngc:Nx1r+1, Ngc:-Ngc]) / 2.0 \
                   + adv.vel1 * (adv.vel1 * dt / g.dx1uc) * \
                   (adv.dens[Ngc-1:Nx1r, Ngc:-Ngc] - adv.dens[Ngc:Nx1r+1, Ngc:-Ngc]) / 2.0
            Res = (flux[1:, :] * g.fS1[1:, :] - flux[:-1, :] * g.fS1[:-1, :]) / g.cVol[:, :]

        # Lax-Wendroff flux in x2 (dx2uc, same reasoning as x1 above)
        if g.Nx2 > 1:
            flux = adv.vel2 * (adv.dens[Ngc:-Ngc, Ngc-1:Nx2r] + adv.dens[Ngc:-Ngc, Ngc:Nx2r+1]) / 2.0 \
                   + adv.vel2 * (adv.vel2 * dt / g.dx2uc) * \
                   (adv.dens[Ngc:-Ngc, Ngc - 1:Nx2r] - adv.dens[Ngc:-Ngc, Ngc:Nx2r+1]) / 2.0
            Res += (flux[:, 1:] * g.fS2[:, 1:] - flux[:, :-1] * g.fS2[:, :-1]) / g.cVol[:, :]

        # Multi-dimensional antidiffusion correction
        if (g.Nx1 > 1 & g.Nx2 > 1):
            Res -= dt * adv.vel1 * adv.vel2 * (
                adv.dens[Ngc-1:Nx1r-1, Ngc-1:Nx2r-1] - adv.dens[Ngc-1:Nx1r-1, Ngc+1:Nx2r+1]
                - adv.dens[Ngc+1:Nx1r+1, Ngc-1:Nx2r-1] + adv.dens[Ngc+1:Nx1r+1, Ngc+1:Nx2r+1]
                ) / 4.0 / g.dx1[Ngc:-Ngc, Ngc:-Ngc] / g.dx2[Ngc:-Ngc, Ngc:-Ngc]
    
    else:
        raise ValueError(
            f"Invalid advection solver: '{par.solver_type}'. "
            f"Expected one of ['adv', 'LW'].")

    return Res
