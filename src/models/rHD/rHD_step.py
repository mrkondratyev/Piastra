# -*- coding: utf-8 -*-
"""
rHD_step.py

Container class and time-stepping routines for 2D special-relativistic
hydrodynamics (rHD).

This module mirrors hydro_one_step.py (NR) for the relativistic Euler equations.
It provides:

- rHD2D class: lightweight container with step_RK() interface
- CFLcondition_rHD: SR-aware CFL timestep
- oneStep_rHD_RK: RK1/RK2/RK3 update with prim <-> cons conversion
- flux_calc_rHD: Godunov-type residuals with 4-velocity reconstruction

Reconstruction strategy
-----------------------
Spatial reconstruction is performed on the 4-velocity components
uⁱ = W vⁱ  (not the 3-velocities) so that the reconstructed state always
satisfies |v_face| < 1.  After reconstruction the 3-velocity is recovered via
    vⁱ = uⁱ / √(1 + |u|²).

References
----------
- Mignone, A. & Bodo, G. (2005), MNRAS 364, 126
- Del Zanna, L. et al. (2003), A&A 400, 397

Author
------
mrkondratyev
"""

import numpy as np
import copy
from src.models.rHD.rHD_phys import (
    prim2cons_rHD,
    cons2prim_rHD,
    Riemann_rHD,
    boundCond_rHD,
)
from src.common.high_order_rec import ( 
    VarReconstruct,
    _swap_troubled,
)


class rHD2D:
    """
    Container class for 2D special-relativistic hydrodynamics routines.

    Attributes
    ----------
    g : Grid
        Grid object with domain sizes, spacing, volumes, and face areas.
    HD : SimState
        Fluid state object containing primitive and conservative variables.
    eos : EOSdata
        Equation of state object.
    par : Parameters
        Simulation parameters including CFL, RK_order, flux_type, rec_type.
    """

    def __init__(self, g, HD, eos, par):
        """
        Initialize the rHD2D container.

        Parameters
        ----------
        g : Grid
        HD : SimState
        eos : EOSdata
        par : Parameters
        """
        self.g   = g
        self.HD  = HD
        self.eos = eos
        self.par = par

    def step_RK(self):
        """
        Perform a single Runge-Kutta timestep.

        Returns
        -------
        HD : SimState
            Updated fluid state.
        """
        dt = min(CFLcondition_rHD(self.g, self.HD, self.eos, self.par.CFL),
                 self.par.timefin - self.par.timenow)
        self.HD = oneStep_rHD_RK(self.g, self.HD, self.eos, self.par, dt)
        self.par.timenow += dt
        return self.HD


# -------------------------
# Small helper: one RK stage applied to all five conservative variables
# -------------------------
def _rk_stage(HD_out, HD_a, HD_b, ResM, Res1, Res2, Res3, ResE, dt, a, b, c):
    """
    Set HD_out.* = a * HD_a.* + b * HD_b.* + c * dt * Res*
 
    For SSP-RK, the standard combinations are:
      Stage 1 (predictor): a=1,    b=0,    c=-1     -> HD_h = HD - dt*R(HD)
      RK2 corrector:       a=0.5,  b=0.5,  c=-0.5
      RK3 stage 2:         a=0.75, b=0.25, c=-0.25
      RK3 stage 3 (final): a=1/3,  b=2/3,  c=-2/3
    """
    HD_out.mass = a * HD_a.mass + b * HD_b.mass + c * dt * ResM
    HD_out.mom1 = a * HD_a.mom1 + b * HD_b.mom1 + c * dt * Res1
    HD_out.mom2 = a * HD_a.mom2 + b * HD_b.mom2 + c * dt * Res2
    HD_out.mom3 = a * HD_a.mom3 + b * HD_b.mom3 + c * dt * Res3
    HD_out.etot = a * HD_a.etot + b * HD_b.etot + c * dt * ResE

# -------------------------
# Function definitions
# -------------------------

def CFLcondition_rHD(g, HD, eos, CFL):
    """
    Compute the maximum stable timestep for 2D special-relativistic
    hydrodynamics using the SR CFL condition.

    The maximum signal speed in each cell is estimated as
        λ_max = max(|Sl|, |Sr|)
    using the SR wave-speed formula from Mignone & Bodo (2005), eqs. (9)-(10).

    A simplified but slightly conservative estimate is used here:
        λ ≈ (|v| + cs) / (1 + |v| cs)
    which bounds the actual fastest characteristic speed.

    Parameters
    ----------
    g   : Grid
    HD  : SimState
    eos : EOSdata
    CFL : float

    Returns
    -------
    dt : float
    
    Notes
    -------
    Lame coefficient hx2 is included in the CFL calculation 
    in order to adjust the correct timestep for the simulations in the polar coordinates, 
    e.g. dt ~ rdφ for the cylindrical polar geometry
    """
    Ngc = g.Ngc

    dens = HD.dens[Ngc:-Ngc, Ngc:-Ngc]
    vel1 = HD.vel1[Ngc:-Ngc, Ngc:-Ngc]
    vel2 = HD.vel2[Ngc:-Ngc, Ngc:-Ngc]
    pres = HD.pres[Ngc:-Ngc, Ngc:-Ngc]

    cs = eos.sound_speed_sr(dens, pres)

    # SR characteristic speed estimate: (|v| + cs) / (1 + |v|*cs)
    # it is an inexact but conservative one 
    lam1 = (np.abs(vel1) + cs) / (1.0 + np.abs(vel1) * cs + 1e-14)
    lam2 = (np.abs(vel2) + cs) / (1.0 + np.abs(vel2) * cs + 1e-14)

    dt_inv = np.max(lam1 / g.dx1[Ngc:-Ngc, Ngc:-Ngc] +
                    lam2 / (g.dx2[Ngc:-Ngc, Ngc:-Ngc] * g.hx2[Ngc:-Ngc, Ngc:-Ngc]))
    
    return CFL / dt_inv


def oneStep_rHD_RK(g, HD, eos, par, dt):
    """
    Perform a single Runge-Kutta timestep for 2D special-relativistic
    hydrodynamics.

    Implements RK1, RK2, and RK3 (Shu-Osher) schemes.

    Parameters
    ----------
    g   : Grid
    HD  : SimState
    eos : EOSdata
    par : Parameters
    dt  : float

    Returns
    -------
    HD : SimState
        Updated fluid state after one RK timestep.
    """
    Ngc = g.Ngc

    HD_h = copy.deepcopy(HD)

    # Conservative variables at the beginning of the timestep
    HD.mass, HD.mom1, HD.mom2, HD.mom3, HD.etot = \
        prim2cons_rHD(
            HD.dens[Ngc:-Ngc, Ngc:-Ngc],
            HD.vel1[Ngc:-Ngc, Ngc:-Ngc],
            HD.vel2[Ngc:-Ngc, Ngc:-Ngc],
            HD.vel3[Ngc:-Ngc, Ngc:-Ngc],
            HD.pres[Ngc:-Ngc, Ngc:-Ngc],
            eos)

    # 1st RK stage (predictor)
    ResM, Res1, Res2, Res3, ResE = flux_calc_rHD(g, HD, par, eos)

    _rk_stage(HD_h, HD, HD, \
        ResM, Res1, Res2, Res3, ResE, dt, 1.0, 0.0, -1.0)

    if par.RK_order == 'RK1':
        
        #simply rewrite the conservative state here for clarity
        HD.mass = HD_h.mass
        HD.mom1 = HD_h.mom1
        HD.mom2 = HD_h.mom2
        HD.mom3 = HD_h.mom3
        HD.etot = HD_h.etot

    elif par.RK_order == 'RK2':
        
        # Primitive recovery after predictor stage
        _prim_recovery(HD_h, Ngc, HD.pres[Ngc:-Ngc, Ngc:-Ngc], eos)

        # 2nd RK stage (corrector)
        ResM, Res1, Res2, Res3, ResE = flux_calc_rHD(g, HD_h, par, eos)

        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy
        _rk_stage(HD, HD, HD_h, \
            ResM, Res1, Res2, Res3, ResE, dt, 1.0/2.0, 1.0/2.0, -1.0/2.0)

    elif par.RK_order == 'RK3':
        
        # Primitive recovery after 1st stage
        _prim_recovery(HD_h, Ngc, HD.pres[Ngc:-Ngc, Ngc:-Ngc], eos)

        # 2nd RK stage
        ResM, Res1, Res2, Res3, ResE = flux_calc_rHD(g, HD_h, par, eos)

        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy        
        _rk_stage(HD_h, HD, HD_h, \
            ResM, Res1, Res2, Res3, ResE, dt, 1.0/4.0, 3.0/4.0, -1.0/4.0)

        # Primitive recovery after 2nd stage
        _prim_recovery(HD_h, Ngc, HD_h.pres[Ngc:-Ngc, Ngc:-Ngc], eos)

        # 3rd RK stage (final)
        ResM, Res1, Res2, Res3, ResE = flux_calc_rHD(g, HD_h, par, eos)

        # Conservative update - final 3rd RK iteration
        # update mass, three components of momentum and total energy
        _rk_stage(HD, HD, HD_h, \
            ResM, Res1, Res2, Res3, ResE, dt, 2.0/3.0, 1.0/3.0, -2.0/3.0)

    else:
        
        raise ValueError(
            f"Invalid RK_order: '{par.RK_order}'. "
            f"Expected one of ['RK1', 'RK2', 'RK3'].")

    # Final primitive variable recovery
    _prim_recovery(HD, Ngc, HD_h.pres[Ngc:-Ngc, Ngc:-Ngc], eos)

    return HD



# ============================================================================
# Helper: call cons2prim_sr_hydro for a SimState object
# ============================================================================
def _prim_recovery(state, Ngc, init_pres, eos):
    """
    Call cons2prim_nr_hydro and write results back into
    state.{dens,vel*,pres}.

    Parameters
    ----------
    state     : SimState with conservative vars populated
    init_pres : initial guess for the unknown pressure 
    Ngc       : int number of ghost cells
    eos       : EOSdata
    """
    (state.dens[Ngc:-Ngc, Ngc:-Ngc],
     state.vel1[Ngc:-Ngc, Ngc:-Ngc],
     state.vel2[Ngc:-Ngc, Ngc:-Ngc],
     state.vel3[Ngc:-Ngc, Ngc:-Ngc],
     state.pres[Ngc:-Ngc, Ngc:-Ngc]) = \
        cons2prim_rHD(
            state.mass, state.mom1, state.mom2, state.mom3, state.etot, \
            init_pres, eos)



def flux_calc_rHD(g, HD, par, eos):
    """
    Compute residuals for conservative variables in 2D special-relativistic
    hydrodynamics.

    Reconstruction is performed on the 4-velocity components uⁱ = W vⁱ to
    ensure |v_face| < 1 after reconstruction.  The recovered 3-velocities at
    the faces are then passed to the SR Riemann solver.

    Parameters
    ----------
    g   : Grid
    HD  : SimState
    par : Parameters
    eos : EOSdata

    Returns
    -------
    ResM, Res1, Res2, Res3, ResE : ndarray
        Residuals for D, m₁, m₂, m₃, E.
    """
    # Apply boundary conditions
    HD = boundCond_rHD(g, par.BC, HD, par.BC_fixed)

    Ngc = g.Ngc

    ResM = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res1 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res2 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res3 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    ResE = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    
    # limiter type for flattening of cells with potentially unphysical behaviour
    # we switch to PLM for such cells
    lim = 'VL' #'VL', 'MM', 'MC', 'KOR', 'PCM', 'NO'

    # --- Precompute 4-velocity components ---
    # W = 1 / sqrt(1 - v1^2 - v2^2 - v3^2)
    v2 = HD.vel1**2 + HD.vel2**2 + HD.vel3**2
    v2 = np.clip(v2, 0.0, 1.0 - 1e-14)
    W_full = 1.0 / np.sqrt(1.0 - v2)

    u1 = W_full * HD.vel1   # 4-velocity component 1
    u2 = W_full * HD.vel2   # 4-velocity component 2
    u3 = W_full * HD.vel3   # 4-velocity component 3

    # Fluxes in x1 direction
    if g.Nx1 > 1:
        # Reconstruct density, pressure, and 4-velocity components
        dens_L, dens_R = VarReconstruct(HD.dens, g, par.rec_type, 1)
        pres_L, pres_R = VarReconstruct(HD.pres, g, par.rec_type, 1)
        u1_L,   u1_R   = VarReconstruct(u1,      g, par.rec_type, 1)
        u2_L,   u2_R   = VarReconstruct(u2,      g, par.rec_type, 1)
        u3_L,   u3_R   = VarReconstruct(u3,      g, par.rec_type, 1)
        
        #Lorenz factors 
        W_L = np.sqrt(1.0 + u1_L**2 + u2_L**2 + u3_L**2)
        W_R = np.sqrt(1.0 + u1_R**2 + u2_R**2 + u3_R**2)
        
        # Detect troubled faces: unphysical states or strong pressure jump
        p_lc = HD.pres[Ngc - 1:g.Nx1r,     Ngc:-Ngc]
        p_rc = HD.pres[Ngc    :g.Nx1r + 1, Ngc:-Ngc]
        troubled = ((dens_L <= 0.0) | (dens_R <= 0.0) |
                    (pres_L <= 0.0) | (pres_R <= 0.0) |
                    (W_L <= 0.0) | (W_R <= 0.0) |
                    (np.abs(p_rc - p_lc) > 0.33 * np.minimum(p_lc, p_rc)))

        # Fallback to PLM with minmod at troubled faces
        dens_L, dens_R = _swap_troubled(dens_L, dens_R, HD.dens, g, 1, lim, troubled)
        pres_L, pres_R = _swap_troubled(pres_L, pres_R, HD.pres, g, 1, lim, troubled)
        u1_L,   u1_R   = _swap_troubled(u1_L,   u1_R,   u1,      g, 1, lim, troubled)
        u2_L,   u2_R   = _swap_troubled(u2_L,   u2_R,   u2,      g, 1, lim, troubled)
        u3_L,   u3_R   = _swap_troubled(u3_L,   u3_R,   u3,      g, 1, lim, troubled)
        
        # Recover 3-velocities from 4-velocities: vⁱ = uⁱ / sqrt(1 + |u|²)
        W_L = np.sqrt(1.0 + u1_L**2 + u2_L**2 + u3_L**2)
        W_R = np.sqrt(1.0 + u1_R**2 + u2_R**2 + u3_R**2)
        vel1_L = u1_L / W_L;  vel1_R = u1_R / W_R
        vel2_L = u2_L / W_L;  vel2_R = u2_R / W_R
        vel3_L = u3_L / W_L;  vel3_R = u3_R / W_R

        #Riemann problem solution to obtain fluxes 
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            Riemann_rHD(
                dens_L, dens_R,
                vel1_L, vel1_R, vel2_L, vel2_R, vel3_L, vel3_R,
                pres_L, pres_R, eos, par.solver_type, 1)

        # residuals update 
        ResM = (Fmass[1:, :] * g.fS1[1:, :] - Fmass[:-1, :] * g.fS1[:-1, :]) / g.cVol
        Res1 = (Fmomx[1:, :] * g.fS1[1:, :] - Fmomx[:-1, :] * g.fS1[:-1, :]) / g.cVol
        Res2 = (Fmomy[1:, :] * g.fS1[1:, :] - Fmomy[:-1, :] * g.fS1[:-1, :]) / g.cVol
        Res3 = (Fmomz[1:, :] * g.fS1[1:, :] - Fmomz[:-1, :] * g.fS1[:-1, :]) / g.cVol
        ResE = (Fetot[1:, :] * g.fS1[1:, :] - Fetot[:-1, :] * g.fS1[:-1, :]) / g.cVol

    # Fluxes in x2 direction
    if g.Nx2 > 1:
        dens_L, dens_R = VarReconstruct(HD.dens, g, par.rec_type, 2)
        pres_L, pres_R = VarReconstruct(HD.pres, g, par.rec_type, 2)
        u1_L,   u1_R   = VarReconstruct(u1,      g, par.rec_type, 2)
        u2_L,   u2_R   = VarReconstruct(u2,      g, par.rec_type, 2)
        u3_L,   u3_R   = VarReconstruct(u3,      g, par.rec_type, 2)
        
        #Lorenz factors 
        W_L = np.sqrt(1.0 + u1_L**2 + u2_L**2 + u3_L**2)
        W_R = np.sqrt(1.0 + u1_R**2 + u2_R**2 + u3_R**2)
        
        # Detect troubled faces: unphysical states or strong pressure jump
        p_lc = HD.pres[Ngc:-Ngc, Ngc - 1:g.Nx2r    ]
        p_rc = HD.pres[Ngc:-Ngc, Ngc    :g.Nx2r + 1]
        troubled = ((dens_L <= 0.0) | (dens_R <= 0.0) |
                    (pres_L <= 0.0) | (pres_R <= 0.0) |
                    (W_L <= 0.0) | (W_R <= 0.0) |
                    (np.abs(p_rc - p_lc) > 0.33 * np.minimum(p_lc, p_rc)))
        
        # Fallback to PLM with minmod at troubled faces
        dens_L, dens_R = _swap_troubled(dens_L, dens_R, HD.dens, g, 2, lim, troubled)
        pres_L, pres_R = _swap_troubled(pres_L, pres_R, HD.pres, g, 2, lim, troubled)
        u1_L,   u1_R   = _swap_troubled(u1_L,   u1_R,   u1,      g, 2, lim, troubled)
        u2_L,   u2_R   = _swap_troubled(u2_L,   u2_R,   u2,      g, 2, lim, troubled)
        u3_L,   u3_R   = _swap_troubled(u3_L,   u3_R,   u3,      g, 2, lim, troubled)
        
        # turn back to 3-velocities 
        W_L = np.sqrt(1.0 + u1_L**2 + u2_L**2 + u3_L**2)
        W_R = np.sqrt(1.0 + u1_R**2 + u2_R**2 + u3_R**2)
        vel1_L = u1_L / W_L;  vel1_R = u1_R / W_R
        vel2_L = u2_L / W_L;  vel2_R = u2_R / W_R
        vel3_L = u3_L / W_L;  vel3_R = u3_R / W_R

        #Riemann problem solution to obtain fluxes 
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            Riemann_rHD(
                dens_L, dens_R,
                vel1_L, vel1_R, vel2_L, vel2_R, vel3_L, vel3_R,
                pres_L, pres_R, eos, par.solver_type, 2)

        #residuals update 
        ResM += (Fmass[:, 1:] * g.fS2[:, 1:] - Fmass[:, :-1] * g.fS2[:, :-1]) / g.cVol
        Res1 += (Fmomx[:, 1:] * g.fS2[:, 1:] - Fmomx[:, :-1] * g.fS2[:, :-1]) / g.cVol
        Res2 += (Fmomy[:, 1:] * g.fS2[:, 1:] - Fmomy[:, :-1] * g.fS2[:, :-1]) / g.cVol
        Res3 += (Fmomz[:, 1:] * g.fS2[:, 1:] - Fmomz[:, :-1] * g.fS2[:, :-1]) / g.cVol
        ResE += (Fetot[:, 1:] * g.fS2[:, 1:] - Fetot[:, :-1] * g.fS2[:, :-1]) / g.cVol

    # Curvature source terms for different curvilinear coordinates
    ST1, ST2, ST3 = curv_source_rHD(g, HD, eos)
    
    # Finally, here we add the external force + curvature source terms
    # Source term for momentum residual 
    Res1 += - HD.dens[Ngc:-Ngc, Ngc:-Ngc] * HD.F1[:,:] - ST1
    Res2 += - HD.dens[Ngc:-Ngc, Ngc:-Ngc] * HD.F2[:,:] - ST2
    Res3 += - ST3

    # Source term for energy residual 
    ResE += -HD.dens[Ngc:-Ngc, Ngc:-Ngc] * (
        HD.F1 * HD.vel1[Ngc:-Ngc, Ngc:-Ngc] +
        HD.F2 * HD.vel2[Ngc:-Ngc, Ngc:-Ngc])

    return ResM, Res1, Res2, Res3, ResE


  
def curv_source_rHD(g, HD, eos):
    """
    Geometric (curvature) source terms for the SPECIAL-RELATIVISTIC
    hydrodynamic equations in curvilinear coordinates (finite-volume form).

    In Cartesian coordinates the relativistic Euler equations are source-free;
    in curvilinear geometries the divergence of the stress tensor expressed in
    a non-Cartesian basis produces momentum source terms. The relativistic form
    differs from the Newtonian one by replacing the momentum-flux prefactor

        dens            ->      rho_h * W**2

    where  rho_h = rho + rho*eps + p  is the relativistic specific enthalpy
    (times rest density) and  W = 1/sqrt(1 - v^2)  is the Lorentz factor. The
    isotropic pressure terms and the geometric factors (1/r, cot(theta)/r) are
    unchanged from the Newtonian case.

    For an ideal-gas (gamma-law) EOS,  rho_h = rho + Gamma/(Gamma-1) * p.

    Parameters
    ----------
    g : object
        Grid: geom ('cart'/'cyl'/'pol'/'sph'), cx1, fx2, Ngc, Nx1, Nx2.
    HD : object
        Primitive state: dens, pres, vel1, vel2, vel3 (3-velocities).
    eos : object
        Equation of state providing GAMMA (adiabatic index).

    Returns
    -------
    ST1, ST2, ST3 : ndarray, shape (Nx1, Nx2)
        Momentum source terms (zero in ghost zones / Cartesian geometry).

    Notes
    -----
    - rho_h * W**2 is the relativistic momentum-density / stress prefactor;
      using the Newtonian 'dens' here is incorrect for v -> 1 (it omits the
      Lorentz-factor and enthalpy enhancement that dominate the SR stress).
    - The spherical cot(theta) is evaluated in a volume-consistent way from
      face values of sin/cos rather than pointwise, matching the FV scheme.
    """
    Ngc = g.Ngc
    ST1 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    ST2 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    ST3 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    
    # source-free; nothing further to do
    if g.geom == 'cart':
        return ST1, ST2, ST3

    # --- interior primitives ---
    r    = g.cx1[Ngc:-Ngc, Ngc:-Ngc]
    dens = HD.dens[Ngc:-Ngc, Ngc:-Ngc]
    pres = HD.pres[Ngc:-Ngc, Ngc:-Ngc]
    v1   = HD.vel1[Ngc:-Ngc, Ngc:-Ngc]
    v2   = HD.vel2[Ngc:-Ngc, Ngc:-Ngc]
    v3   = HD.vel3[Ngc:-Ngc, Ngc:-Ngc]

    # --- relativistic momentum-flux prefactor:  rho_h * W**2 ---
    v2sq = v1 * v1 + v2 * v2 + v3 * v3
    W2 = 1.0 / (1.0 - v2sq) # Lorentz factor squared
    rhoh = dens * eos.enthalpy_sr(dens, pres) # dens * enthalpy
    D = rhoh * W2

    # --- cylindrical (R, Z) ---
    if g.geom == 'cyl':
        ST1 = (pres + D * v3 * v3) / r
        ST3 = -D * v3 * v1 / r

    # --- polar (R, phi) ---
    elif g.geom == 'pol':
        ST1 = (pres + D * v2 * v2) / r
        ST2 = -D * v2 * v1 / r

    # --- spherical-polar (r, theta) ---
    elif g.geom == 'sph':
        if g.Nx2 > 1:
            sin_theta = np.sin(g.fx2[Ngc:g.Nx1 + Ngc, Ngc:g.Nx2 + Ngc + 1])
            cos_theta = np.cos(g.fx2[Ngc:g.Nx1 + Ngc, Ngc:g.Nx2 + Ngc + 1])
            cot = (sin_theta[:, 1:] - sin_theta[:, :-1]) / \
                  (cos_theta[:, :-1] - cos_theta[:, 1:])
        else:
            cot = np.zeros_like(r)

        ST1 = (2.0 * pres + D * (v2 * v2 + v3 * v3)) / r
        ST2 = (pres + D * v3 * v3) * cot / r - D * v1 * v2 / r
        ST3 = -(D * v2 * v3) * cot / r - D * v1 * v3 / r

    return ST1, ST2, ST3
