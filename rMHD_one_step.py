# -*- coding: utf-8 -*-
"""
rMHD_one_step.py
================

Container class and time-stepping routines for 2D Special-Relativistic
Magnetohydrodynamics (SRMHD) with Constrained Transport (CT) divergence control.

This module mirrors MHD_one_step_CT.py for the relativistic MHD equations.
Spatial reconstruction is performed on the 4-velocity components  u^i = W v^i
(identical strategy to rHD_one_step.py) to guarantee  |v_face| < 1 after
reconstruction.  The magnetic field is advanced using the Constrained Transport
method ('flux-CT') so that div B = 0 is preserved to machine precision.

Components
----------
  rMHD2D_CT               container class with  step_RK()  interface
  CFLcondition_rMHD       SR CFL timestep using fast magnetosonic speed
  oneStep_rMHD_RK_CT      TVD Runge-Kutta update (RK1 / RK2 / RK3)
  flux_calc_rMHD_CT       residual computation (Godunov fluxes + CT electric field)
  boundCond_electric_field_rMHD   ghost-cell fill for face-centred electric field

References
----------
  Del Zanna, Bucciantini & Londrillo (2003), A&A 400, 397
  Mignone & Bodo (2006), MNRAS 368, 1040
  Shu & Osher (1988), J. Comput. Phys. 77, 439
  Evans & Hawley (1988), ApJ 332, 659   (Constrained Transport)

Author
------
mrkondratyev
"""

import copy
import numpy as np

from src.grid.grid_misc import interp_face_to_cell, div_face_vector
from src.common.high_order_rec import VarReconstruct

from src.models.rMHD.rMHD_phys import (
    prim2cons_sr_MHD,
    cons2prim_sr_MHD,
    fast_magnetosonic_speed_sr,
    Riemann_sr_MHD,
    boundCond_rMHD,
    _lorentz,
)


# ============================================================================
# Container class
# ============================================================================

class rMHD2D_CT:
    """
    Container class for 2D Special-Relativistic MHD with Constrained Transport.

    Attributes
    ----------
    grid : Grid
    MHD  : SimState
    eos  : EOSdata
    par  : Parameters
    """

    def __init__(self, grid, MHD, eos, par):
        self.grid = grid
        self.MHD  = MHD
        self.eos  = eos
        self.par  = par

    def step_RK(self):
        """Advance the SRMHD state by one Runge-Kutta timestep."""
        dt = min(
            CFLcondition_rMHD(self.grid, self.MHD, self.eos, self.par.CFL),
            self.par.timefin - self.par.timenow,
        )
        self.MHD = oneStep_rMHD_RK_CT(self.grid, self.MHD, self.eos, self.par, dt)
        self.par.timenow += dt
        return self.MHD


# ============================================================================
# CFL condition
# ============================================================================

def CFLcondition_rMHD(grid, MHD, eos, CFL):
    """
    Compute the stable timestep for SRMHD using the fast magnetosonic speed.

    The signal speed in each direction accounts for the relativistic velocity
    addition formula:  lam = (|v| + c_f) / (1 + |v| c_f).

    Parameters
    ----------
    grid : Grid
    MHD  : SimState
    eos  : EOSdata
    CFL  : float

    Returns
    -------
    dt : float
    """
    Ngc = grid.Ngc

    dens = MHD.dens[Ngc:-Ngc, Ngc:-Ngc]
    vel1 = MHD.vel1[Ngc:-Ngc, Ngc:-Ngc]
    vel2 = MHD.vel2[Ngc:-Ngc, Ngc:-Ngc]
    vel3 = MHD.vel3[Ngc:-Ngc, Ngc:-Ngc]
    pres = MHD.pres[Ngc:-Ngc, Ngc:-Ngc]
    B1   = MHD.bfi1[Ngc:-Ngc, Ngc:-Ngc]
    B2   = MHD.bfi2[Ngc:-Ngc, Ngc:-Ngc]
    B3   = MHD.bfi3[Ngc:-Ngc, Ngc:-Ngc]

    cf = fast_magnetosonic_speed_sr(dens, pres, vel1, vel2, vel3, B1, B2, B3, eos)

    # Relativistic signal speed in each direction
    lam1 = (np.abs(vel1) + cf) / (1.0 + np.abs(vel1) * cf + 1e-14)
    lam2 = (np.abs(vel2) + cf) / (1.0 + np.abs(vel2) * cf + 1e-14)

    dt_inv = np.max(
        lam1 / grid.dx1[Ngc:-Ngc, Ngc:-Ngc] +
        lam2 / grid.dx2[Ngc:-Ngc, Ngc:-Ngc]
    )
    return CFL / dt_inv


# ============================================================================
# Single RK timestep
# ============================================================================

def oneStep_rMHD_RK_CT(grid, MHD, eos, par, dt):
    """
    Advance the SRMHD state by one timestep using RK1, RK2, or RK3 (TVD).

    Mirrors oneStep_MHD_RK_CT but uses SR conservative variables and
    4-velocity reconstruction.

    Parameters
    ----------
    grid : Grid
    MHD  : SimState
    eos  : EOSdata
    par  : Parameters
    dt   : float

    Returns
    -------
    MHD : SimState  updated state
    """
    Ngc   = grid.Ngc
    MHD_h = copy.deepcopy(MHD)

    # ---- prim -> cons at beginning of timestep --------------------------
    (MHD.mass, MHD.mom1, MHD.mom2, MHD.mom3, MHD.etot,
     MHD.bcon1, MHD.bcon2, MHD.bcon3) = \
        prim2cons_sr_MHD(
            MHD.dens[Ngc:-Ngc, Ngc:-Ngc],
            MHD.vel1[Ngc:-Ngc, Ngc:-Ngc],
            MHD.vel2[Ngc:-Ngc, Ngc:-Ngc],
            MHD.vel3[Ngc:-Ngc, Ngc:-Ngc],
            MHD.pres[Ngc:-Ngc, Ngc:-Ngc],
            MHD.bfi1[Ngc:-Ngc, Ngc:-Ngc],
            MHD.bfi2[Ngc:-Ngc, Ngc:-Ngc],
            MHD.bfi3[Ngc:-Ngc, Ngc:-Ngc],
            eos)

    # Store x = rho h W^2 for the Newton solver initial guess
    MHD._x_guess = MHD.dens[Ngc:-Ngc, Ngc:-Ngc] + \
        MHD.pres[Ngc:-Ngc, Ngc:-Ngc] * eos.GAMMA / (eos.GAMMA - 1.0)
    
    # ---- 1st RK stage (predictor) --------------------------------------
    ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = \
        flux_calc_rMHD_CT(grid, MHD, par, eos)

    MHD_h.mass  = MHD.mass  - dt * ResM
    MHD_h.mom1  = MHD.mom1  - dt * ResV1
    MHD_h.mom2  = MHD.mom2  - dt * ResV2
    MHD_h.mom3  = MHD.mom3  - dt * ResV3
    MHD_h.etot  = MHD.etot  - dt * ResE
    MHD_h.fb1   = MHD.fb1   - dt * ResB1
    MHD_h.fb2   = MHD.fb2   - dt * ResB2
    MHD_h.bcon3 = MHD.bcon3 - dt * ResB3

    # ---- RK1 -----------------------------------------------------------
    if par.RK_order == 'RK1':
        MHD.mass  = MHD_h.mass
        MHD.mom1  = MHD_h.mom1
        MHD.mom2  = MHD_h.mom2
        MHD.mom3  = MHD_h.mom3
        MHD.etot  = MHD_h.etot
        MHD.fb1   = MHD_h.fb1
        MHD.fb2   = MHD_h.fb2
        MHD.bcon3 = MHD_h.bcon3

    # ---- RK2 -----------------------------------------------------------
    if par.RK_order == 'RK2':
        MHD_h.bcon1, MHD_h.bcon2 = interp_face_to_cell(grid, MHD_h.fb1, MHD_h.fb2)
        _prim_recovery(MHD_h, MHD._x_guess, Ngc, eos)

        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = \
            flux_calc_rMHD_CT(grid, MHD_h, par, eos)

        MHD.mass  = (MHD_h.mass  + MHD.mass)  / 2.0 - dt * ResM  / 2.0
        MHD.mom1  = (MHD_h.mom1  + MHD.mom1)  / 2.0 - dt * ResV1 / 2.0
        MHD.mom2  = (MHD_h.mom2  + MHD.mom2)  / 2.0 - dt * ResV2 / 2.0
        MHD.mom3  = (MHD_h.mom3  + MHD.mom3)  / 2.0 - dt * ResV3 / 2.0
        MHD.etot  = (MHD_h.etot  + MHD.etot)  / 2.0 - dt * ResE  / 2.0
        MHD.fb1   = (MHD_h.fb1   + MHD.fb1)   / 2.0 - dt * ResB1 / 2.0
        MHD.fb2   = (MHD_h.fb2   + MHD.fb2)   / 2.0 - dt * ResB2 / 2.0
        MHD.bcon3 = (MHD_h.bcon3 + MHD.bcon3) / 2.0 - dt * ResB3 / 2.0

    # ---- RK3 (Shu-Osher) -----------------------------------------------
    if par.RK_order == 'RK3':
        # Stage 2
        MHD_h.bcon1, MHD_h.bcon2 = interp_face_to_cell(grid, MHD_h.fb1, MHD_h.fb2)
        _prim_recovery(MHD_h, MHD._x_guess, Ngc, eos)

        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = \
            flux_calc_rMHD_CT(grid, MHD_h, par, eos)

        MHD_h.mass  = (MHD_h.mass  + 3.0 * MHD.mass)  / 4.0 - dt * ResM  / 4.0
        MHD_h.mom1  = (MHD_h.mom1  + 3.0 * MHD.mom1)  / 4.0 - dt * ResV1 / 4.0
        MHD_h.mom2  = (MHD_h.mom2  + 3.0 * MHD.mom2)  / 4.0 - dt * ResV2 / 4.0
        MHD_h.mom3  = (MHD_h.mom3  + 3.0 * MHD.mom3)  / 4.0 - dt * ResV3 / 4.0
        MHD_h.etot  = (MHD_h.etot  + 3.0 * MHD.etot)  / 4.0 - dt * ResE  / 4.0
        MHD_h.fb1   = (MHD_h.fb1   + 3.0 * MHD.fb1)   / 4.0 - dt * ResB1 / 4.0
        MHD_h.fb2   = (MHD_h.fb2   + 3.0 * MHD.fb2)   / 4.0 - dt * ResB2 / 4.0
        MHD_h.bcon3 = (MHD_h.bcon3 + 3.0 * MHD.bcon3) / 4.0 - dt * ResB3 / 4.0

        # Stage 3
        MHD_h.bcon1, MHD_h.bcon2 = interp_face_to_cell(grid, MHD_h.fb1, MHD_h.fb2)
        _prim_recovery(MHD_h, MHD._x_guess, Ngc, eos)

        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = \
            flux_calc_rMHD_CT(grid, MHD_h, par, eos)

        MHD.mass  = (2.0 * MHD_h.mass  + MHD.mass)  / 3.0 - 2.0 * dt * ResM  / 3.0
        MHD.mom1  = (2.0 * MHD_h.mom1  + MHD.mom1)  / 3.0 - 2.0 * dt * ResV1 / 3.0
        MHD.mom2  = (2.0 * MHD_h.mom2  + MHD.mom2)  / 3.0 - 2.0 * dt * ResV2 / 3.0
        MHD.mom3  = (2.0 * MHD_h.mom3  + MHD.mom3)  / 3.0 - 2.0 * dt * ResV3 / 3.0
        MHD.etot  = (2.0 * MHD_h.etot  + MHD.etot)  / 3.0 - 2.0 * dt * ResE  / 3.0
        MHD.fb1   = (2.0 * MHD_h.fb1   + MHD.fb1)   / 3.0 - 2.0 * dt * ResB1 / 3.0
        MHD.fb2   = (2.0 * MHD_h.fb2   + MHD.fb2)   / 3.0 - 2.0 * dt * ResB2 / 3.0
        MHD.bcon3 = (2.0 * MHD_h.bcon3 + MHD.bcon3) / 3.0 - 2.0 * dt * ResB3 / 3.0

    # ---- Final prim recovery and divB ----------------------------------
    MHD.bcon1, MHD.bcon2 = interp_face_to_cell(grid, MHD.fb1, MHD.fb2)
    _prim_recovery(MHD, MHD._x_guess, Ngc, eos)
    MHD.divB = div_face_vector(grid, MHD.fb1, MHD.fb2)

    return MHD


# ============================================================================
# Helper: call cons2prim_sr_MHD for a SimState object
# ============================================================================

def _prim_recovery(state, x_guess, Ngc, eos):
    """
    Call cons2prim_sr_MHD and write results back into
    state.{dens,vel*,pres,bfi*}.

    Parameters
    ----------
    state   : SimState  with conservative vars populated
    x_guess : ndarray   initial guess for x = rho h W^2
    Ngc     : int       number of ghost cells
    eos     : EOSdata
    """
    (state.dens[Ngc:-Ngc, Ngc:-Ngc],
     state.vel1[Ngc:-Ngc, Ngc:-Ngc],
     state.vel2[Ngc:-Ngc, Ngc:-Ngc],
     state.vel3[Ngc:-Ngc, Ngc:-Ngc],
     state.pres[Ngc:-Ngc, Ngc:-Ngc],
     state.bfi1[Ngc:-Ngc, Ngc:-Ngc],
     state.bfi2[Ngc:-Ngc, Ngc:-Ngc],
     state.bfi3[Ngc:-Ngc, Ngc:-Ngc]) = \
        cons2prim_sr_MHD(
            state.mass, state.mom1, state.mom2, state.mom3, state.etot,
            state.bcon1, state.bcon2, state.bcon3,
            x_guess, eos)


# ============================================================================
# Residual computation
# ============================================================================

def flux_calc_rMHD_CT(grid, MHD, par, eos):
    """
    Compute residuals for SRMHD conservative variables using Godunov fluxes
    and the flux-CT divergence control.

    Reconstruction is performed on the 4-velocity  u^i = W v^i  (same
    strategy as in rHD_one_step.py) so that the reconstructed face state
    always satisfies  |v_face| < 1.

    Parameters
    ----------
    grid : Grid
    MHD  : SimState  (primitives must be valid + ghost cells filled)
    par  : Parameters
    eos  : EOSdata

    Returns
    -------
    ResM, ResV1, ResV2, ResV3, ResE : ndarray  shape (Nx1, Nx2)
        Residuals for D, S_1, S_2, S_3, E.
    ResB1 : ndarray  shape (Nx1+1, Nx2)   residual for face B-field (x1)
    ResB2 : ndarray  shape (Nx1, Nx2+1)   residual for face B-field (x2)
    ResB3 : ndarray  shape (Nx1, Nx2)     residual for cell-centred Bz
    """
    MHD = boundCond_rMHD(grid, par.BC, MHD)

    Ngc  = grid.Ngc
    Nx1  = grid.Nx1
    Nx2  = grid.Nx2
    Nx1r = grid.Nx1r
    Nx2r = grid.Nx2r

    MHD.divB[:, :] = 0.0

    ResM  = np.zeros((Nx1, Nx2))
    ResV1 = np.zeros((Nx1, Nx2))
    ResV2 = np.zeros((Nx1, Nx2))
    ResV3 = np.zeros((Nx1, Nx2))
    ResE  = np.zeros((Nx1, Nx2))
    ResB1 = np.zeros((Nx1 + 1, Nx2))
    ResB2 = np.zeros((Nx1, Nx2 + 1))
    ResB3 = np.zeros((Nx1, Nx2))

    fluxB21 = np.zeros_like(grid.fx1)   # B2 flux through x1-faces (for CT)
    fluxB12 = np.zeros_like(grid.fx2)   # B1 flux through x2-faces (for CT)

    # ----------------------------------------------------------------
    # Precompute 4-velocity  u^i = W v^i
    # ----------------------------------------------------------------
    W_full = _lorentz(MHD.vel1, MHD.vel2, MHD.vel3)
    u1 = W_full * MHD.vel1
    u2 = W_full * MHD.vel2
    u3 = W_full * MHD.vel3

    # ----------------------------------------------------------------
    # Fluxes in x1-direction
    # ----------------------------------------------------------------
    if grid.Nx1 > 1:

        dens_L, dens_R = VarReconstruct(MHD.dens, grid, par.rec_type, 1)
        pres_L, pres_R = VarReconstruct(MHD.pres, grid, par.rec_type, 1)
        bfi2_L, bfi2_R = VarReconstruct(MHD.bfi2, grid, par.rec_type, 1)
        bfi3_L, bfi3_R = VarReconstruct(MHD.bfi3, grid, par.rec_type, 1)
        u1_L,   u1_R   = VarReconstruct(u1, grid, par.rec_type, 1)
        u2_L,   u2_R   = VarReconstruct(u2, grid, par.rec_type, 1)
        u3_L,   u3_R   = VarReconstruct(u3, grid, par.rec_type, 1)

        # Recover 3-velocities from 4-velocities
        W_L = np.sqrt(1.0 + u1_L**2 + u2_L**2 + u3_L**2)
        W_R = np.sqrt(1.0 + u1_R**2 + u2_R**2 + u3_R**2)
        v1_L = u1_L / W_L;  v1_R = u1_R / W_R
        v2_L = u2_L / W_L;  v2_R = u2_R / W_R
        v3_L = u3_L / W_L;  v3_R = u3_R / W_R

        (Fmass, Fmom1, Fmom2, Fmom3, Fetot,
         Fbfi1, fluxB21[Ngc:Nx1r+1, Ngc:-Ngc], Fbfi3) = \
            Riemann_sr_MHD(
                dens_L, dens_R,
                v1_L, v1_R, v2_L, v2_R, v3_L, v3_R,
                pres_L, pres_R,
                MHD.fb1[:, :], MHD.fb1[:, :],
                bfi2_L, bfi2_R, bfi3_L, bfi3_R,
                eos, par.flux_type, 1)

        ResM  = (Fmass[1:, :] * grid.fS1[1:, :] - Fmass[:-1, :] * grid.fS1[:-1, :]) / grid.cVol
        ResV1 = (Fmom1[1:, :] * grid.fS1[1:, :] - Fmom1[:-1, :] * grid.fS1[:-1, :]) / grid.cVol
        ResV2 = (Fmom2[1:, :] * grid.fS1[1:, :] - Fmom2[:-1, :] * grid.fS1[:-1, :]) / grid.cVol
        ResV3 = (Fmom3[1:, :] * grid.fS1[1:, :] - Fmom3[:-1, :] * grid.fS1[:-1, :]) / grid.cVol
        ResE  = (Fetot[1:, :] * grid.fS1[1:, :] - Fetot[:-1, :] * grid.fS1[:-1, :]) / grid.cVol
        ResB3 = (Fbfi3[1:, :] * grid.edg2[1:, :] - Fbfi3[:-1, :] * grid.edg2[:-1, :]) / grid.fS3

    # ----------------------------------------------------------------
    # Fluxes in x2-direction
    # ----------------------------------------------------------------
    if grid.Nx2 > 1:

        dens_L, dens_R = VarReconstruct(MHD.dens, grid, par.rec_type, 2)
        pres_L, pres_R = VarReconstruct(MHD.pres, grid, par.rec_type, 2)
        bfi1_L, bfi1_R = VarReconstruct(MHD.bfi1, grid, par.rec_type, 2)
        bfi3_L, bfi3_R = VarReconstruct(MHD.bfi3, grid, par.rec_type, 2)
        u1_L,   u1_R   = VarReconstruct(u1, grid, par.rec_type, 2)
        u2_L,   u2_R   = VarReconstruct(u2, grid, par.rec_type, 2)
        u3_L,   u3_R   = VarReconstruct(u3, grid, par.rec_type, 2)

        W_L = np.sqrt(1.0 + u1_L**2 + u2_L**2 + u3_L**2)
        W_R = np.sqrt(1.0 + u1_R**2 + u2_R**2 + u3_R**2)
        v1_L = u1_L / W_L;  v1_R = u1_R / W_R
        v2_L = u2_L / W_L;  v2_R = u2_R / W_R
        v3_L = u3_L / W_L;  v3_R = u3_R / W_R

        (Fmass, Fmom1, Fmom2, Fmom3, Fetot,
         fluxB12[Ngc:-Ngc, Ngc:Nx2r+1], Fbfi2, Fbfi3) = \
            Riemann_sr_MHD(
                dens_L, dens_R,
                v1_L, v1_R, v2_L, v2_R, v3_L, v3_R,
                pres_L, pres_R,
                bfi1_L, bfi1_R,
                MHD.fb2[:, :], MHD.fb2[:, :],
                bfi3_L, bfi3_R,
                eos, par.flux_type, 2)

        ResM  += (Fmass[:, 1:] * grid.fS2[:, 1:] - Fmass[:, :-1] * grid.fS2[:, :-1]) / grid.cVol
        ResV1 += (Fmom1[:, 1:] * grid.fS2[:, 1:] - Fmom1[:, :-1] * grid.fS2[:, :-1]) / grid.cVol
        ResV2 += (Fmom2[:, 1:] * grid.fS2[:, 1:] - Fmom2[:, :-1] * grid.fS2[:, :-1]) / grid.cVol
        ResV3 += (Fmom3[:, 1:] * grid.fS2[:, 1:] - Fmom3[:, :-1] * grid.fS2[:, :-1]) / grid.cVol
        ResE  += (Fetot[:, 1:] * grid.fS2[:, 1:] - Fetot[:, :-1] * grid.fS2[:, :-1]) / grid.cVol
        ResB3 += (Fbfi3[:, 1:] * grid.edg1[:, 1:] - Fbfi3[:, :-1] * grid.edg1[:, :-1]) / grid.fS3

    # ----------------------------------------------------------------
    # CT: electric field at cell corners  E_3 = -(v x B)_3
    # ----------------------------------------------------------------
    fluxB21, fluxB12 = boundCond_electric_field_rMHD(
        grid, fluxB21, fluxB12, par.BC)
    #arithemtic average in 2D and 1D 
    if ((grid.Nx1 > 1) & (grid.Nx2 > 1)):
        ave = 4.0
    else: 
        ave = 2.0
        
    #average electric field on the edges
    Efld3 = (
        -(fluxB21[Ngc:Nx1r+1, Ngc-1:Nx2r  ] + fluxB21[Ngc:Nx1r+1, Ngc:Nx2r+1]) / ave
        + (fluxB12[Ngc-1:Nx1r,  Ngc:Nx2r+1] + fluxB12[Ngc:Nx1r+1, Ngc:Nx2r+1]) / ave
        )
    
    #residual update    
    ResB1 = (Efld3[:, 1:] * grid.edg3[:, 1:] - Efld3[:, :-1] * grid.edg3[:, :-1]) / (grid.fS1 + 1e-30)
    ResB2 = -(Efld3[1:, :] * grid.edg3[1:, :] - Efld3[:-1, :] * grid.edg3[:-1, :]) / (grid.fS2 + 1e-30)

    # ----------------------------------------------------------------
    # External force source terms (gravity, etc.)
    # ----------------------------------------------------------------
    ResV1 += -MHD.dens[Ngc:-Ngc, Ngc:-Ngc] * MHD.F1
    ResV2 += -MHD.dens[Ngc:-Ngc, Ngc:-Ngc] * MHD.F2
    ResE  += -MHD.dens[Ngc:-Ngc, Ngc:-Ngc] * (
        MHD.F1 * MHD.vel1[Ngc:-Ngc, Ngc:-Ngc] +
        MHD.F2 * MHD.vel2[Ngc:-Ngc, Ngc:-Ngc])

    return ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3


# ============================================================================
# Electric-field boundary conditions (same structure as MHD_one_step_CT.py)
# ============================================================================

def boundCond_electric_field_rMHD(grid, Efld3x1, Efld3x2, BC):
    """
    Apply boundary conditions to the face-centred z-electric field for CT.

    Identical to boundCond_electric_field in MHD_one_step_CT.py.

    Parameters
    ----------
    grid    : Grid
    Efld3x1 : ndarray  E_z on x1-faces
    Efld3x2 : ndarray  E_z on x2-faces
    BC      : list of 4 str

    Returns
    -------
    Efld3x1, Efld3x2 : ndarray  with ghost-face values filled
    """
    Nx1 = grid.Nx1
    Nx2 = grid.Nx2
    Ngc = grid.Ngc

    for i in range(Ngc):
        # inner x2 boundary (acts on Efld3x1 along x2 direction)
        if BC[1] == 'free':
            Efld3x1[:, i] = Efld3x1[:, 2 * Ngc - 1 - i]
        elif BC[1] == 'wall':
            Efld3x1[:, i] = -Efld3x1[:, 2 * Ngc - 1 - i]
        elif BC[1] == 'peri':
            Efld3x1[:, i] = Efld3x1[:, Nx2 + i]
        # outer x2
        if BC[3] == 'free':
            Efld3x1[:, Nx2 + Ngc + i] = Efld3x1[:, Nx2 + Ngc - 1 - i]
        elif BC[3] == 'wall':
            Efld3x1[:, Nx2 + Ngc + i] = Efld3x1[:, Nx2 + Ngc - 1 - i]
        elif BC[3] == 'peri':
            Efld3x1[:, Nx2 + Ngc + i] = Efld3x1[:, Ngc + i]

    for i in range(Ngc):
        # inner x1 boundary (acts on Efld3x2 along x1 direction)
        if BC[0] == 'free':
            Efld3x2[i, :] = Efld3x2[2 * Ngc - 1 - i, :]
        elif BC[0] in ('wall', 'axis'):
            Efld3x2[i, :] = -Efld3x2[2 * Ngc - 1 - i, :]
        elif BC[0] == 'peri':
            Efld3x2[i, :] = Efld3x2[Nx1 + i, :]
        # outer x1
        if BC[2] == 'free':
            Efld3x2[Nx1 + Ngc + i, :] = Efld3x2[Nx1 + Ngc - 1 - i, :]
        elif BC[2] == 'wall':
            Efld3x2[Nx1 + Ngc + i, :] = -Efld3x2[Nx1 + Ngc - 1 - i, :]
        elif BC[2] == 'peri':
            Efld3x2[Nx1 + Ngc + i, :] = Efld3x2[Ngc + i, :]

    return Efld3x1, Efld3x2
