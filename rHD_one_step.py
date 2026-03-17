# -*- coding: utf-8 -*-
"""
rHD_one_step.py

Container class and time-stepping routines for 2D special-relativistic
hydrodynamics (rHD).

This module mirrors hydro_one_step.py for the relativistic Euler equations.
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
from rHD_phys import (
    prim2cons_sr_hydro,
    cons2prim_sr_hydro,
    sound_speed_sr,
    Riemann_sr_hydro,
    boundCond_rHD,
)
from reconstruction import VarReconstruct


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
    """
    Ngc = g.Ngc

    dens = HD.dens[Ngc:-Ngc, Ngc:-Ngc]
    vel1 = HD.vel1[Ngc:-Ngc, Ngc:-Ngc]
    vel2 = HD.vel2[Ngc:-Ngc, Ngc:-Ngc]
    pres = HD.pres[Ngc:-Ngc, Ngc:-Ngc]

    cs = sound_speed_sr(dens, pres, eos)

    # SR characteristic speed estimate: (|v| + cs) / (1 + |v|*cs)
    lam1 = (np.abs(vel1) + cs) / (1.0 + np.abs(vel1) * cs + 1e-14)
    lam2 = (np.abs(vel2) + cs) / (1.0 + np.abs(vel2) * cs + 1e-14)

    dt_inv = np.max(lam1 / g.dx1[Ngc:-Ngc, Ngc:-Ngc] +
                    lam2 / g.dx2[Ngc:-Ngc, Ngc:-Ngc])
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
        prim2cons_sr_hydro(
            HD.dens[Ngc:-Ngc, Ngc:-Ngc],
            HD.vel1[Ngc:-Ngc, Ngc:-Ngc],
            HD.vel2[Ngc:-Ngc, Ngc:-Ngc],
            HD.vel3[Ngc:-Ngc, Ngc:-Ngc],
            HD.pres[Ngc:-Ngc, Ngc:-Ngc],
            eos)

    # 1st RK stage (predictor)
    ResM, Res1, Res2, Res3, ResE = flux_calc_rHD(g, HD, par, eos)

    HD_h.mass = HD.mass - dt * ResM
    HD_h.mom1 = HD.mom1 - dt * Res1
    HD_h.mom2 = HD.mom2 - dt * Res2
    HD_h.mom3 = HD.mom3 - dt * Res3
    HD_h.etot = HD.etot - dt * ResE

    if par.RK_order == 'RK1':
        HD.mass = HD_h.mass
        HD.mom1 = HD_h.mom1
        HD.mom2 = HD_h.mom2
        HD.mom3 = HD_h.mom3
        HD.etot = HD_h.etot

    if par.RK_order == 'RK2':
        # Primitive recovery after predictor stage
        HD_h.dens[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.vel2[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.pres[Ngc:-Ngc, Ngc:-Ngc] = \
            cons2prim_sr_hydro(
                HD_h.mass, HD_h.mom1, HD_h.mom2, HD_h.mom3, HD_h.etot,
                HD_h.pres[Ngc:-Ngc, Ngc:-Ngc], eos)

        # 2nd RK stage (corrector)
        ResM, Res1, Res2, Res3, ResE = flux_calc_rHD(g, HD_h, par, eos)

        HD.mass = (HD_h.mass + HD.mass) / 2.0 - dt * ResM / 2.0
        HD.mom1 = (HD_h.mom1 + HD.mom1) / 2.0 - dt * Res1 / 2.0
        HD.mom2 = (HD_h.mom2 + HD.mom2) / 2.0 - dt * Res2 / 2.0
        HD.mom3 = (HD_h.mom3 + HD.mom3) / 2.0 - dt * Res3 / 2.0
        HD.etot = (HD_h.etot + HD.etot) / 2.0 - dt * ResE / 2.0

    if par.RK_order == 'RK3':
        # Primitive recovery after 1st stage
        HD_h.dens[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.vel2[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.pres[Ngc:-Ngc, Ngc:-Ngc] = \
            cons2prim_sr_hydro(
                HD_h.mass, HD_h.mom1, HD_h.mom2, HD_h.mom3, HD_h.etot,
                HD_h.pres[Ngc:-Ngc, Ngc:-Ngc], eos)

        # 2nd RK stage
        ResM, Res1, Res2, Res3, ResE = flux_calc_rHD(g, HD_h, par, eos)

        HD_h.mass = (HD_h.mass + 3.0 * HD.mass) / 4.0 - dt * ResM / 4.0
        HD_h.mom1 = (HD_h.mom1 + 3.0 * HD.mom1) / 4.0 - dt * Res1 / 4.0
        HD_h.mom2 = (HD_h.mom2 + 3.0 * HD.mom2) / 4.0 - dt * Res2 / 4.0
        HD_h.mom3 = (HD_h.mom3 + 3.0 * HD.mom3) / 4.0 - dt * Res3 / 4.0
        HD_h.etot = (HD_h.etot + 3.0 * HD.etot) / 4.0 - dt * ResE / 4.0

        # Primitive recovery after 2nd stage
        HD_h.dens[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.vel2[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.pres[Ngc:-Ngc, Ngc:-Ngc] = \
            cons2prim_sr_hydro(
                HD_h.mass, HD_h.mom1, HD_h.mom2, HD_h.mom3, HD_h.etot,
                HD_h.pres[Ngc:-Ngc, Ngc:-Ngc], eos)

        # 3rd RK stage (final)
        ResM, Res1, Res2, Res3, ResE = flux_calc_rHD(g, HD_h, par, eos)

        HD.mass = (2.0 * HD_h.mass + HD.mass) / 3.0 - 2.0 * dt * ResM / 3.0
        HD.mom1 = (2.0 * HD_h.mom1 + HD.mom1) / 3.0 - 2.0 * dt * Res1 / 3.0
        HD.mom2 = (2.0 * HD_h.mom2 + HD.mom2) / 3.0 - 2.0 * dt * Res2 / 3.0
        HD.mom3 = (2.0 * HD_h.mom3 + HD.mom3) / 3.0 - 2.0 * dt * Res3 / 3.0
        HD.etot = (2.0 * HD_h.etot + HD.etot) / 3.0 - 2.0 * dt * ResE / 3.0

    # Final primitive variable recovery
    HD.dens[Ngc:-Ngc, Ngc:-Ngc], \
        HD.vel1[Ngc:-Ngc, Ngc:-Ngc], \
        HD.vel2[Ngc:-Ngc, Ngc:-Ngc], \
        HD.vel3[Ngc:-Ngc, Ngc:-Ngc], \
        HD.pres[Ngc:-Ngc, Ngc:-Ngc] = \
        cons2prim_sr_hydro(
            HD.mass, HD.mom1, HD.mom2, HD.mom3, HD.etot,
            HD.pres[Ngc:-Ngc, Ngc:-Ngc], eos)

    return HD


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
    HD = boundCond_rHD(g, par.BC, HD)

    Ngc = g.Ngc

    ResM = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res1 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res2 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res3 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    ResE = np.zeros((g.Nx1, g.Nx2), dtype=np.double)

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

        # Recover 3-velocities from 4-velocities: vⁱ = uⁱ / sqrt(1 + |u|²)
        W_L = np.sqrt(1.0 + u1_L**2 + u2_L**2 + u3_L**2)
        W_R = np.sqrt(1.0 + u1_R**2 + u2_R**2 + u3_R**2)
        vel1_L = u1_L / W_L;  vel1_R = u1_R / W_R
        vel2_L = u2_L / W_L;  vel2_R = u2_R / W_R
        vel3_L = u3_L / W_L;  vel3_R = u3_R / W_R

        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            Riemann_sr_hydro(
                dens_L, dens_R,
                vel1_L, vel1_R, vel2_L, vel2_R, vel3_L, vel3_R,
                pres_L, pres_R, eos, par.flux_type, 1)

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

        W_L = np.sqrt(1.0 + u1_L**2 + u2_L**2 + u3_L**2)
        W_R = np.sqrt(1.0 + u1_R**2 + u2_R**2 + u3_R**2)
        vel1_L = u1_L / W_L;  vel1_R = u1_R / W_R
        vel2_L = u2_L / W_L;  vel2_R = u2_R / W_R
        vel3_L = u3_L / W_L;  vel3_R = u3_R / W_R

        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            Riemann_sr_hydro(
                dens_L, dens_R,
                vel1_L, vel1_R, vel2_L, vel2_R, vel3_L, vel3_R,
                pres_L, pres_R, eos, par.flux_type, 2)

        ResM += (Fmass[:, 1:] * g.fS2[:, 1:] - Fmass[:, :-1] * g.fS2[:, :-1]) / g.cVol
        Res1 += (Fmomx[:, 1:] * g.fS2[:, 1:] - Fmomx[:, :-1] * g.fS2[:, :-1]) / g.cVol
        Res2 += (Fmomy[:, 1:] * g.fS2[:, 1:] - Fmomy[:, :-1] * g.fS2[:, :-1]) / g.cVol
        Res3 += (Fmomz[:, 1:] * g.fS2[:, 1:] - Fmomz[:, :-1] * g.fS2[:, :-1]) / g.cVol
        ResE += (Fetot[:, 1:] * g.fS2[:, 1:] - Fetot[:, :-1] * g.fS2[:, :-1]) / g.cVol

    # External force source terms (gravity, etc.)
    Res1 += -HD.dens[Ngc:-Ngc, Ngc:-Ngc] * HD.F1
    Res2 += -HD.dens[Ngc:-Ngc, Ngc:-Ngc] * HD.F2
    ResE += -HD.dens[Ngc:-Ngc, Ngc:-Ngc] * (
        HD.F1 * HD.vel1[Ngc:-Ngc, Ngc:-Ngc] +
        HD.F2 * HD.vel2[Ngc:-Ngc, Ngc:-Ngc])

    return ResM, Res1, Res2, Res3, ResE
