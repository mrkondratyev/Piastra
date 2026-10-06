# -*- coding: utf-8 -*-
"""
rMHD_riemann_approx.py

Approximate Riemann solvers for special-relativistic MHD (SRMHD).

Implemented solvers, in increasing order of accuracy/cost:

    LLF   - Local Lax-Friedrichs / Rusanov (1961)
    HLL   - Harten, Lax, van Leer (1983)

All solvers share the same calling convention as the non-relativistic
ones in MHD_riemann_approx.py. They are called from Riemann_rMHD in
rMHD_phys.py, which rotates the states so that x is always the normal
direction, selects the solver (par.solver_type) and rotates the fluxes
back.

The small helpers needed here (Lorentz factor, magnetic invariant b^2,
fast magnetosonic speed) are deliberately copied from rMHD_phys.py
rather than imported, so that this file can be read on its own: all SRMHD
formulas used by the solvers are written out below.

Each Riemann solver routine has the following i/o structure:

   Parameters
   ----------
   rhol, rhor : ndarray
       Left and right densities.
   vxl, vxr, vyl, vyr, vzl, vzr : ndarray
       Left and right 3-velocity components (x = normal direction).
   pl, pr : ndarray
       Left and right gas pressures.
   bxl, bxr, byl, byr, bzl, bzr : ndarray
       Left and right lab-frame magnetic field components.
   eos : object
       Equation of state object (enthalpy_sr, sound_speed_sr).

   Returns
   -------
   Fmass : ndarray
       Flux of D = rho W.
   Fmomx, Fmomy, Fmomz : ndarray
       Fluxes of the momentum components S_x, S_y, S_z.
   Fetot : ndarray
       Flux of the total energy E.
   Fbfix, Fbfiy, Fbfiz : ndarray
       Fluxes of the magnetic field components (Fbfix = 0).

SRMHD relations used (c = 1, Mignone & Bodo 2006, MNRAS 368, 1040)
------------------------------------------------------------------
  W     = 1 / sqrt(1 - v^2)
  b^2   = B^2 / W^2 + (v.B)^2
  D     = rho W
  S_i   = (rho h W^2 + B^2) v_i - (v.B) B_i
  E     = rho h W^2 - p + (B^2 + v^2 B^2 - (v.B)^2) / 2
  p_tot = p + b^2 / 2
  b^i   = B^i / W + W (v.B) v^i                 (spatial part of the 4-vector b)

  F_D   = D v_x
  F_S_i = S_i v_x - B_x b^i / W + p_tot delta_ix
  F_E   = S_x
  F_B_y = B_y v_x - B_x v_y,   F_B_z = B_z v_x - B_x v_z

Author: mrkondratyev
"""
import numpy as np


# -------------------------
# Small helpers (copies of the ones in rMHD_phys.py)
# -------------------------
def _lorentz(v1, v2, v3):
    """
    Lorentz factor  W = 1 / sqrt(1 - v^2).

    Velocities are clipped to ensure v^2 < 1 (sub-luminal flow).
    """
    vsq = np.clip(v1**2 + v2**2 + v3**2, 0.0, 1.0 - 1e-14)
    return 1.0 / np.sqrt(1.0 - vsq)


def _b_squared(W, v1, v2, v3, B1, B2, B3):
    """
    Covariant magnetic invariant  b^2 = B^2/W^2 + (v.B)^2.

    Returns
    -------
    b2     : ndarray  covariant magnetic invariant
    vdB    : ndarray  dot product  v.B
    B2lab  : ndarray  B1^2 + B2^2 + B3^2
    """
    vdB   = v1 * B1 + v2 * B2 + v3 * B3
    B2lab = B1**2 + B2**2 + B3**2
    b2    = B2lab / W**2 + vdB**2
    return b2, vdB, B2lab


def _fast_speed_sr(dens, pres, vel1, vel2, vel3, B1, B2, B3, eos):
    """
    Upper bound on the fast magnetosonic speed in the fluid frame
    (Leismann et al. 2005; PLUTO code):

        v_A^2 = b^2 / (rho h + b^2)      relativistic Alfven speed^2
        cs^2  = GAMMA p / (rho h)         SR sound speed^2
        c_f^2 = cs^2 + vA^2 - cs^2 vA^2
    """
    W    = _lorentz(vel1, vel2, vel3)
    enth = eos.enthalpy_sr(dens, pres)
    cs2  = eos.sound_speed_sr(dens, pres)**2

    b2, _, _ = _b_squared(W, vel1, vel2, vel3, B1, B2, B3)
    vA2 = b2 / (dens * enth + b2 + 1e-14)

    cf2 = np.clip(cs2 + vA2 - cs2 * vA2, 0.0, 1.0 - 1e-14)
    return np.sqrt(cf2)


# -------------------------
# Conservative SRMHD variables + their fluxes along Ox for one state
# -------------------------
def rMHD_cons_and_flux(rho, vx, vy, vz, p, bx, by, bz, eos):
    """
    Conservative variables and their Ox-normal fluxes for one SRMHD state.

    Parameters
    ----------
    rho, vx, vy, vz, p, bx, by, bz : ndarray
        Primitive state (density, 3-velocity, gas pressure, lab-frame B).
        bx is the normal field (the same on both sides of the interface).
    eos : object
        Equation of state object.

    Returns
    -------
    (D, Sx, Sy, Sz, E, By, Bz) : tuple of ndarray
        Conservative variables (the normal field Bx is not evolved here).
    (FD, FSx, FSy, FSz, FE, FBy, FBz) : tuple of ndarray
        Their fluxes along x.
    """
    W    = _lorentz(vx, vy, vz)
    enth = eos.enthalpy_sr(rho, p)
    b2, vdB, Bsq = _b_squared(W, vx, vy, vz, bx, by, bz)
    vsq  = vx**2 + vy**2 + vz**2

    # conservative variables: S_i = (rho h W^2 + B^2) v_i - (v.B) B_i
    z    = rho * enth * W**2 + Bsq
    D    = rho * W
    Sx   = z * vx - vdB * bx
    Sy   = z * vy - vdB * by
    Sz   = z * vz - vdB * bz
    E    = rho * enth * W**2 - p + 0.5 * (Bsq + Bsq * vsq - vdB**2)
    ptot = p + 0.5 * b2

    # spatial part of the 4-vector b: b^i = B^i / W + W (v.B) v^i
    bbx  = bx / W + W * vdB * vx
    bby  = by / W + W * vdB * vy
    bbz  = bz / W + W * vdB * vz

    # fluxes along x
    FD   = D * vx
    FSx  = Sx * vx - bx * bbx / W + ptot
    FSy  = Sy * vx - bx * bby / W
    FSz  = Sz * vx - bx * bbz / W
    FE   = Sx                     # SR identity: energy flux = momentum density
    FBy  = by * vx - bx * vy
    FBz  = bz * vx - bx * vz

    return (D, Sx, Sy, Sz, E, by, bz), (FD, FSx, FSy, FSz, FE, FBy, FBz)


def _signal_speeds(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr,
                   bxn, byl, byr, bzl, bzr, eos):
    """
    Leftmost / rightmost signal speeds: the fluid-frame fast speed c_f,
    Doppler-shifted by relativistic velocity addition along x,
        lambda_-+ = (v_x -+ c_f) / (1 -+ v_x c_f),
    taking the extreme value over the left and right states.
    """
    cfl = _fast_speed_sr(rhol, pl, vxl, vyl, vzl, bxn, byl, bzl, eos)
    cfr = _fast_speed_sr(rhor, pr, vxr, vyr, vzr, bxn, byr, bzr, eos)

    Sl = np.minimum((vxl - cfl) / (1.0 - vxl * cfl + 1e-14),
                    (vxr - cfr) / (1.0 - vxr * cfr + 1e-14))
    Sr = np.maximum((vxl + cfl) / (1.0 + vxl * cfl + 1e-14),
                    (vxr + cfr) / (1.0 + vxr * cfr + 1e-14))
    return Sl, Sr


# -------------------------
# LLF (Local Lax-Friedrichs / Rusanov)
# -------------------------
def LLF_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr,
             bxl, bxr, byl, byr, bzl, bzr, eos):
    """
    Local Lax-Friedrichs (Rusanov 1961) flux for SRMHD:
        F = (F_L + F_R)/2 - lam (U_R - U_L)/2,   lam = max(|S_L|, |S_R|).
    """
    # normal field: one value per interface
    bxn = 0.5 * (bxl + bxr)

    (Dl, Sxl, Syl, Szl, El, Byl, Bzl), (FDl, FSxl, FSyl, FSzl, FEl, FByl, FBzl) = \
        rMHD_cons_and_flux(rhol, vxl, vyl, vzl, pl, bxn, byl, bzl, eos)
    (Dr, Sxr, Syr, Szr, Er, Byr, Bzr), (FDr, FSxr, FSyr, FSzr, FEr, FByr, FBzr) = \
        rMHD_cons_and_flux(rhor, vxr, vyr, vzr, pr, bxn, byr, bzr, eos)

    Sl, Sr = _signal_speeds(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr,
                            bxn, byl, byr, bzl, bzr, eos)
    lam = np.maximum(np.abs(Sl), np.abs(Sr))

    Fmass = 0.5 * (FDl  + FDr  - lam * (Dr  - Dl ))
    Fmomx = 0.5 * (FSxl + FSxr - lam * (Sxr - Sxl))
    Fmomy = 0.5 * (FSyl + FSyr - lam * (Syr - Syl))
    Fmomz = 0.5 * (FSzl + FSzr - lam * (Szr - Szl))
    Fetot = 0.5 * (FEl  + FEr  - lam * (Er  - El ))
    Fbfix = np.zeros_like(Fmass)
    Fbfiy = 0.5 * (FByl + FByr - lam * (Byr - Byl))
    Fbfiz = 0.5 * (FBzl + FBzr - lam * (Bzr - Bzl))

    return Fmass, Fmomx, Fmomy, Fmomz, Fetot, Fbfix, Fbfiy, Fbfiz


# -------------------------
# HLL (Harten, Lax, van Leer 1983)
# -------------------------
def HLL_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr,
             bxl, bxr, byl, byr, bzl, bzr, eos):
    """
    HLL flux for SRMHD (one intermediate state between the fastest
    left- and right-going waves):
        F = (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L),
    with S_L <= 0 <= S_R (upwind fluxes are recovered automatically).
    """
    # normal field: one value per interface
    bxn = 0.5 * (bxl + bxr)

    (Dl, Sxl, Syl, Szl, El, Byl, Bzl), (FDl, FSxl, FSyl, FSzl, FEl, FByl, FBzl) = \
        rMHD_cons_and_flux(rhol, vxl, vyl, vzl, pl, bxn, byl, bzl, eos)
    (Dr, Sxr, Syr, Szr, Er, Byr, Bzr), (FDr, FSxr, FSyr, FSzr, FEr, FByr, FBzr) = \
        rMHD_cons_and_flux(rhor, vxr, vyr, vzr, pr, bxn, byr, bzr, eos)

    Sl, Sr = _signal_speeds(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr,
                            bxn, byl, byr, bzl, bzr, eos)
    Sl = np.minimum(Sl, 0.0)
    Sr = np.maximum(Sr, 0.0)

    Fmass = (Sr * FDl  - Sl * FDr  + Sr * Sl * (Dr  - Dl )) / (Sr - Sl)
    Fmomx = (Sr * FSxl - Sl * FSxr + Sr * Sl * (Sxr - Sxl)) / (Sr - Sl)
    Fmomy = (Sr * FSyl - Sl * FSyr + Sr * Sl * (Syr - Syl)) / (Sr - Sl)
    Fmomz = (Sr * FSzl - Sl * FSzr + Sr * Sl * (Szr - Szl)) / (Sr - Sl)
    Fetot = (Sr * FEl  - Sl * FEr  + Sr * Sl * (Er  - El )) / (Sr - Sl)
    Fbfix = np.zeros_like(Fmass)
    Fbfiy = (Sr * FByl - Sl * FByr + Sr * Sl * (Byr - Byl)) / (Sr - Sl)
    Fbfiz = (Sr * FBzl - Sl * FBzr + Sr * Sl * (Bzr - Bzl)) / (Sr - Sl)

    return Fmass, Fmomx, Fmomy, Fmomz, Fetot, Fbfix, Fbfiy, Fbfiz