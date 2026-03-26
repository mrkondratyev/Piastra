# -*- coding: utf-8 -*-
"""
rMHD_phys.py
============

Core physics routines for 2D Special-Relativistic Magnetohydrodynamics (SRMHD).

This module provides:
  - Primitive <-> conservative variable conversions
  - Newton-Raphson conservative-to-primitive inversion
  - SR fast magnetosonic wave speed estimation
  - Approximate Riemann solvers: LLF, HLL
  - Boundary conditions for SRMHD primitive variables

Conservative variables (flat Minkowski spacetime, c = 1)
---------------------------------------------------------
Notation follows Del Zanna et al. (2003, A&A 400 397) and the PLUTO code
(Mignone et al. 2012, ApJS 198 7).

  W     = 1 / sqrt(1 - v^2)          Lorentz factor
  h     = 1 + Gp / (rho(G-1))        specific enthalpy
  b^2   = B^2/W^2 + (v.B)^2          covariant magnetic invariant
  p_tot = p + b^2/2                   total (gas + magnetic) pressure

  D     = rho W                       baryon number density
  S_i   = (rho h W^2 + B^2) v_i - (v.B) B_i   momentum density  [i=1,2,3]
  E     = rho h W^2 - p + (B^2 + |vxB|^2)/2   total energy density
  B_i                                 unchanged (ideal MHD)

Physical fluxes in the x-direction
-----------------------------------
  F^x_D     = D vx
  F^x_{S_1} = (rho h + b^2) W^2 vx^2 - bx bx + p_tot
  F^x_{S_2} = (rho h + b^2) W^2 vx vy - bx by
  F^x_{S_3} = (rho h + b^2) W^2 vx vz - bx bz
  F^x_E     = S_1             (SR identity: energy flux = momentum density)
  F^x_{B_y} = B_y vx - B_x vy
  F^x_{B_z} = B_z vx - B_x vz

where bx = Bx/W + W(v.B)vx  is the lab-frame 4-vector b component.

Primitive-variable recovery (cons -> prim)
-----------------------------------------
Given conserved (D, S_i, E, B_i), one must solve a non-linear equation
for x = rho h W^2 using Newton-Raphson.  See:
  - Mignone & McKinney (2007), MNRAS 378, 1118
  - Noble et al. (2006), ApJ Suppl. 164, 536

References
----------
  Del Zanna, Bucciantini & Londrillo (2003), A&A 400, 397
  Mignone & Bodo (2006), MNRAS 368, 1040
  Noble et al. (2006), ApJ Suppl. 164, 536
  Mignone et al. (2012), ApJS 198, 7  (PLUTO code paper)

Author
------
mrkondratyev
"""

import numpy as np
from boundaries import apply_bc_scalar, apply_bc_vector


# ============================================================================
# Helper: derived quantities from primitives
# ============================================================================

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

    Parameters
    ----------
    W            : Lorentz factor ndarray
    v1,v2,v3     : 3-velocity components
    B1,B2,B3     : lab-frame magnetic field components

    Returns
    -------
    b2     : ndarray  covariant magnetic invariant
    vdB    : ndarray  dot product  v.B
    B2_lab : ndarray  B1^2 + B2^2 + B3^2
    """
    vdB   = v1 * B1 + v2 * B2 + v3 * B3
    B2lab = B1**2 + B2**2 + B3**2
    b2    = B2lab / W**2 + vdB**2
    return b2, vdB, B2lab


# ============================================================================
# Primitive -> conservative
# ============================================================================

def prim2cons_sr_MHD(dens, vel1, vel2, vel3, pres, B1, B2, B3, eos):
    """
    Convert primitive to conservative variables for an ideal-gas SRMHD fluid.

    Parameters
    ----------
    dens             : ndarray  rest-mass density rho
    vel1,vel2,vel3   : ndarray  3-velocity components (|v| < 1)
    pres             : ndarray  thermal pressure p
    B1,B2,B3         : ndarray  lab-frame magnetic field components
    eos              : EOSdata  equation of state (provides GAMMA)

    Returns
    -------
    D              : ndarray   D = rho W
    S1, S2, S3     : ndarray   momentum density components
    E              : ndarray   total energy density
    B1, B2, B3     : ndarray   unchanged (passed through for convenience)
    """
    W  = _lorentz(vel1, vel2, vel3)
    W2 = W**2
    enth = 1.0 + pres / (dens + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)

    b2, vdB, Bsq = _b_squared(W, vel1, vel2, vel3, B1, B2, B3)
    vsq = vel1**2 + vel2**2 + vel3**2

    D  = dens * W
    S1 = (dens * enth * W2 + Bsq) * vel1 - vdB * B1
    S2 = (dens * enth * W2 + Bsq) * vel2 - vdB * B2
    S3 = (dens * enth * W2 + Bsq) * vel3 - vdB * B3
    E  = dens * enth * W2 - pres + 0.5 * (Bsq + Bsq * vsq - vdB**2)

    return D, S1, S2, S3, E, B1, B2, B3


# ============================================================================
# Conservative -> primitive  (Newton-Raphson inversion)
# ============================================================================

def cons2prim_sr_MHD(mass, mom1, mom2, mom3, ener, B1, B2, B3, x_init, eos):
    """
    Recover primitive variables from conservative variables for SRMHD.

    Uses the scalar variable  x = rho h W^2  and solves f(x) = 0 with
    Newton-Raphson (numerical derivative, following the rHD approach).

    The velocity is recovered analytically from the momentum equation:
        v_i = (S_i + (S.B / x) B_i) / (x + B^2)

    Parameters
    ----------
    mass           : ndarray  D  (baryon density)
    mom1,mom2,mom3 : ndarray  S_i (momentum density)
    ener           : ndarray  E  (total energy density)
    B1,B2,B3       : ndarray  magnetic field components
    x_init         : ndarray  initial guess for x = rho h W^2
    eos            : EOSdata

    Returns
    -------
    dens,vel1,vel2,vel3,pres,B1,B2,B3 : ndarray  primitive variables
    """
    msqr = mom1**2 + mom2**2 + mom3**2
    SdB  = mom1 * B1 + mom2 * B2 + mom3 * B3
    Bsqr = B1**2 + B2**2 + B3**2
    SdB2 = SdB**2
    gamma_r = eos.GAMMA / (eos.GAMMA - 1.0)

    x = _newton_rMHD(x_init, mass, ener, SdB2, msqr, Bsqr, gamma_r)

    # Lorentz factor from the velocity magnitude
    vsq = (msqr * x**2 + SdB2 * (2.0 * x + Bsqr)) / \
          (x**2 * (x + Bsqr)**2 + 1e-28)
    vsq = np.clip(vsq, 0.0, 1.0 - 1e-14)
    W  = 1.0 / np.sqrt(1.0 - vsq)

    # baryon density
    dens = mass / W

    # pressure from the definition of x = rho h W^2
    pres = (x - mass * W) / (gamma_r * W**2)
    pres = np.maximum(pres, 1e-14)

    # velocity from the momentum equation:
    #   v_i = [S_i + (S.B / x) B_i] / [x + B^2]
    vel1 = (mom1 + SdB * B1 / (x + 1e-28)) / (x + Bsqr + 1e-28)
    vel2 = (mom2 + SdB * B2 / (x + 1e-28)) / (x + Bsqr + 1e-28)
    vel3 = (mom3 + SdB * B3 / (x + 1e-28)) / (x + Bsqr + 1e-28)

    return dens, vel1, vel2, vel3, pres, B1, B2, B3


# ============================================================================
#   Nonlinear rMHD solver (Newton-Raphson with numerical derivative)
# ============================================================================

def _x_eqn_rMHD(x, mass, etot, SdB2, msqr, Bsqr, gamma_r):
    """
    Nonlinear equation f(x) = 0 whose root gives x = rho h W^2.

    f(x) = x - p(x) + (1 - 1/(2 W^2)) B^2 - (S.B)^2 / (2 x^2) - E

    where  W^2  and  p  are functions of  x  derived from the conservative
    state.

    Parameters
    ----------
    x       : ndarray  current guess for rho h W^2
    mass    : ndarray  D (baryon density)
    etot    : ndarray  E (total energy)
    SdB2    : ndarray  (S . B)^2
    msqr    : ndarray  |S|^2
    Bsqr    : ndarray  |B|^2
    gamma_r : float    GAMMA / (GAMMA - 1)

    Returns
    -------
    func : ndarray  residual f(x)
    """
    # velocity magnitude squared
    vsq = (msqr * x**2 + SdB2 * (2.0 * x + Bsqr)) / \
          (x**2 * (x + Bsqr)**2 + 1e-28)
    vsq = np.clip(vsq, 0.0, 1.0 - 1e-14)

    W2 = 1.0 / (1.0 - vsq)
    W  = np.sqrt(W2)

    # pressure from x = rho h W^2  =>  p = (x - D W) / (gamma_r W^2)
    pg = (x - mass * W) / (gamma_r * W2)

    func = x - pg + (1.0 - 0.5 / W2) * Bsqr - SdB2 / (2.0 * x**2 + 1e-28) - etot

    return func


def _newton_rMHD(x_init, mass, etot, SdB2, msqr, Bsqr, gamma_r):
    """
    Newton-Raphson iteration to solve f(x) = 0 for x = rho h W^2.

    Uses numerical differentiation (finite-difference derivative), consistent
    with the rHD pressure solver approach.  Convergence is declared when
    both the residual max-norm and the relative update are below ``tol``.

    Parameters
    ----------
    x_init  : ndarray  initial guess for x
    mass    : ndarray  D
    etot    : ndarray  E
    SdB2    : ndarray  (S . B)^2
    msqr    : ndarray  |S|^2
    Bsqr    : ndarray  |B|^2
    gamma_r : float    GAMMA / (GAMMA - 1)

    Returns
    -------
    x : ndarray  converged x = rho h W^2
    """
    tol    = 1.0e-8
    dx_rel = 1.0e-12   # relative step for numerical derivative
    maxitr = 100

    x = np.maximum(x_init, 1.0e-14)

    res  = _x_eqn_rMHD(x, mass, etot, SdB2, msqr, Bsqr, gamma_r)
    eps1 = np.max(np.abs(res))
    eps2 = 1.0

    for itr in range(maxitr):
        if eps1 <= tol and eps2 <= tol:
            break

        dx   = x * (1.0 + dx_rel)
        f0   = _x_eqn_rMHD(x,      mass, etot, SdB2, msqr, Bsqr, gamma_r)
        f1   = _x_eqn_rMHD(x + dx, mass, etot, SdB2, msqr, Bsqr, gamma_r)
        deriv = (f1 - f0) / (dx + 1e-28)

        update = f0 / (deriv + 1e-28)
        x    = x - update
        x    = np.maximum(x, 1.0e-14)

        res  = _x_eqn_rMHD(x, mass, etot, SdB2, msqr, Bsqr, gamma_r)
        eps1 = np.max(np.abs(res))
        eps2 = np.max(np.abs(update / (x + 1e-28)))
    else:
        print(f"[rMHD] Newton solver: did not converge after {maxitr} iterations "
              f"(residual = {eps1:.3e})")

    return x


# ============================================================================
# SR fast magnetosonic speed (upper bound)
# ============================================================================

def sound_speed_sr_MHD(dens, pres, eos):
    """
    Adiabatic sound speed for an ideal relativistic gas (same as rHD).

       cs^2 = GAMMA p / (rho h)

    Parameters
    ----------
    dens, pres : ndarray
    eos        : EOSdata

    Returns
    -------
    cs : ndarray   (0 < cs < 1)
    """
    enth = 1.0 + pres / (dens + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)
    cs2  = eos.GAMMA * pres / (dens * enth + 1e-14)
    return np.sqrt(np.clip(cs2, 0.0, 1.0 - 1e-14))


def fast_magnetosonic_speed_sr(dens, pres, vel1, vel2, vel3, B1, B2, B3, eos):
    """
    Upper bound on the fast magnetosonic speed for SRMHD.

    Uses the standard approximation (Leismann et al. 2005; PLUTO code):

        v_A^2 = b^2 / (rho h + b^2)      relativistic Alfven speed^2
        cs^2  = GAMMA p / (rho h)         SR sound speed^2
        c_f^2 ~ cs^2 + vA^2 - cs^2 vA^2  (relativistic analogue)

    Parameters
    ----------
    dens, pres        : ndarray
    vel1, vel2, vel3  : ndarray
    B1, B2, B3        : ndarray
    eos               : EOSdata

    Returns
    -------
    c_f : ndarray  fast magnetosonic speed upper bound  (0 < c_f < 1)
    """
    W    = _lorentz(vel1, vel2, vel3)
    enth = 1.0 + pres / (dens + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)
    cs2  = eos.GAMMA * pres / (dens * enth + 1e-14)

    b2, _, _ = _b_squared(W, vel1, vel2, vel3, B1, B2, B3)
    vA2 = b2 / (dens * enth + b2 + 1e-14)

    cf2 = np.clip(cs2 + vA2 - cs2 * vA2, 0.0, 1.0 - 1e-14)
    return np.sqrt(cf2)


# ============================================================================
# Approximate Riemann solvers (LLF and HLL)
# ============================================================================

def Riemann_sr_MHD(rhol, rhor,
                   vxl, vxr, vyl, vyr, vzl, vzr,
                   pl, pr,
                   bxl, bxr, byl, byr, bzl, bzr,
                   eos, flux_type, dim):
    """
    Approximate Riemann fluxes for the SRMHD equations.

    Supports LLF (Local Lax-Friedrichs) and HLL solvers.
    For dim=2 the system is solved after rotating coordinates so that the
    x-direction is always the normal direction.

    Parameters
    ----------
    rhol, rhor               : ndarray  left/right density
    vxl,vxr, vyl,vyr, vzl,vzr : ndarray  left/right 3-velocities
    pl, pr                   : ndarray  left/right pressure
    bxl,bxr, byl,byr, bzl,bzr : ndarray  left/right B-field components
    eos                      : EOSdata
    flux_type                : {'LLF', 'HLL'}
    dim                      : int  1 or 2

    Returns
    -------
    Fmass,Fmom1,Fmom2,Fmom3,Fetot,Fbfix,Fbfiy,Fbfiz : ndarray
        Interface fluxes for D, S_1, S_2, S_3, E, B_x, B_y, B_z.
    """
    # ----------------------------------------------------------------
    # Coordinate rotation for dim=2
    # ----------------------------------------------------------------
    if dim == 2:
        vxl, vxr, vyl, vyr = vyl, vyr, -vxl, -vxr
        bxl, bxr, byl, byr = byl, byr, -bxl, -bxr

    # Normal B-field: use arithmetic average (as in NR CT-MHD)
    Bxn = 0.5 * (bxl + bxr)

    # ----------------------------------------------------------------
    # Derived quantities (left state)
    # ----------------------------------------------------------------
    Wl    = _lorentz(vxl, vyl, vzl)
    enthl = 1.0 + pl / (rhol + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)
    b2l, vdBl, Bsql = _b_squared(Wl, vxl, vyl, vzl, Bxn, byl, bzl)
    vsql = vxl**2 + vyl**2 + vzl**2

    # Left conservative state: S_i = (rho h W^2 + B^2) v_i - (v.B) B_i
    zl    = rhol * enthl * Wl**2 + Bsql
    Dl    = rhol * Wl
    S1l   = zl * vxl - vdBl * Bxn
    S2l   = zl * vyl - vdBl * byl
    S3l   = zl * vzl - vdBl * bzl
    El    = rhol * enthl * Wl**2 - pl + 0.5 * (Bsql + Bsql * vsql - vdBl**2)
    ptotl = pl + 0.5 * b2l

    # 4-vector b components for momentum flux: b^i = B^i/W + W(v.B) v^i
    zfl   = (rhol * enthl + b2l) * Wl**2
    bbxl  = Bxn / Wl + Wl * vdBl * vxl
    bbyl  = byl / Wl + Wl * vdBl * vyl
    bbzl  = bzl / Wl + Wl * vdBl * vzl

    # Left physical fluxes (x-direction)
    FDl   = Dl * vxl
    FS1l  = zfl * vxl**2 - bbxl**2 + ptotl
    FS2l  = zfl * vxl * vyl - bbxl * bbyl
    FS3l  = zfl * vxl * vzl - bbxl * bbzl
    FEl   = S1l           # SR identity: energy flux = momentum density
    FByl  = byl * vxl - Bxn * vyl
    FBzl  = bzl * vxl - Bxn * vzl

    # ----------------------------------------------------------------
    # Derived quantities (right state)
    # ----------------------------------------------------------------
    Wr    = _lorentz(vxr, vyr, vzr)
    enthr = 1.0 + pr / (rhor + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)
    b2r, vdBr, Bsqr = _b_squared(Wr, vxr, vyr, vzr, Bxn, byr, bzr)
    vsqr = vxr**2 + vyr**2 + vzr**2

    zr    = rhor * enthr * Wr**2 + Bsqr
    Dr    = rhor * Wr
    S1r   = zr * vxr - vdBr * Bxn
    S2r   = zr * vyr - vdBr * byr
    S3r   = zr * vzr - vdBr * bzr
    Er    = rhor * enthr * Wr**2 - pr + 0.5 * (Bsqr + Bsqr * vsqr - vdBr**2)
    ptotr = pr + 0.5 * b2r

    zfr   = (rhor * enthr + b2r) * Wr**2
    bbxr  = Bxn / Wr + Wr * vdBr * vxr
    bbyr  = byr / Wr + Wr * vdBr * vyr
    bbzr  = bzr / Wr + Wr * vdBr * vzr

    # Right physical fluxes (x-direction)
    FDr   = Dr * vxr
    FS1r  = zfr * vxr**2 - bbxr**2 + ptotr
    FS2r  = zfr * vxr * vyr - bbxr * bbyr
    FS3r  = zfr * vxr * vzr - bbxr * bbzr
    FEr   = S1r
    FByr  = byr * vxr - Bxn * vyr
    FBzr  = bzr * vxr - Bxn * vzr

    # ----------------------------------------------------------------
    # SR wave-speed estimates (Doppler-shifted fast magnetosonic speed)
    # ----------------------------------------------------------------
    cfl = fast_magnetosonic_speed_sr(rhol, pl, vxl, vyl, vzl, Bxn, byl, bzl, eos)
    cfr = fast_magnetosonic_speed_sr(rhor, pr, vxr, vyr, vzr, Bxn, byr, bzr, eos)

    # Relativistic signal speeds (Mignone & Bodo 2005 / PLUTO-style estimate)
    Sl = np.minimum((vxl - cfl) / (1.0 - vxl * cfl + 1e-14),
                    (vxr - cfr) / (1.0 - vxr * cfr + 1e-14))
    Sr = np.maximum((vxl + cfl) / (1.0 + vxl * cfl + 1e-14),
                    (vxr + cfr) / (1.0 + vxr * cfr + 1e-14))

    # ----------------------------------------------------------------
    # LLF (Local Lax-Friedrichs / Rusanov)
    # ----------------------------------------------------------------
    if flux_type == 'LLF':

        lam = np.maximum(np.abs(Sl), np.abs(Sr))

        Fmass = 0.5 * (FDl  + FDr  - lam * (Dr  - Dl ))
        Fmom1 = 0.5 * (FS1l + FS1r - lam * (S1r - S1l))
        Fmom2 = 0.5 * (FS2l + FS2r - lam * (S2r - S2l))
        Fmom3 = 0.5 * (FS3l + FS3r - lam * (S3r - S3l))
        Fetot = 0.5 * (FEl  + FEr  - lam * (Er  - El ))
        Fbfix = np.zeros_like(Fmass)
        Fbfiy = 0.5 * (FByl + FByr - lam * (byr - byl))
        Fbfiz = 0.5 * (FBzl + FBzr - lam * (bzr - bzl))

    # ----------------------------------------------------------------
    # HLL (Harten-Lax-van Leer)
    # ----------------------------------------------------------------
    elif flux_type == 'HLL':

        Sl = np.minimum(Sl, 0.0)
        Sr = np.maximum(Sr, 0.0)

        Fmass = (Sr * FDl  - Sl * FDr  + Sr * Sl * (Dr  - Dl )) / (Sr - Sl)
        Fmom1 = (Sr * FS1l - Sl * FS1r + Sr * Sl * (S1r - S1l)) / (Sr - Sl)
        Fmom2 = (Sr * FS2l - Sl * FS2r + Sr * Sl * (S2r - S2l)) / (Sr - Sl)
        Fmom3 = (Sr * FS3l - Sl * FS3r + Sr * Sl * (S3r - S3l)) / (Sr - Sl)
        Fetot = (Sr * FEl  - Sl * FEr  + Sr * Sl * (Er  - El )) / (Sr - Sl)
        Fbfix = np.zeros_like(Fmass)
        Fbfiy = (Sr * FByl - Sl * FByr + Sr * Sl * (byr - byl)) / (Sr - Sl)
        Fbfiz = (Sr * FBzl - Sl * FBzr + Sr * Sl * (bzr - bzl)) / (Sr - Sl)

    else:
        raise ValueError(
            f"Unknown flux_type '{flux_type}'. "
            "Expected 'LLF' or 'HLL'.  "
            "(HLLD for rMHD is not yet implemented.)"
        )

    # ----------------------------------------------------------------
    # Undo coordinate rotation for dim=2
    # ----------------------------------------------------------------
    if dim == 2:
        Fmom1, Fmom2 = -Fmom2, Fmom1
        Fbfix, Fbfiy = -Fbfiy, Fbfix

    return Fmass, Fmom1, Fmom2, Fmom3, Fetot, Fbfix, Fbfiy, Fbfiz


# ============================================================================
# Boundary conditions
# ============================================================================

def boundCond_rMHD(grid, BC, fluid):
    """
    Apply boundary conditions to SRMHD primitive variables.

    Mirrors the structure of boundCond_MHD in MHD_phys.py, applying BCs
    to density, pressure, velocity, and cell-centred magnetic field.

    Parameters
    ----------
    grid  : Grid
    BC    : array of 4 str  --  [x1_inner, x2_inner, x1_outer, x2_outer]
    fluid : SimState

    Returns
    -------
    fluid : SimState  (modified in-place)
    """
    Ngc = grid.Ngc

    for axis, side, bc in [
        (1, 'inner', BC[0]),
        (2, 'inner', BC[1]),
        (1, 'outer', BC[2]),
        (2, 'outer', BC[3]),
    ]:
        fluid.dens = apply_bc_scalar(fluid.dens, Ngc, bc, axis=axis, side=side)
        fluid.pres = apply_bc_scalar(fluid.pres, Ngc, bc, axis=axis, side=side)

        fluid.vel1, fluid.vel2, fluid.vel3 = apply_bc_vector(
            fluid.vel1, fluid.vel2, fluid.vel3, Ngc, bc, axis=axis, side=side)

        fluid.bfi1, fluid.bfi2, fluid.bfi3 = apply_bc_vector(
            fluid.bfi1, fluid.bfi2, fluid.bfi3, Ngc, bc, axis=axis, side=side)

    return fluid
