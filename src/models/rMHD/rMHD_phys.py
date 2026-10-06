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
Implementation follows Mignone & Bodo (2006), MNRAS 368, 1040.

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
  Mignone et al. (2007), ApJS (PLUTO code paper)

Author
------
mrkondratyev
"""

import numpy as np
from src.common.boundaries import apply_bc_scalar, apply_bc_vector
from src.models.rMHD.rMHD_riemann_approx import LLF_flux, HLL_flux

# ============================================================================
# Helpers: derived quantities from primitives
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

def prim2cons_rMHD(dens, vel1, vel2, vel3, pres, B1, B2, B3, eos):
    """
    Convert primitive to conservative variables for an ideal-gas SRMHD.

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

def cons2prim_rMHD(mass, mom1, mom2, mom3, ener, Bcon1, Bcon2, Bcon3, x_init, eos):
    """
    Recover primitive variables from conservative variables for SRMHD.

    Uses the scalar variable  x = rho h W^2  and solves f(x) = 0 with
    Newton-Raphson (numerical derivative in contrast to the rHD approach).

    The velocity is recovered analytically from the momentum equation:
        v_i = (S_i + (S.B / x) B_i) / (x + B^2)

    Parameters
    ----------
    mass              : ndarray  D  (baryon density)
    mom1,mom2,mom3    : ndarray  S_i (momentum density)
    ener              : ndarray  E  (total energy density)
    Bcon1,Bcon2,Bcon3 : ndarray  magnetic field components
    x_init            : ndarray  initial guess for x = rho h W^2
    eos               : EOSdata

    Returns
    -------
    dens,vel1,vel2,vel3,pres,B1,B2,B3 : ndarray  primitive variables
    """
    
    #floor variables for density, pressure, and Lorenz factor 
    dens_floor=1.0e-10; pres_floor=1.0e-10; W_ceiling=1.0e4
    
    msqr = mom1**2 + mom2**2 + mom3**2
    SdB  = mom1 * Bcon1 + mom2 * Bcon2 + mom3 * Bcon3
    Bsqr = Bcon1**2 + Bcon2**2 + Bcon3**2
    SdB2 = SdB**2
    gamma_r = eos.GAMMA / (eos.GAMMA - 1.0)
    
    B1 = Bcon1; B2 = Bcon2; B3 = Bcon3

    x = _newton_rMHD(x_init, mass, ener, SdB2, msqr, Bsqr, gamma_r)
    
    #squared velocity and Lorenz factor 
    vsq = (msqr * x**2 + SdB2 * (2.0 * x + Bsqr)) / \
        (x**2 * (x + Bsqr)**2 + 1e-28)
    vsq = np.clip(vsq, 0.0, 1.0 - 1e-14)
    W   = 1.0 / np.sqrt(1.0 - vsq)
    
    # baryon density
    dens = np.maximum(mass / W, dens_floor)
    
    # velocity from the momentum equation:
    #   v_i = [S_i + (S.B / x) B_i] / [x + B^2]
    vel1 = (mom1 + SdB * B1 / (x + 1e-28)) / (x + Bsqr + 1e-28)
    vel2 = (mom2 + SdB * B2 / (x + 1e-28)) / (x + Bsqr + 1e-28)
    vel3 = (mom3 + SdB * B3 / (x + 1e-28)) / (x + Bsqr + 1e-28)

    # pressure from the definition of x = rho h W^2
    pres = np.maximum((x - mass * W) / (gamma_r * W**2), pres_floor)
    
    # --- Super-luminal safeguard ---
    # here we clip the maximal Lorenz factor with W_ceiling for problematic cells
    v2       = vel1**2 + vel2**2 + vel3**2
    too_fast = v2 >= 1.0
    n_clip   = int(np.count_nonzero(too_fast))
    if n_clip > 0:
        v_max = np.sqrt(1.0 - 1.0 / W_ceiling**2)
        fac   = np.where(too_fast, v_max / np.sqrt(v2 + 1e-30), 1.0)
        vel1 *= fac; vel2 *= fac; vel3 *= fac
        print(f"[rMHD] cons2prim: clipped {n_clip} super-luminal cell(s) "
              f"to W = {W_ceiling:.1f}")

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

    func = x - pg + (1.0 - 0.5 / W2) * Bsqr - SdB2 / (2.0 * x**2 + 1e-30) - etot

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
    dx_rel = 1.0e-8   # relative step for numerical derivative
    maxitr = 50
    eps_f  = 1.0e-10     # cushion above the x > mass floor

    # Start from the previous-step x, projected into the valid domain
    x_lim = (1.0 + eps_f) * mass # per-cell physical lower edge
    x = np.maximum(x_init, x_lim)
    active = np.ones_like(x, dtype=bool)

    for itr in range(maxitr):
        if not active.any():
            break
        
        #calculate the numerical derivative, since the pressure eqn is cumbersome 
        dx   = x * dx_rel
        f0   = _x_eqn_rMHD(x,      mass, etot, SdB2, msqr, Bsqr, gamma_r)
        f1   = _x_eqn_rMHD(x + dx, mass, etot, SdB2, msqr, Bsqr, gamma_r)
        deriv = (f1 - f0) / (dx + 1e-30)
        
        deriv = np.where(np.abs(deriv) > 1.0e-30, deriv, -1.0e-30)
        
        x_new = x- f0 / deriv

        # If the full Newton step leaves the valid region or
        # is non-finite, bisect back toward the boundary instead of crossing it
        bad   = ~np.isfinite(x_new) | (x_new <= x_lim)
        x_new = np.where(bad, 0.5 * (x + x_lim), x_new)

        # Move only the still-active cells; freeze the rest
        x_prev = x
        x = np.where(active, x_new, x)

        # Per-cell convergence: small residual AND small relative change
        res = _x_eqn_rMHD(x, mass, etot, SdB2, msqr, Bsqr, gamma_r)
        rel = np.abs(x - x_prev) / (np.abs(x) + 1e-30)
        converged = (np.abs(res) <= tol) & (rel <= tol)
        active = active & ~converged

    # Report the cells that failed to converge 
    n_bad = int(np.count_nonzero(active))
    if n_bad > 0:
        worst = float(np.nanmax(np.abs(
            _x_eqn_rMHD(x, mass, etot, SdB2, msqr, Bsqr, gamma_r))[active]))
        print(f"[rMHD] Newton: {n_bad} cell(s) unconverged after {maxitr} "
              f"iters (max residual = {worst:.3e})")

    return x



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
    enth = eos.enthalpy_sr(dens, pres)
    cs2  = eos.sound_speed_sr(dens, pres)**2

    b2, _, _ = _b_squared(W, vel1, vel2, vel3, B1, B2, B3)
    vA2 = b2 / (dens * enth + b2 + 1e-14)

    cf2 = np.clip(cs2 + vA2 - cs2 * vA2, 0.0, 1.0 - 1e-14)
    return np.sqrt(cf2)


# ============================================================================
# Approximate Riemann solvers (LLF and HLL)
# ============================================================================

def Riemann_rMHD(rhol, rhor,
                   vxl, vxr, vyl, vyr, vzl, vzr,
                   pl, pr,
                   bxl, bxr, byl, byr, bzl, bzr,
                   eos, solver_type, dim):
    """
    Approximate Riemann fluxes for the SRMHD equations.

    Supports LLF (Rusanov) and HLL solvers.
    For dim=2 the system is solved after rotating coordinates so that the
    x-direction is always the normal direction, 
    see Mignone & Bodo (2006), MNRAS 368, 1040.

    Parameters
    ----------
    rhol, rhor               : ndarray  left/right density
    vxl,vxr, vyl,vyr, vzl,vzr : ndarray  left/right 3-velocities
    pl, pr                   : ndarray  left/right pressure
    bxl,bxr, byl,byr, bzl,bzr : ndarray  left/right B-field components
    eos                      : EOSdata
    solver_type                : {'LLF', 'HLL'}
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

    # ----------------------------------------------------------------
    # LLF (Local Lax-Friedrichs / Rusanov)
    # ----------------------------------------------------------------
    if solver_type == 'LLF':

        Fmass, Fmom1, Fmom2, Fmom3, Fetot, Fbfix, Fbfiy, Fbfiz = \
              LLF_flux(rhol,rhor, vxl,vxr, vyl,vyr, vzl,vzr, pl,pr, bxl,bxr, byl,byr, bzl,bzr, eos)

    # ----------------------------------------------------------------
    # HLL (Harten-Lax-van Leer)
    # ----------------------------------------------------------------
    elif solver_type == 'HLL':

        Fmass, Fmom1, Fmom2, Fmom3, Fetot, Fbfix, Fbfiy, Fbfiz = \
              HLL_flux(rhol,rhor, vxl,vxr, vyl,vyr, vzl,vzr, pl,pr, bxl,bxr, byl,byr, bzl,bzr, eos)

    else:
        
        #solver_type is incorrect -> throw an error
        raise ValueError(
            f"Unknown rMHD solver_type '{solver_type}'. " 
            f"Expected one of ['LLF', 'HLL'].")

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

def boundCond_rMHD(grid, BC, BCm, MHD):
    """
    Apply boundary conditions to SRMHD primitive variables.

    Mirrors the structure of boundCond_MHD in MHD_phys.py, applying BCs
    to density, pressure, velocity, and cell-centred magnetic field.
    
    Notes
    ----------
    Fixed boundaries are not supproted yet for CT rMHD

    Parameters
    ----------
    grid  : Grid
    BC    : array of 4 str  --  [x1_inner, x2_inner, x1_outer, x2_outer]
    BCm   : array of 4 str  --  [x1_inner, x2_inner, x1_outer, x2_outer]
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
        MHD.dens = apply_bc_scalar(MHD.dens, Ngc, bc, axis=axis, side=side)
        MHD.pres = apply_bc_scalar(MHD.pres, Ngc, bc, axis=axis, side=side)

        MHD.vel1, MHD.vel2, MHD.vel3 = apply_bc_vector(
            MHD.vel1, MHD.vel2, MHD.vel3, Ngc, bc, axis=axis, side=side)

    for axis, side, bc in [
        (1, 'inner', BCm[0]),
        (2, 'inner', BCm[1]),
        (1, 'outer', BCm[2]),
        (2, 'outer', BCm[3]),
    ]:
        MHD.bfi1, MHD.bfi2, MHD.bfi3 = apply_bc_vector(
            MHD.bfi1, MHD.bfi2, MHD.bfi3, Ngc, bc, axis=axis, side=side)

    return MHD
