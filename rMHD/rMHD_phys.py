# -*- coding: utf-8 -*-
"""
rMHD_phys.py
============

Core physics routines for 2D Special-Relativistic Magnetohydrodynamics (SRMHD).

This module provides:
  - Primitive <-> conservative variable conversions
  - SR fast magnetosonic wave speed estimation
  - Approximate Riemann solvers: LLF, HLL
  - Boundary conditions for SRMHD primitive variables

Conservative variables (flat Minkowski spacetime, c = 1)
---------------------------------------------------------
Notation follows Del Zanna et al. (2003, A&A 400 397) and the PLUTO code
(Mignone et al. 2012, ApJS 198 7).

  W     = 1 / sqrt(1 - v²)          Lorentz factor
  h     = 1 + Γp / (ρ(Γ-1))         specific enthalpy
  b²    = B²/W² + (v·B)²            covariant magnetic invariant
  p_tot = p + b²/2                   total (gas + magnetic) pressure

  D     = ρ W                        baryon number density
  S_i   = (ρhW² + B²) v_i - (v·B) B_i   momentum density  [i=1,2,3]
  E     = ρhW² - p + (B² + |v×B|²)/2    total energy density
  B_i                                unchanged (ideal MHD: ∂_t B = -∇×E_ideal)

where |v×B|² = B²v² - (v·B)²   (Lagrange identity).

Physical fluxes in the x-direction
-----------------------------------
These have the SAME FORM as non-relativistic MHD; the SR physics enters
only through the definitions of S_i and E.

  F^x_D     = D vx
  F^x_{S_1} = S_1 vx - B_x² + p_tot
  F^x_{S_2} = S_2 vx - B_x B_y
  F^x_{S_3} = S_3 vx - B_x B_z
  F^x_E     = (E + p_tot) vx - B_x (v·B)
  F^x_{B_x} = 0                          (CT: normal flux zero)
  F^x_{B_y} = B_y vx - B_x vy            (ideal induction)
  F^x_{B_z} = B_z vx - B_x vz

Primitive-variable recovery (cons → prim)
-----------------------------------------
This is the hard part of SRMHD.  Given conserved (D, S_i, E, B_i), one must
solve a non-linear system for (ρ, v_i, p).  The standard approach reduces the
system to a single scalar equation in one unknown (e.g. W or p) which is then
solved with Newton-Raphson.  See:

  - Noble et al.   (2006), ApJ Suppl. 164, 536
  - Mignone & McKinney (2007), MNRAS 378, 1118

The function `cons2prim_sr_MHD` below is intentionally left as a stub —
fill in your preferred inversion scheme.

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

import os
import sys
import numpy as np

# Allow importing Piastra modules from the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from boundaries import apply_bc_scalar, apply_bc_vector


# ============================================================================
# Helper: derived quantities from primitives
# ============================================================================

def _lorentz(v1, v2, v3):
    """
    Lorentz factor W = 1 / sqrt(1 - v²).

    Velocities are clipped to ensure v² < 1 (sub-luminal flow).
    """
    v2 = np.clip(v1**2 + v2**2 + v3**2, 0.0, 1.0 - 1e-14)
    return 1.0 / np.sqrt(1.0 - v2)


def _b_squared(W, v1, v2, v3, B1, B2, B3):
    """
    Covariant magnetic invariant  b² = B²/W² + (v·B)².

    Parameters
    ----------
    W            : Lorentz factor ndarray
    v1,v2,v3     : 3-velocity components
    B1,B2,B3     : lab-frame magnetic field components

    Returns
    -------
    b2  : ndarray  covariant magnetic invariant
    vdB : ndarray  dot product  v·B
    B2_lab : ndarray  B₁² + B₂² + B₃²
    """
    vdB   = v1 * B1 + v2 * B2 + v3 * B3
    B2lab = B1**2 + B2**2 + B3**2
    b2    = B2lab / W**2 + vdB**2
    return b2, vdB, B2lab


# ============================================================================
# Primitive → conservative
# ============================================================================

def prim2cons_sr_MHD(dens, vel1, vel2, vel3, pres, B1, B2, B3, eos):
    """
    Convert primitive to conservative variables for an ideal-gas SRMHD fluid.

    Parameters
    ----------
    dens             : ndarray  rest-mass density  ρ
    vel1,vel2,vel3   : ndarray  3-velocity components  vⁱ   (|v| < 1)
    pres             : ndarray  thermal pressure  p
    B1,B2,B3         : ndarray  lab-frame magnetic field components  Bⁱ
    eos              : EOSdata  equation of state (provides GAMMA)

    Returns
    -------
    D    : ndarray   D = ρW
    S1   : ndarray   x-momentum
    S2   : ndarray   y-momentum
    S3   : ndarray   z-momentum
    E    : ndarray   total energy  = ρhW² - p + (B² + |v×B|²)/2
    B1,B2,B3 : ndarray  unchanged (passed through for convenience)
    """
    W    = _lorentz(vel1, vel2, vel3)
    W2 = W**2
    enth = 1.0 + pres / (dens + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)

    b2, vdB, Bsq = _b_squared(W, vel1, vel2, vel3, B1, B2, B3)
    vsq  = vel1**2 + vel2**2 + vel3**2

    D  = dens * W
    S1 = (dens * enth * W2 + Bsq) * vel1 - vdB * B1
    S2 = (dens * enth * W2 + Bsq) * vel2 - vdB * B2
    S3 = (dens * enth * W2 + Bsq) * vel3 - vdB * B3
    E  = dens * enth * W2 - pres + 0.5 * (Bsq + Bsq * vsq - vdB**2)

    return D, S1, S2, S3, E


# ============================================================================
# Conservative → primitive  (STUB — fill in your inversion scheme)
# ============================================================================

def cons2prim_sr_MHD(mass, mom1, mom2, mom3, ener, B1, B2, B3, x_init, eos):
    """
    Recover primitive variables from conservative variables for SRMHD.

    THIS FUNCTION IS A STUB.
    ----------------------------------
    The con→prim inversion in SRMHD is a non-trivial implicit problem.
    We use the scalar solver for ideal-gas EOS, written in 
    Mignone & Bodo 2005 (see also Noble et al. 2006):

      1.  Define  x ≡ ρhW²  (unknown scalar).

      2.  From the momentum equations:
              (v·B) = S·B / (z - B²)


      A single Newton-Raphson iteration in x converges quickly
      for typical astrophysical conditions.

    Parameters
    ----------
    D,S1,S2,S3,E   : ndarray  conservative variables (cell interiors only)
    B1,B2,B3       : ndarray  magnetic field (same shape)
    pres_init      : ndarray  pressure guess (from previous timestep)
    eos            : EOSdata

    Returns
    -------
    dens,vel1,vel2,vel3,pres,B1,B2,B3 : ndarray  primitive variables
    """
    
    msqr = mom1**2 + mom2**2 + mom3**2
    S = mom1 * B1 + mom2 * B2 + mom3 * B3
    Bsqr = B1**2 + B2**2 + B3**2     
    Ssqr = S**2
    gamma_r = eos.GAMMA/(eos.GAMMA-1.0)    
    
    x = _newton_rMHD(x_init, mass, etot, Ssqr, msqr, Bsqr, gamma_r)
    
    #Lorentz factor 
    W =  np.sqrt(np.maximum( 1.0/(1.0 - \
        (Ssqr * (2.0 * x + Bsqr) + msqr * x**2) / \
        (x**2 + Bsqr)**2 / x**2), 1.0 ))
            
    #number density conservation 
    dens = mass / W
    #energy conervation
    pres = (x - mass * W)/(gamma_r * W**2)
    
    #from the momentum eqn:
    # v_i   = [mom_i + (v·B) B_i / x] / [x + B^2]
    vel1 = (mom1 + S * B1/x)/(x + Bsqr)  
    vel1 = (mom2 + S * B2/x)/(x + Bsqr) 
    vel1 = (mom3 + S * B3/x)/(x + Bsqr) 

    return dens,vel1,vel2,vel3,pres,B1,B2,B3

# ============================================================================
#   Nonlinear rMHD solver (Newton-Raphson)
# ============================================================================

def _x_eqn_rMHD(x, mass, etot, S2, m2, B2, gamma_r):
    """
    Nonlinear equation f(x) = 0 whose root gives the x = ρ h W².
    
    Here we follow the approach from Mignone & Bodo (2005)

    Parameters and Returns: arrays of same shape as `x`.
    
    """
    
    GAMMA2 = np.maximum( 1.0/(1.0 - \
        (S2 * (2.0 * x + B2) + m2 * x**2) / \
        (x**2 + B2)**2 / x**2), 1.0 )
    GAMMA = np.sqrt(GAMMA2)
    
    pg = (x - mass*GAMMA)/(gamma_r*GAMMA2)
    
    func = x - pg + (1.0 - 1.0/2.0/GAMMA2)*B2 - S2/2.0/x**2 - etot
    
    return func
    
    
def _x_der_rMHD(x, mass, etot, S2, B2, gamma_r):
    """
    Derivative  df(x)/dx for root finding procedure in rMHD.

    Parameters and Returns: arrays of same shape as `pres`.
    """
    
    GAMMA2 = np.maximum( 1.0/(1.0 - \
        (S2 * (2.0 * x + B2) + m2 * x**2) / \
        (x**2 + B2)**2 / x**2), 1.0 )
    GAMMA = np.sqrt(GAMMA2)
    
    dgammadx = -GAMMA**3 * (2.0 * S2 * (3.0 * x**2 + 3 * x * B2 + B2**2) + \
        m2 * x**3 )/(2.0 * x**3 * (x + B2)**3
        
    dpgdx = (GAMMA * (1.0 + mass * dgammadx) - 2.0 * x * dgammadx)/ \
        gamma_r / GAMMA**3
    
    der = 1.0 - dpgdx + B2 * dgammadx / GAMMA**3 + S2 / x**3
    
    return der


def _newton_rMHD(x_init, mass, etot, S2, m2, B2, gamma_r):
    """
    Newton-Raphson iteration to solve f(x) = 0 for x = ρhW²

    Convergence is declared when both the residual max-norm and the relative
    update are below `tol`.

    Parameters
    ----------
    pres_init : ndarray   –  initial pressure guess
    mass, mom1, mom2, mom3, etot : ndarray  –  conservative state
    GAMMA : float

    Returns
    -------
    pres : ndarray   –  converged pressure field
    """
    tol    = 1.0e-8
    maxitr = 100

    x_init = np.maximum(x_init, 1.0e-14)

    res   = _x_eqn_rMHD(x_init, mass, etot, S2, m2, B2, gamma_r)
    eps1  = np.max(np.abs(res))
    eps2  = 1.0

    for itr in range(maxitr):
        if eps1 <= tol and eps2 <= tol:
            break

        f0    = _x_eqn_rMHD(x, mass, etot, S2, m2, B2, gamma_r)
        der = _x_der_rMHD(x, mass, etot, S2, B2, gamma_r)

        update = f0 / (der + 1.0e-28)
        x   = x - update
        x   = np.maximum(x, 1.0e-14)

        res  = _x_eqn_rMHD(x, mass, etot, S2, m2, B2, gamma_r)
        eps1 = np.max(np.abs(res))
        eps2 = np.max(np.abs(update / (pres + 1.0e-28)))
    else:
        print(f"[rHD] pressure Newton solver: did not converge after {maxitr} iterations "
              f"(residual = {eps1:.3e})")

    return x
    


# ============================================================================
# SR fast magnetosonic speed (upper bound)
# ============================================================================

def sound_speed_sr_MHD(dens, pres, eos):
    """
    Adiabatic sound speed for an ideal relativistic gas (same as rHD).

       cs² = Γ p / (ρ h)

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

        v_A² = b² / (ρh + b²)      relativistic Alfvén speed²
        cs²  = Γp / (ρh)           SR sound speed²
        c_f² ≈ cs² + vA² - cs²·vA² (relativistic analogue of NR formula)

    The actual fast speed requires solving a quartic; this bound is
    sufficiently accurate for the CFL condition and HLL/LLF wave speeds.

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
    vA2  = b2 / (dens * enth + b2 + 1e-14)

    cf2  = np.clip(cs2 + vA2 - cs2 * vA2, 0.0, 1.0 - 1e-14)
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
    An HLLD solver for SRMHD is not implemented; fill it in following
    Mignone & Bodo (2006) or Mignone, Ugliano & Bodo (2009).

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
    Bxn  = 0.5 * (bxl + bxr)

    # ----------------------------------------------------------------
    # Derived quantities (left state)
    # ----------------------------------------------------------------
    Wl    = _lorentz(vxl, vyl, vzl)
    enthl = 1.0 + pl / (rhol + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)
    b2l, vdBl, Bsql = _b_squared(Wl, vxl, vyl, vzl, Bxn, byl, bzl)
    vsql  = vxl**2 + vyl**2 + vzl**2

    # Left conservative state
    zl    = rhol * enthl * Wl**2 + Bsql
    Dl    = rhol * Wl
    S1l   = zl * vxl - vdBl * Bxn
    S2l   = zl * vyl - vdBl * byl
    S3l   = zl * vzl - vdBl * bzl
    El    = rhol * enthl * Wl**2 - pl + 0.5 * (Bsql + Bsql * vsql - vdBl**2)
    ptotl = pl + 0.5 * b2l
    zl = (rhol * enthl + b2l) * Wl**2 #rewrite it, now for fluxes 
    bbxl = bxn / Wl + Wl * vdBl * vxl
    bbyl = byl / Wl + Wl * vdBl * vyl
    bbzl = bzl / Wl + Wl * vdBl * vzl
    
    # Left physical fluxes (x-direction)
    FDl   = Dl  * vxl
    FS1l  = zl * vxl**2 - bbxl**2 + ptotl
    FS2l  = zl * vxl * vyl - bbxl * bbyl
    FS3l  = zl * vxl * vzl - bbxl * bbzl
    FEl   = S1l
    FByl  = byl * vxl - Bxn * vyl
    FBzl  = bzl * vxl - Bxn * vzl

    # ----------------------------------------------------------------
    # Derived quantities (right state)
    # ----------------------------------------------------------------
    Wr    = _lorentz(vxr, vyr, vzr)
    enthr = 1.0 + pr / (rhor + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)
    b2r, vdBr, Bsqr = _b_squared(Wr, vxr, vyr, vzr, Bxn, byr, bzr)
    vsqr  = vxr**2 + vyr**2 + vzr**2
    
    zr    = rhor * enthr * Wr**2 + Bsqr
    Dr    = rhor * Wr
    S1r   = zr * vxr - vdBr * Bxn
    S2r   = zr * vyr - vdBr * byr
    S3r   = zr * vzr - vdBr * bzr
    Er    = rhor * enthr * Wr**2 - pr + 0.5 * (Bsqr + Bsqr * vsqr - vdBr**2)
    ptotr = pr + 0.5 * b2r
    zr = (rhor * enthr + b2r) * Wr**2 #rewrite it, now for fluxes 
    bbxr = bxn / Wr + Wr * vdBr * vxr
    bbyr = byr / Wr + Wr * vdBr * vyr
    bbzr = bzr / Wr + Wr * vdBr * vzr
    
    # Right physical fluxes (x-direction)
    FDr   = Dr  * vxr
    FS1r  = zr * vxr**2 - bbxr**2 + ptotr
    FS2r  = zr * vxr * vyr - bbxr * bbyr
    FS3r  = zr * vxr * vzr - bbxr * bbzr
    FEr   = S1r
    FByr  = byr * vxr - Bxn * vyr
    FBzr  = bzr * vxr - Bxn * vzr

    # ----------------------------------------------------------------
    # SR wave-speed estimates (Doppler-shifted fast magnetosonic speed)
    # The fast speed uses the approximation from fast_magnetosonic_speed_sr.
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

    Mirrors the structure of boundCond_MHD in MHD_phys.py.

    Parameters
    ----------
    grid  : Grid
    BC    : array of 4 str  –  [x1_inner, x2_inner, x1_outer, x2_outer]
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
