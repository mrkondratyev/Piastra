# -*- coding: utf-8 -*-
"""
rMHD_riemann_approx.py

Approximate Riemann solvers for special-relativistic MHD.

Implemented solvers, in increasing order of accuracy/cost:

    LLF   - Local Lax-Friedrichs / Rusanov (1961)
    HLL   - Harten, Lax, van Leer (1983)

All solvers share the same calling convention (see Riemann_rMHD in 
                                               rMHD_phys file).

Each Riemann solver routine has the following i/o sturcture:

   Parameters
   ----------
   rhol, rhor : ndarray
       Left and right densities.
   vxl, vxr, vyl, vyr, vzl, vzr : ndarray
       Left and right velocity components.
   pl, pr : ndarray
       Left and right pressures.
   bxl, bxr, byl, byr, bzl, bzr : ndarray
       Left and right magnetic field components.
   eos : object
       Equation of state object.
  

   Returns
   -------
   Fmass : ndarray
       Flux of mass density.
   Fmomx, Fmomy, Fmomz : ndarray
       Fluxes of momentum components.
   Fetot : ndarray
       Flux of total energy.
   Fbfix, Fbfiy, Fbfiz : ndarray
       Fluxes of magnetic field components.
   

Author: mrkondratyev
"""
import numpy as np



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

    # Normal B-field: use arithmetic average (as in NR MHD)
    Bxn = 0.5 * (bxl + bxr)

    # ----------------------------------------------------------------
    # Derived quantities (left state)
    # ----------------------------------------------------------------
    Wl    = _lorentz(vxl, vyl, vzl)
    enthl = eos.enthalpy_sr(rhol, pl)
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
    bbxl  = Bxn / Wl + Wl * vdBl * vxl
    bbyl  = byl / Wl + Wl * vdBl * vyl
    bbzl  = bzl / Wl + Wl * vdBl * vzl

    # Left physical fluxes (x-direction)
    FDl   = Dl * vxl
    FS1l  = S1l * vxl - Bxn * bbxl / Wl + ptotl
    FS2l  = S2l * vxl - Bxn * bbyl / Wl
    FS3l  = S3l * vxl - Bxn * bbzl / Wl
    FEl   = S1l           # SR identity: energy flux = momentum density
    FByl  = byl * vxl - Bxn * vyl
    FBzl  = bzl * vxl - Bxn * vzl

    # ----------------------------------------------------------------
    # Derived quantities (right state)
    # ----------------------------------------------------------------
    Wr    = _lorentz(vxr, vyr, vzr)
    enthr = eos.enthalpy_sr(rhor, pr)
    b2r, vdBr, Bsqr = _b_squared(Wr, vxr, vyr, vzr, Bxn, byr, bzr)
    vsqr = vxr**2 + vyr**2 + vzr**2

    zr    = rhor * enthr * Wr**2 + Bsqr
    Dr    = rhor * Wr
    S1r   = zr * vxr - vdBr * Bxn
    S2r   = zr * vyr - vdBr * byr
    S3r   = zr * vzr - vdBr * bzr
    Er    = rhor * enthr * Wr**2 - pr + 0.5 * (Bsqr + Bsqr * vsqr - vdBr**2)
    ptotr = pr + 0.5 * b2r

    bbxr  = Bxn / Wr + Wr * vdBr * vxr
    bbyr  = byr / Wr + Wr * vdBr * vyr
    bbzr  = bzr / Wr + Wr * vdBr * vzr

    # Right physical fluxes (x-direction)
    FDr   = Dr * vxr
    FS1r  = S1r * vxr - Bxn * bbxr / Wr + ptotr
    FS2r  = S2r * vxr - Bxn * bbyr / Wr
    FS3r  = S3r * vxr - Bxn * bbzr / Wr
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
    if solver_type == 'LLF':

        lam = np.maximum(np.abs(Sl), np.abs(Sr)) # 1.0

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
    elif solver_type == 'HLL':

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







