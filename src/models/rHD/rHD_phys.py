# -*- coding: utf-8 -*-
"""
rHD_phys.py

Core routines for special-relativistic hydrodynamics (rHD) solvers
===================================================================

This module mirrors hydro_phys.py for the special-relativistic Euler equations.
It provides:

- Primitive ↔ conservative variable conversions for an ideal-gas SR fluid
- A Newton-Raphson pressure solver for the con→prim inversion
- SR adiabatic sound speed
- Approximate Riemann solvers: LLF, HLL, HLLC
- Boundary condition filler for SR fluid state

Conservative variables (per unit coordinate volume)
----------------------------------------------------
    D  = ρ W                               (baryon number density)
    mᵢ = ρ h W² vᵢ  (i = 1, 2, 3)        (momentum density components)
    E  = ρ h W² − p                        (total energy density, = T⁰⁰)

where W = 1/√(1 − v²) is the Lorentz factor and h = 1 + Γp/(ρ(Γ−1)) the
specific enthalpy for an ideal Γ-law gas.

Primitive ↔ conservative inversion
------------------------------------
Unlike NR hydro, the mapping from (D, m, E) to (ρ, v, p) is implicit.
We solve the scalar nonlinear equation (Mignone & Bodo 2005, eq. 3)

    f(p) = D W(p) + Γ/(Γ−1) · p · W(p)² − E − p = 0

using a Newton-Raphson iteration, where  W(p) = (E+p)/√((E+p)² − |m|²).

Reconstruction
--------------
Spatial reconstruction is performed on the 4-velocity components
uⁱ = W vⁱ  (not the 3-velocities) to guarantee that the reconstructed
state satisfies |v| < 1.  After reconstruction the 3-velocity is
recovered via  vⁱ = uⁱ/√(1 + |u|²).

References
----------
- Mignone, A. & Bodo, G. (2005), MNRAS 364, 126
  "An HLLC Riemann solver for relativistic flows"
- Del Zanna, L. et al. (2003), A&A 400, 397
- Toro, E.F. (2009), "Riemann Solvers and Numerical Methods for Fluid Dynamics"

Author
------
mrkondratyev
"""

import numpy as np
from src.common.boundaries import (
    apply_bc_scalar, 
    apply_bc_vector)


# ============================================================================
#   Primitive <-> conservative conversions
# ============================================================================

def prim2cons_sr_hydro(dens, vel1, vel2, vel3, pres, eos):
    """
    Convert primitive to conservative variables for a special-relativistic ideal gas.

    Parameters
    ----------
    dens : ndarray   –  rest-mass density  ρ
    vel1, vel2, vel3 : ndarray  –  3-velocity components  vⁱ  (|v| < 1)
    pres : ndarray   –  thermal pressure  p
    eos  : EOSdata   –  equation of state (provides GAMMA)

    Returns
    -------
    mass : ndarray   –  D  = ρ W
    mom1, mom2, mom3 : ndarray  –  mᵢ = ρ h W² vᵢ
    etot : ndarray   –  E  = ρ h W² − p
    """
    W    = 1.0 / np.sqrt(1.0 - vel1**2 - vel2**2 - vel3**2)
    enth = eos.enthalpy_sr(dens, pres)

    mass = dens * W
    mom1 = mass * enth * W * vel1
    mom2 = mass * enth * W * vel2
    mom3 = mass * enth * W * vel3
    etot = mass * enth * W - pres

    return mass, mom1, mom2, mom3, etot


def cons2prim_sr_hydro(mass, mom1, mom2, mom3, etot, pres_init, eos):
    """
    Recover primitive variables from conservative variables via Newton-Raphson.

    Parameters
    ----------
    mass : ndarray   –  D
    mom1, mom2, mom3 : ndarray  –  mᵢ
    etot : ndarray   –  E
    pres_init : ndarray  –  initial pressure guess (e.g., previous timestep value)
    eos  : EOSdata

    Returns
    -------
    dens, vel1, vel2, vel3, pres : ndarray
    """
    pres = _newton_pres_sr(pres_init, mass, mom1, mom2, mom3, etot, eos.GAMMA)

    W    = (etot + pres) / np.sqrt((etot + pres)**2 - (mom1**2 + mom2**2 + mom3**2))
    dens = mass / W
    enth = eos.enthalpy_sr(dens, pres)
    
    vel1 = mom1 / (mass * enth * W)
    vel2 = mom2 / (mass * enth * W)
    vel3 = mom3 / (mass * enth * W)
    
    # safety prescription -- set den and pres above floor values of 1e-12
    dens = np.maximum(dens, 1e-12)
    pres = np.maximum(pres, 1e-12)

    # Clip velocity to sub-luminal
    v2 = vel1**2 + vel2**2 + vel3**2
    too_fast = v2 >= 1.0
    if np.any(too_fast):
        fac = np.where(too_fast, 0.9999 / np.sqrt(v2 + 1e-28), 1.0)
        vel1 *= fac; vel2 *= fac; vel3 *= fac

    return dens, vel1, vel2, vel3, pres


# ============================================================================
#   Nonlinear pressure solver (Newton-Raphson)
# ============================================================================

def _pres_eqn_sr(pres, mass, m2, etot, GAMMA):
    """
    Nonlinear equation f(p) = 0 whose root gives the SR pressure.

    f(p) = D·W(p) + Γ/(Γ−1)·p·W(p)² − E − p

    where W(p) = (E+p)/√((E+p)²−|m|²).

    Parameters and Returns: arrays of same shape as `pres`.
    """
    Ep   = etot + pres
    Epm2 = Ep**2 - m2
    W2   = Ep**2 / Epm2
    
    func = mass * np.sqrt(np.maximum(W2, 1.0)) + GAMMA / (GAMMA - 1.0) * pres * W2 - Ep
    
    return func


def _pres_der_sr(pres, mass, m2, etot, GAMMA):
    """
    Derivative  df(p)/dp for root finding procedure in SR hydrodynamics.

    f(p) = D·W(p) + Γ/(Γ−1)·p·W(p)² − E − p

    where W(p) = (E+p)/√((E+p)²−|m|²).

    Parameters and Returns: arrays of same shape as `pres`.
    """
    Ep   = etot + pres
    Epm2 = Ep**2 - m2
    
    der = -mass * m2 / Epm2**1.5 + \
        GAMMA / (GAMMA - 1.0) * Ep * (Ep**3 - m2*(Ep+2.0*pres)) / Epm2**2 - 1.0
    
    return der


def _newton_pres_sr(pres_init, mass, mom1, mom2, mom3, etot, GAMMA):
    """
    Newton-Raphson iteration to solve f(p) = 0 for SR pressure.

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
    maxitr = 50

    m2   = mom1**2 + mom2**2 + mom3**2
    pres = np.maximum(pres_init, 1e-12)

    res   = _pres_eqn_sr(pres, mass, m2, etot, GAMMA)
    eps1  = np.max(np.abs(res))
    eps2  = 1.0

    for itr in range(maxitr):
        if eps1 <= tol and eps2 <= tol:
            break

        f0    = _pres_eqn_sr(pres, mass, m2, etot, GAMMA)
        deriv = _pres_der_sr(pres, mass, m2, etot, GAMMA)

        update = f0 / (deriv + 1e-16)
        pres   = pres - update
        pres   = np.maximum(pres, 1e-12)

        res  = _pres_eqn_sr(pres, mass, m2, etot, GAMMA)
        eps1 = np.max(np.abs(res))
        eps2 = np.max(np.abs(update / (pres + 1e-16)))
    else:
        print(f"[rHD] pressure Newton solver: did not converge after {maxitr} iterations "
              f"(residual = {eps1:.3e})")

    return pres


# -------------------------
# Small helper: conservative SR hydro variables + their fluxes along Ox 
# -------------------------
def sr_hydro_cons_and_flux(rho, vx, vy, vz, p, eos):
    
    # --- specific enthalpy ---
    ent = eos.enthalpy_sr(rho, p)

    # --- Lorentz factor ---
    W = 1.0 / np.sqrt(1.0 - vx**2 - vy**2 - vz**2)
    
    # --- conservative state ---
    #number density 
    m   = rho * W
    #temporary variable 
    tmp = m * ent * W 
    #momentum components 
    mx = tmp * vx; my = tmp * vy; mz = tmp * vz
    #total energy 
    e  = tmp - p

    # --- conservative fluxes ---
    Fm = m * vx
    Fx = mx * vx + p; Fy = mx * vy; Fz = mx * vz
    Fe = mx
    
    #output -- conservative state + fluxes 
    return m, mx, my, mz, e, \
        ent, W, Fm, Fx, Fy, Fz, Fe



# ============================================================================
#   Approximate SR Riemann solvers
# ============================================================================

def Riemann_sr_hydro(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos, flux_type, dim):
    """
    Approximate Riemann fluxes for the special-relativistic Euler equations.

    The Riemann problem is always solved in the normal (x₁) direction; for
    dim=2 a coordinate rotation is applied before and undone after.

    Supported solvers
    -----------------
    ``'LLF'``  – Local Lax-Friedrichs / Rusanov  (most diffusive, most robust)
    ``'HLL'``  – Harten-Lax-van Leer (1983)
    ``'HLLC'`` – HLL with contact restoration (Mignone & Bodo 2005)

    Wave-speed estimates follow Mignone & Bodo (2005) eqs. (9)–(10).

    Parameters
    ----------
    rhol, rhor : ndarray   –  left/right rest-mass density
    vxl, vxr   : ndarray   –  left/right normal velocity
    vyl, vyr   : ndarray   –  left/right tangential velocity 1
    vzl, vzr   : ndarray   –  left/right tangential velocity 2
    pl, pr     : ndarray   –  left/right pressure
    eos        : EOSdata
    flux_type  : str        –  'LLF', 'HLL', or 'HLLC'
    dim        : int        –  1 (normal) or 2 (tangential → rotate)

    Returns
    -------
    Fmass, Fmomx, Fmomy, Fmomz, Fetot : ndarray
        Interface fluxes for D, m₁, m₂, m₃, E.
    """
    # --- coordinate rotation for dim=2 ---
    if dim == 2:
        vxl, vxr, vyl, vyr = vyl, vyr, -vxl, -vxr

    #left state and fluxes 
    Dl, momxl, momyl, momzl, El, \
    entl, Wl, \
    FDl, Fmxl, Fmyl, Fmzl, FEl  = \
        sr_hydro_cons_and_flux(rhol, vxl, vyl, vzl, pl, eos)
        
    #right state and fluxes 
    Dr, momxr, momyr, momzr, Er, \
    entr, Wr, \
    FDr, Fmxr, Fmyr, Fmzr, FEr  = \
        sr_hydro_cons_and_flux(rhor, vxr, vyr, vzr, pr, eos)
    
    # --- SR wave-speed estimates (Mignone & Bodo 2005, eqs. 9-10) ---
    cs2l = eos.sound_speed_sr(rhol, pl)**2
    cs2r = eos.sound_speed_sr(rhor, pr)**2

    sigl = cs2l / (Wl**2 * (1.0 - cs2l) + 1e-14)
    sigr = cs2r / (Wr**2 * (1.0 - cs2r) + 1e-14)

    bl_m = (vxl - np.sqrt(sigl * (1.0 - vxl**2 + sigl))) / (1.0 + sigl)
    bl_p = (vxl + np.sqrt(sigl * (1.0 - vxl**2 + sigl))) / (1.0 + sigl)
    br_m = (vxr - np.sqrt(sigr * (1.0 - vxr**2 + sigr))) / (1.0 + sigr)
    br_p = (vxr + np.sqrt(sigr * (1.0 - vxr**2 + sigr))) / (1.0 + sigr)

    Sl = np.minimum(bl_m, br_m)
    Sr = np.maximum(bl_p, br_p)

    # ----------------------------------------------------------------
    if flux_type == 'LLF':

        lam  = np.maximum(np.abs(Sl), np.abs(Sr))
        Fmass = 0.5 * (FDl  + FDr  - lam * (Dr    - Dl   ))
        Fmomx = 0.5 * (Fmxl + Fmxr - lam * (momxr - momxl))
        Fmomy = 0.5 * (Fmyl + Fmyr - lam * (momyr - momyl))
        Fmomz = 0.5 * (Fmzl + Fmzr - lam * (momzr - momzl))
        Fetot = 0.5 * (FEl  + FEr  - lam * (Er    - El   ))

    elif flux_type == 'HLL':

        Sl = np.minimum(Sl, 0.0)
        Sr = np.maximum(Sr, 0.0)

        Fmass = (Sr * FDl  - Sl * FDr  + Sr * Sl * (Dr    - Dl   )) / (Sr - Sl)
        Fmomx = (Sr * Fmxl - Sl * Fmxr + Sr * Sl * (momxr - momxl)) / (Sr - Sl)
        Fmomy = (Sr * Fmyl - Sl * Fmyr + Sr * Sl * (momyr - momyl)) / (Sr - Sl)
        Fmomz = (Sr * Fmzl - Sl * Fmzr + Sr * Sl * (momzr - momzl)) / (Sr - Sl)
        Fetot = (Sr * FEl  - Sl * FEr  + Sr * Sl * (Er    - El   )) / (Sr - Sl)

    elif flux_type == 'HLLC':

        Sl = np.minimum(Sl, 0.0)
        Sr = np.maximum(Sr, 0.0)

        # HLL intermediate state (needed for contact speed calculation)
        momx_hll = (Sr * momxr - Sl * momxl + Fmxl - Fmxr) / (Sr - Sl)
        etot_hll = (Sr * Er   - Sl * El   + FEl  - FEr ) / (Sr - Sl)
        Fmx_hll  = (Sr * Fmxl - Sl * Fmxr + Sr * Sl * (momxr - momxl)) / (Sr - Sl)
        FE_hll   = (Sr * FEl  - Sl * FEr  + Sr * Sl * (Er   - El  )) / (Sr - Sl)

        # Contact wave speed  "Ss" (Mignone & Bodo 2005, eq. 18)
        disc  = np.maximum((etot_hll + Fmx_hll)**2 - 4.0 * momx_hll * FE_hll, 0.0)
        Ss = ((etot_hll + Fmx_hll) - np.sqrt(disc)) / (2.0 * FE_hll + 1e-28)

        # Starred pressure (Mignone & Bodo 2005, eq. 17)
        Pstar = (pl + Sl * Ss * El - (Ss + Sl - vxl) * momxl) / (1.0 - Sl * Ss) 
        #Pstar = (pr + Sr * Ss * Er - (Ss + Sr - vxr) * momxr) / (1.0 - Sr * Ss)

        Dl_s    = Dl     * (Sl - vxl) / (Sl - Ss)
        momxl_s = (momxl * (Sl - vxl) + Pstar - pl) / (Sl - Ss)
        momyl_s = momyl  * (Sl - vxl) / (Sl - Ss)
        momzl_s = momzl  * (Sl - vxl) / (Sl - Ss)
        El_s    = (El    * (Sl - vxl) + Pstar * Ss - pl * vxl) / (Sl - Ss)

        Dr_s    = Dr     * (Sr - vxr) / (Sr - Ss)
        momxr_s = (momxr * (Sr - vxr) + Pstar - pr) / (Sr - Ss)
        momyr_s = momyr  * (Sr - vxr) / (Sr - Ss)
        momzr_s = momzr  * (Sr - vxr) / (Sr - Ss)
        Er_s    = (Er    * (Sr - vxr) + Pstar * Ss - pr * vxr) / (Sr - Ss)

        def _hllc_state(FL, FR, UL, UR, ULs, URs):
            return np.where(
                Sl >= 0.0, FL,
                np.where((Sl < 0.0) & (Ss >= 0.0), FL + Sl * (ULs - UL),
                np.where((Ss < 0.0) & (Sr >= 0.0), FR + Sr * (URs - UR), FR)))

        Fmass = _hllc_state(FDl,  FDr,  Dl,    Dr,    Dl_s,    Dr_s   )
        Fmomx = _hllc_state(Fmxl, Fmxr, momxl, momxr, momxl_s, momxr_s)
        Fmomy = _hllc_state(Fmyl, Fmyr, momyl, momyr, momyl_s, momyr_s)
        Fmomz = _hllc_state(Fmzl, Fmzr, momzl, momzr, momzl_s, momzr_s)
        Fetot = _hllc_state(FEl,  FEr,  El,    Er,    El_s,    Er_s   )

    else:
        raise ValueError(f"Unknown rHD flux_type '{flux_type}'. Choose 'LLF', 'HLL', or 'HLLC'.")

    # --- undo coordinate rotation for dim=2 ---
    if dim == 2:
        Fmomx, Fmomy = -Fmomy, Fmomx

    return Fmass, Fmomx, Fmomy, Fmomz, Fetot


# ============================================================================
#   Boundary conditions
# ============================================================================

def boundCond_rHD(grid, BC, fluid):
    """
    Apply boundary conditions to the rHD primitive variables.

    Identical in structure to boundCond_HD: scalars (dens, pres) use
    apply_bc_scalar; velocity components use apply_bc_vector.

    Parameters
    ----------
    grid  : Grid
    BC    : array of 4 str  –  [x1_inner, x2_inner, x1_outer, x2_outer]
    fluid : SimState

    Returns
    -------
    fluid : SimState  (modified in place)
    """
    Ngc = grid.Ngc

    # inner x1
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[0], axis=1, side='inner')
    fluid.pres = apply_bc_scalar(fluid.pres, Ngc, BC[0], axis=1, side='inner')
    fluid.vel1, fluid.vel2, fluid.vel3 = \
        apply_bc_vector(fluid.vel1, fluid.vel2, fluid.vel3, Ngc, BC[0], axis=1, side='inner')

    # inner x2
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[1], axis=2, side='inner')
    fluid.pres = apply_bc_scalar(fluid.pres, Ngc, BC[1], axis=2, side='inner')
    fluid.vel1, fluid.vel2, fluid.vel3 = \
        apply_bc_vector(fluid.vel1, fluid.vel2, fluid.vel3, Ngc, BC[1], axis=2, side='inner')

    # outer x1
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[2], axis=1, side='outer')
    fluid.pres = apply_bc_scalar(fluid.pres, Ngc, BC[2], axis=1, side='outer')
    fluid.vel1, fluid.vel2, fluid.vel3 = \
        apply_bc_vector(fluid.vel1, fluid.vel2, fluid.vel3, Ngc, BC[2], axis=1, side='outer')

    # outer x2
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[3], axis=2, side='outer')
    fluid.pres = apply_bc_scalar(fluid.pres, Ngc, BC[3], axis=2, side='outer')
    fluid.vel1, fluid.vel2, fluid.vel3 = \
        apply_bc_vector(fluid.vel1, fluid.vel2, fluid.vel3, Ngc, BC[3], axis=2, side='outer')

    return fluid
