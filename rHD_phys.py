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
from boundaries import apply_bc_scalar, apply_bc_vector


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
    enth = 1.0 + pres / (dens + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)

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
    enth = 1.0 + pres / (dens + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)

    vel1 = mom1 / (mass * enth * W)
    vel2 = mom2 / (mass * enth * W)
    vel3 = mom3 / (mass * enth * W)

    return dens, vel1, vel2, vel3, pres


# ============================================================================
#   Nonlinear pressure solver (Newton-Raphson)
# ============================================================================

def _pres_eqn_sr(pres, mass, mom1, mom2, mom3, etot, GAMMA):
    """
    Nonlinear equation f(p) = 0 whose root gives the SR pressure.

    f(p) = D·W(p) + Γ/(Γ−1)·p·W(p)² − E − p

    where W(p) = (E+p)/√((E+p)²−|m|²).

    Parameters and Returns: arrays of same shape as `pres`.
    """
    Ep   = etot + pres
    m2   = mom1**2 + mom2**2 + mom3**2
    W2   = Ep**2 / (Ep**2 - m2 + 1e-28)
    W    = np.sqrt(np.maximum(W2, 1.0))
    return mass * W + GAMMA / (GAMMA - 1.0) * pres * W2 - etot - pres


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
    dp_rel = 1.0e-12   # relative step for numerical derivative
    maxitr = 100

    pres = np.maximum(pres_init, 1.0e-14)

    res   = _pres_eqn_sr(pres, mass, mom1, mom2, mom3, etot, GAMMA)
    eps1  = np.max(np.abs(res))
    eps2  = 1.0

    for itr in range(maxitr):
        if eps1 <= tol and eps2 <= tol:
            break

        dp    = pres * (1.0 + dp_rel)
        f0    = _pres_eqn_sr(pres,      mass, mom1, mom2, mom3, etot, GAMMA)
        f1    = _pres_eqn_sr(pres + dp, mass, mom1, mom2, mom3, etot, GAMMA)
        deriv = (f1 - f0) / dp

        update = f0 / (deriv + 1.0e-28)
        pres   = pres - update
        pres   = np.maximum(pres, 1.0e-14)

        res  = _pres_eqn_sr(pres, mass, mom1, mom2, mom3, etot, GAMMA)
        eps1 = np.max(np.abs(res))
        eps2 = np.max(np.abs(update / (pres + 1.0e-28)))
    else:
        print(f"[rHD] pressure Newton solver: did not converge after {maxitr} iterations "
              f"(residual = {eps1:.3e})")

    return pres


# ============================================================================
#   SR sound speed
# ============================================================================

def sound_speed_sr(dens, pres, eos):
    """
    Adiabatic sound speed for an ideal relativistic gas.

        cs² = Γ p / (ρ h)

    where h = 1 + Γ p / (ρ (Γ−1)) is the specific enthalpy.

    Parameters
    ----------
    dens, pres : ndarray
    eos : EOSdata

    Returns
    -------
    cs : ndarray   –  sound speed  (0 < cs < 1)
    """
    enth = 1.0 + pres / (dens + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)
    cs2  = eos.GAMMA * pres / (dens * enth + 1e-14)
    return np.sqrt(np.clip(cs2, 0.0, 1.0 - 1e-14))


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

    # --- specific enthalpies ---
    entl = 1.0 + pl / (rhol + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)
    entr = 1.0 + pr / (rhor + 1e-14) * eos.GAMMA / (eos.GAMMA - 1.0)

    # --- Lorentz factors ---
    Wl = 1.0 / np.sqrt(1.0 - vxl**2 - vyl**2 - vzl**2)
    Wr = 1.0 / np.sqrt(1.0 - vxr**2 - vyr**2 - vzr**2)

    # --- left conservative state ---
    Dl   = rhol * Wl
    momxl = Dl * entl * Wl * vxl
    momyl = Dl * entl * Wl * vyl
    momzl = Dl * entl * Wl * vzl
    El   = Dl * entl * Wl - pl

    # --- right conservative state ---
    Dr   = rhor * Wr
    momxr = Dr * entr * Wr * vxr
    momyr = Dr * entr * Wr * vyr
    momzr = Dr * entr * Wr * vzr
    Er   = Dr * entr * Wr - pr

    # --- left physical fluxes ---
    FDl    = Dl   * vxl
    Fmxl   = momxl * vxl + pl
    Fmyl   = momxl * vyl
    Fmzl   = momxl * vzl
    FEl    = momxl

    # --- right physical fluxes ---
    FDr    = Dr   * vxr
    Fmxr   = momxr * vxr + pr
    Fmyr   = momxr * vyr
    Fmzr   = momxr * vzr
    FEr    = momxr

    # --- SR wave-speed estimates (Mignone & Bodo 2005, eqs. 9-10) ---
    cs2l = eos.GAMMA * pl / (rhol * entl + 1e-14)
    cs2r = eos.GAMMA * pr / (rhor * entr + 1e-14)

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
        Fmass = 0.5 * (FDl  + FDr  - lam * (Dr   - Dl  ))
        Fmomx = 0.5 * (Fmxl + Fmxr - lam * (momxr - momxl))
        Fmomy = 0.5 * (Fmyl + Fmyr - lam * (momyr - momyl))
        Fmomz = 0.5 * (Fmzl + Fmzr - lam * (momzr - momzl))
        Fetot = 0.5 * (FEl  + FEr  - lam * (Er   - El  ))

    elif flux_type == 'HLL':

        Sl_m = np.minimum(Sl, 0.0)
        Sr_p = np.maximum(Sr, 0.0)
        dS   = Sr_p - Sl_m + 1e-14

        Fmass = (Sr_p * FDl  - Sl_m * FDr  + Sr_p * Sl_m * (Dr   - Dl  )) / dS
        Fmomx = (Sr_p * Fmxl - Sl_m * Fmxr + Sr_p * Sl_m * (momxr - momxl)) / dS
        Fmomy = (Sr_p * Fmyl - Sl_m * Fmyr + Sr_p * Sl_m * (momyr - momyl)) / dS
        Fmomz = (Sr_p * Fmzl - Sl_m * Fmzr + Sr_p * Sl_m * (momzr - momzl)) / dS
        Fetot = (Sr_p * FEl  - Sl_m * FEr  + Sr_p * Sl_m * (Er   - El  )) / dS

    elif flux_type == 'HLLC':

        Sl_m = np.minimum(Sl, 0.0)
        Sr_p = np.maximum(Sr, 0.0)
        dS   = Sr_p - Sl_m + 1e-14

        # HLL intermediate state (needed for contact speed calculation)
        momx_hll = (Sr_p * momxr - Sl_m * momxl + Fmxl - Fmxr) / dS
        etot_hll  = (Sr_p * Er   - Sl_m * El   + FEl  - FEr ) / dS
        Fmx_hll  = (Sr_p * Fmxl - Sl_m * Fmxr + Sr_p * Sl_m * (momxr - momxl)) / dS
        FE_hll   = (Sr_p * FEl  - Sl_m * FEr  + Sr_p * Sl_m * (Er   - El  )) / dS

        # Contact wave speed (Mignone & Bodo 2005, eq. 18)
        disc  = np.maximum((etot_hll + Fmx_hll)**2 - 4.0 * momx_hll * FE_hll, 0.0)
        Sstar = ((etot_hll + Fmx_hll) - np.sqrt(disc)) / (2.0 * FE_hll + 1e-28)

        # Starred pressure (Mignone & Bodo 2005, eq. 17)
        Pstarl = (pl + Sl_m * Sstar * El - (Sstar + Sl_m - vxl) * momxl) / (1.0 - Sl_m * Sstar + 1e-28)
        Pstarr = (pr + Sr_p * Sstar * Er - (Sstar + Sr_p - vxr) * momxr) / (1.0 - Sr_p * Sstar + 1e-28)

        # Starred conservative states
        def _star(U, FU, S, Pstar):
            return (S * U - FU + Pstar * np.array([0.0])) / (S - Sstar + 1e-28)

        dSl = Sl_m - Sstar + 1e-28
        dSr = Sr_p - Sstar + 1e-28

        Dl_s    = Dl   * (Sl_m - vxl) / dSl
        momxl_s = (momxl * (Sl_m - vxl) + Pstarl - pl) / dSl
        momyl_s = momyl  * (Sl_m - vxl) / dSl
        momzl_s = momzl  * (Sl_m - vxl) / dSl
        El_s    = (El    * (Sl_m - vxl) + Pstarl * Sstar - pl * vxl) / dSl

        Dr_s    = Dr   * (Sr_p - vxr) / dSr
        momxr_s = (momxr * (Sr_p - vxr) + Pstarr - pr) / dSr
        momyr_s = momyr  * (Sr_p - vxr) / dSr
        momzr_s = momzr  * (Sr_p - vxr) / dSr
        Er_s    = (Er    * (Sr_p - vxr) + Pstarr * Sstar - pr * vxr) / dSr

        def _hllc_flux(FL, FR, UL, UR, ULs, URs):
            return np.where(
                Sl_m >= 0.0, FL,
                np.where((Sl_m < 0.0) & (Sstar >= 0.0), FL + Sl_m * (ULs - UL),
                np.where((Sstar < 0.0) & (Sr_p >= 0.0), FR + Sr_p * (URs - UR), FR)))

        Fmass = _hllc_flux(FDl,  FDr,  Dl,    Dr,    Dl_s,    Dr_s   )
        Fmomx = _hllc_flux(Fmxl, Fmxr, momxl, momxr, momxl_s, momxr_s)
        Fmomy = _hllc_flux(Fmyl, Fmyr, momyl, momyr, momyl_s, momyr_s)
        Fmomz = _hllc_flux(Fmzl, Fmzr, momzl, momzr, momzl_s, momzr_s)
        Fetot = _hllc_flux(FEl,  FEr,  El,    Er,    El_s,    Er_s   )

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
