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
    apply_bc_vector,
    apply_bc_fixed)
from src.models.rHD.rHD_riemann_approx import (
    LLF_flux,
    HLL_flux,
    HLLC_flux)

# ============================================================================
#   Primitive -> conservative conversion
# ============================================================================
def prim2cons_rHD(dens, vel1, vel2, vel3, pres, eos):
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


# ============================================================================
#   Conservative -> primitive inversion
# ============================================================================
def cons2prim_rHD(mass, mom1, mom2, mom3, etot, pres_init, eos):
    """
    Recover (rho, v1, v2, v3, p) from (D, m1, m2, m3, E) for an SR ideal gas.

    Ordering matters and is deliberate:
      1. solve for p (Newton stays in the physical domain, so W is real);
      2. build W and FLOOR the density BEFORE recovering velocity, so a
         floored cell does not feed a tiny pre-floor `mass` into the velocity;
      3. recover v_i = m_i / (E + p), since  E + p = rho h W^2  is exactly the
         positive quantity the Newton solve already pinned down -- this avoids
         recomputing the enthalpy and avoids a separate divide that could blow up;
      4. floor pressure;
      5. clip any residual super-luminal velocity, but count it
    """
    
    #floor variables for density, pressure, and Lorenz factor 
    dens_floor=1.0e-11; pres_floor=1.0e-11; W_ceiling=1.0e4
    
    # --- pressure Newton solver 
    pres = _newton_pres_sr(pres_init, mass, mom1, mom2, mom3, etot, eos.GAMMA,
                           p_floor=pres_floor)

    # --- Lorentz factor and density
    m2   = mom1**2 + mom2**2 + mom3**2
    Ep   = etot + pres
    W    = Ep / np.sqrt(Ep**2 - m2)
    dens = mass / W
    dens = np.maximum(dens, dens_floor)

    # --- velocity
    inv_Ep = 1.0 / Ep                       
    vel1 = mom1 * inv_Ep
    vel2 = mom2 * inv_Ep
    vel3 = mom3 * inv_Ep

    # --- pressure floor ---
    pres = np.maximum(pres, pres_floor)

    # --- Super-luminal safeguard ---
    # here we clip the maximal Lorenz factor with W_ceiling for problematic cells
    v2 = vel1**2 + vel2**2 + vel3**2
    too_fast = v2 >= 1.0
    n_clip = int(np.count_nonzero(too_fast))
    if n_clip > 0:
        v_max = np.sqrt(1.0 - 1.0 / W_ceiling**2) # |v| for W = W_ceiling
        fac = np.where(too_fast, v_max / np.sqrt(v2 + 1.0e-30), 1.0)
        vel1 *= fac; vel2 *= fac; vel3 *= fac
        print(f"[rHD] cons2prim: clipped {n_clip} super-luminal cell(s) "
              f"to W = {W_ceiling:.1f}")

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
    Ep = etot + pres;  Epm2 = Ep**2 - m2;  W2 = Ep**2 / Epm2
    func = mass * np.sqrt(np.maximum(W2, 1.0)) + GAMMA / (GAMMA - 1.0) * pres * W2 - Ep
    
    return func



def _pres_der_sr(pres, mass, m2, etot, GAMMA):
    """
    Derivative  df(p)/dp for root finding procedure in SR hydrodynamics.

    f(p) = D·W(p) + Γ/(Γ−1)·p·W(p)² − E − p

    where W(p) = (E+p)/√((E+p)²−|m|²).

    Parameters and Returns: arrays of same shape as `pres`.
    """
    Ep   = etot + pres;  Epm2 = Ep**2 - m2
    der = -mass * m2 / Epm2**1.5 + \
        GAMMA / (GAMMA - 1.0) * Ep * (Ep**3 - m2*(Ep + 2.0*pres)) / Epm2**2 - 1.0
    
    return der



def _newton_pres_sr(pres_init, mass, mom1, mom2, mom3, etot, GAMMA,
                    p_floor=1.0e-12):
    """
    Solve f(p)=0 per cell with a domain-safeguarded, per-cell Newton iteration.

    Robustness features vs. a plain vectorised Newton:/
      * the iterate is held strictly inside the physical domain p > |m| - E,
        so W(p) can never go imaginary and the derivative can never NaN;
      * any Newton step that would leave the domain (or is non-finite) is
        replaced by a backtracking step toward the domain boundary;
      * convergence is tracked PER CELL via an active mask, so one pathological
        cell cannot hold the whole grid hostage, and a single bad cell cannot
        disguise itself as "nothing converged" through a global max-norm.
    """
    tol    = 1.0e-8
    maxitr = 50
    # eps_W bounds the *in-iteration* Lorentz factor to ~1/sqrt(2*eps_W) ~ 7e4,
    # far above any physical flow, while keeping (E+p)^2 - m2 safely positive.
    eps_W  = 1.0e-10

    m2 = mom1**2 + mom2**2 + mom3**2

    # Lower edge of the valid domain, per cell
    p_lim = np.maximum(p_floor, (1.0 + eps_W) * np.sqrt(m2) - etot)

    # Start from the previous-step pressure, projected into the valid domain.
    pres   = np.maximum(pres_init, p_lim)
    active = np.ones_like(pres, dtype=bool) # cells still being iterated

    for itr in range(maxitr):
        if not active.any():
            break

        f0    = _pres_eqn_sr(pres, mass, m2, etot, GAMMA)
        deriv = _pres_der_sr(pres, mass, m2, etot, GAMMA)

        # Guard only against a divide-by-zero
        deriv = np.where(np.abs(deriv) > 1.0e-30, deriv, -1.0e-30)

        p_new = pres - f0 / deriv

        # If the full Newton step leaves the valid region or
        # is non-finite, bisect back toward the boundary instead of crossing it
        bad = ~np.isfinite(p_new) | (p_new <= p_lim)
        p_new = np.where(bad, 0.5 * (pres + p_lim), p_new)

        # Move only the still-active cells; freeze the rest
        p_prev = pres
        pres = np.where(active, p_new, pres)

        # Per-cell convergence: small residual AND small relative change
        res = _pres_eqn_sr(pres, mass, m2, etot, GAMMA)
        rel = np.abs(pres - p_prev) / (np.abs(pres) + 1.0e-30)
        converged = (np.abs(res) <= tol) & (rel <= tol)
        active = active & ~converged

    # Report the cells that failed to converge 
    n_bad = int(np.count_nonzero(active))
    if n_bad > 0:
        worst = float(np.nanmax(np.abs(_pres_eqn_sr(pres, mass, m2, etot, GAMMA))[active]))
        print(f"[rHD] pressure Newton: {n_bad} cell(s) unconverged after "
              f"{maxitr} iters (max residual = {worst:.3e})")

    return pres



# ============================================================================
#   Approximate SR Riemann solvers
# ============================================================================

def Riemann_rHD(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos, solver_type, dim):
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
    solver_type: str        –  'LLF', 'HLL', or 'HLLC'
    dim        : int        –  1 (normal) or 2 (tangential → rotate)

    Returns
    -------
    Fmass, Fmomx, Fmomy, Fmomz, Fetot : ndarray
        Interface fluxes for D, m₁, m₂, m₃, E.
    """
    # --- coordinate rotation for dim=2 ---
    if dim == 2:
        vxl, vxr, vyl, vyr = vyl, vyr, -vxl, -vxr

    #here we calculate the flux using various Riemann solvers
    if solver_type == 'LLF':
        
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            LLF_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos)
        
    elif solver_type == 'HLL':  
        
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            HLL_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos)
               
    elif solver_type == 'HLLC':
        
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            HLLC_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos)

    else:
        
        #solver_type is incorrect -> throw an error
        raise ValueError(
            f"Unknown rHD solver_type '{solver_type}'. " 
            f"Expected one of ['LLF', 'HLL', 'HLLC'].")

    # --- undo coordinate rotation for dim=2 ---
    if dim == 2:
        Fmomx, Fmomy = -Fmomy, Fmomx

    return Fmass, Fmomx, Fmomy, Fmomz, Fetot



# ============================================================================
#   Boundary conditions
# ============================================================================
def boundCond_rHD(grid, BC, fluid, BC_fixed=None):
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

    # --- fixed (Dirichlet) ghost-fill, applied LAST ---
    if BC_fixed is not None:
        N1, N2 = fluid.dens.shape
        sf = {'dens': fluid.dens, 'pres': fluid.pres,
              'vel1': fluid.vel1, 'vel2': fluid.vel2, 'vel3': fluid.vel3}
        for face in (0, 1, 2, 3):
            if BC_fixed.get(face):
                apply_bc_fixed(sf, Ngc, N1, N2, face, BC_fixed[face])
        
    return fluid
