# -*- coding: utf-8 -*-
"""
rHD_riemann_approx.py

Approximate Riemann solvers for special-relativistic hydrodynamics.

Implemented solvers, in increasing order of accuracy/cost:

    LLF   - Local Lax-Friedrichs / Rusanov (1961)
    HLL   - Harten, Lax, van Leer (1983)
    HLLC  - HLL with contact restoration (Mignone & Bodo 2005)
    
All solvers share the same calling convention (see Riemann_rHD in rHD_phys.py,
which dispatches to one of these by par.solver_type).

Each Riemann solver routine has the following i/o structure:

   Parameters
   ----------
   rhol, rhor : ndarray
       Left and right densities.
   vxl, vxr, vyl, vyr, vzl, vzr : ndarray
       Velocity components (x, y, z) for left and right states.
   pl, pr : ndarray
       Left and right pressures.
   eos : object
       Equation of state object

   Returns
   -------
   Fmass : ndarray
       Flux of mass density.
   Fmomx, Fmomy, Fmomz : ndarray
       Fluxes of momentum density in x, y, z.
   Fetot : ndarray
       Flux of total energy density.
   

Author: mrkondratyev
"""

import numpy as np



# -------------------------
# Small helper: conservative SR hydro variables + their fluxes along Ox 
# -------------------------
def cons_and_flux_rHD(rho, vx, vy, vz, p, eos):
    """
    Conservative variables and their Ox-normal fluxes for one SR state.

    Parameters
    ----------
    rho, vx, vy, vz, p : ndarray
        Primitive state (rest-mass density, 3-velocity components in
        units of c, pressure).
    eos : object
        Equation of state object.

    Returns
    -------
    m, mx, my, mz, e : ndarray
        Conservative variables (D = rho*W, momentum x/y/z, total energy).
    ent, W : ndarray
        Specific enthalpy and Lorentz factor (returned for reuse by the
        calling Riemann solver, e.g. for the wavespeed estimate).
    Fm, Fx, Fy, Fz, Fe : ndarray
        Their fluxes normal to the face (along the local x/Ox direction).
    """
    # --- specific enthalpy ---
    ent = eos.enthalpy_sr(rho, p)

    # --- Lorentz factor ---
    W = 1.0 / np.sqrt(1.0 - vx**2 - vy**2 - vz**2)
    
    # --- conservative state ---
    #number density 
    m   = rho * W
    #temporary variable (rho*h*W^2)
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


def LLF_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos):
    """
    Local Lax-Friedrichs (Rusanov) flux, special-relativistic.

    Parameters / Returns: see the module docstring above -- every solver
    in this file shares the same (rhol, rhor, vxl, vxr, ..., pl, pr, eos)
    signature and (Fmass, Fmomx, Fmomy, Fmomz, Fetot) return.
    """

    #left state and fluxes 
    Dl, momxl, momyl, momzl, El, \
    entl, Wl, \
    FDl, Fmxl, Fmyl, Fmzl, FEl  = \
        cons_and_flux_rHD(rhol, vxl, vyl, vzl, pl, eos)
        
    #right state and fluxes 
    Dr, momxr, momyr, momzr, Er, \
    entr, Wr, \
    FDr, Fmxr, Fmyr, Fmzr, FEr  = \
        cons_and_flux_rHD(rhor, vxr, vyr, vzr, pr, eos)
    
    # --- SR wave-speed estimates (Mignone & Bodo 2005, eqs. 9-10) ---
    cs2l = eos.sound_speed_sr(rhol, pl)**2
    cs2r = eos.sound_speed_sr(rhor, pr)**2

    sigl = cs2l / (Wl**2 * (1.0 - cs2l) + 1e-14)
    sigr = cs2r / (Wr**2 * (1.0 - cs2r) + 1e-14)

    bl_m = (vxl - np.sqrt(sigl * (1.0 - vxl**2 + sigl))) / (1.0 + sigl)
    bl_p = (vxl + np.sqrt(sigl * (1.0 - vxl**2 + sigl))) / (1.0 + sigl)
    br_m = (vxr - np.sqrt(sigr * (1.0 - vxr**2 + sigr))) / (1.0 + sigr)
    br_p = (vxr + np.sqrt(sigr * (1.0 - vxr**2 + sigr))) / (1.0 + sigr)

    Sl = np.minimum(bl_m, br_m); Sr = np.maximum(bl_p, br_p)

    #maximal eugenvalue in the system 
    lam  = np.maximum(np.abs(Sl), np.abs(Sr))
    
    Fmass = 0.5 * (FDl  + FDr  - lam * (Dr    - Dl   ))
    Fmomx = 0.5 * (Fmxl + Fmxr - lam * (momxr - momxl))
    Fmomy = 0.5 * (Fmyl + Fmyr - lam * (momyr - momyl))
    Fmomz = 0.5 * (Fmzl + Fmzr - lam * (momzr - momzl))
    Fetot = 0.5 * (FEl  + FEr  - lam * (Er    - El   ))
        
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot



def HLL_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos):
    """
    Harten, Lax, and Van Leer (HLL) flux, special-relativistic.

    Parameters / Returns: see the module docstring above -- every solver
    in this file shares the same (rhol, rhor, vxl, vxr, ..., pl, pr, eos)
    signature and (Fmass, Fmomx, Fmomy, Fmomz, Fetot) return.
    """

    #left state and fluxes 
    Dl, momxl, momyl, momzl, El, \
    entl, Wl, \
    FDl, Fmxl, Fmyl, Fmzl, FEl  = \
        cons_and_flux_rHD(rhol, vxl, vyl, vzl, pl, eos)
        
    #right state and fluxes 
    Dr, momxr, momyr, momzr, Er, \
    entr, Wr, \
    FDr, Fmxr, Fmyr, Fmzr, FEr  = \
        cons_and_flux_rHD(rhor, vxr, vyr, vzr, pr, eos)
    
    # --- SR wave-speed estimates (Mignone & Bodo 2005, eqs. 9-10) ---
    cs2l = eos.sound_speed_sr(rhol, pl)**2
    cs2r = eos.sound_speed_sr(rhor, pr)**2

    sigl = cs2l / (Wl**2 * (1.0 - cs2l) + 1e-14)
    sigr = cs2r / (Wr**2 * (1.0 - cs2r) + 1e-14)

    bl_m = (vxl - np.sqrt(sigl * (1.0 - vxl**2 + sigl))) / (1.0 + sigl)
    bl_p = (vxl + np.sqrt(sigl * (1.0 - vxl**2 + sigl))) / (1.0 + sigl)
    br_m = (vxr - np.sqrt(sigr * (1.0 - vxr**2 + sigr))) / (1.0 + sigr)
    br_p = (vxr + np.sqrt(sigr * (1.0 - vxr**2 + sigr))) / (1.0 + sigr)

    Sl = np.minimum(bl_m, br_m); Sr = np.maximum(bl_p, br_p)

    Sl = np.minimum(Sl, 0.0); Sr = np.maximum(Sr, 0.0)

    Fmass = (Sr * FDl  - Sl * FDr  + Sr * Sl * (Dr    - Dl   )) / (Sr - Sl)
    Fmomx = (Sr * Fmxl - Sl * Fmxr + Sr * Sl * (momxr - momxl)) / (Sr - Sl)
    Fmomy = (Sr * Fmyl - Sl * Fmyr + Sr * Sl * (momyr - momyl)) / (Sr - Sl)
    Fmomz = (Sr * Fmzl - Sl * Fmzr + Sr * Sl * (momzr - momzl)) / (Sr - Sl)
    Fetot = (Sr * FEl  - Sl * FEr  + Sr * Sl * (Er    - El   )) / (Sr - Sl)
    
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot
    


def HLLC_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos):
    """
    Harten, Lax, and Van Leer + Contact wave (HLLC) flux, special-
    relativistic (Mignone & Bodo 2005).

    Parameters / Returns: see the module docstring above -- every solver
    in this file shares the same (rhol, rhor, vxl, vxr, ..., pl, pr, eos)
    signature and (Fmass, Fmomx, Fmomy, Fmomz, Fetot) return.
    """

    #left state and fluxes 
    Dl, momxl, momyl, momzl, El, \
    entl, Wl, \
    FDl, Fmxl, Fmyl, Fmzl, FEl  = \
        cons_and_flux_rHD(rhol, vxl, vyl, vzl, pl, eos)
        
    #right state and fluxes 
    Dr, momxr, momyr, momzr, Er, \
    entr, Wr, \
    FDr, Fmxr, Fmyr, Fmzr, FEr  = \
        cons_and_flux_rHD(rhor, vxr, vyr, vzr, pr, eos)
    
    # --- SR wave-speed estimates (Mignone & Bodo 2005, eqs. 9-10) ---
    cs2l = eos.sound_speed_sr(rhol, pl)**2
    cs2r = eos.sound_speed_sr(rhor, pr)**2

    sigl = cs2l / (Wl**2 * (1.0 - cs2l) + 1e-14)
    sigr = cs2r / (Wr**2 * (1.0 - cs2r) + 1e-14)

    bl_m = (vxl - np.sqrt(sigl * (1.0 - vxl**2 + sigl))) / (1.0 + sigl)
    bl_p = (vxl + np.sqrt(sigl * (1.0 - vxl**2 + sigl))) / (1.0 + sigl)
    br_m = (vxr - np.sqrt(sigr * (1.0 - vxr**2 + sigr))) / (1.0 + sigr)
    br_p = (vxr + np.sqrt(sigr * (1.0 - vxr**2 + sigr))) / (1.0 + sigr)

    Sl = np.minimum(bl_m, br_m); Sr = np.maximum(bl_p, br_p)

    Sl = np.minimum(Sl, 0.0); Sr = np.maximum(Sr, 0.0)

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

    return Fmass, Fmomx, Fmomy, Fmomz, Fetot
