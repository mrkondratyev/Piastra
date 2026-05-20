# -*- coding: utf-8 -*-
"""
riemann_approx.py

Approximate Riemann solvers for non-relativistic hydrodynamics.

Implemented solvers, in increasing order of accuracy/cost:

    LLF   - Local Lax-Friedrichs / Rusanov (1961)
    HLL   - Harten, Lax, van Leer (1983)
    HLLC  - HLL with contact restoration (Toro et al. 1994)
    Roe   - Linearized Roe solver (Roe 1981)

All solvers share the same calling convention (see Riemann_nr_hydro in 
                                               hydro_phys file).

Each Riemann solver routine has the following i/o sturcture:

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
# Small helper: conservative hydro variables + their fluxes along Ox 
# -------------------------
def nr_hydro_cons_and_flux(rho, vx, vy, vz, p, eos):
    
    #conservative variables 
    #mass
    m = rho
    #momentum
    mx = rho * vx; my = rho * vy; mz = rho * vz
    #total energy
    e = eos.eint(rho, p) + rho*(vx**2 + vy**2 + vz**2) / 2.0
    
    #fluxes 
    #mass
    Fm = mx
    #momentum
    Fx = mx * vx + p; Fy = my * vx; Fz = mz * vx
    #total enery 
    Fe = vx * (p + e)
    
    #output -- conservative state + fluxes 
    return m, mx, my, mz, e, \
        Fm, Fx, Fy, Fz, Fe

    

"""
Local Lax-Friedrichs (Rusanov) flux
"""
def LLF_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos):
    
    #left state conservatives and fluxes 
    mass_L, momx_L, momy_L, momz_L, etot_L, \
    Fmass_L, Fmomx_L, Fmomy_L, Fmomz_L, Fetot_L = \
        nr_hydro_cons_and_flux(rhol, vxl, vyl, vzl, pl, eos)
        
    #right state conservatives and fluxes 
    mass_R, momx_R, momy_R, momz_R, etot_R, \
    Fmass_R, Fmomx_R, Fmomy_R, Fmomz_R, Fetot_R = \
        nr_hydro_cons_and_flux(rhor, vxr, vyr, vzr, pr, eos)

    #maximal absolute value of eigenvalues  
    Sr = np.maximum(eos.sound_speed_nr(rhol, pl) + np.abs(vxl), \
                    eos.sound_speed_nr(rhor, pr) + np.abs(vxr))
    
    #calculation of the flux (dissipation ~ to max wavespeed added)
    Fmass = ( Fmass_L + Fmass_R ) / 2.0 - Sr * (mass_R - mass_L) / 2.0
    Fmomx = ( Fmomx_L + Fmomx_R ) / 2.0 - Sr * (momx_R - momx_L) / 2.0
    Fmomy = ( Fmomy_L + Fmomy_R ) / 2.0 - Sr * (momy_R - momy_L) / 2.0
    Fmomz = ( Fmomz_L + Fmomz_R ) / 2.0 - Sr * (momz_R - momz_L) / 2.0
    Fetot = ( Fetot_L + Fetot_R ) / 2.0 - Sr * (etot_R - etot_L) / 2.0
    
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot



"""
Harten, Lax, and Van Leer (HLL) flux
"""
def HLL_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos):
    
    #left state conservatives and fluxes 
    mass_L, momx_L, momy_L, momz_L, etot_L, \
    Fmass_L, Fmomx_L, Fmomy_L, Fmomz_L, Fetot_L = \
        nr_hydro_cons_and_flux(rhol, vxl, vyl, vzl, pl, eos)
        
    #right state conservatives and fluxes 
    mass_R, momx_R, momy_R, momz_R, etot_R, \
    Fmass_R, Fmomx_R, Fmomy_R, Fmomz_R, Fetot_R = \
        nr_hydro_cons_and_flux(rhor, vxr, vyr, vzr, pr, eos)
    
    #left and right sound speeds 
    csl = eos.sound_speed_nr(rhol, pl)
    csr = eos.sound_speed_nr(rhor, pr)
    
    #one-line form of maximal and minimal eigenvalues HLL estimate according to Davis (1988)
    Sl = np.minimum(np.minimum(vxl, vxr) - np.maximum(csl, csr), 0.0)
    Sr = np.maximum(np.maximum(vxl, vxr) + np.maximum(csl, csr), 0.0)
    
    #calculation of the flux using HLL approximate Riemann fan (3 states between two shocks)
    Fmass = ( Sr * Fmass_L - Sl * Fmass_R + Sr * Sl * (mass_R - mass_L) ) / (Sr - Sl)
    Fmomx = ( Sr * Fmomx_L - Sl * Fmomx_R + Sr * Sl * (momx_R - momx_L) ) / (Sr - Sl)
    Fmomy = ( Sr * Fmomy_L - Sl * Fmomy_R + Sr * Sl * (momy_R - momy_L) ) / (Sr - Sl)
    Fmomz = ( Sr * Fmomz_L - Sl * Fmomz_R + Sr * Sl * (momz_R - momz_L) ) / (Sr - Sl)
    Fetot = ( Sr * Fetot_L - Sl * Fetot_R + Sr * Sl * (etot_R - etot_L) ) / (Sr - Sl)
    
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot



"""
Harten, Lax, and Van Leer + Contact wave (HLLC) flux
"""
def HLLC_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos):
    
    #left state conservatives and fluxes 
    mass_L, momx_L, momy_L, momz_L, etot_L, \
    Fmass_L, Fmomx_L, Fmomy_L, Fmomz_L, Fetot_L = \
        nr_hydro_cons_and_flux(rhol, vxl, vyl, vzl, pl, eos)
        
    #right state conservatives and fluxes 
    mass_R, momx_R, momy_R, momz_R, etot_R, \
    Fmass_R, Fmomx_R, Fmomy_R, Fmomz_R, Fetot_R = \
        nr_hydro_cons_and_flux(rhor, vxr, vyr, vzr, pr, eos)
    
    #left and right sound speeds 
    csl = eos.sound_speed_nr(rhol, pl)
    csr = eos.sound_speed_nr(rhor, pr)
    
    #one-line form of maximal and minimal eigenvalues HLL estimate according to Davis (1988)
    Sl = np.minimum(np.minimum(vxl, vxr) - np.maximum(csl, csr), 0.0)
    Sr = np.maximum(np.maximum(vxl, vxr) + np.maximum(csl, csr), 0.0)
    
    #contact wave speed in HLLC approximation
    Sstar = (pr - pl + rhol * vxl * (Sl - vxl) - 
        rhor * vxr * (Sr - vxr)) / (rhol * (Sl - vxl) - rhor * (Sr - vxr))
    
    #conservative fluid state in the regions in both sides from the contact discontinuity
    #left starred state
    massS_L = rhol * (Sl - vxl) / (Sl - Sstar) 
    momxS_L = massS_L * Sstar;    momyS_L = massS_L * vyl;    momzS_L = massS_L * vzl 
    etotS_L = massS_L * ( etot_L / rhol + (Sstar - vxl) * (Sstar + pl / rhol / (Sl - vxl)) ) 
    
    #right starred state
    massS_R = rhor * (Sr - vxr) / (Sr - Sstar)
    momxS_R = massS_R * Sstar ;    momyS_R = massS_R * vyr;    momzS_R = massS_R * vzr 
    etotS_R = massS_R * ( etot_R / rhor + (Sstar - vxr) * (Sstar + pr / rhor / (Sr - vxr)) ) 
    
    # helper -- calculation of the flux using HLLC approximate Riemann fan 
    # 4 states between left shock, contact wave, and right shock
    def _hllc_state(FL, FR, UL, UR, ULs, URs):
        return np.where(
            Sl >= 0.0, FL,
            np.where((Sl < 0.0) & (Sstar >= 0.0), FL + Sl * (ULs - UL),
            np.where((Sstar < 0.0) & (Sr >= 0.0), FR + Sr * (URs - UR), 
            FR)))
    
    #final fluxes 
    Fmass = _hllc_state(Fmass_L, Fmass_R, mass_L, mass_R, massS_L, massS_R)
    Fmomx = _hllc_state(Fmomx_L, Fmomx_R, momx_L, momx_R, momxS_L, momxS_R)
    Fmomy = _hllc_state(Fmomy_L, Fmomy_R, momy_L, momy_R, momyS_L, momyS_R)
    Fmomz = _hllc_state(Fmomz_L, Fmomz_R, momz_L, momz_R, momzS_L, momzS_R)
    Fetot = _hllc_state(Fetot_L, Fetot_R, etot_L, etot_R, etotS_L, etotS_R)
    
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot



"""
Linearized acoustic system (Roe) flux

Note: Roe solver works only with ideal gamma-law EOS! 
"""
def Roe_flux(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos):

    #left state conservatives and fluxes 
    mass_L, momx_L, momy_L, momz_L, etot_L, \
    Fmass_L, Fmomx_L, Fmomy_L, Fmomz_L, Fetot_L = \
        nr_hydro_cons_and_flux(rhol, vxl, vyl, vzl, pl, eos)
        
    #right state conservatives and fluxes 
    mass_R, momx_R, momy_R, momz_R, etot_R, \
    Fmass_R, Fmomx_R, Fmomy_R, Fmomz_R, Fetot_R = \
        nr_hydro_cons_and_flux(rhor, vxr, vyr, vzr, pr, eos)    

    #left and rigth enthalpies
    entl = (eos.eint(rhol, pl) + pl)/rhol + (vxl**2 + vyl**2 + vzl**2)/2.0 
    entr = (eos.eint(rhor, pr) + pr)/rhor + (vxr**2 + vyr**2 + vzr**2)/2.0  
    
    #left and right sound speeds 
    csl = eos.sound_speed_nr(rhol, pl)
    csr = eos.sound_speed_nr(rhor, pr)
    
    #Roe-averaged density
    rhos = np.sqrt(rhol*rhor)
    
    #square roots of L/R densities 
    sqrt_rhol = np.sqrt(rhol)
    sqrt_rhor = np.sqrt(rhor)
    
    #Roe-averaged velocity
    vxs = (sqrt_rhol*vxl + sqrt_rhor*vxr)/(sqrt_rhol + sqrt_rhor)
    vys = (sqrt_rhol*vyl + sqrt_rhor*vyr)/(sqrt_rhol + sqrt_rhor)
    vzs = (sqrt_rhol*vzl + sqrt_rhor*vzr)/(sqrt_rhol + sqrt_rhor)
    
    #Roe-averaged enthalpy
    ents = (sqrt_rhol*entl + sqrt_rhor*entr)/(sqrt_rhol + sqrt_rhor)
    
    #Roe-averaged sound speed 
    css = np.sqrt( (eos.GAMMA - 1.0)*(ents - (vxs**2 + vys**2 + vzs**2)/2.0) )
    
    # Alternatively, we can use the following prescription for css, 
    # which does not use substraction of (possibly) two large numbers
    #css = np.sqrt( (sqrt_rhol*csl**2 + \
        #sqrt_rhor*csr**2)/(sqrt_rhol + sqrt_rhor) + \
        #(eos.GAMMA - 1.0)*rhos/(sqrt_rhol + sqrt_rhor)**2 * \
        #((vxr-vxl)**2 + (vyr-vyl)**2 + (vzr-vzl)**2)/2.0 ) 
    
    #arrays of right eugenvectors
    rv = np.zeros((5, 5, *rhos.shape))
    
    #v - cs 
    rv[0,0,:,:] = np.ones_like(rhos)
    rv[0,1,:,:] = vxs - css
    rv[0,2,:,:] = vys
    rv[0,3,:,:] = vzs
    rv[0,4,:,:] = ents - vxs*css
    
    # v (1)
    rv[1,0,:,:] = 2.0*np.ones_like(rhos)
    rv[1,1,:,:] = 2.0*vxs
    rv[1,2,:,:] = 2.0*vys
    rv[1,3,:,:] = 2.0*vzs
    rv[1,4,:,:] = vxs**2 + vys**2 + vzs**2
    
    # v (2)
    rv[2,0,:,:] = np.zeros_like(rhos)
    rv[2,1,:,:] = np.zeros_like(rhos)
    rv[2,2,:,:] = 2.0*css
    rv[2,3,:,:] = np.zeros_like(rhos)
    rv[2,4,:,:] = 2.0*css*vys
    
    # v (3)
    rv[3,0,:,:] = np.zeros_like(rhos)
    rv[3,1,:,:] = np.zeros_like(rhos)
    rv[3,2,:,:] = np.zeros_like(rhos)
    rv[3,3,:,:] = 2.0*css
    rv[3,4,:,:] = 2.0*css*vzs
    
    # v + cs 
    rv[4,0,:,:] = np.ones_like(rhos)
    rv[4,1,:,:] = vxs + css
    rv[4,2,:,:] = vys
    rv[4,3,:,:] = vzs
    rv[4,4,:,:] = ents + vxs*css
    
    #array of absolute value of eugenvalues
    eugen = np.zeros((5, *rhos.shape))
    eugen[0,:,:] = np.abs(np.minimum(vxs - css, vxl - csl)) # entropy fix
    eugen[1,:,:] = np.abs(vxs); eugen[2,:,:] = np.abs(vxs); eugen[3,:,:] = np.abs(vxs) 
    eugen[4,:,:] = np.abs(np.maximum(vxs + css, vxr + csr)) # entropy fix 
    
    #array of left eugenvectors residuals
    dS = np.zeros((5, *rhos.shape))
    dS[0,:,:] = ( (pr - pl) - rhos*css*(vxr - vxl) )/css**2/2.0
    dS[1,:,:] = ( css**2*(rhor - rhol) - (pr - pl) )/css**2/2.0
    dS[2,:,:] = rhos*(vyr - vyl)/css/2.0
    dS[3,:,:] = rhos*(vzr - vzl)/css/2.0
    dS[4,:,:] = ( (pr - pl) + rhos*css*(vxr - vxl) )/css**2/2.0
    
    #array of flux residuals
    dF = np.zeros((5, *rhos.shape))
    
    # calculation of dF (basically, it is our numerical diffusion, that stabilizes the method)
    dF = np.sum(eugen[:, np.newaxis, :, :] * rv[:, :, :, :] * dS[:, np.newaxis, :, :], axis=0)

    #final values of conservative fluxes, obtained from linearized Riemann problem solution
    Fmass = (Fmass_L + Fmass_R)/2.0 - dF[0,:,:]/2.0
    Fmomx = (Fmomx_L + Fmomx_R)/2.0 - dF[1,:,:]/2.0
    Fmomy = (Fmomy_L + Fmomy_R)/2.0 - dF[2,:,:]/2.0
    Fmomz = (Fmomz_L + Fmomz_R)/2.0 - dF[3,:,:]/2.0
    Fetot = (Fetot_L + Fetot_R)/2.0 - dF[4,:,:]/2.0    
    
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot