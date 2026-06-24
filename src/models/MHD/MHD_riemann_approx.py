# -*- coding: utf-8 -*-
"""
MHD_riemann_approx.py

Approximate Riemann solvers for non-relativistic MHD.

Implemented solvers, in increasing order of accuracy/cost:

    LLF   - Local Lax-Friedrichs / Rusanov (1961)
    HLL   - Harten, Lax, van Leer (1983)
    HLLC  - HLL with contact restoration (Li 2005)
    HLLD  - HLL with contact and Alfven restoration (Miyoshi and Kusano 2005)

All solvers share the same calling convention (see Riemann_MHD in 
                                               MHD_phys file).

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


# -------------------------
# Small helper: conservative MHD variables + their fluxes along Ox
# -------------------------
def nr_MHD_cons_and_flux(rho, vx, vy, vz, p, bx, by, bz, eos):
    
    '''conservative variables'''
    #mass
    m = rho
    #momentum
    mx = rho * vx; my = rho * vy; mz = rho * vz
    #squared B-field
    b2 = bx**2 + by**2 + bz**2
    #total pressure 
    ptot = p + b2 / 2.0
    #total energy
    e = eos.eint(rho, p) + rho*(vx**2 + vy**2 + vz**2) / 2.0 + b2 / 2.0
    #magnetic field (only normal components, since Fbx = 0)
    By = by; Bz = bz
    
    '''fluxes''' 
    #mass
    Fm = mx
    #momentum
    Fmx = mx * vx + ptot - bx**2; Fmy = my * vx - by * bx; Fmz = mz * vx - bz * bx
    #total enery 
    Fe = vx * (ptot + e) - bx * (bx * vx + by * vy + bz * vz)
    #magnetic field (only normal components, since Fbx = 0)
    Fby = vx * by - vy * bx; Fbz = vx * bz - vz * bx
    
    #output -- conservative state + fluxes + ptot 
    return m, mx, my, mz, e, By, Bz, ptot, b2, \
        Fm, Fmx, Fmy, Fmz, Fe, Fby, Fbz


    

"""
Local Lax-Friedrichs (Rusanov) flux
"""
def LLF_flux(rhol,rhor, vxl,vxr, vyl,vyr, vzl,vzr, pl,pr, bxl,bxr, byl,byr, bzl,bzr, eos):
    
    #normal B-field and total pressures on the left and on the right side
    bxn = (bxl + bxr)/2.0  
    
    #left state conservatives and fluxes 
    mass_L, momx_L, momy_L, momz_L, etot_L, bfiy_L, bfiz_L, ptot_L, b2l, \
    Fmass_L, Fmomx_L, Fmomy_L, Fmomz_L, Fetot_L, Fbfiy_L, Fbfiz_L = \
            nr_MHD_cons_and_flux(rhol, vxl, vyl, vzl, pl, bxn, byl, bzl, eos)

    #right state conservatives and fluxes 
    mass_R, momx_R, momy_R, momz_R, etot_R, bfiy_R, bfiz_R, ptot_R, b2r, \
    Fmass_R, Fmomx_R, Fmomy_R, Fmomz_R, Fetot_R, Fbfiy_R, Fbfiz_R = \
            nr_MHD_cons_and_flux(rhor, vxr, vyr, vzr, pr, bxn, byr, bzr, eos)

    #left and right squared sound speeds 
    csl2 = eos.sound_speed_nr(rhol, pl)**2
    csr2 = eos.sound_speed_nr(rhor, pr)**2
    
    #left and right fast magnetosonic speeds 
    cfl = np.sqrt( (csl2 + b2l/rhol)/2.0 + np.sqrt((csl2 + b2l/rhol)**2 - 4.0*csl2*bxn**2/rhol)/2.0 )
    cfr = np.sqrt( (csr2 + b2r/rhor)/2.0 + np.sqrt((csr2 + b2r/rhor)**2 - 4.0*csr2*bxn**2/rhor)/2.0 )
    
    #maximal absolute value of eigenvalues  
    Sr = np.maximum(cfl + np.abs(vxl), cfr + np.abs(vxr))
    
    #calculation of the flux (dissipation ~ max wavespeed added)
    Fmass = ( Fmass_L + Fmass_R ) / 2.0 - Sr * (mass_R - mass_L) / 2.0
    Fmomx = ( Fmomx_L + Fmomx_R ) / 2.0 - Sr * (momx_R - momx_L) / 2.0
    Fmomy = ( Fmomy_L + Fmomy_R ) / 2.0 - Sr * (momy_R - momy_L) / 2.0
    Fmomz = ( Fmomz_L + Fmomz_R ) / 2.0 - Sr * (momz_R - momz_L) / 2.0
    Fetot = ( Fetot_L + Fetot_R ) / 2.0 - Sr * (etot_R - etot_L) / 2.0
    Fbfix = np.zeros_like(Fmass)
    Fbfiy = ( Fbfiy_L + Fbfiy_R ) / 2.0 - Sr * (bfiy_R - bfiy_L) / 2.0
    Fbfiz = ( Fbfiz_L + Fbfiz_R ) / 2.0 - Sr * (bfiz_R - bfiz_L) / 2.0
        
    #return approximate Riemann flux for MHD 
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot, Fbfix, Fbfiy, Fbfiz



"""
Harten, Lax, and Van Leer (HLL) flux
"""
def HLL_flux(rhol,rhor, vxl,vxr, vyl,vyr, vzl,vzr, pl,pr, bxl,bxr, byl,byr, bzl,bzr, eos):
    
    #normal B-field and total pressures on the left and on the right side
    bxn = (bxl + bxr)/2.0  
    
    #left state conservatives and fluxes 
    mass_L, momx_L, momy_L, momz_L, etot_L, bfiy_L, bfiz_L, ptot_L, b2l, \
    Fmass_L, Fmomx_L, Fmomy_L, Fmomz_L, Fetot_L, Fbfiy_L, Fbfiz_L = \
            nr_MHD_cons_and_flux(rhol, vxl, vyl, vzl, pl, bxn, byl, bzl, eos)

    #right state conservatives and fluxes 
    mass_R, momx_R, momy_R, momz_R, etot_R, bfiy_R, bfiz_R, ptot_R, b2r, \
    Fmass_R, Fmomx_R, Fmomy_R, Fmomz_R, Fetot_R, Fbfiy_R, Fbfiz_R = \
            nr_MHD_cons_and_flux(rhor, vxr, vyr, vzr, pr, bxn, byr, bzr, eos)

    #left and right squared sound speeds 
    csl2 = eos.sound_speed_nr(rhol, pl)**2
    csr2 = eos.sound_speed_nr(rhor, pr)**2
    
    #left and right fast magnetosonic speeds 
    cfl = np.sqrt( (csl2 + b2l/rhol)/2.0 + np.sqrt((csl2 + b2l/rhol)**2 - 4.0*csl2*bxn**2/rhol)/2.0 )
    cfr = np.sqrt( (csr2 + b2r/rhor)/2.0 + np.sqrt((csr2 + b2r/rhor)**2 - 4.0*csr2*bxn**2/rhor)/2.0 )
    
    #one-line form of maximal and minimal eigenvalues HLL estimate according to Davis (1988)
    Sl = np.minimum(np.minimum(vxl, vxr) - np.maximum(cfl, cfr), 0.0)
    Sr = np.maximum(np.maximum(vxl, vxr) + np.maximum(cfl, cfr), 0.0)
    
    #calculation of the flux using HLL approximate Riemann fan (3 states between two shocks)
    Fmass = ( Sr * Fmass_L - Sl * Fmass_R + Sr * Sl * (mass_R - mass_L) ) / (Sr - Sl)
    Fmomx = ( Sr * Fmomx_L - Sl * Fmomx_R + Sr * Sl * (momx_R - momx_L) ) / (Sr - Sl)
    Fmomy = ( Sr * Fmomy_L - Sl * Fmomy_R + Sr * Sl * (momy_R - momy_L) ) / (Sr - Sl)
    Fmomz = ( Sr * Fmomz_L - Sl * Fmomz_R + Sr * Sl * (momz_R - momz_L) ) / (Sr - Sl)
    Fetot = ( Sr * Fetot_L - Sl * Fetot_R + Sr * Sl * (etot_R - etot_L) ) / (Sr - Sl)
    Fbfix = np.zeros_like(Fmass)
    Fbfiy = ( Sr * Fbfiy_L - Sl * Fbfiy_R + Sr * Sl * (bfiy_R - bfiy_L) ) / (Sr - Sl)
    Fbfiz = ( Sr * Fbfiz_L - Sl * Fbfiz_R + Sr * Sl * (bfiz_R - bfiz_L) ) / (Sr - Sl)
    
    #return approximate Riemann flux for MHD 
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot, Fbfix, Fbfiy, Fbfiz


"""
Harten, Lax, and Van Leer + Contact wave (HLLC) flux for Newtonian MHD,
following Li (2005). Single contact between two fast waves; the transverse
magnetic field in the star region is the HLL average (no Alfven separation).
"""
def HLLC_flux(rhol,rhor, vxl,vxr, vyl,vyr, vzl,vzr, pl,pr, bxl,bxr, byl,byr, bzl,bzr, eos):

    #normal B-field (continuous across the fan)
    bxn = (bxl + bxr)/2.0

    #left state conservatives and fluxes
    mass_L, momx_L, momy_L, momz_L, etot_L, bfiy_L, bfiz_L, ptot_L, b2l, \
    Fmass_L, Fmomx_L, Fmomy_L, Fmomz_L, Fetot_L, Fbfiy_L, Fbfiz_L = \
            nr_MHD_cons_and_flux(rhol, vxl, vyl, vzl, pl, bxn, byl, bzl, eos)

    #right state conservatives and fluxes
    mass_R, momx_R, momy_R, momz_R, etot_R, bfiy_R, bfiz_R, ptot_R, b2r, \
    Fmass_R, Fmomx_R, Fmomy_R, Fmomz_R, Fetot_R, Fbfiy_R, Fbfiz_R = \
            nr_MHD_cons_and_flux(rhor, vxr, vyr, vzr, pr, bxn, byr, bzr, eos)

    #left and right squared sound speeds
    csl2 = eos.sound_speed_nr(rhol, pl)**2
    csr2 = eos.sound_speed_nr(rhor, pr)**2

    #left and right fast magnetosonic speeds
    cfl = np.sqrt( (csl2 + b2l/rhol)/2.0 + np.sqrt((csl2 + b2l/rhol)**2 - 4.0*csl2*bxn**2/rhol)/2.0 )
    cfr = np.sqrt( (csr2 + b2r/rhor)/2.0 + np.sqrt((csr2 + b2r/rhor)**2 - 4.0*csr2*bxn**2/rhor)/2.0 )

    #maximal and minimal eigenvalues HLL estimate according to Davis (1988)
    Sl = np.minimum(vxl, vxr) - np.maximum(cfl, cfr)
    Sr = np.maximum(vxl, vxr) + np.maximum(cfl, cfr)

    #mass fluxes across the left and right fast waves
    Jl = rhol*(Sl - vxl)
    Jr = rhor*(Sr - vxr)

    #contact velocity and total pressure between the fast waves 
    Sm  = ( Jr*vxr - Jl*vxl - (ptot_R - ptot_L) )/( Jr - Jl )
    pts = ( Jr*ptot_L - Jl*ptot_R + Jl*Jr*(vxr - vxl) )/( Jr - Jl )

    #star region densities
    rhosl = Jl/(Sl - Sm);    rhosr = Jr/(Sr - Sm)

    #transverse magnetic field in the star region = HLL average (single value,
    #continuous across the contact: this is what distinguishes HLLC from HLLD)
    byss = ( Sr*byr - Sl*byl - (Fbfiy_R - Fbfiy_L) )/( Sr - Sl )
    bzss = ( Sr*bzr - Sl*bzl - (Fbfiz_R - Fbfiz_L) )/( Sr - Sl )

    #transverse velocities consistent with the HLL transverse field
    #(Jl < 0 and Jr > 0 strictly, so these divisions are safe)
    vysl = vyl - bxn*(byss - byl)/Jl;    vzsl = vzl - bxn*(bzss - bzl)/Jl
    vysr = vyr - bxn*(byss - byr)/Jr;    vzsr = vzr - bxn*(bzss - bzr)/Jr

    #LEFT STARRED STATE
    massS_L = rhosl
    momxS_L = rhosl*Sm;    momyS_L = rhosl*vysl;    momzS_L = rhosl*vzsl
    etotS_L = ( (Sl - vxl)*etot_L - ptot_L*vxl + pts*Sm + \
        bxn*(vxl*bxn + vyl*byl + vzl*bzl - Sm*bxn - vysl*byss - vzsl*bzss) )/(Sl - Sm )
    bfiyS_L = byss;          bfizS_L = bzss

    #RIGHT STARRED STATE
    massS_R = rhosr
    momxS_R = rhosr*Sm;    momyS_R = rhosr*vysr;    momzS_R = rhosr*vzsr
    etotS_R = ( (Sr - vxr)*etot_R - ptot_R*vxr + pts*Sm + \
        bxn*(vxr*bxn + vyr*byr + vzr*bzr - Sm*bxn - vysr*byss - vzsr*bzss) )/(Sr - Sm )
    bfiyS_R = byss;          bfizS_R = bzss

    # calculation of the state using HLLC approximate Riemann fan
    # 2 star states between left fast wave, contact wave and right fast wave
    def _hllc_state(FL, FR, UL, UR, ULs, URs):
        return np.where(Sl >= 0.0, FL,
            np.where((Sl < 0.0) & (Sm >= 0.0), FL + Sl * (ULs - UL),
            np.where((Sm < 0.0) & (Sr >= 0.0), FR + Sr * (URs - UR), FR)))

    # calculation of the flux using HLLC approximate Riemann fan
    Fmass = _hllc_state(Fmass_L, Fmass_R, mass_L, mass_R, massS_L, massS_R)
    Fmomx = _hllc_state(Fmomx_L, Fmomx_R, momx_L, momx_R, momxS_L, momxS_R)
    Fmomy = _hllc_state(Fmomy_L, Fmomy_R, momy_L, momy_R, momyS_L, momyS_R)
    Fmomz = _hllc_state(Fmomz_L, Fmomz_R, momz_L, momz_R, momzS_L, momzS_R)
    Fetot = _hllc_state(Fetot_L, Fetot_R, etot_L, etot_R, etotS_L, etotS_R)
    Fbfix = np.zeros_like(Fmass)
    Fbfiy = _hllc_state(Fbfiy_L, Fbfiy_R, bfiy_L, bfiy_R, bfiyS_L, bfiyS_R)
    Fbfiz = _hllc_state(Fbfiz_L, Fbfiz_R, bfiz_L, bfiz_R, bfizS_L, bfizS_R)

    #return approximate Riemann flux for MHD
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot, Fbfix, Fbfiy, Fbfiz


"""
Harten, Lax, and Van Leer + Contact wave + Alfven discontinuity (HLLD) MHD flux,
following Miyoshi & Kusano (2005).
"""
def HLLD_flux(rhol,rhor, vxl,vxr, vyl,vyr, vzl,vzr, pl,pr, bxl,bxr, byl,byr, bzl,bzr, eos):
    
    #normal B-field and total pressures on the left and on the right side
    bxn = (bxl + bxr)/2.0  
    
    #left state conservatives and fluxes 
    mass_L, momx_L, momy_L, momz_L, etot_L, bfiy_L, bfiz_L, ptot_L, b2l, \
    Fmass_L, Fmomx_L, Fmomy_L, Fmomz_L, Fetot_L, Fbfiy_L, Fbfiz_L = \
            nr_MHD_cons_and_flux(rhol, vxl, vyl, vzl, pl, bxn, byl, bzl, eos)

    #right state conservatives and fluxes 
    mass_R, momx_R, momy_R, momz_R, etot_R, bfiy_R, bfiz_R, ptot_R, b2r, \
    Fmass_R, Fmomx_R, Fmomy_R, Fmomz_R, Fetot_R, Fbfiy_R, Fbfiz_R = \
            nr_MHD_cons_and_flux(rhor, vxr, vyr, vzr, pr, bxn, byr, bzr, eos)

    #left and right squared sound speeds 
    csl2 = eos.sound_speed_nr(rhol, pl)**2
    csr2 = eos.sound_speed_nr(rhor, pr)**2
    
    #left and right fast magnetosonic speeds 
    cfl = np.sqrt( (csl2 + b2l/rhol)/2.0 + np.sqrt((csl2 + b2l/rhol)**2 - 4.0*csl2*bxn**2/rhol)/2.0 )
    cfr = np.sqrt( (csr2 + b2r/rhor)/2.0 + np.sqrt((csr2 + b2r/rhor)**2 - 4.0*csr2*bxn**2/rhor)/2.0 )
    
    #maximal and minimal eigenvalues HLL estimate according to Davis (1988)
    Sl = np.minimum(vxl, vxr) - np.maximum(cfl, cfr)
    Sr = np.maximum(vxl, vxr) + np.maximum(cfl, cfr)
    
    #normal magnetic field sign
    sgnBx = np.sign(bxn)
    
    #contact velocity and total pressure (Jl,Jr - mass fluxes)
    Jl = rhol*(Sl - vxl)
    Jr = rhor*(Sr - vxr)
    #velocity between the shocks
    Sm = ( Jr*vxr - Jl*vxl - (ptot_R - ptot_L) )/( Jr - Jl )
    #total pressure between the shocks
    pts = ( Jr*ptot_L - Jl*ptot_R + Jl*Jr*(vxr - vxl) )/( Jr - Jl )

    #star region densities
    rhosl = Jl/(Sl - Sm);    rhosr = Jr/(Sr - Sm)
    
    #square roots of densities in the star region
    sqrt_rhosl = np.sqrt(rhosl);    sqrt_rhosr = np.sqrt(rhosr)

    #Alfven velocities
    Ssl = Sm - np.abs(bxn)/sqrt_rhosl;    Ssr = Sm + np.abs(bxn)/sqrt_rhosr
    
    #LEFT STARRED STATE 
    vysl = np.where(np.abs( Jl*(Sl - Sm) - bxn**2 ) > 1e-12, vyl - bxn*byl*(Sm - vxl)/( Jl*(Sl - Sm) - bxn**2 + 1e-30), vyl)
    vzsl = np.where(np.abs( Jl*(Sl - Sm) - bxn**2 ) > 1e-12, vzl - bxn*bzl*(Sm - vxl)/( Jl*(Sl - Sm) - bxn**2 + 1e-30), vzl)
    bysl = np.where(np.abs( Jl*(Sl - Sm) - bxn**2 ) > 1e-12, byl*( Jl*(Sl - vxl) - bxn**2 )/( Jl*(Sl - Sm) - bxn**2 + 1e-30), byl)
    bzsl = np.where(np.abs( Jl*(Sl - Sm) - bxn**2 ) > 1e-12, bzl*( Jl*(Sl - vxl) - bxn**2 )/( Jl*(Sl - Sm) - bxn**2 + 1e-30), bzl)
    
    #conservative state inside the star region (L)
    massS_L = rhosl
    momxS_L = rhosl*Sm;    momyS_L = rhosl*vysl;    momzS_L = rhosl*vzsl
    etotS_L = ( (Sl - vxl)*etot_L - ptot_L*vxl + pts*Sm + \
        bxn*(vxl*bxn + vyl*byl + vzl*bzl - Sm*bxn - vysl*bysl - vzsl*bzsl) )/(Sl - Sm )
    bfiyS_L = bysl;          bfizS_L = bzsl

    #RIGHT STARRED STATE 
    vysr = np.where(np.abs( Jr*(Sr - Sm) - bxn**2 ) > 1e-12, vyr - bxn*byr*(Sm - vxr)/( Jr*(Sr - Sm) - bxn**2 + 1e-30), vyr)
    vzsr = np.where(np.abs( Jr*(Sr - Sm) - bxn**2 ) > 1e-12, vzr - bxn*bzr*(Sm - vxr)/( Jr*(Sr - Sm) - bxn**2 + 1e-30), vzr)
    bysr = np.where(np.abs( Jr*(Sr - Sm) - bxn**2 ) > 1e-12, byr*( Jr*(Sr - vxr) - bxn**2 )/( Jr*(Sr - Sm) - bxn**2 + 1e-30), byr)
    bzsr = np.where(np.abs( Jr*(Sr - Sm) - bxn**2 ) > 1e-12, bzr*( Jr*(Sr - vxr) - bxn**2 )/( Jr*(Sr - Sm) - bxn**2 + 1e-30), bzr)
    
    #conservative state inside the star region (R)
    massS_R = rhosr
    momxS_R = rhosr*Sm;    momyS_R = rhosr*vysr;    momzS_R = rhosr*vzsr
    etotS_R = ( (Sr - vxr)*etot_R - ptot_R*vxr + pts*Sm + \
        bxn*(vxr*bxn + vyr*byr + vzr*bzr - Sm*bxn - vysr*bysr - vzsr*bzsr) )/(Sr - Sm )
    bfiyS_R = bysr;          bfizS_R = bzsr
    
    #TWO STARS REGION
    vyss = ( sqrt_rhosl*vysl + sqrt_rhosr*vysr + (bysr - bysl)*sgnBx )/( sqrt_rhosl + sqrt_rhosr )
    byss = ( sqrt_rhosl*bysr + sqrt_rhosr*bysl + np.sqrt(rhosr*rhosl)*(vysr - vysl)*sgnBx )/( sqrt_rhosl + sqrt_rhosr )
    vzss = ( sqrt_rhosl*vzsl + sqrt_rhosr*vzsr + (bzsr - bzsl)*sgnBx )/( sqrt_rhosl + sqrt_rhosr )
    bzss = ( sqrt_rhosl*bzsr + sqrt_rhosr*bzsl + np.sqrt(rhosr*rhosl)*(vzsr - vzsl)*sgnBx )/( sqrt_rhosl + sqrt_rhosr )

    #conservative state inside two stars region (L)
    massSS_L = rhosl
    momxSS_L = rhosl*Sm;    momySS_L = rhosl*vyss;    momzSS_L = rhosl*vzss
    etotSS_L = etotS_L - sqrt_rhosl*( Sm*bxn + vysl*bysl + vzsl*bzsl - Sm*bxn - vyss*byss - vzss*bzss )*sgnBx
    bfiySS_L = byss;          bfizSS_L = bzss
    
    #conservative state inside two stars region (R)        
    massSS_R = rhosr
    momxSS_R = rhosr*Sm;    momySS_R = rhosr*vyss;    momzSS_R = rhosr*vzss
    etotSS_R = etotS_R + sqrt_rhosr*( Sm*bxn + vysr*bysr + vzsr*bzsr - Sm*bxn - vyss*byss - vzss*bzss )*sgnBx
    bfiySS_R = byss;          bfizSS_R = bzss
    
    # calculation of the state using HLLD approximate Riemann fan 
    # 6 states between left shock, left Alfven disc., contact wave, right Alfven disc. and right shock
    def _hlld_state(FL, FR, UL, UR, ULs, URs, ULss, URss):
        return np.where(Sl >= 0.0, FL, 
            np.where((Sl < 0.0) & (Ssl >= 0.0), FL + Sl * (ULs - UL),
            np.where((Ssl < 0.0) & (Sm >= 0.0), FL + Sl * (ULs - UL) + Ssl * (ULss - ULs),
            np.where((Sm < 0.0) & (Ssr >= 0.0), FR + Sr * (URs - UR) + Ssr * (URss - URs), 
            np.where((Ssr < 0.0) & (Sr >= 0.0), FR + Sr * (URs - UR), FR)))))
    
    # calculation of the flux using HLLD approximate Riemann fan 
    Fmass = _hlld_state(Fmass_L, Fmass_R, mass_L, mass_R, massS_L, massS_R, massSS_L, massSS_R)
    Fmomx = _hlld_state(Fmomx_L, Fmomx_R, momx_L, momx_R, momxS_L, momxS_R, momxSS_L, momxSS_R)
    Fmomy = _hlld_state(Fmomy_L, Fmomy_R, momy_L, momy_R, momyS_L, momyS_R, momySS_L, momySS_R)
    Fmomz = _hlld_state(Fmomz_L, Fmomz_R, momz_L, momz_R, momzS_L, momzS_R, momzSS_L, momzSS_R)
    Fetot = _hlld_state(Fetot_L, Fetot_R, etot_L, etot_R, etotS_L, etotS_R, etotSS_L, etotSS_R)
    Fbfix = np.zeros_like(Fmass)
    Fbfiy = _hlld_state(Fbfiy_L, Fbfiy_R, bfiy_L, bfiy_R, bfiyS_L, bfiyS_R, bfiySS_L, bfiySS_R)
    Fbfiz = _hlld_state(Fbfiz_L, Fbfiz_R, bfiz_L, bfiz_R, bfizS_L, bfizS_R, bfizSS_L, bfizSS_R)        
    
    #return approximate Riemann flux for MHD 
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot, Fbfix, Fbfiy, Fbfiz

