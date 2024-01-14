"""
this function calculates the approximate solution of the Riemann problem for gas dynamics 
it is assumed, that the Riemann problem is solved along x-direction (1-direction)
other coordinate directions can be obtained by simple rotations of a coordinate system
flux_type == 'LLF' - Local Lax-Frierdichs (Rusanov (1961)) solver - the most diffusive but most stable
flux_type == 'HLL' - Harten, Lax & van Leer (1983) flux - assumes, that the Riemann fan consists of only 2 shocks
flux_type = 'HLLC' - HLL(Contact) (Toro et al (1994)) - assumes 2 shocks and a contact discontinuity between them 
The last option is less diffusive and allows to resolve stationary contact waves exactly  
@author: mrkondratyev
"""


import numpy as np


def FluxCalcRS(rhol, rhor, vxl, vxr, vyl, vyr, vzl, vzr, pl, pr, eos, flux_type, dim):
    
    
    #check in what direction we solve the problem
    if dim == 2: #2-direction -- rotate the coordinate system
        templ, tempr = vxl, vxr
        vxl, vxr = vyl, vyr
        vyl, vyr = -templ, -tempr
        
        
    #left conservative state
    mass_L = rhol
    momx_L = rhol * vxl
    momy_L = rhol * vyl
    momz_L = rhol * vzl
    etot_L = pl / (eos.GAMMA - 1.0) + rhol * vxl * vxl / 2.0 + \
        rhol * vyl * vyl / 2.0 + rhor * vzl * vzl / 2.0
    
    
    #right conservative state
    mass_R = rhor
    momx_R = rhor * vxr
    momy_R = rhor * vyr
    momz_R = rhor * vzr
    etot_R = pr / (eos.GAMMA - 1.0) + rhor * vxr * vxr / 2.0 + \
        rhor * vyr * vyr / 2.0 + rhor * vzr * vzr / 2.0
    
    
    #left fluxes
    Fmass_L = rhol * vxl
    Fmomx_L = rhol * vxl * vxl + pl
    Fmomy_L = rhol * vyl * vxl
    Fmomz_L = rhol * vzl * vxl
    Fetot_L = vxl * (pl + etot_L)
    
    
    #right fluxes
    Fmass_R = rhor * vxr
    Fmomx_R = rhor * vxr * vxr + pr
    Fmomy_R = rhor * vyr * vxr
    Fmomz_R = rhor * vzr * vxr
    Fetot_R = vxr * (pr + etot_R)


    #left and right sound speeds 
    csl = np.sqrt(eos.GAMMA * pl / rhol)
    csr = np.sqrt(eos.GAMMA * pr / rhor)
    
    
    #maximal and minimal eigenvalues estimate according to Davis (1988)
    Sl = np.minimum(vxl, vxr) - np.maximum(csl, csr)
    Sr = np.maximum(vxl, vxr) + np.maximum(csl, csr)
    
    
    #here we calculate the flux using LLF, HLL or HLLC approximate Riemann solvers
    if flux_type == 'LLF':
        
        #maximal absolute value of eigenvalues  
        Sr = np.maximum(csl + np.abs(vxl), csr + np.abs(vxr))
        
        #calculation of the flux (the dissipation, proportional to the maximal wavespeed, is added)
        Fmass = ( Fmass_L + Fmass_R ) / 2.0 - Sr * (mass_R - mass_L) / 2.0
        Fmomx = ( Fmomx_L + Fmomx_R ) / 2.0 - Sr * (momx_R - momx_L) / 2.0
        Fmomy = ( Fmomy_L + Fmomy_R ) / 2.0 - Sr * (momy_R - momy_L) / 2.0
        Fmomz = ( Fmomz_L + Fmomz_R ) / 2.0 - Sr * (momz_R - momz_L) / 2.0
        Fetot = ( Fetot_L + Fetot_R ) / 2.0 - Sr * (etot_R - etot_L) / 2.0
        
        
    elif flux_type == 'HLL':  
        
        #maximal and minimal eigenvalues for one-line form of HLL flux
        Sl = np.minimum(Sl, 0.0)
        Sr = np.maximum(Sr, 0.0)
        
        #calculation of the flux using HLL approximate Riemann fan (3 states between two shocks)
        Fmass = ( Sr * Fmass_L - Sl * Fmass_R + Sr * Sl * (mass_R - mass_L) ) / (Sr - Sl + 1e-14)
        Fmomx = ( Sr * Fmomx_L - Sl * Fmomx_R + Sr * Sl * (momx_R - momx_L) ) / (Sr - Sl + 1e-14)
        Fmomy = ( Sr * Fmomy_L - Sl * Fmomy_R + Sr * Sl * (momy_R - momy_L) ) / (Sr - Sl + 1e-14)
        Fmomz = ( Sr * Fmomz_L - Sl * Fmomz_R + Sr * Sl * (momz_R - momz_L) ) / (Sr - Sl + 1e-14)
        Fetot = ( Sr * Fetot_L - Sl * Fetot_R + Sr * Sl * (etot_R - etot_L) ) / (Sr - Sl + 1e-14)
            
            
    elif flux_type == 'HLLC':
        
        #contact wave speed in HLLC approximation
        Sstar = (pr - pl + rhol * vxl * (Sl - vxl) - 
            rhor * vxr * (Sr - vxr)) / (rhol * (Sl - vxl) - rhor * (Sr - vxr))
        
        
        #conservative fluid state in the regions in both sides from the contact discontinuity
        #left starred state
        massS_L = rhol * (Sl - vxl) / (Sl - Sstar) 
        momxS_L = massS_L * Sstar 
        momyS_L = massS_L * vyl 
        momzS_L = massS_L * vzl 
        etotS_L = massS_L * ( etot_L / rhol + (Sstar - vxl) * (Sstar + pl / rhol / (Sl - vxl)) ) 
        
        #right starred state
        massS_R = rhor * (Sr - vxr) / (Sr - Sstar)
        momxS_R = massS_R * Sstar 
        momyS_R = massS_R * vyr 
        momzS_R = massS_R * vzr 
        etotS_R = massS_R * ( etot_R / rhor + (Sstar - vxr) * (Sstar + pr / rhor / (Sr - vxr)) ) 
        
        
        # calculation of the flux using HLLC approximate Riemann fan 
        # 4 states between left shock, contact wave, and right shock
        Fmass = np.where(Sl >= 0.0, Fmass_L, 
                         np.where((Sl < 0.0) & (Sstar >= 0.0), Fmass_L + Sl * (massS_L - mass_L),
                                  np.where((Sstar < 0.0) & (Sr >= 0.0), Fmass_R + Sr * (massS_R - mass_R), Fmass_R)))
        
        Fmomx = np.where(Sl >= 0.0, Fmomx_L, 
                          np.where((Sl < 0.0) & (Sstar >= 0.0), Fmomx_L + Sl * (momxS_L - momx_L),
                                   np.where((Sstar < 0.0) & (Sr >= 0.0), Fmomx_R + Sr * (momxS_R - momx_R), Fmomx_R)))

        Fmomy = np.where(Sl >= 0.0, Fmomy_L, 
                          np.where((Sl < 0.0) & (Sstar >= 0.0), Fmomy_L + Sl * (momyS_L - momy_L),
                                   np.where((Sstar < 0.0) & (Sr >= 0.0), Fmomy_R + Sr * (momyS_R - momy_R), Fmomy_R)))

        Fmomz = np.where(Sl >= 0.0, Fmomz_L, 
                          np.where((Sl < 0.0) & (Sstar >= 0.0), Fmomz_L + Sl * (momzS_L - momz_L),
                                   np.where((Sstar < 0.0) & (Sr >= 0.0), Fmomz_R + Sr * (momzS_R - momz_R), Fmomz_R)))

        Fetot = np.where(Sl >= 0.0, Fetot_L, 
                          np.where((Sl < 0.0) & (Sstar >= 0.0), Fetot_L + Sl * (etotS_L - etot_L),
                                   np.where((Sstar < 0.0) & (Sr >= 0.0), Fetot_R + Sr * (etotS_R - etot_R), Fetot_R)))
    
    
    #check in what direction we solve the problem
    if dim == 2: #2-direction -- rotate the coordinate system
        temp = Fmomx
        Fmomx = -Fmomy
        Fmomy = temp
        
        
    #return approximate Riemann flux for gas dynamics -- 
    #5 fluxes for conservative variables (mass, three components of momentum and energy)
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot

