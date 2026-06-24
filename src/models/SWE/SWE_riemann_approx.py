# -*- coding: utf-8 -*-
"""
SWE_riemann_approx.py

Approximate Riemann solvers for shallow water equations.

Implemented solvers, in increasing order of accuracy/cost:

    LLF   - Local Lax-Friedrichs / Rusanov (1961)
    HLL   - Harten, Lax, van Leer (1983)

All solvers share the same calling convention (see Riemann_SWE in 
                                               SWE_phys file).

Each Riemann solver routine has the following i/o sturcture:

   Parameters
   ----------
   hl, hr : ndarray
       Left and right fluid heights.
   vxl, vxr, vyl, vyr : ndarray
       Velocity components (x, y) for left and right states.
   g_ff : float
       free-fall acceleration

   Returns
   -------
   Fh : ndarray
       Flux of the fluid height.
   Fx, Fy : ndarray
       Fluxes of "momentum density" in x, y.   

Author: mrkondratyev
"""

import numpy as np

# ============================================================================
#   Approximate SWE Riemann solvers
# ============================================================================

"""
Local Lax-Friedrichs (Rusanov) flux
"""
def LLF_flux(hl, hr, vxl, vxr, vyl, vyr, g_ff):
    
    #left fluxes
    Fh_L = hl * vxl
    Fx_L = hl * vxl * vxl + 0.5 * g_ff * hl**2 
    Fy_L = hl * vyl * vxl
    
    #right fluxes
    Fh_R = hr * vxr
    Fx_R = hr * vxr * vxr + 0.5 * g_ff * hr**2
    Fy_R = hr * vyr * vxr

    #left and right speeds of gravity waves 
    gwavel = np.sqrt(g_ff * hl); gwaver = np.sqrt(g_ff * hr)     
        
    #maximal absolute value of eigenvalues  
    Sr = np.maximum(gwavel + np.abs(vxl), gwaver + np.abs(vxr))
    
    #Rusanov -- diffusion ~ to maximal system eugenvalue is added 
    Fh = (Fh_L + Fh_R) / 2.0 - Sr * (hr     - hl    ) / 2.0
    Fx = (Fx_L + Fx_R) / 2.0 - Sr * (hr*vxr - hl*vxl) / 2.0
    Fy = (Fy_L + Fy_R) / 2.0 - Sr * (hr*vyr - hl*vyl) / 2.0
    
    #return Rusanov flux for SWE
    return Fh, Fx, Fy



"""
Harten, Lax, and Van Leer (HLL) flux
"""
def HLL_flux(hl, hr, vxl, vxr, vyl, vyr, g_ff):
    
    #left fluxes
    Fh_L = hl * vxl
    Fx_L = hl * vxl * vxl + 0.5 * g_ff * hl**2 
    Fy_L = hl * vyl * vxl
    
    #right fluxes
    Fh_R = hr * vxr
    Fx_R = hr * vxr * vxr + 0.5 * g_ff * hr**2
    Fy_R = hr * vyr * vxr

    #left and right speeds of gravity waves 
    gwavel = np.sqrt(g_ff * hl); gwaver = np.sqrt(g_ff * hr)    
        
    #maximal and minimal eigenvalues estimate according to Davis (1988)
    Sl = np.minimum(vxl, vxr) - np.maximum(gwavel, gwaver)
    Sr = np.maximum(vxl, vxr) + np.maximum(gwavel, gwaver)
        
    #maximal and minimal eigenvalues for one-line form of HLL flux
    Sl = np.minimum(Sl, 0.0)
    Sr = np.maximum(Sr, 0.0)
        
    #HLL -- 3 states between two shocks
    Fh = (Sr*Fh_L - Sl*Fh_R + Sr*Sl*(hr     - hl    ))/(Sr - Sl)
    Fx = (Sr*Fx_L - Sl*Fx_R + Sr*Sl*(hr*vxr - hl*vxl))/(Sr - Sl)
    Fy = (Sr*Fy_L - Sl*Fy_R + Sr*Sl*(hr*vyr - hl*vyl))/(Sr - Sl)
    
    #return HLL flux for SWE
    return Fh, Fx, Fy
