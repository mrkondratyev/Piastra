# -*- coding: utf-8 -*-
"""
===============================================================================
SWE_phys.py
===============================================================================

Core physics routines for the 2D Shallow Water Equations (SWE).

The SWE describe depth-averaged flow of a thin fluid layer and read:

    ∂h/∂t  + ∂(h v₁)/∂x₁ + ∂(h v₂)/∂x₂ = 0
    ∂(h v₁)/∂t + ∂(h v₁² + g h²/2)/∂x₁ + ∂(h v₁ v₂)/∂x₂ = -g h ∂b/∂x₁ + f_c h v₂
    ∂(h v₂)/∂t + ∂(h v₁ v₂)/∂x₁ + ∂(h v₂² + g h²/2)/∂x₂ = -g h ∂b/∂x₂ - f_c h v₁

where:
    h   -- fluid column height
    v₁  -- velocity in x₁ direction
    v₂  -- velocity in x₂ direction
    g   -- gravitational acceleration
    b   -- bathymetry (bed elevation)
    f_c -- Coriolis parameter

Conservative form:  ∂U/∂t + ∂F/∂x₁ + ∂G/∂x₂ = S

with conserved variables  U = (h, h v₁, h v₂)  and source terms
from bathymetry gradient and Coriolis force handled via Strang splitting.

Riemann solver
--------------
HLL (Harten-Lax-van Leer) with wave-speed estimates following Davis (1988):
    S⁻ = min(v - √(gh), v - √(gh)) over left/right states
    S⁺ = max(v + √(gh), v + √(gh)) over left/right states

Boundary conditions
-------------------
Mirrors the structure of hydro_phys.py: apply_bc_scalar and apply_bc_vector
from boundaries.py are used for h and (v₁, v₂) respectively.

References
----------
- Toro, E. F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics.
- Harten, A., Lax, P. D. & van Leer, B. (1983). SIAM Rev. 25, 35.
- Davis, S. F. (1988). SIAM J. Sci. Stat. Comput. 9, 445.

Author: mrkondratyev
"""

import numpy as np
from src.common.boundaries import apply_bc_scalar, apply_bc_vector
from src.models.SWE.SWE_riemann_exact import exact_swe_godunov_state
from src.models.SWE.SWE_riemann_approx import (
    LLF_flux,
    HLL_flux)

def boundCond_SWE(grid, BC, SWE):
    """
    Apply boundary conditions to hydrodynamic variables.

    Parameters
    ----------
    grid : object
        Grid object containing domain information (Nx1, Nx2, Ngc).
    BC : list of str
        Boundary types for each boundary [inner_x1, inner_x2, outer_x1, outer_x2].
        Supported: 'free', 'wall', 'peri', 'axis'.
    SWE : object
        SWE state object with attributes h, vel1, vel2.

    Returns
    -------
    fluid : object
        Fluid object with ghost cells updated according to BCs.
    """
    Ngc = grid.Ngc
    
    # Apply BCs for density
    SWE.h = apply_bc_scalar(SWE.h, Ngc, BC[0], axis=1, side='inner')
    SWE.h = apply_bc_scalar(SWE.h, Ngc, BC[1], axis=2, side='inner')
    SWE.h = apply_bc_scalar(SWE.h, Ngc, BC[2], axis=1, side='outer')
    SWE.h = apply_bc_scalar(SWE.h, Ngc, BC[3], axis=2, side='outer')
    
    #auxilary third component of the velocity in order to use the vector function for hydro flows 
    vel3_dummy = np.zeros_like(SWE.vel1)
    
    # Apply BCs for velocity
    SWE.vel1, SWE.vel2, vel3_dummy = \
        apply_bc_vector(SWE.vel1, SWE.vel2, vel3_dummy, Ngc, BC[0], axis=1, side='inner')
    SWE.vel1, SWE.vel2, vel3_dummy = \
        apply_bc_vector(SWE.vel1, SWE.vel2, vel3_dummy, Ngc, BC[1], axis=2, side='inner')
    SWE.vel1, SWE.vel2, vel3_dummy = \
        apply_bc_vector(SWE.vel1, SWE.vel2, vel3_dummy, Ngc, BC[2], axis=1, side='outer')
    SWE.vel1, SWE.vel2, vel3_dummy = \
        apply_bc_vector(SWE.vel1, SWE.vel2, vel3_dummy, Ngc, BC[3], axis=2, side='outer')
    
    return SWE


def Riemann_SWE(hl, hr, vxl, vxr, vyl, vyr, g_ff, solver_type, dim):
    """
   Approximate Riemann solver for the Euler equations of gas dynamics.

   Parameters
   ----------
   hl, hr : ndarray
       Left and right fluid heights.
   vxl, vxr, vyl, vyr : ndarray
       Velocity components (x, y) for left and right states.
   g_ff : float
       gravity acceleration.
   solver_type : str
       Type of flux solver: 'LLF', 'HLL'.
   dim : int
       Coordinate direction (1 or 2). Other directions obtained by rotation.

   Returns
   -------
   Fh : ndarray
       Flux of mass density.
   Fx, Fy : ndarray
       Fluxes of momentum density in x, y.
   """
    
    #check in what direction we solve the problem
    if dim == 2: #2-direction -- rotate the coordinate system
        templ, tempr = vxl, vxr
        vxl, vxr = vyl, vyr
        vyl, vyr = -templ, -tempr   
    
    #here we calculate the flux using various Riemann solvers
    if solver_type == 'LLF':
        
        Fh, Fx, Fy = LLF_flux(hl, hr, vxl, vxr, vyl, vyr, g_ff)
        
    elif solver_type == 'HLL':  
        
        Fh, Fx, Fy = HLL_flux(hl, hr, vxl, vxr, vyl, vyr, g_ff)
    
    elif solver_type == 'Exact':
        
        # exact Riemann problem solution for SWE/barotropic HD 
        h0, vx0, vy0 = exact_swe_godunov_state(hl, hr, vxl, vxr, vyl, vyr, g_ff)
        
        # flux calculation
        Fh = h0 * vx0
        Fx = h0 * vx0**2 + 0.5 * g_ff * h0**2
        Fy = h0 * vx0 * vy0
        
    else:

        #solver_type is incorrect -> throw an error
        raise ValueError(
            f"Unknown SWE solver_type '{solver_type}'. " 
            f"Expected one of ['LLF', 'HLL', 'Exact'].")

    #check in what direction we solve the problem    #если решаем ЗР вдоль (Y) -- повернем систему координат в исходное состояние 
    if dim == 2: #2-direction -- rotate the coordinate system
        temp = Fx
        Fx = -Fy
        Fy = temp
        
    #return Riemann flux for SWE -- 
    #3 fluxes for conservative variables
    return Fh, Fx, Fy
