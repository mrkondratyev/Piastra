"""
Non-relativistic Magnetohydrodynamics (MHD) Routines
====================================================

This module provides functions for evolving the equations of
non-relativistic ideal magnetohydrodynamics (MHD) using
finite-volume Godunov-type schemes. The routines handle conversion
between primitive and conservative variables, calculation of wave
speeds, approximate Riemann solvers (LLF, HLL, HLLD), and divergence
cleaning with the GLM method.

Implemented features
--------------------
- Conversion between primitive and conservative variables
- Calculation of maximum characteristic wave speeds (fast modes)
- Boundaries handling for cell-center variables
- Approximate Riemann solvers:
  * LLF (Local Lax-Friedrichs / Rusanov)
  * HLL (Harten–Lax–van Leer)
  * HLLD (Harten–Lax–van Leer with Discontinuities)
- Dedner GLM divergence cleaning subsystem (solver will be supported in future)

Assumptions
-----------
- Ideal gas equation of state with constant gamma.
- Non-relativistic limit (velocities << c).
- Conservative formulation with eight variables:
  mass density, three momentum components, total energy,
  and three magnetic field components.

References
----------
- Rusanov (1961), USSR J. Comp. Math. Phys.
- Harten, Lax & van Leer (1983), SIAM Rev.
- Davis (1988), SIAM J. Sci. Stat. Comput.
- Miyoshi & Kusano (2005), J. Comp. Phys.
- Dedner et al. (2002), J. Comp. Phys.

Author
------
mrkondratyev, 2024–2025
"""

import numpy as np
from boundaries import apply_bc_scalar, apply_bc_vector



def prim2cons_nr_MHD(dens, vel1, vel2, vel3, pres, bfld1, bfld2, bfld3, eos):
    """
    Convert primitive variables to conservative variables (non-relativistic MHD).

    Parameters
    ----------
    dens : ndarray
       Mass density.
    vel1, vel2, vel3 : ndarray
       Velocity components in Cartesian directions.
    pres : ndarray
       Gas pressure.
    bfld1, bfld2, bfld3 : ndarray
       Magnetic field components (primitive state).
    eos : object
       Equation of state object with attribute `GAMMA`.

    Returns
    -------
    mass : ndarray
       Mass density (conserved).
    mom1, mom2, mom3 : ndarray
       Momentum density components.
    etot : ndarray
       Total energy density.
    bcon1, bcon2, bcon3 : ndarray
       Magnetic field components (unchanged in conservative form).
    """
    
    mass = dens
    mom1 = dens * vel1
    mom2 = dens * vel2
    mom3 = dens * vel3
    etot = dens * (vel1**2 + vel2**2 + vel3**2) / 2.0 + pres / (eos.GAMMA - 1.0) + (bfld1**2 + bfld2**2 + bfld3**2) / 2.0
    bcon1 = bfld1
    bcon2 = bfld2
    bcon3 = bfld3
    
    return mass, mom1, mom2, mom3, etot, bcon1, bcon2, bcon3




def cons2prim_nr_MHD(mass, mom1, mom2, mom3, etot, bcon1, bcon2, bcon3, eos):
    """
    Convert conservative variables to primitive variables (non-relativistic MHD).

    Parameters
    ----------
    mass : ndarray
        Mass density.
    mom1, mom2, mom3 : ndarray
        Momentum density components.
    etot : ndarray
        Total energy density (kinetic + thermal + magnetic).
    bcon1, bcon2, bcon3 : ndarray
        Magnetic field components (conservative state).
    eos : object
        Equation of state object with attribute `GAMMA`.

    Returns
    -------
    dens : ndarray
        Mass density.
    vel1, vel2, vel3 : ndarray
        Velocity components.
    pres : ndarray
        Gas pressure.
    bfld1, bfld2, bfld3 : ndarray
        Magnetic field components (identical to input).
    """
    dens = mass
    vel1 = mom1 / dens
    vel2 = mom2 / dens
    vel3 = mom3 / dens
    bfld1 = bcon1
    bfld2 = bcon2
    bfld3 = bcon3
    pres = (eos.GAMMA - 1.0) * (etot - dens * (vel1**2 + vel2**2 + vel3**2) / 2.0 - (bfld1**2 + bfld2**2 + bfld3**2) / 2.0)
    
    return dens, vel1, vel2, vel3, pres, bfld1, bfld2, bfld3




def max_wavespeed_MHD(csound, b1, b2, b3, dens):
    """
    Compute the maximum fast magnetosonic speed.

    Parameters
    ----------
    csound : ndarray
        Sound speed.
    b1, b2, b3 : ndarray
        Magnetic field components.
    dens : ndarray
        Mass density.

    Returns
    -------
    cfast : ndarray
        Maximum fast magnetosonic speed.
    """
    cfast = np.sqrt( csound**2 + (b1**2 + b2**2 + b3**2) / dens )

    return cfast




def boundCond_MHD(grid, BC, fluid):
    """
    Apply boundary conditions to MHD variables.

    Parameters
    ----------
    grid : object
        Grid object containing domain information (Nx1, Nx2, Ngc).
    BC : list of str
        Boundary types for each boundary [inner_x1, inner_x2, outer_x1, outer_x2].
        Supported: 'free', 'wall', 'peri', 'axis'.
    fluid : object
        MHD state object with attributes dens, pres, vel1, vel2, vel3, bfi1, bfi2, bfi3.

    Returns
    -------
    fluid : object
        MHD object with ghost cells updated according to BCs.
    """
    Ngc = grid.Ngc
    
    # Apply BCs for density and pressure
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[0], axis=1, side='inner')
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[1], axis=2, side='inner')
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[2], axis=1, side='outer')
    fluid.dens = apply_bc_scalar(fluid.dens, Ngc, BC[3], axis=2, side='outer')
    
    fluid.pres = apply_bc_scalar(fluid.pres, Ngc, BC[0], axis=1, side='inner')
    fluid.pres = apply_bc_scalar(fluid.pres, Ngc, BC[1], axis=2, side='inner')
    fluid.pres = apply_bc_scalar(fluid.pres, Ngc, BC[2], axis=1, side='outer')
    fluid.pres = apply_bc_scalar(fluid.pres, Ngc, BC[3], axis=2, side='outer')
    
    # Apply BCs for velocity
    fluid.vel1, fluid.vel2, fluid.vel3 = \
        apply_bc_vector(fluid.vel1, fluid.vel2, fluid.vel3, Ngc, BC[0], axis=1, side='inner')
    fluid.vel1, fluid.vel2, fluid.vel3 = \
        apply_bc_vector(fluid.vel1, fluid.vel2, fluid.vel3, Ngc, BC[1], axis=2, side='inner')
    fluid.vel1, fluid.vel2, fluid.vel3 = \
        apply_bc_vector(fluid.vel1, fluid.vel2, fluid.vel3, Ngc, BC[2], axis=1, side='outer')
    fluid.vel1, fluid.vel2, fluid.vel3 = \
        apply_bc_vector(fluid.vel1, fluid.vel2, fluid.vel3, Ngc, BC[3], axis=2, side='outer')
    
    # Apply BCs for magnetic fields
    fluid.bfi1, fluid.bfi2, fluid.bfi3 = \
        apply_bc_vector(fluid.bfi1, fluid.bfi2, fluid.bfi3, Ngc, BC[0], axis=1, side='inner')
    fluid.bfi1, fluid.bfi2, fluid.bfi3 = \
        apply_bc_vector(fluid.bfi1, fluid.bfi2, fluid.bfi3, Ngc, BC[1], axis=2, side='inner')
    fluid.bfi1, fluid.bfi2, fluid.bfi3 = \
        apply_bc_vector(fluid.bfi1, fluid.bfi2, fluid.bfi3, Ngc, BC[2], axis=1, side='outer')
    fluid.bfi1, fluid.bfi2, fluid.bfi3 = \
        apply_bc_vector(fluid.bfi1, fluid.bfi2, fluid.bfi3, Ngc, BC[3], axis=2, side='outer')
    
    return fluid




def Riemann_flux_nr_MHD(rhol,rhor, vxl,vxr, vyl,vyr, vzl,vzr, pl,pr, bxl,bxr, byl,byr, bzl,bzr, eos, flux_type, dim):
    """
    Compute approximate Riemann fluxes for non-relativistic MHD.

    Supports LLF, HLL, and HLLD solvers.

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
        Equation of state object with attribute `GAMMA`.
    flux_type : {'LLF', 'HLL', 'HLLD'}
        Choice of Riemann solver.
    dim : int
        Normal direction (1 or 2). If `dim == 2`, system is rotated.

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
    """
    
    #check in what direction we solve the problem
    if dim == 2: #2-direction -- rotate the coordinate system
        templ, tempr = vxl, vxr
        vxl, vxr = vyl, vyr
        vyl, vyr = -templ, -tempr
        
        templ, tempr = bxl, bxr
        bxl, bxr = byl, byr
        byl, byr = -templ, -tempr
    
    #normal B-field and total pressures on the left and on the right side
    bxn = (bxl + bxr)/2.0  

    b2l = bxn * bxn + byl * byl + bzl * bzl 
    b2r = bxn * bxn + byr * byr + bzr * bzr
    ptot_L = pl + b2l / 2.0
    ptot_R = pr + b2r / 2.0

    #left conservative state
    mass_L = rhol
    momx_L = rhol * vxl
    momy_L = rhol * vyl
    momz_L = rhol * vzl
    etot_L = pl / (eos.GAMMA - 1.0) + rhol * ( vxl * vxl + vyl * vyl + vzl * vzl ) / 2.0 + b2l / 2.0
    bfix_L = bxn
    bfiy_L = byl
    bfiz_L = bzl
    
    #right conservative state
    mass_R = rhor
    momx_R = rhor * vxr
    momy_R = rhor * vyr
    momz_R = rhor * vzr
    etot_R = pr / (eos.GAMMA - 1.0) + rhor * ( vxr * vxr + vyr * vyr + vzr * vzr ) / 2.0 + b2r / 2.0
    bfix_R = bxn
    bfiy_R = byr
    bfiz_R = bzr
    
    #left fluxes
    Fmass_L = rhol * vxl
    Fmomx_L = rhol * vxl * vxl + ptot_L - bxn * bxn
    Fmomy_L = rhol * vyl * vxl - byl * bxn
    Fmomz_L = rhol * vzl * vxl - bzl * bxn
    Fetot_L = vxl * (ptot_L + etot_L) - bxn * (bxn*vxl + byl*vyl + bzl*vzl)
    Fbfix_L = 0.0
    Fbfiy_L = vxl * byl - vyl * bxn
    Fbfiz_L = vxl * bzl - vzl * bxn
    
    #right fluxes
    Fmass_R = rhor * vxr
    Fmomx_R = rhor * vxr * vxr + ptot_R - bxn * bxn
    Fmomy_R = rhor * vyr * vxr - byr * bxn
    Fmomz_R = rhor * vzr * vxr - bzr * bxn
    Fetot_R = vxr * (ptot_R + etot_R) - bxn * (bxn*vxr + byr*vyr + bzr*vzr)
    Fbfix_R = 0.0
    Fbfiy_R = vxr * byr - vyr * bxn
    Fbfiz_R = vxr * bzr - vzr * bxn

    #left and right squared sound speeds 
    csl2 = eos.GAMMA * pl / rhol
    csr2 = eos.GAMMA * pr / rhor
    
    #left and right fast magnetosonic speeds 
    cfl = np.sqrt( (csl2 + b2l/rhol)/2.0 + np.sqrt((csl2 + b2l/rhol)**2 - 4.0*csl2*bxn**2/rhol)/2.0 )
    cfr = np.sqrt( (csr2 + b2r/rhor)/2.0 + np.sqrt((csr2 + b2r/rhor)**2 - 4.0*csr2*bxn**2/rhor)/2.0 )
    
    #here we calculate the flux using LLF approximate Riemann solver (Rusanov (1961))
    if flux_type == 'LLF':
        
        #maximal absolute value of eigenvalues  
        Sr = np.maximum(cfl + np.abs(vxl), cfr + np.abs(vxr))
        
        #calculation of the flux (the dissipation, proportional to the maximal wavespeed, is added)
        Fmass = ( Fmass_L + Fmass_R ) / 2.0 - Sr * (mass_R - mass_L) / 2.0
        Fmomx = ( Fmomx_L + Fmomx_R ) / 2.0 - Sr * (momx_R - momx_L) / 2.0
        Fmomy = ( Fmomy_L + Fmomy_R ) / 2.0 - Sr * (momy_R - momy_L) / 2.0
        Fmomz = ( Fmomz_L + Fmomz_R ) / 2.0 - Sr * (momz_R - momz_L) / 2.0
        Fetot = ( Fetot_L + Fetot_R ) / 2.0 - Sr * (etot_R - etot_L) / 2.0
        Fbfix = np.zeros_like(Fmass)
        Fbfiy = ( Fbfiy_L + Fbfiy_R ) / 2.0 - Sr * (bfiy_R - bfiy_L) / 2.0
        Fbfiz = ( Fbfiz_L + Fbfiz_R ) / 2.0 - Sr * (bfiz_R - bfiz_L) / 2.0
        
    #here we calculate the flux using HLL approximate Riemann solver (3 states between two fast shocks)
    #solution of Riemann problem according to Harten, Lax and van Leer, SIAM (1983)
    #see also Miyoshi and Kusano, JCP (2005)
    elif flux_type == 'HLL':  
        
        #maximal and minimal eigenvalues estimate according to Davis (1988)
        Sl = np.minimum(vxl, vxr) - np.maximum(cfl, cfr)
        Sr = np.maximum(vxl, vxr) + np.maximum(cfl, cfr)
        
        #maximal and minimal eigenvalues for one-line form of HLL flux
        Sl = np.minimum(Sl, 0.0)
        Sr = np.maximum(Sr, 0.0)
        
        #calculation of the flux using HLL approximate Riemann fan (3 states between two shocks)
        Fmass = ( Sr * Fmass_L - Sl * Fmass_R + Sr * Sl * (mass_R - mass_L) ) / (Sr - Sl)
        Fmomx = ( Sr * Fmomx_L - Sl * Fmomx_R + Sr * Sl * (momx_R - momx_L) ) / (Sr - Sl)
        Fmomy = ( Sr * Fmomy_L - Sl * Fmomy_R + Sr * Sl * (momy_R - momy_L) ) / (Sr - Sl)
        Fmomz = ( Sr * Fmomz_L - Sl * Fmomz_R + Sr * Sl * (momz_R - momz_L) ) / (Sr - Sl)
        Fetot = ( Sr * Fetot_L - Sl * Fetot_R + Sr * Sl * (etot_R - etot_L) ) / (Sr - Sl)
        Fbfix = np.zeros_like(Fmass)
        Fbfiy = ( Sr * Fbfiy_L - Sl * Fbfiy_R + Sr * Sl * (bfiy_R - bfiy_L) ) / (Sr - Sl)
        Fbfiz = ( Sr * Fbfiz_L - Sl * Fbfiz_R + Sr * Sl * (bfiz_R - bfiz_L) ) / (Sr - Sl)
        
    #here we calculate the flux using HLLD approximate Riemann solver 
    #(6 states between two fast shocks, two Alfven discontinuities and contact surface)
    #solution of Riemann problem according to Miyoshi and Kusano, JCP (2005) 
    elif flux_type == 'HLLD':
        
        #maximal and minimal eigenvalues estimate according to Davis (1988)
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
        rhosl = Jl/(Sl - Sm)
        rhosr = Jr/(Sr - Sm)
        
        #square roots of densities in the star region
        sqrt_rhosl = np.sqrt(rhosl)
        sqrt_rhosr = np.sqrt(rhosr)

        #Alfven velocities
        Ssl = Sm - np.abs(bxn)/sqrt_rhosl
        Ssr = Sm + np.abs(bxn)/sqrt_rhosr
        
        #LEFT STARRED STATE 
        vysl = np.where(np.abs( Jl*(Sl - Sm) - bxn**2 ) > 1e-12, vyl - bxn*byl*(Sm - vxl)/( Jl*(Sl - Sm) - bxn**2 ), vyl)
        vzsl = np.where(np.abs( Jl*(Sl - Sm) - bxn**2 ) > 1e-12, vzl - bxn*bzl*(Sm - vxl)/( Jl*(Sl - Sm) - bxn**2 ), vzl)
        bysl = np.where(np.abs( Jl*(Sl - Sm) - bxn**2 ) > 1e-12, byl*( Jl*(Sl - vxl) - bxn**2 )/( Jl*(Sl - Sm) - bxn**2 ), byl)
        bzsl = np.where(np.abs( Jl*(Sl - Sm) - bxn**2 ) > 1e-12, bzl*( Jl*(Sl - vxl) - bxn**2 )/( Jl*(Sl - Sm) - bxn**2 ), bzl)
        
        #conservative state inside the star region (L)
        massS_L = rhosl
        momxS_L = rhosl*Sm
        momyS_L = rhosl*vysl
        momzS_L = rhosl*vzsl
        etotS_L = ( (Sl - vxl)*etot_L - ptot_L*vxl + pts*Sm + \
            bxn*(vxl*bxn + vyl*byl + vzl*bzl - Sm*bxn - vysl*bysl - vzsl*bzsl) )/(Sl - Sm )
        bfixS_L = bxn
        bfiyS_L = bysl
        bfizS_L = bzsl

        #RIGHT STARRED STATE 
        vysr = np.where(np.abs( Jr*(Sr - Sm) - bxn**2 ) > 1e-12, vyr - bxn*byr*(Sm - vxr)/( Jr*(Sr - Sm) - bxn**2 ), vyr)
        vzsr = np.where(np.abs( Jr*(Sr - Sm) - bxn**2 ) > 1e-12, vzr - bxn*bzr*(Sm - vxr)/( Jr*(Sr - Sm) - bxn**2 ), vzr)
        bysr = np.where(np.abs( Jr*(Sr - Sm) - bxn**2 ) > 1e-12, byr*( Jr*(Sr - vxr) - bxn**2 )/( Jr*(Sr - Sm) - bxn**2 ), byr)
        bzsr = np.where(np.abs( Jr*(Sr - Sm) - bxn**2 ) > 1e-12, bzr*( Jr*(Sr - vxr) - bxn**2 )/( Jr*(Sr - Sm) - bxn**2 ), bzr)
        
        #conservative state inside the star region (R)
        massS_R = rhosr
        momxS_R = rhosr*Sm
        momyS_R = rhosr*vysr
        momzS_R = rhosr*vzsr
        etotS_R = ( (Sr - vxr)*etot_R - ptot_R*vxr + pts*Sm + \
            bxn*(vxr*bxn + vyr*byr + vzr*bzr - Sm*bxn - vysr*bysr - vzsr*bzsr) )/(Sr - Sm )
        bfixS_R = bxn
        bfiyS_R = bysr
        bfizS_R = bzsr
        
        #TWO STARS REGION
        vyss = ( sqrt_rhosl*vysl + sqrt_rhosr*vysr + (bysr - bysl)*sgnBx )/( sqrt_rhosl + sqrt_rhosr )
        byss = ( sqrt_rhosl*bysr + sqrt_rhosr*bysl + np.sqrt(rhosr*rhosl)*(vysr - vysl)*sgnBx )/( sqrt_rhosl + sqrt_rhosr )
        vzss = ( sqrt_rhosl*vzsl + sqrt_rhosr*vzsr + (bzsr - bzsl)*sgnBx )/( sqrt_rhosl + sqrt_rhosr )
        bzss = ( sqrt_rhosl*bzsr + sqrt_rhosr*bzsl + np.sqrt(rhosr*rhosl)*(vzsr - vzsl)*sgnBx )/( sqrt_rhosl + sqrt_rhosr )

        #conservative state inside two stars region (L)
        massSS_L = rhosl
        momxSS_L = rhosl*Sm
        momySS_L = rhosl*vyss
        momzSS_L = rhosl*vzss
        etotSS_L = etotS_L - sqrt_rhosl*( Sm*bxn + vysl*bysl + vzsl*bzsl - Sm*bxn - vyss*byss - vzss*bzss )*sgnBx
        bfixSS_L = bxn
        bfiySS_L = byss
        bfizSS_L = bzss
        
        #conservative state inside two stars region (R)        
        massSS_R = rhosr
        momxSS_R = rhosr*Sm
        momySS_R = rhosr*vyss
        momzSS_R = rhosr*vzss
        etotSS_R = etotS_R + sqrt_rhosr*( Sm*bxn + vysr*bysr + vzsr*bzsr - Sm*bxn - vyss*byss - vzss*bzss )*sgnBx
        bfixSS_R = bxn
        bfiySS_R = byss
        bfizSS_R = bzss
        
        # calculation of the flux using HLLD approximate Riemann fan 
        # 6 states between left shock, left Alfven disc., contact wave, right Alfven disc. and right shock
        #final mass flux
        Fmass = np.where(Sl >= 0.0, Fmass_L, 
            np.where((Sl < 0.0) & (Ssl >= 0.0), Fmass_L + Sl * (massS_L - mass_L),
            np.where((Ssl < 0.0) & (Sm >= 0.0), Fmass_L + Sl * (massS_L - mass_L) + Ssl * (massSS_L - massS_L),
            np.where((Sm < 0.0) & (Ssr >= 0.0), Fmass_R + Sr * (massS_R - mass_R) + Ssr * (massSS_R - massS_R), 
            np.where((Ssr < 0.0) & (Sr >= 0.0), Fmass_R + Sr * (massS_R - mass_R), 
            Fmass_R)))))
        
        #final x-momentum flux
        Fmomx = np.where(Sl >= 0.0, Fmomx_L, 
            np.where((Sl < 0.0) & (Ssl >= 0.0), Fmomx_L + Sl * (momxS_L - momx_L),
            np.where((Ssl < 0.0) & (Sm >= 0.0), Fmomx_L + Sl * (momxS_L - momx_L) + Ssl * (momxSS_L - momxS_L),
            np.where((Sm < 0.0) & (Ssr >= 0.0), Fmomx_R + Sr * (momxS_R - momx_R) + Ssr * (momxSS_R - momxS_R), 
            np.where((Ssr < 0.0) & (Sr >= 0.0), Fmomx_R + Sr * (momxS_R - momx_R), 
            Fmomx_R)))))
        
        #final y-momentum flux
        Fmomy = np.where(Sl >= 0.0, Fmomy_L, 
            np.where((Sl < 0.0) & (Ssl >= 0.0), Fmomy_L + Sl * (momyS_L - momy_L),
            np.where((Ssl < 0.0) & (Sm >= 0.0), Fmomy_L + Sl * (momyS_L - momy_L) + Ssl * (momySS_L - momyS_L),
            np.where((Sm < 0.0) & (Ssr >= 0.0), Fmomy_R + Sr * (momyS_R - momy_R) + Ssr * (momySS_R - momyS_R), 
            np.where((Ssr < 0.0) & (Sr >= 0.0), Fmomy_R + Sr * (momyS_R - momy_R), 
            Fmomy_R)))))
        
        #final z-momentum flux
        Fmomz = np.where(Sl >= 0.0, Fmomz_L, 
            np.where((Sl < 0.0) & (Ssl >= 0.0), Fmomz_L + Sl * (momzS_L - momz_L),
            np.where((Ssl < 0.0) & (Sm >= 0.0), Fmomz_L + Sl * (momzS_L - momz_L) + Ssl * (momzSS_L - momzS_L),
            np.where((Sm < 0.0) & (Ssr >= 0.0), Fmomz_R + Sr * (momzS_R - momz_R) + Ssr * (momzSS_R - momzS_R), 
            np.where((Ssr < 0.0) & (Sr >= 0.0), Fmomz_R + Sr * (momzS_R - momz_R), 
            Fmomz_R)))))

        #final energy flux
        Fetot = np.where(Sl >= 0.0, Fetot_L, 
            np.where((Sl < 0.0) & (Ssl >= 0.0), Fetot_L + Sl * (etotS_L - etot_L),
            np.where((Ssl < 0.0) & (Sm >= 0.0), Fetot_L + Sl * (etotS_L - etot_L) + Ssl * (etotSS_L - etotS_L),
            np.where((Sm < 0.0) & (Ssr >= 0.0), Fetot_R + Sr * (etotS_R - etot_R) + Ssr * (etotSS_R - etotS_R), 
            np.where((Ssr < 0.0) & (Sr >= 0.0), Fetot_R + Sr * (etotS_R - etot_R), 
            Fetot_R)))))
        
        #final Bx-field flux (always equals to zero) 
        Fbfix = np.zeros_like(Fmass)
        
        #final By-field flux
        Fbfiy = np.where(Sl >= 0.0, Fbfiy_L, 
            np.where((Sl < 0.0) & (Ssl >= 0.0), Fbfiy_L + Sl * (bfiyS_L - bfiy_L),
            np.where((Ssl < 0.0) & (Sm >= 0.0), Fbfiy_L + Sl * (bfiyS_L - bfiy_L) + Ssl * (bfiySS_L - bfiyS_L),
            np.where((Sm < 0.0) & (Ssr >= 0.0), Fbfiy_R + Sr * (bfiyS_R - bfiy_R) + Ssr * (bfiySS_R - bfiyS_R), 
            np.where((Ssr < 0.0) & (Sr >= 0.0), Fbfiy_R + Sr * (bfiyS_R - bfiy_R), 
            Fbfiy_R)))))
        
        #final Bz-field flux
        Fbfiz = np.where(Sl >= 0.0, Fbfiz_L, 
            np.where((Sl < 0.0) & (Ssl >= 0.0), Fbfiz_L + Sl * (bfizS_L - bfiz_L),
            np.where((Ssl < 0.0) & (Sm >= 0.0), Fbfiz_L + Sl * (bfizS_L - bfiz_L) + Ssl * (bfizSS_L - bfizS_L),
            np.where((Sm < 0.0) & (Ssr >= 0.0), Fbfiz_R + Sr * (bfizS_R - bfiz_R) + Ssr * (bfizSS_R - bfizS_R), 
            np.where((Ssr < 0.0) & (Sr >= 0.0), Fbfiz_R + Sr * (bfizS_R - bfiz_R), 
            Fbfiz_R)))))
        
    else:
        
        #flux_type is incorrect -> throw an error
        raise ValueError(f"Unknown flux_type: {flux_type}. Expected one of ['LLF', 'HLL', 'HLLD'].")
        
    #check in what direction we solve the problem
    if dim == 2: #2-direction -- rotate the coordinate system
        temp = Fmomx
        Fmomx = -Fmomy
        Fmomy = temp
        
        temp = Fbfix
        Fbfix = -Fbfiy
        Fbfiy = temp
        
    #return approximate Riemann flux for gas dynamics -- 
    #8 fluxes for conservative variables 
    #(mass, three components of momentum, energy, three components of the B-field)
    return Fmass, Fmomx, Fmomy, Fmomz, Fetot, Fbfix, Fbfiy, Fbfiz




def divB_clean_GLM_sol_MHD(c_h, bnl,bnr, psil,psir):
    """
    Solve GLM divergence cleaning subsystem.

    Implements mixed hyperbolic–parabolic divergence control
    according to Dedner et al., JCP (2002).

    Parameters
    ----------
    c_h : float
        Cleaning wave speed.
    bnl, bnr : ndarray
        Left and right normal magnetic field components.
    psil, psir : ndarray
        Left and right scalar potentials.

    Returns
    -------
    bnf : ndarray
        Flux for the normal magnetic field.
    psif : ndarray
        Flux for the scalar potential.
    """    
    bnf = np.where(np.abs(c_h) > 1e-14, (bnr + bnl)/2.0 - (psir - psil)/c_h/2.0, (bnr + bnl)/2.0)
    psif = np.where(np.abs(c_h) > 1e-14, (psir + psil)/2.0 - (bnr - bnl)*c_h/2.0, 0.0)
    
    return bnf, psif 
