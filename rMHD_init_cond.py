# -*- coding: utf-8 -*-
"""
rMHD_init_cond.py
=================

Initial conditions for Special-Relativistic Magnetohydrodynamics (SRMHD) tests.

All functions follow the same interface as the other Piastra IC modules:

    IC_rMHD_<name>(grid, state, par)  ->  (grid, state, par, eos)

where *state* is a SimState object with attributes:
    dens, vel1, vel2, vel3, pres   -- primitive variables (with ghost cells)
    bfi1, bfi2, bfi3               -- cell-centred magnetic field
    fb1, fb2                       -- face-centred (staggered) magnetic field
    F1, F2                         -- external force (gravity, etc.)

The function sets:
    par.timefin   -- final physical time
    par.BC        -- boundary-condition array  [x1_in, x2_in, x1_out, x2_out]

and creates and returns an EOSdata object.

Currently implemented
---------------------
  IC_rMHD1D_BW,RP2,RP3,RP4     : 1D relativistic MHD shock tubes (Mignone & Bodo 2006)
  IC_rMHD2D_rotor     : 2D relativistic rotor (Del Zanna et al. 2003)
  IC_rMHD2D_blast    : 2D relativistic blast wave (Komissarov 1999)
  IC_rMHD_user_defined : placeholder for custom ICs

Author
------
mrkondratyev
"""

import numpy as np
from src.common.eos_setup import EOSdata


# ============================================================================
# User-defined placeholder
# ============================================================================
def IC_rMHD_user_defined(grid, state, par):
    """
    Placeholder for user-defined SRMHD initial conditions.

    Fill in custom values for:
        state.dens, state.vel1-3, state.pres
        state.bfi1-3, state.fb1, state.fb2
        par.timefin
        par.BC

    Parameters
    ----------
    grid  : Grid
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos
    """
    print("rMHD -- user-defined problem")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timefin = 0.4
    par.timenow = 0.0
    par.BC[:] = 'free'

    eos = EOSdata(4.0 / 3.0)

    state.dens[:, :] = 1.0
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0
    state.pres[:, :] = 1.0
    state.bfi1[:, :] = 0.0
    state.bfi2[:, :] = 0.0
    state.bfi3[:, :] = 0.0
    state.fb1[:, :]  = 0.0
    state.fb2[:, :]  = 0.0

    raise ValueError(
        "User-defined rMHD problem – see 'rMHD_init_cond.py', "
        "set your ICs and remove this line."
    )

    return grid, state, par, eos



# ============================================================================
# 1D Brio-Wu relativistic MHD problem
# ============================================================================
def IC_rMHD1D_BW(grid, MHD, par):
    """
    Brio–Wu 1D rMHD extension.
    
    A classical Riemann problem in magnetized fluids used to test the 
    ability of numerical schemes to capture fast/slow shocks, 
    rarefactions, and compound waves.
    
    Parameters
    ----------
    grid : object
        Grid object with geometry and ghost cell info.
    MHD : object
        Container for MHD variables (density, pressure, velocity, B-fields).
    par : object
        Simulation parameters (time control, boundary conditions).
    
    Returns
    -------
    grid : object
        Updated grid object with Cartesian coordinates set.
    MHD : object
        Initialized MHD fields (left/right states).
    par : object
        Updated parameters (timefin, timenow, boundary conditions).
    eos : EOSdata
        Equation of state object with γ = 2.0.
    """
    print("Brio-Wu 1D relativistic MHD shock tube test")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    MHD.vel1[:, :] = 0.0
    MHD.vel2[:, :] = 0.0
    MHD.vel3[:, :] = 0.0    
    MHD.bfi3[:, :] = 0.0
    
    MHD.bfi1[:, :] = 0.0 + 0.75    
        
    MHD.fb1[:,:] = 0.75
    
    par.timefin = 0.1
    par.timenow = 0.0
    
    eos = EOSdata(2.0)
    
    for i in range(grid.Nx1):
        if (grid.fx1[i+grid.Ngc,1]<0.5):
            MHD.fb2[i, :] = 1.0
        else: 
            MHD.fb2[i, :] = -1.0
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r+1):
            if grid.fx1[i, j] < 0.5:
                MHD.dens[i, j] = 1.0
                MHD.pres[i, j] = 1.0
                MHD.bfi2[i, j] = 1.0
            else:
                MHD.dens[i, j] = 0.125
                MHD.pres[i, j] = 0.1
                MHD.bfi2[i, j] = -1.0
                
    par.BC[:] = 'free'
    
    return grid, MHD, par, eos




# ============================================================================
# RP2 relativistic MHD problem
# ============================================================================
def IC_rMHD1D_RP2(grid, MHD, par):
    """
    Riemann problem 2 from 
    Mignone & Bodo "An HLLC Solver for Relativistic Flows – II.
    Magnetohydrodynamics", MNRAS (2006).
    
    Parameters
    ----------
    grid : object
        Grid object with geometry and ghost cell info.
    MHD : object
        Container for MHD variables (density, pressure, velocity, B-fields).
    par : object
        Simulation parameters (time control, boundary conditions).
    
    Returns
    -------
    grid : object
        Updated grid object with Cartesian coordinates set.
    MHD : object
        Initialized MHD fields (left/right states).
    par : object
        Updated parameters (timefin, timenow, boundary conditions).
    eos : EOSdata
        Equation of state object with γ = 2.0.
    """
    print("RP2 rMHD shock tube 1D test")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    MHD.vel1[:, :] = 0.0
    MHD.vel2[:, :] = 0.0
    MHD.vel3[:, :] = 0.0    
    
    MHD.bfi1[:, :] = 0.0 + 5.0    
        
    MHD.fb1[:,:] = 5.0
    
    par.timefin = 0.4
    par.timenow = 0.0
    
    eos = EOSdata(5.0/3.0)
    
    for i in range(grid.Nx1):
        if (grid.fx1[i+grid.Ngc,1]<0.5):
            MHD.fb2[i, :] = 6.0
        else: 
            MHD.fb2[i, :] = 0.7
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r+1):
            if grid.fx1[i, j] < 0.5:
                MHD.dens[i, j] = 1.0
                MHD.pres[i, j] = 30.0
                MHD.bfi2[i, j] = 6.0
                MHD.bfi3[i, j] = 6.0
            else:
                MHD.dens[i, j] = 1.0
                MHD.pres[i, j] = 1.0
                MHD.bfi2[i, j] = 0.7
                MHD.bfi3[i, j] = 0.7
                
    par.BC[:] = 'free'
    
    return grid, MHD, par, eos



# ============================================================================
# RP3 relativistic MHD problem
# ============================================================================
def IC_rMHD1D_RP3(grid, MHD, par):
    """
    Riemann problem 3 with a strong shock from 
    Mignone & Bodo "An HLLC Solver for Relativistic Flows – II.
    Magnetohydrodynamics", MNRAS (2006).
    
    Parameters
    ----------
    grid : object
        Grid object with geometry and ghost cell info.
    MHD : object
        Container for MHD variables (density, pressure, velocity, B-fields).
    par : object
        Simulation parameters (time control, boundary conditions).
    
    Returns
    -------
    grid : object
        Updated grid object with Cartesian coordinates set.
    MHD : object
        Initialized MHD fields (left/right states).
    par : object
        Updated parameters (timefin, timenow, boundary conditions).
    eos : EOSdata
        Equation of state object with γ = 2.0.
    """
    print("RP3 rMHD shock tube 1D test -- strong shock")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    MHD.vel1[:, :] = 0.0
    MHD.vel2[:, :] = 0.0
    MHD.vel3[:, :] = 0.0    
    MHD.bfi3[:, :] = 0.0
    
    MHD.bfi1[:, :] = 0.0 + 10.0    
        
    MHD.fb1[:,:] = 10.0
    
    par.timefin = 0.4
    par.timenow = 0.0
    
    eos = EOSdata(5.0/3.0)
    
    for i in range(grid.Nx1):
        if (grid.fx1[i+grid.Ngc,1] < 0.5):
            MHD.fb2[i, :] = 7.0
        else: 
            MHD.fb2[i, :] = 0.7
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r+1):
            if grid.fx1[i, j] < 0.5:
                MHD.dens[i, j] = 1.0
                MHD.pres[i, j] = 1000.0
                MHD.bfi2[i, j] = 7.0
                MHD.bfi3[i, j] = 7.0
            else:
                MHD.dens[i, j] = 1.0
                MHD.pres[i, j] = 0.1
                MHD.bfi2[i, j] = 0.7
                MHD.bfi3[i, j] = 0.7
                
    par.BC[:] = 'free'
    
    return grid, MHD, par, eos



# ============================================================================
# RP4 relativistic MHD problem
# ============================================================================
def IC_rMHD1D_RP4(grid, MHD, par):
    """
    Riemann problem 4 with ultrarelativistic motion from 
    Mignone & Bodo "An HLLC Solver for Relativistic Flows – II.
    Magnetohydrodynamics", MNRAS (2006).
    
    Parameters
    ----------
    grid : object
        Grid object with geometry and ghost cell info.
    MHD : object
        Container for MHD variables (density, pressure, velocity, B-fields).
    par : object
        Simulation parameters (time control, boundary conditions).
    
    Returns
    -------
    grid : object
        Updated grid object with Cartesian coordinates set.
    MHD : object
        Initialized MHD fields (left/right states).
    par : object
        Updated parameters (timefin, timenow, boundary conditions).
    eos : EOSdata
        Equation of state object with γ = 2.0.
    """
    print("RP4 rMHD shock tube 1D test -- ultrarelativistic shocks")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    MHD.vel1[:, :] = 0.0
    MHD.vel2[:, :] = 0.0
    MHD.vel3[:, :] = 0.0    
    MHD.bfi3[:, :] = 0.0
    
    MHD.bfi1[:, :] = 0.0 + 10.0    
        
    MHD.fb1[:,:] = 10.0
    
    par.timefin = 0.4
    par.timenow = 0.0
    
    eos = EOSdata(5.0/3.0)
    
    for i in range(grid.Nx1):
        if (grid.fx1[i+grid.Ngc,1]<0.5):
            MHD.fb2[i, :] = 7.0
        else: 
            MHD.fb2[i, :] = -7.0
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r+1):
            if grid.cx1[i, j] < 0.5:
                MHD.dens[i, j] = 1.0
                MHD.pres[i, j] = 0.1
                MHD.bfi2[i, j] = 7.0
                MHD.bfi3[i, j] = 7.0
                MHD.vel1[i, j] = 0.999
            else:
                MHD.dens[i, j] = 1.0
                MHD.pres[i, j] = 0.1
                MHD.bfi2[i, j] = -7.0
                MHD.bfi3[i, j] = -7.0
                MHD.vel1[i, j] = -0.999
                
    par.BC[:] = 'free'
    
    return grid, MHD, par, eos


# ============================================================================
# 2D relativistic MHD explosion (cylindrical blast wave)
# ============================================================================
def IC_rMHD2D_blast(grid, MHD, par):
    """
    2D relativistic MHD explosion problem (cylindrical blast).
    
    Standard RMHD test used in many codes (e.g. PLUTO, ATHENA++).
    A high-pressure region expands into a magnetized ambient medium.
    
    Parameters
    ----------
    grid : object
        Grid object with geometry and ghost cell info.
    MHD : object
        Container for MHD variables.
    par : object
        Simulation parameters.
    
    Returns
    -------
    grid, MHD, par, eos
    """
    
    print("2D rMHD explosion test -- cylindrical blast wave")
    
    # ------------------------------------------------------------------
    # Domain
    # ------------------------------------------------------------------
    x1ini, x1fin = -6.0, 6.0
    x2ini, x2fin = -6.0, 6.0
    
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    # ------------------------------------------------------------------
    # Initial uniform state (ambient medium)
    # ------------------------------------------------------------------
    MHD.vel1[:, :] = 0.0
    MHD.vel2[:, :] = 0.0
    MHD.vel3[:, :] = 0.0
    
    MHD.bfi1[:, :] = 0.1   # uniform magnetic field (x-direction)
    MHD.bfi2[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0
    
    MHD.fb1[:, :] = 0.1
    MHD.fb2[:, :] = 0.0
    
    # ------------------------------------------------------------------
    # Time control
    # ------------------------------------------------------------------
    par.timefin = 4.0
    par.timenow = 0.0
    
    # Relativistic EOS
    eos = EOSdata(4.0/3.0)
    
    # ------------------------------------------------------------------
    # Explosion parameters
    # ------------------------------------------------------------------
    x0 = 0.5
    y0 = 0.5
    r0 = 0.08   # blast radius
    
    # Ambient state
    rho_out = 1.0e-4
    p_out   = 3.0e-5
    
    # Inner (explosion) state
    rho_in = 0.01
    p_in   = 1.0   # high pressure → explosion
    
    # ------------------------------------------------------------------
    # Fill domain
    # ------------------------------------------------------------------
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            
            x = grid.cx1[i, j]
            y = grid.cx2[i, j]
            
            r = ((x - x0)**2 + (y - y0)**2)**0.5
            
            if r < r0:
                MHD.dens[i, j] = rho_in
                MHD.pres[i, j] = p_in
            elif r < 1.0:
                MHD.dens[i, j] = (rho_in*(1.0 - r) + rho_out*(r - r0))/(1.0 - r0)
                MHD.pres[i, j] = (p_in*(1.0 - r) + p_out*(r - r0))/(1.0 - r0)
            else:
                MHD.dens[i, j] = rho_out
                MHD.pres[i, j] = p_out
    
    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    par.BC[:] = 'free'
    
    return grid, MHD, par, eos



# ============================================================================
# 2D relativistic rotor
# ============================================================================
def IC_rMHD2D_rotor(grid, state, par):
    """
    2D relativistic rotor (Del Zanna et al. 2003 test).

    A uniformly rotating dense cylinder (r < 0.1) embedded in a uniform
    ambient medium with background magnetic field B_x = 1.
    Adiabatic index GAMMA = 5/3.

    Domain: [-0.5, 0.5] x [-0.5, 0.5].

    Parameters
    ----------
    grid  : Grid
    state : SimState
    par   : Parameters

    Returns
    -------
    grid, state, par, eos
    """
    print("rMHD 2D -- relativistic rotor (Del Zanna et al. 2003)")

    x1ini, x1fin = -0.5, 0.5
    x2ini, x2fin = -0.5, 0.5
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timefin = 0.4
    par.timenow = 0.0
    par.BC = np.array(['free', 'free', 'free', 'free'], dtype=object)

    eos = EOSdata(5.0 / 3.0)

    x = grid.cx1
    y = grid.cx2
    r = np.sqrt(x**2 + y**2)

    # Ambient medium
    state.dens[:, :] = 1.0
    state.vel1[:, :] = 0.0
    state.vel2[:, :] = 0.0
    state.vel3[:, :] = 0.0
    state.pres[:, :] = 1.0
    state.bfi1[:, :] = 1.0
    state.bfi2[:, :] = 0.0
    state.bfi3[:, :] = 0.0

    # Rotating cylinder
    r0    = 0.1
    omega = 9.95   # chosen so that v_max ~ 0.995 at r = r0
    mask  = r < r0
    state.dens[mask] = 10.0
    state.vel1[mask] = -omega * y[mask]
    state.vel2[mask] =  omega * x[mask]

    # Clip velocity to avoid superluminal values
    v2   = state.vel1**2 + state.vel2**2 + state.vel3**2
    vmax = np.sqrt(np.max(v2))
    if vmax >= 1.0:
        fac = 0.995 / vmax
        state.vel1[mask] *= fac
        state.vel2[mask] *= fac

    # Initialise staggered B (uniform Bx)
    state.fb1[:, :] = 1.0
    state.fb2[:, :] = 0.0

    return grid, state, par, eos
