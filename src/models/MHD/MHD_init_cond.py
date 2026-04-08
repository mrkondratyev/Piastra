# -*- coding: utf-8 -*-
"""
MHD Initial Conditions Module

This module provides functions to set up 1D and 2D MHD
test problems. It initializes the grid, fluid state, 
and simulation time parameters.

Author: mrkondratyev
Date: June 17, 2024
"""

import numpy as np
from src.common.eos_setup import EOSdata




def IC_MHD_user_defined(grid, MHD, par):
    """
    user-defined MHD problem.


    Parameters
    ----------
    grid : object
        Cartesian grid in [0,1] × [0,1].
    MHD : object
        MHD state container (density, pressure, velocity, magnetic fields).
    par : object
        Simulation parameters.

    Returns
    -------
    grid : object
        Cartesian grid with Orszag–Tang vortex initialized.
    MHD : object
        Initialized fluid and magnetic fields.
    par : object
        Parameters with periodic BCs.
    eos : EOSdata
        Equation of state with γ = 5/3.
    """
    
    print("user-defined problem for MHD")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    #grid.CylindricalGrid(x1ini, x1fin, x2ini, x2fin)
    
    MHD.vel3[:, :] = 0.0    
    MHD.bfi3[:, :] = 0.0
    
    MHD.dens[:, :] = 25.0/36.0/np.pi
    MHD.pres[:, :] = 5.0/12.0/np.pi  
        
    par.timefin = 0.5
    par.timenow = 0.0
    
    eos = EOSdata(5.0/3.0)
    
    for i in range(grid.Nx1+1):
        MHD.fb1[i, :] = -np.sin(2.0 * np.pi * grid.cx2[i,grid.Ngc:-grid.Ngc])/np.sqrt(4.0 * np.pi)
        
    for j in range(grid.Nx2+1):
        MHD.fb2[:, j] = np.sin(4.0 * np.pi * grid.cx1[grid.Ngc:-grid.Ngc,j])/np.sqrt(4.0 * np.pi)
        
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            MHD.bfi1[i, j] = -np.sin(2.0 * np.pi * grid.cx2[i,j])/np.sqrt(4.0 * np.pi)
            MHD.bfi2[i, j] = np.sin(4.0 * np.pi * grid.cx1[i,j])/np.sqrt(4.0 * np.pi)
            MHD.vel1[i, j] = -np.sin(2.0 * np.pi * grid.cx2[i,j])
            MHD.vel2[i, j] = np.sin(2.0 * np.pi * grid.cx1[i,j])
        
    #boundary conditions
    #all support walls, periodic and free-outflow boundaries, BC[0] supports axis for cylindrical grids
    par.BC[:] = 'peri'
    
    raise ValueError("User-defined MHD problem, see file 'MHD_init_cond.py', adjust ICs and delete this line.")
    
    return grid, MHD, par, eos



def IC_MHD1D_BW(grid, MHD, par):
    """
    Brio–Wu (1988) 1D MHD shock tube test.
    
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
    print("Brio-Wu (1988) 1D MHD shock tube test")
    
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
    
    eos = EOSdata(10.0/5.0)
    
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




def IC_MHD1D_Toth(grid, MHD, par):
    """
    Tóth (2000) 1D MHD shock tube test.

    A strong shock tube problem with large pressure and velocity 
    discontinuities. Used to test robustness of MHD solvers 
    against shock interactions.

    Parameters
    ----------
    grid : object
        Grid object with Cartesian geometry.
    MHD : object
        Container for MHD variables.
    par : object
        Simulation parameters.

    Returns
    -------
    grid : object
        Grid initialized for Cartesian coordinates.
    MHD : object
        Initialized MHD fields.
    par : object
        Updated simulation parameters (final time, BCs).
    eos : EOSdata
        Equation of state with γ = 5/3.
    """    
    print("1D Toth (2000) MHD shock tube test")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    MHD.vel1[:, :] = 0.0
    MHD.vel2[:, :] = 0.0
    MHD.vel3[:, :] = 0.0    
    MHD.bfi3[:, :] = 0.0
    
    MHD.bfi1[:, :] = 0.0 + 5.0/np.sqrt(4.0*np.pi)
    MHD.fb1[:, :] = 0.0 + 5.0/np.sqrt(4.0*np.pi)
    MHD.dens[:, :] = 1.0
    
    par.timefin = 0.08
    par.timenow = 0.0
    MHD.fb2[:,:] = 0.0 + 5.0/np.sqrt(4.0*np.pi)
    
    eos = EOSdata(5.0/3.0)
    
    for i in range(grid.Ngc, grid.Nx1r):           
        for j in range(grid.Ngc, grid.Nx2r+1):
            if grid.fx1[i, j] < 0.5:
                MHD.pres[i, j] = 20.0
                MHD.vel1[i, j] = 10.0
                MHD.bfi2[i, j] = 0.0 + 5.0/np.sqrt(4.0*np.pi)
            else:
                MHD.pres[i, j] = 1.0
                MHD.vel1[i, j] = -10.0
                MHD.bfi2[i, j] = 0.0 + 5.0/np.sqrt(4.0*np.pi)
                
    par.BC[:] = 'free'
    
    return grid, MHD, par, eos




def IC_MHD2D_blast_cart(grid, MHD, par):
    """
    2D magnetized explosion test (planar Cartesian geometry).

    A standard blast wave problem in a magnetized medium. 
    A high-pressure circular region is initialized in the center 
    of a uniform low-pressure medium with a diagonal background magnetic field.

    Parameters
    ----------
    grid : object
        Cartesian grid.
    MHD : object
        MHD state container.
    par : object
        Simulation parameters.

    Returns
    -------
    grid : object
        Updated Cartesian grid.
    MHD : object
        Initialized density, pressure, velocity, and magnetic fields.
    par : object
        Parameters with timefin = 0.2 and free BCs.
    eos : EOSdata
        Equation of state with γ = 7/5.
    """    
    print("magnetized explosion test in 2D planar Cartesian geometry")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    MHD.vel1[:, :] = 0.0
    MHD.vel2[:, :] = 0.0
    MHD.vel3[:, :] = 0.0    
    MHD.bfi3[:, :] = 0.0
    
    MHD.dens[:, :] = 1.0
    MHD.bfi1[:, :] = 1.0/np.sqrt(2.0)
    MHD.bfi2[:, :] = 1.0/np.sqrt(2.0)    
    
    MHD.fb1[:, :] = 1.0/np.sqrt(2.0)
    MHD.fb2[:, :] = 1.0/np.sqrt(2.0)    
    
    par.timefin = 0.2
    par.timenow = 0.0
    
    eos = EOSdata(7.0/5.0)
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            rad = np.sqrt(np.abs(grid.fx1[i, j] - 0.5)**2 + np.abs(grid.fx2[i, j] - 0.5)**2) 
            if rad < 0.1:
                MHD.pres[i, j] = 10.0
            else:
                MHD.pres[i, j] = 0.1
            
    par.BC[:] = 'free'
    
    return grid, MHD, par, eos




def IC_MHD2D_blast_cyl(grid, MHD, par):
    """
    2D magnetized explosion test (cylindrical axisymmetry).
    
    Cylindrical version of the blast wave problem. The explosion 
    is initialized in an axisymmetric (r, z) domain with background 
    axial magnetic field.
    
    Notes
    -----
    Coordinate extents:
    - r ∈ [0.0, 0.5]
    - z ∈ [-0.5, 0.5]
    
    Parameters
    ----------
    grid : object
        Cylindrical grid.
    MHD : object
        MHD state container.
    par : object
        Simulation parameters.
    
    Returns
    -------
    grid : object
        Cylindrical grid with explosion initialized.
    MHD : object
        Initialized MHD fields.
    par : object
        Parameters with reflecting boundary on axis, free elsewhere.
    eos : EOSdata
        Equation of state with γ = 7/5.
    """
    print("magnetized explosion test in 2D cylindrical axisymmetry")
    
    #coordinate range in each direction, by default r and z are in range [0..0.5, -0.5..0.5]
    x1ini, x1fin = 0.0, 0.5
    x2ini, x2fin = -0.5, 0.5

    #filling the grid arrays with grid data 
    grid.CylindricalGrid(x1ini, x1fin, x2ini, x2fin)
    
    MHD.vel1[:, :] = 0.0
    MHD.vel2[:, :] = 0.0
    MHD.vel3[:, :] = 0.0    
    MHD.bfi3[:, :] = 0.0
    
    MHD.dens[:, :] = 1.0
    MHD.bfi1[:, :] = 0.0
    MHD.bfi2[:, :] = 1.0
    
    MHD.fb1[:, :] = 0.0
    MHD.fb2[:, :] = 1.0 
    
    par.timefin = 0.2
    par.timenow = 0.0
    
    eos = EOSdata(7.0/5.0)
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            rad = np.sqrt(grid.cx1[i, j]**2 + grid.cx2[i, j]**2) 
            if rad < 0.1:
                MHD.pres[i, j] = 10.0
            else:
                MHD.pres[i, j] = 0.1
            
    par.BC[0] = 'axis'
    par.BC[1] = 'free'
    par.BC[2] = 'free'
    par.BC[3] = 'free'
    
    return grid, MHD, par, eos




def IC_MHD2D_OT(grid, MHD, par):
    """
    2D Orszag–Tang vortex problem.

    A widely used MHD turbulence benchmark. Initial conditions 
    generate interacting shocks and vortices that quickly 
    evolve into MHD turbulence.

    Parameters
    ----------
    grid : object
        Cartesian grid in [0,1] × [0,1].
    MHD : object
        MHD state container (density, pressure, velocity, magnetic fields).
    par : object
        Simulation parameters.

    Returns
    -------
    grid : object
        Cartesian grid with Orszag–Tang vortex initialized.
    MHD : object
        Initialized fluid and magnetic fields.
    par : object
        Parameters with periodic BCs.
    eos : EOSdata
        Equation of state with γ = 5/3.
    """
    
    print("2D Orszag-Tang vortex problem in 2D MHD")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    MHD.vel3[:, :] = 0.0    
    MHD.bfi3[:, :] = 0.0
    
    MHD.dens[:, :] = 25.0/36.0/np.pi
    MHD.pres[:, :] = 5.0/12.0/np.pi  
        
    par.timefin = 0.5
    par.timenow = 0.0
    
    eos = EOSdata(5.0/3.0)
    
    for i in range(grid.Nx1+1):
        MHD.fb1[i, :] = -np.sin(2.0 * np.pi * grid.cx2[i,grid.Ngc:-grid.Ngc])/np.sqrt(4.0 * np.pi)
        
    for j in range(grid.Nx2+1):
        MHD.fb2[:, j] = np.sin(4.0 * np.pi * grid.cx1[grid.Ngc:-grid.Ngc,j])/np.sqrt(4.0 * np.pi)
        
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            MHD.bfi1[i, j] = -np.sin(2.0 * np.pi * grid.cx2[i,j])/np.sqrt(4.0 * np.pi)
            MHD.bfi2[i, j] = np.sin(4.0 * np.pi * grid.cx1[i,j])/np.sqrt(4.0 * np.pi)
            MHD.vel1[i, j] = -np.sin(2.0 * np.pi * grid.cx2[i,j])
            MHD.vel2[i, j] = np.sin(2.0 * np.pi * grid.cx1[i,j])
                 
    par.BC[:] = 'peri'
    
    return grid, MHD, par, eos




# ============================================================================
# 2D MHD rotor problem
# ============================================================================
def IC_MHD2D_rotor(grid, MHD, par):
    """
    2D MHD rotor problem.

    A dense, rapidly rotating disk embedded in a static medium.
    Generates strong torsional Alfvén waves and shocks.

    Returns
    -------
    grid, MHD, par, eos
    """
    
    print("2D MHD rotor problem")
    
    # ------------------------------------------------------------------
    # Domain
    # ------------------------------------------------------------------
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    # ------------------------------------------------------------------
    # Initial uniform medium
    # ------------------------------------------------------------------
    MHD.vel1[:, :] = 0.0
    MHD.vel2[:, :] = 0.0
    MHD.vel3[:, :] = 0.0
    
    MHD.bfi1[:, :] = 5.0 / np.sqrt(4.0 * np.pi)
    MHD.bfi2[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0
    
    MHD.fb1[:, :] = 5.0 / np.sqrt(4.0 * np.pi)
    MHD.fb2[:, :] = 0.0
    
    MHD.pres[:, :] = 1.0
    
    # Rotor parameters
    x0, y0 = 0.5, 0.5
    r0 = 0.1
    r1 = 0.115
    
    rho_in = 10.0
    rho_out = 1.0
    
    omega = 20.0
    
    par.timefin = 0.15
    par.timenow = 0.0
    
    eos = EOSdata(7.0/5.0)
    
    # ------------------------------------------------------------------
    # Fill domain
    # ------------------------------------------------------------------
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            
            x = grid.cx1[i, j]
            y = grid.cx2[i, j]
            
            dx = x - x0
            dy = y - y0
            r = np.sqrt(dx*dx + dy*dy)
            
            if r <= r0:
                MHD.dens[i, j] = rho_in
                MHD.vel1[i, j] = -omega * dy
                MHD.vel2[i, j] =  omega * dx
                
            elif r <= r1:
                f = (r1 - r) / (r1 - r0)
                MHD.dens[i, j] = rho_out + (rho_in - rho_out) * f
                MHD.vel1[i, j] = -omega * dy * f
                MHD.vel2[i, j] =  omega * dx * f
                
            else:
                MHD.dens[i, j] = rho_out
                MHD.vel1[i, j] = 0.0
                MHD.vel2[i, j] = 0.0
    
    par.BC[:] = 'free'
    
    return grid, MHD, par, eos





def IC_MHD1D_RJ(grid, MHD, par):
    """
    Ryu & Jones (1995) 1D MHD shock tube, test 2a.

    A standard MHD Riemann problem that produces all seven MHD wave
    families (fast/slow shocks, rotational discontinuities, contact).
    Widely used to validate HLLD-type solvers.

    Left  state: rho=1.08, v=(1.2, 0.01, 0.5), p=0.95, B=(2/sqrt(4pi), 3.6/sqrt(4pi), 2/sqrt(4pi))
    Right state: rho=1,    v=(0, 0, 0),         p=1,    B=(2/sqrt(4pi), 4/sqrt(4pi), 2/sqrt(4pi))
    Gamma = 5/3, t_fin = 0.2

    Parameters
    ----------
    grid : object
    MHD : object
    par : object

    Returns
    -------
    grid, MHD, par, eos : objects

    References
    ----------
    Ryu, D. & Jones, T. W. (1995), ApJ 442, 228
    """
    print("Ryu & Jones (1995) 1D MHD shock tube test 2a")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timefin = 0.2
    par.timenow = 0.0
    eos = EOSdata(5.0 / 3.0)

    Bx = 2.0 / np.sqrt(4.0 * np.pi)
    MHD.bfi1[:, :] = Bx
    MHD.fb1[:, :] = Bx

    for i in range(grid.Nx1):
        if grid.fx1[i + grid.Ngc, 1] < 0.5:
            MHD.fb2[i, :] = 3.6 / np.sqrt(4.0 * np.pi)
        else:
            MHD.fb2[i, :] = 4.0 / np.sqrt(4.0 * np.pi)

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r + 1):
            if grid.fx1[i, j] < 0.5:
                MHD.dens[i, j] = 1.08
                MHD.vel1[i, j] = 1.2
                MHD.vel2[i, j] = 0.01
                MHD.vel3[i, j] = 0.5
                MHD.pres[i, j] = 0.95
                MHD.bfi2[i, j] = 3.6 / np.sqrt(4.0 * np.pi)
                MHD.bfi3[i, j] = 2.0 / np.sqrt(4.0 * np.pi)
            else:
                MHD.dens[i, j] = 1.0
                MHD.vel1[i, j] = 0.0
                MHD.vel2[i, j] = 0.0
                MHD.vel3[i, j] = 0.0
                MHD.pres[i, j] = 1.0
                MHD.bfi2[i, j] = 4.0 / np.sqrt(4.0 * np.pi)
                MHD.bfi3[i, j] = 2.0 / np.sqrt(4.0 * np.pi)

    par.BC[:] = 'free'

    return grid, MHD, par, eos




def IC_MHD2D_current_sheet(grid, MHD, par):
    """
    2D current sheet (magnetic reconnection) test.

    Two anti-parallel current sheets are initialized with a small
    velocity perturbation to trigger tearing-mode instability and
    magnetic reconnection. The problem tests the code's ability to
    handle thin current layers and resistive-like numerical dissipation.

    Domain: [0, 2] x [0, 2], periodic in both directions.
    B_x = +1 for 0.5 < y < 1.5, -1 otherwise (anti-parallel sheets)
    rho = 1, p = 0.1, small v_y perturbation at sheet locations.

    Parameters
    ----------
    grid : object
    MHD : object
    par : object

    Returns
    -------
    grid, MHD, par, eos : objects

    References
    ----------
    Gardiner, T. A. & Stone, J. M. (2005), J. Comput. Phys. 205, 509
    """
    print("2D MHD current sheet / reconnection test")

    x1ini, x1fin = 0.0, 2.0
    x2ini, x2fin = 0.0, 2.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timefin = 5.0
    par.timenow = 0.0
    eos = EOSdata(5.0 / 3.0)

    MHD.dens[:, :] = 1.0
    MHD.pres[:, :] = 0.1
    MHD.vel1[:, :] = 0.0
    MHD.vel3[:, :] = 0.0
    MHD.bfi2[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0
    MHD.fb2[:, :] = 0.0

    B0 = 1.0
    amp = 0.1  # perturbation amplitude

    for i in range(grid.Nx1 + 1):
        y_face = grid.cx2[i, grid.Ngc:-grid.Ngc] if i < grid.Nx1 + 1 else grid.cx2[grid.Ngc, grid.Ngc:-grid.Ngc]
        # Use cell center y for face-centered Bx
        for jj in range(grid.Nx2):
            yc = grid.cx2[grid.Ngc + i if i < grid.Nx1 else grid.Ngc, grid.Ngc + jj]
            if 0.5 < yc < 1.5:
                MHD.fb1[i, jj] = B0
            else:
                MHD.fb1[i, jj] = -B0

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            y = grid.cx2[i, j]
            x = grid.cx1[i, j]
            if 0.5 < y < 1.5:
                MHD.bfi1[i, j] = B0
            else:
                MHD.bfi1[i, j] = -B0

            # Small velocity perturbation at each current sheet
            MHD.vel2[i, j] = amp * (
                np.sin(2.0 * np.pi * x / 2.0)
                * (np.exp(-((y - 0.5) / 0.05)**2) + np.exp(-((y - 1.5) / 0.05)**2))
            )

    par.BC[:] = 'peri'

    return grid, MHD, par, eos




def IC_MHD2D_field_loop(grid, MHD, par):
    """
    2D magnetic field loop advection test (Gardiner & Stone 2005).

    A weak circular magnetic field loop is advected diagonally across
    the domain. The magnetic pressure is much less than the gas
    pressure (beta >> 1), so the field is passively advected. This
    tests the ability of the CT scheme to maintain the shape and
    divergence-free nature of a weak magnetic structure.

    Domain: [-1, 1] x [-0.5, 0.5], periodic
    Background: rho=1, p=1, v=(2, 1, 0)
    Field loop: A_z = A0 * (R0 - r) for r < R0, with R0=0.3, A0=1e-3
    B = curl(A_z z-hat), so B_x = dA_z/dy, B_y = -dA_z/dx
    Gamma = 5/3, t_fin = 1 (one full crossing)

    Parameters
    ----------
    grid : object
    MHD : object
    par : object

    Returns
    -------
    grid, MHD, par, eos : objects

    References
    ----------
    Gardiner, T. A. & Stone, J. M. (2005), J. Comput. Phys. 205, 509
    """
    print("2D magnetic field loop advection (Gardiner & Stone 2005)")

    x1ini, x1fin = -1.0, 1.0
    x2ini, x2fin = -0.5, 0.5
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timefin = 1.0
    par.timenow = 0.0
    eos = EOSdata(5.0 / 3.0)

    # Background state
    MHD.dens[:, :] = 1.0
    MHD.pres[:, :] = 1.0
    MHD.vel1[:, :] = 2.0
    MHD.vel2[:, :] = 1.0
    MHD.vel3[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0

    R0 = 0.3
    A0 = 1.0e-3

    # Compute cell-centred B from vector potential A_z
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            x = grid.cx1[i, j]
            y = grid.cx2[i, j]
            r = np.sqrt(x**2 + y**2)
            if r < R0 and r > 1e-14:
                MHD.bfi1[i, j] = A0 * (-y / r)
                MHD.bfi2[i, j] = A0 * (x / r)
            else:
                MHD.bfi1[i, j] = 0.0
                MHD.bfi2[i, j] = 0.0

    # Face-centred B from A_z at face midpoints
    # fb1 (B_x on x1-faces): B_x = dA_z/dy ~ A0*(-y/r)
    for i in range(grid.Nx1 + 1):
        for jj in range(grid.Nx2):
            xf = grid.fx1[i + grid.Ngc, jj + grid.Ngc]
            yf = grid.cx2[i + grid.Ngc, jj + grid.Ngc]
            r = np.sqrt(xf**2 + yf**2)
            if r < R0 and r > 1e-14:
                MHD.fb1[i, jj] = A0 * (-yf / r)
            else:
                MHD.fb1[i, jj] = 0.0

    # fb2 (B_y on x2-faces): B_y = -dA_z/dx ~ A0*(x/r)
    for ii in range(grid.Nx1):
        for j in range(grid.Nx2 + 1):
            xf = grid.cx1[ii + grid.Ngc, j + grid.Ngc]
            yf = grid.fx2[ii + grid.Ngc, j + grid.Ngc]
            r = np.sqrt(xf**2 + yf**2)
            if r < R0 and r > 1e-14:
                MHD.fb2[ii, j] = A0 * (xf / r)
            else:
                MHD.fb2[ii, j] = 0.0

    par.BC[:] = 'peri'

    return grid, MHD, par, eos




def IC_MHD2D_disk(grid, MHD, par):
    """
    Axisymmetric MHD accretion disk in cylindrical (R, Z) coordinates.

    Equilibrium configuration from the PLUTO code paper. A disk in
    hydrostatic and centrifugal equilibrium threaded by a weak
    vertical magnetic field. Tests the ability of the code to
    maintain MHD equilibrium and resolve the magneto-rotational
    instability (MRI) if resolved.

    Coordinate system: cylindrical (R, Z)
    Domain: R in [0.5, 3], Z in [0, 1]
    Disk: rho = rho0 * R^(-3/2), v_phi = R^(-1/2) (Keplerian),
          p balanced by gravity, B_z = beta_0^(-1/2) * p^(1/2)
    Gamma = 5/3, t_fin = 10 (orbits at R=1)

    Parameters
    ----------
    grid : object
    MHD : object
    par : object

    Returns
    -------
    grid, MHD, par, eos : objects

    References
    ----------
    Mignone, A. et al. (2007), ApJS 170, 228 (PLUTO code paper)
    """
    print("2D axisymmetric MHD disk (Mignone et al. 2007)")

    R_in, R_out = 0.5, 3.0
    Z_in, Z_out = 0.0, 1.0
    grid.CylindricalGrid(R_in, R_out, Z_in, Z_out)

    par.timenow = 0.0
    par.timefin = 10.0 * 2.0 * np.pi   # 10 orbits at R=1
    eos = EOSdata(5.0 / 3.0)

    # Disk parameters
    rho0 = 1.0
    GM = 1.0           # central mass (gravitational parameter)
    beta0 = 100.0       # plasma beta
    h_over_r = 0.1      # disk aspect ratio

    MHD.vel1[:, :] = 0.0    # v_R = 0
    MHD.vel3[:, :] = 0.0    # v_phi handled below
    MHD.bfi1[:, :] = 0.0    # B_R = 0
    MHD.bfi3[:, :] = 0.0    # B_phi = 0
    MHD.fb1[:, :] = 0.0     # face-B_R = 0

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            R = grid.cx1[i, j]

            # Keplerian rotation
            v_K = np.sqrt(GM / R)
            MHD.vel2[i, j] = 0.0       # v_Z = 0
            MHD.vel3[i, j] = v_K       # v_phi = Keplerian (stored in vel3)

            # Density and pressure
            rho = rho0 * R**(-1.5)
            cs = h_over_r * v_K
            pres = rho * cs**2

            MHD.dens[i, j] = rho
            MHD.pres[i, j] = pres

            # Weak vertical magnetic field (constant beta)
            B_z = np.sqrt(2.0 * pres / beta0)
            MHD.bfi2[i, j] = B_z

            # Source terms: stellar gravity
            MHD.F1[i - grid.Ngc, j - grid.Ngc] = -GM / R**2 * rho
            MHD.F2[i - grid.Ngc, j - grid.Ngc] = 0.0

    # Face-centred B_z
    for ii in range(grid.Nx1):
        for j in range(grid.Nx2 + 1):
            R = grid.cx1[ii + grid.Ngc, grid.Ngc]
            v_K = np.sqrt(GM / R)
            cs = h_over_r * v_K
            rho = rho0 * R**(-1.5)
            pres = rho * cs**2
            MHD.fb2[ii, j] = np.sqrt(2.0 * pres / beta0)

    par.BC[0] = 'axis'
    par.BC[1] = 'wall'
    par.BC[2] = 'free'
    par.BC[3] = 'free'

    return grid, MHD, par, eos




def IC_MHD2D_shock_cloud(grid, MHD, par):
    """
    2D MHD shock-cloud interaction.

    A magnetised shock impacts a dense spherical cloud embedded in a
    uniform magnetised ambient medium. The magnetic field modifies
    the cloud's disruption by suppressing Kelvin-Helmholtz instabilities
    along field lines while enhancing them perpendicular to B. This is
    an astrophysically important problem modelling supernova blast
    waves hitting interstellar clouds.

    Domain: [0, 1] x [0, 1]
    Cloud: centre (0.25, 0.5), radius 0.1, rho=10
    Ambient (pre-shock): rho=1, p=1, B_x=2/sqrt(4pi)
    Post-shock (x < 0.05): Rankine-Hugoniot jump for Mach 10

    Parameters
    ----------
    grid : object
    MHD : object
    par : object

    Returns
    -------
    grid, MHD, par, eos : objects

    References
    ----------
    Orlando, S. et al. (2008), ApJ 678, 274
    Shin, M.-S., Stone, J. M. & Snyder, G. F. (2008), ApJ 680, 336
    """
    print("2D MHD shock-cloud interaction")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timefin = 0.06
    par.timenow = 0.0
    eos = EOSdata(5.0 / 3.0)

    # Cloud parameters
    xc, yc = 0.25, 0.5
    rc = 0.1
    rho_cloud = 10.0

    # Pre-shock (ambient)
    rho_amb = 1.0
    p_amb = 1.0
    B0 = 2.0 / np.sqrt(4.0 * np.pi)

    # Post-shock state (Mach 10 in magnetised gas)
    rho_post = 3.857143
    v1_post = 11.2
    p_post = 167.0

    x_shock = 0.05

    MHD.vel3[:, :] = 0.0
    MHD.bfi2[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0
    MHD.fb2[:, :] = 0.0

    # Uniform B_x everywhere (unaffected by HD shock to first approx)
    MHD.bfi1[:, :] = B0
    MHD.fb1[:, :] = B0

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            x = grid.cx1[i, j]
            y = grid.cx2[i, j]
            r = np.sqrt((x - xc)**2 + (y - yc)**2)

            if x < x_shock:
                MHD.dens[i, j] = rho_post
                MHD.vel1[i, j] = v1_post
                MHD.vel2[i, j] = 0.0
                MHD.pres[i, j] = p_post
            elif r < rc:
                MHD.dens[i, j] = rho_cloud
                MHD.vel1[i, j] = 0.0
                MHD.vel2[i, j] = 0.0
                MHD.pres[i, j] = p_amb
            else:
                MHD.dens[i, j] = rho_amb
                MHD.vel1[i, j] = 0.0
                MHD.vel2[i, j] = 0.0
                MHD.pres[i, j] = p_amb

    par.BC[:] = 'free'

    return grid, MHD, par, eos




def IC_MHD1D_Alfven(grid, MHD, par):
    """
    1D circularly polarised Alfven wave test.

    A circularly polarised Alfven wave propagates along a uniform
    background field. This is an exact nonlinear solution of the
    ideal MHD equations and provides an excellent convergence test.
    The wave should maintain its shape and amplitude indefinitely.

    Domain: [0, 1], periodic
    B_x = 1, B_y = 0.1*sin(2*pi*x), B_z = 0.1*cos(2*pi*x)
    v_y = -B_y/sqrt(rho), v_z = -B_z/sqrt(rho)  (forward Alfven wave)
    rho = 1, p = 0.1, Gamma = 5/3
    t_fin = 1 (one full period)

    Parameters
    ----------
    grid : object
    MHD : object
    par : object

    Returns
    -------
    grid, MHD, par, eos : objects

    References
    ----------
    Toth, G. (2000), J. Comput. Phys. 161, 605
    """
    print("1D circularly polarised Alfven wave test")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timefin = 1.0
    par.timenow = 0.0
    eos = EOSdata(5.0 / 3.0)

    rho0 = 1.0
    p0 = 0.1
    B0 = 1.0
    amp = 0.1

    MHD.dens[:, :] = rho0
    MHD.pres[:, :] = p0
    MHD.vel1[:, :] = 0.0

    MHD.bfi1[:, :] = B0
    MHD.fb1[:, :] = B0

    for i in range(grid.Nx1):
        x_face = grid.fx1[i + grid.Ngc, grid.Ngc]
        MHD.fb2[i, :] = amp * np.sin(2.0 * np.pi * x_face)

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r + 1):
            x = grid.cx1[i, j]
            MHD.bfi2[i, j] = amp * np.sin(2.0 * np.pi * x)
            MHD.bfi3[i, j] = amp * np.cos(2.0 * np.pi * x)
            # Forward-propagating Alfven wave
            MHD.vel2[i, j] = -MHD.bfi2[i, j] / np.sqrt(rho0)
            MHD.vel3[i, j] = -MHD.bfi3[i, j] / np.sqrt(rho0)

    par.BC[:] = 'peri'

    return grid, MHD, par, eos




