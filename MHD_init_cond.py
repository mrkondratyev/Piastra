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
from eos_setup import EOSdata




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

