# -*- coding: utf-8 -*-
"""
Advection Initial Conditions Module

This module provides functions to set up simple 1D and 2D linear advection 
test problems. It initializes the advected quantity, velocity fields, 
and simulation time parameters.

Author: mrkondratyev
Date: June 14, 2024
"""

import numpy as np




def IC_advection_user_defined(grid, adv, par):
    """
    Initialize a linear advection problem according to initial conditions introduced by user.

    Parameters
    ----------
    grid : object
        Grid object containing cell coordinates and ghost cells.
    adv : object
        Advected state object with attribute `dens` (2D array of advected quantity)
        and velocity components `vel1` and `vel2`.
    par : object
        Simulation parameters including `timefin` and `timenow`.

    Returns
    -------
    grid, adv, par : objects
        Updated advected state and simulation parameters.

    Notes
    -----
    - The user is offered to adjust the initial and boundary conditions as well as other parameters here
    """
    print("Linear advection of user-defined profile")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    #grid.CylindricalGrid(x1ini, x1fin, x2ini, x2fin)
    
    adv.dens[:, :] = 0.0
    par.timefin = 1.0
    par.timenow = 0.0

    x0 = 0.5
    y0 = 0.5

    adv.vel1 = 1.0
    adv.vel2 = 1.0
    
    #boundary conditions
    #all support walls, periodic and free-outflow boundaries, BC[0] supports axis for handling cylindrical problems
    par.BC[:] = 'peri'

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            rad = np.sqrt((grid.fx1[i, j] - x0)**2 + (grid.fx2[i, j] - y0)**2)
            adv.dens[i,j] = grid.cx1[i,j] + grid.cx2[i,j] + 1.0 #......
                
    raise ValueError("User-defined advection problem, see file 'advection_init_cond.py', adjust ICs and delete this line.")
        
    return grid, adv, par




def IC_advection1D_smooth(grid, adv, par):
    """
    Initialize a 1D linear advection test problem with a smooth profile

    Parameters
    ----------
    grid : object
        Grid object containing cell coordinates and ghost cells.
    adv : object
        Advected state object with attribute `dens` (2D array of advected quantity)
        and velocity components `vel1` and `vel2`.
    par : object
        Simulation parameters including `timefin` and `timenow`.

    Returns
    -------
    grid, adv, par : objects
        Updated advected state and simulation parameters.

    Notes
    -----
    - The initial condition consists of a smooth Gaussian profile in x1.
    - Velocities are set to `vel1=1.0`, `vel2=0.0`.
    - The time integration will run from `timenow=0.0` to `timefin=1.0`.
    """
    print("Linear 1D advection of smooth profile")
    
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    adv.dens[:, :] = 0.0
    par.timefin = 1.0
    par.timenow = 0.0
    adv.vel1 = 1.0
    adv.vel2 = 0.0
    
    par.BC[:] = 'peri'
    
    Len = grid.x1fin - grid.x1ini

    x0 = 0.3
        
    #profile width
    delta = 0.1
    
    #предполагаем что наша область периодическая, 
    #т.е. вещество, что выходит с одной стороны, сразу входит в область с другой стороны с сохранением параметров
    x = x0 - np.floor(x0/Len) * Len

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            adv.dens[i, j] = np.exp(-(grid.cx1[i,j] - x)**2/delta**2) + \
            np.exp(-(grid.cx1[i,j] - x - np.sign(adv.vel1)*Len)**2/delta**2)

    return grid, adv, par




def IC_advection1D_disc(grid, adv, par):
    """
    Initialize a 1D linear advection test problem with a discontinuous profile

    Parameters
    ----------
    grid : object
        Grid object containing cell coordinates and ghost cells.
    adv : object
        Advected state object with attribute `dens` (2D array of advected quantity)
        and velocity components `vel1` and `vel2`.
    par : object
        Simulation parameters including `timefin` and `timenow`.

    Returns
    -------
    grid, adv, par : objects
        Updated advected state and simulation parameters.

    Notes
    -----
    - The initial condition consists of a piecewise constant profile in x1.
    - Velocities are set to `vel1=1.0`, `vel2=0.0`.
    - The time integration will run from `timenow=0.0` to `timefin=1.0`.
    """
    print("Linear 1D advection of discontinuous profile")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    adv.dens[:, :] = 0.0
    par.timefin = 1.0
    par.timenow = 0.0
    adv.vel1 = 1.0
    adv.vel2 = 0.0
    
    par.BC[:] = 'peri'

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.2:
                adv.dens[i, j] = 1.0
            elif grid.fx1[i, j] < 0.4:
                adv.dens[i, j] = 2.0
            else:
                adv.dens[i, j] = 1.0

    return grid, adv, par




def IC_advection2D_smooth(grid, adv, par):
    """
    Initialize a 2D linear advection test problem.

    Parameters
    ----------
    grid : object
        Grid object containing cell coordinates and ghost cells.
    adv : object
        Advected state object with attribute `dens` (2D array of advected quantity)
        and velocity components `vel1` and `vel2`.
    par : object
        Simulation parameters including `timefin` and `timenow`.

    Returns
    -------
    adv, par, grid : objects
        Updated advected state and simulation parameters.

    Notes
    -----
    - The initial condition consists of a circular region of high value 
      (`adv=1.0`) centered at (x0, y0) with radius `rad0=0.1`.
    - Outside the circle, the advected quantity is zero.
    - Velocities are set to `vel1=1.0`, `vel2=1.0`.
    - The time integration runs from `timenow=0.0` to `timefin=1.0`.
    """
    print("Linear 2D advection of smooth profile")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    adv.dens[:, :] = 0.0
    par.timefin = 1.0
    par.timenow = 0.0

    x0 = 0.5
    y0 = 0.5

    delta = 0.1
    
    adv.vel1 = 1.0
    adv.vel2 = 1.0
    
    par.BC[:] = 'peri'

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            rad = np.sqrt((grid.fx1[i, j] - x0)**2 + (grid.fx2[i, j] - y0)**2)
            adv.dens[i, j] = np.exp(-rad**2/delta**2)

    return grid, adv, par




def IC_advection2D_disc(grid, adv, par):
    """
    Initialize a 2D linear advection test problem.

    Parameters
    ----------
    grid : object
        Grid object containing cell coordinates and ghost cells.
    adv : object
        Advected state object with attribute `dens` (2D array of advected quantity)
        and velocity components `vel1` and `vel2`.
    par : object
        Simulation parameters including `timefin` and `timenow`.

    Returns
    -------
    grid, adv, par : objects
        Updated advected state and simulation parameters.

    Notes
    -----
    - The initial condition consists of a circular region of high value 
      (`adv=1.0`) centered at (x0, y0) with radius `rad0=0.1`.
    - Outside the circle, the advected quantity is zero.
    - Velocities are set to `vel1=1.0`, `vel2=1.0`.
    - The time integration runs from `timenow=0.0` to `timefin=1.0`.
    """
    print("Linear 2D advection of discontinuous profile")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    adv.dens[:, :] = 0.0
    par.timefin = 1.0
    par.timenow = 0.0

    rad0 = 0.1
    x0 = 0.5
    y0 = 0.5

    adv.vel1 = 1.0
    adv.vel2 = 1.0
    
    par.BC[:] = 'peri'

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            rad = np.sqrt((grid.fx1[i, j] - x0)**2 + (grid.fx2[i, j] - y0)**2)
            if rad < rad0:
                adv.dens[i, j] = 1.0
            else:
                adv.dens[i, j] = 0.0

    return grid, adv, par
