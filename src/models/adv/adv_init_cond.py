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


# ============================================================================
#   User-defined template
# ============================================================================
def IC_adv_user_defined(grid, adv, par):
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
    
    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timefin = 1.0; par.timenow = 0.0
    
    adv.dens[:, :] = 1.0
    adv.vel1 = 1.0; adv.vel2 = 1.0
    
    #boundary conditions
    #all support walls, axis, periodic and free-outflow boundaries, BC[0] supports axis for handling cylindrical problems
    par.BC[:] = 'peri'
                
    raise ValueError(
        "User-defined advection problem – see 'adv_init_cond.py', "
        "set your ICs and remove this line."
    )  
    
    return grid, adv, par



# ============================================================================
#   1D problems
# ============================================================================
def IC_adv1D_smooth(grid, adv, par):
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
    
    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 1.0
    
    #domain length 
    Len = grid.x1fin - grid.x1ini

    x0 = 0.3 # profile center 
    delta = 0.1 #profile semi-width
    
    adv.vel1 = 1.0; adv.vel2 = 0.0 #velocity 
    
    x = x0 - np.floor(x0/Len) * Len
    adv.dens[:, :] = np.exp(-(grid.cx1 - x)**2/delta**2) + \
    np.exp(-(grid.cx1 - x - np.sign(adv.vel1)*Len)**2/delta**2)

    #assume periodic domain,
    # i.e. matter, which leaves the domain, enters it from the other side
    par.BC[:] = 'peri'

    return grid, adv, par



def IC_adv1D_disc(grid, adv, par):
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
    
    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 1.0

    x0 = 0.3 # profile center 
    delta = 0.1 #profile semi-width
    
    adv.vel1 = 1.0; adv.vel2 = 0.0 #velocity 
    
    #discontinunous profile 
    adv.dens = np.where(grid.cx1 < x0 - delta, 1.0,
        np.where(grid.cx1 < x0 + delta, 2.0, 1.0))
                
    par.BC[:] = 'peri'

    return grid, adv, par



# ============================================================================
#   2D problems
# ============================================================================
def IC_adv2D_smooth(grid, adv, par):
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
    
    #grid creation 
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 1.0
    
    # center position
    x0 = x1ini + 0.5 * (x1fin - x1ini)
    y0 = x2ini + 0.5 * (x2fin - x2ini)

    delta = 0.1 #semi-width
    
    adv.vel1 = 1.0; adv.vel2 = 1.0 #advection velocity 
    
    #smooth gaussian profile 
    rad = np.sqrt((grid.cx1 - x0)**2 + (grid.cx2 - y0)**2)
    adv.dens[:, :] = np.exp(-rad**2/delta**2)            

    par.BC[:] = 'peri'
    
    return grid, adv, par



def IC_adv2D_disc(grid, adv, par):
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
    
    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 1.0

    rad0 = 0.1 #radius of the bump
    
    # center position
    x0 = x1ini + 0.5 * (x1fin - x1ini)
    y0 = x2ini + 0.5 * (x2fin - x2ini)
    
    adv.vel1 = 1.0; adv.vel2 = 1.0 #advection velocity 
    
    #discontinuous 2D profile 
    rad = np.sqrt((grid.cx1 - x0)**2 + (grid.cx2 - y0)**2)
    adv.dens[:, :] = np.where(rad < rad0, 1.0, 0.0)  
    
    par.BC[:] = 'peri'

    return grid, adv, par
