# -*- coding: utf-8 -*-
"""
Initial conditions for 1D/2D hydrodynamics test problems.

This module provides functions to set up initial states for various standard 
benchmark tests used in compressible hydrodynamics simulations. Each function
initializes density, pressure, velocity, boundary conditions, and EOS parameters
for a specific test problem.

Available tests
---------------
1D:
    - Sod shock tube in various coordinate systems
    - Strong shock tube
    - Double blast wave (DBW)

2D:
    - Kelvin-Helmholtz instability (KH)
    - Rayleigh-Taylor instability (RT)
    - Cylindrical Sod shock tube in Cartesian domain
    - Planar Sedov-Taylor explosion
    - Cylindrical Sedov-Taylor explosion
    
aux:
    -user-defined 

Notes
-----
- Each function returns updated `fluid` and `par` objects, as well as an EOS object.
- Boundary conditions are set via the `par.BC` array.
- The `EOSdata` class is used to initialize the adiabatic index (gamma).
- Grid data is filled inside these routines.

Author
------
mrkondratyev
"""

import numpy as np
from eos_setup import EOSdata





def IC_hydro_user_defined(grid, fluid, par):
    """
    Initialize user-defined problem.

    Parameters
    ----------
    grid : object
        Grid object.
    fluid : object
        FluidState object to be initialized.
    par : object
        Simulation parameters including BC, timefin, timenow.

    Returns
    -------
    grid, fluid, par, eos : objects
        Updated grid data, fluid state, parameters, and EOS object.

    """
    print("user-defined problem for hydrodynamics")
    
    x1ini, x1fin = 0.0, 0.5
    x2ini, x2fin = 0.0, 0.5
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    #grid.CylindricalGrid(x1ini, x1fin, x2ini, x2fin)
    
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    fluid.dens[:, :] = 1.0
    eos = EOSdata(7.0/5.0)
    par.timefin = 0.2
    par.timenow = 0.0

    volume = 0.0
    rad0 = 0.02
    energ = 0.25

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            rad = np.sqrt(grid.fx1[i, j]**2 + grid.fx2[i, j]**2)
            if rad < rad0:
                volume += grid.cVol[i, j]

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            rad = np.sqrt(grid.fx1[i, j]**2 + grid.fx2[i, j]**2)
            if rad < rad0:
                fluid.pres[i, j] = (eos.GAMMA - 1.0) * energ / volume
            else:
                fluid.pres[i, j] = 1e-4
       
    #boundary conditions
    #all support walls, periodic and free-outflow boundaries, BC[0] supports axis for handling cylindrical problems
    par.BC[0] = 'wall'
    par.BC[1] = 'wall'
    par.BC[2] = 'free'
    par.BC[3] = 'free'
    
    raise ValueError("User-defined hydro problem, see file 'hydro_init_cond.py', adjust ICs and delete this line.")
    
    return grid, fluid, par, eos




def IC_hydro1D_Sod(grid, fluid, par, geom):
    """
    Initialize the 1D Sod shock tube test in various geometries.

    Parameters
    ----------
    grid : object
        Grid object with Nx1r, Nx2r, fx1, fx2, etc.
    fluid : object
        FluidState object containing vel1, vel2, vel3, dens, pres.
    par : object
        Simulation parameters object with BC, timefin, timenow.

    Returns
    -------
    grid : object
        Grid object containing filled arrays of grid data.
    fluid : object
        Updated fluid state.
    par : object
        Updated simulation parameters.
    eos : object
        Equation of state object with gamma=1.4.

    Notes
    -----
    The domain is divided at x=0.5. Left state: rho=1, p=1; right state: rho=0.125, p=0.1.
    Boundary conditions are set to 'wall'.
    """
    
    
    print("1D Sod shock tube test (G.A. Sod (1978))")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    
    #filling the grid arrays with grid data 
    if geom == 'cart':
        grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    if geom == 'cyl':
        grid.CylindricalGrid(x1ini, x1fin, x2ini, x2fin)
    if geom == 'pol':
        grid.PolarGrid(x1ini, x1fin, x2ini, x2fin)
        
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    par.timefin = 0.2
    par.timenow = 0.0
    eos = EOSdata(7.0/5.0)

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.5:
                fluid.dens[i, j] = 1.0
                fluid.pres[i, j] = 1.0
            else:
                fluid.dens[i, j] = 0.125
                fluid.pres[i, j] = 0.1

    par.BC[:] = 'wall'
    
    return grid, fluid, par, eos




def IC_hydro1D_strong_shock(grid, fluid, par):
    """
    Initialize a 1D strong shock tube test in Cartesian coordinates.

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects

    Notes
    -----
    Left state: rho=1, p=1000; right state: rho=1, p=0.01. Boundary conditions: wall.
    """
    print("1D Shock tube test with a strong shock")
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    
    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    par.timefin = 0.008
    par.timenow = 0.0
    eos = EOSdata(7.0/5.0)

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.5:
                fluid.dens[i, j] = 1.0
                fluid.pres[i, j] = 1000.0
            else:
                fluid.dens[i, j] = 1.0
                fluid.pres[i, j] = 0.01

    par.BC[:] = 'wall'
    return grid, fluid, par, eos




def IC_hydro1D_DBW(grid, fluid, par):
    """
    Initialize the 1D double blast wave test (Woodward & Colella 1984) in Cartesian coordinates.

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects

    Notes
    -----
    Initial pressure distribution: high-low-high across the domain.
    """
    print("1D Double blast wave test by Woodward and Collela (1984)")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    
    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    fluid.dens[:, :] = 1.0
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    par.timefin = 0.038
    par.timenow = 0.0
    eos = EOSdata(7.0/5.0)

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.1:
                fluid.pres[i, j] = 1000.0
            elif grid.fx1[i, j] < 0.9:
                fluid.pres[i, j] = 0.01
            else:
                fluid.pres[i, j] = 100.0

    par.BC[:] = 'wall'
    
    return grid, fluid, par, eos




def IC_hydro2D_KHI(grid, fluid, par):
    """
    Initialize the 2D Kelvin-Helmholtz instability.

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects

    Notes
    -----
    Sets a shear velocity profile with small sinusoidal perturbation in vel1.
    Boundary conditions: wall-peri-wall-peri.
    """
    print("Kelvin-Helmholtz instability in 2D")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    
    #filling the grid arrays with grid data 
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    fluid.vel3[:,:] = 0.0
    fluid.pres[:,:] = 2.5
    eos = EOSdata(5.0/3.0)
    par.timefin = 2.0
    par.timenow = 0.0

    sigma1 = 0.05/np.sqrt(2.0)
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if np.abs(grid.fx2[i, j] - 0.5) > 0.25:
                fluid.vel1[i, j] = -0.5
                fluid.dens[i, j] = 1.0
            else:
                fluid.vel1[i, j] = 0.5
                fluid.dens[i, j] = 2.0
            fluid.vel2[i,j] = 0.1*np.sin(4.0*np.pi*grid.cx1[i, j])*\
                (np.exp(-(grid.cx2[i, j] - 0.25)**2/2/sigma1**2)+\
                 np.exp(-(grid.cx2[i, j] - 0.75)**2/2/sigma1**2))

    par.BC[0] = 'peri'
    par.BC[1] = 'wall'
    par.BC[2] = 'peri'
    par.BC[3] = 'wall'
    
    return grid, fluid, par, eos




def IC_hydro2D_RTI(grid, fluid, par):
    """
    Initialize the 2D Rayleigh-Taylor instability problem.

    Parameters
    ----------
    grid : object
        Grid object used to create the domain.
    fluid : object
        FluidState object to be initialized.
    par : object
        Simulation parameters including BC, timefin, timenow.

    Returns
    -------
    grid, fluid, par, eos : objects
        Updated grid data, fluid state, parameters, and EOS object.

    Notes
    -----
    - Sets up a two-layer fluid with heavier fluid on top of lighter fluid.
    - Applies a small interface perturbation for instability growth.
    - Hydrostatic equilibrium is satisfied in the vertical direction.
    - Boundary conditions: wall-peri-wall-peri.
    """
    print("Rayleigh-Taylor instability in 2D")
    
    x1ini, x1fin = -0.5, 0.5
    x2ini, x2fin = -1.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0

    eos = EOSdata(7.0/5.0)

    rho_u, rho_d = 2.0, 1.0
    g_ff = -0.5
    P0 = 10.0/7.0 + 0.25
    P1 = 10.0/7.0 - 0.25

    fluid.F1[:, :] = 0.0
    fluid.F2[:, :] = g_ff
    par.timefin = 5.0
    par.timenow = 0.0

    h0 = 0.03
    kappa = 2.0 * np.pi

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx2[i, j] > h0 * np.cos(grid.cx1[i, j] * kappa + np.pi):
                fluid.dens[i, j] = rho_u
                fluid.pres[i, j] = P1 + grid.cx2[i, j] * g_ff * rho_u
            else:
                fluid.dens[i, j] = rho_d
                fluid.pres[i, j] = P0 + (grid.cx2[i, j] + 1.0) * g_ff * rho_d

    par.BC[0] = 'peri'
    par.BC[1] = 'wall'
    par.BC[2] = 'peri'
    par.BC[3] = 'wall'
    
    return grid, fluid, par, eos




def IC_hydro2D_Sod(grid, fluid, par):
    """
    Initialize the 2D cylindrical Sod shock tube problem (quadrant symmetry).
    in Cartesian coordinates 
    
    Parameters
    ----------
    grid : object
        Grid object.
    fluid : object
        FluidState object to be initialized.
    par : object
        Simulation parameters including BC, timefin, timenow.

    Returns
    -------
    grid, fluid, par, eos : objects
        Updated grid data, fluid state, parameters, and EOS object.

    Notes
    -----
    - Uses radial symmetry: radius = sqrt(x^2 + y^2).
    - Inner region (r < 0.5): rho=1, p=1; outer region: rho=0.125, p=0.1.
    - Velocity is zero everywhere initially.
    - Boundary conditions: wall-wall-free-free.
    """
    print("Cylindrical 2D Sod shock tube test in Cartesian geometry")
    
    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    eos = EOSdata(7.0/5.0)
    par.timefin = 0.2
    par.timenow = 0.0

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            rad = np.sqrt(grid.fx1[i, j]**2 + grid.fx2[i, j]**2)
            if rad < 0.5:
                fluid.dens[i, j] = 1.0
                fluid.pres[i, j] = 1.0
            else:
                fluid.dens[i, j] = 0.125
                fluid.pres[i, j] = 0.1

    par.BC[0] = 'wall'
    par.BC[1] = 'wall'
    par.BC[2] = 'free'
    par.BC[3] = 'free'
    
    return grid, fluid, par, eos




def IC_hydro2D_Sedov_cart(grid, fluid, par):
    """
    Initialize the 2D Sedov-Taylor explosion test in Cartesian coordinates.

    Parameters
    ----------
    grid : object
        Grid object.
    fluid : object
        FluidState object to be initialized.
    par : object
        Simulation parameters including BC, timefin, timenow.

    Returns
    -------
    grid, fluid, par, eos : objects
        Updated grid data, fluid state, parameters, and EOS object.

    Notes
    -----
    - Sets initial energy in a small circular region at the origin.
    - Outer region density set to 1.0, pressure near zero.
    - Velocity initially zero everywhere.
    - Boundary conditions: wall-wall-free-free.
    - Uses quadrant symmetry.
    """
    print("Flat 2D Sedov-Taylor explosion test in Cartesian geometry")
    
    x1ini, x1fin = 0.0, 0.5
    x2ini, x2fin = 0.0, 0.5
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    fluid.dens[:, :] = 1.0
    eos = EOSdata(7.0/5.0)
    par.timefin = 0.2
    par.timenow = 0.0

    volume = 0.0
    rad0 = 0.02
    energ = 0.25

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            rad = np.sqrt(grid.fx1[i, j]**2 + grid.fx2[i, j]**2)
            if rad < rad0:
                volume += grid.cVol[i, j]

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            rad = np.sqrt(grid.fx1[i, j]**2 + grid.fx2[i, j]**2)
            if rad < rad0:
                fluid.pres[i, j] = (eos.GAMMA - 1.0) * energ / volume
            else:
                fluid.pres[i, j] = 1e-4

    par.BC[0] = 'wall'
    par.BC[1] = 'wall'
    par.BC[2] = 'free'
    par.BC[3] = 'free'
    
    return grid, fluid, par, eos




def IC_hydro2D_Sedov_cyl(grid, fluid, par):
    """
    Initialize the 3D Sedov-Taylor explosion test in Cartesian coordinates.

    Parameters
    ----------
    grid : object
        Grid object.
    fluid : object
        FluidState object to be initialized.
    par : object
        Simulation parameters including BC, timefin, timenow.

    Returns
    -------
    grid, fluid, par, eos : objects
        Updated grid data, fluid state, parameters, and EOS object.

    Notes
    -----
    - Sets initial energy in a small circular region at the origin.
    - Outer region density set to 1.0, pressure near zero.
    - Velocity initially zero everywhere.
    - Boundary conditions: wall-wall-free-free.
    - Uses quadrant symmetry.
    """
    print("Sedov-Taylor explosion test in Cylindrical (R,Z) geometry")
    
    x1ini, x1fin = 0.0, 0.5
    x2ini, x2fin = 0.0, 0.5
    grid.CylindricalGrid(x1ini, x1fin, x2ini, x2fin)
    
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    fluid.dens[:, :] = 1.0
    eos = EOSdata(7.0/5.0)
    par.timefin = 0.2
    par.timenow = 0.0

    volume = 0.0
    rad0 = 0.02
    energ = 0.25

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            rad = np.sqrt(grid.fx1[i, j]**2 + grid.fx2[i, j]**2)
            if rad < rad0:
                volume += grid.cVol[i, j]

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            rad = np.sqrt(grid.fx1[i, j]**2 + grid.fx2[i, j]**2)
            if rad < rad0:
                fluid.pres[i, j] = (eos.GAMMA - 1.0) * energ / volume
            else:
                fluid.pres[i, j] = 1e-4

    par.BC[0] = 'wall'
    par.BC[1] = 'wall'
    par.BC[2] = 'free'
    par.BC[3] = 'free'
    
    return grid, fluid, par, eos




