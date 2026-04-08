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
    - Noh problem (Noh 1987) – infinite-strength shock
    - Shu-Osher problem (Shu & Osher 1989) – shock-entropy wave interaction
    - Einfeldt rarefaction (Einfeldt et al. 1991) – 1-2-3 problem

2D:
    - Kelvin-Helmholtz instability (KH)
    - Rayleigh-Taylor instability (RT)
    - Cylindrical Sod shock tube in Cartesian domain
    - Planar Sedov-Taylor explosion
    - Cylindrical Sedov-Taylor explosion
    - Four-quadrant 2D Riemann problem (Lax & Liu 1998)
    - Implosion problem (Liska & Wendroff 2003)
    - Double Mach reflection (Woodward & Colella 1984)
    - Gresho vortex (Gresho & Chan 1990) – angular momentum preservation
    - Shock-cloud interaction (Klein et al. 1994) – astrophysical
    - Gap opening in protoplanetary disk (polar coordinates) – astrophysical
    - Axisymmetric non-relativistic jet (cylindrical coordinates) – astrophysical

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
from src.common.eos_setup import EOSdata





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
    Initialize the 3D Sedov-Taylor explosion test in 2D Cylindrical coordinates.

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
    print("3D Sedov-Taylor explosion test in Cylindrical (R,Z) geometry")
    
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




def IC_hydro1D_Noh(grid, fluid, par):
    """
    Initialize the 1D Noh problem (Noh 1987).

    Two uniform cold streams collide at the origin, producing an
    infinite-strength accretion shock propagating outward. This is a
    severe test for the numerical treatment of wall heating artefacts.

    Initial state: rho=1, |v|=1 (inward), p=1e-6 (nearly zero)
    Gamma = 5/3

    The exact post-shock density is (Gamma+1)/(Gamma-1) = 4 for Gamma=5/3.

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    Noh, W. F. (1987), J. Comput. Phys. 72, 78
    """
    print("1D Noh problem (Noh 1987)")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    fluid.dens[:, :] = 1.0
    fluid.vel1[:, :] = -1.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    fluid.pres[:, :] = 1e-6
    par.timefin = 0.5
    par.timenow = 0.0
    eos = EOSdata(5.0 / 3.0)

    par.BC[0] = 'wall'
    par.BC[1] = 'free'
    par.BC[2] = 'wall'
    par.BC[3] = 'free'

    return grid, fluid, par, eos




def IC_hydro1D_ShuOsher(grid, fluid, par):
    """
    Initialize the 1D Shu-Osher problem (Shu & Osher 1989).

    A Mach-3 shock interacts with a sinusoidal density perturbation.
    This test evaluates the ability of numerical schemes to resolve
    small-scale smooth features behind a strong shock.

    Domain: x in [-5, 5]
    Left  state (x < -4): rho=3.857143, v=2.629369, p=10.33333
    Right state (x > -4): rho=1+0.2*sin(5x), v=0, p=1

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    Shu, C.-W. & Osher, S. (1989), J. Comput. Phys. 83, 32
    """
    print("1D Shu-Osher shock-entropy wave interaction (Shu & Osher 1989)")

    x1ini, x1fin = -5.0, 5.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    par.timefin = 1.8
    par.timenow = 0.0
    eos = EOSdata(7.0 / 5.0)

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < -4.0:
                fluid.dens[i, j] = 3.857143
                fluid.vel1[i, j] = 2.629369
                fluid.pres[i, j] = 10.33333
            else:
                fluid.dens[i, j] = 1.0 + 0.2 * np.sin(5.0 * grid.cx1[i, j])
                fluid.vel1[i, j] = 0.0
                fluid.pres[i, j] = 1.0

    par.BC[:] = 'free'

    return grid, fluid, par, eos




def IC_hydro2D_RP2D(grid, fluid, par):
    """
    Initialize a 2D Riemann problem (Configuration 3 from Lax & Liu 1998).

    The unit square is divided into four quadrants with different constant
    states. The interaction of shocks, contacts, and rarefactions produces
    complex 2D wave structures including spiralling slip lines.

    +-----------+-----------+
    |  rho=0.53 |  rho=1.5  |
    |  v=1.206  |  v=0      |
    |  u=0      |  u=0      |
    |  p=0.3    |  p=1.5    |
    |  (x<0.5,  |  (x>0.5,  |
    |   y>0.5)  |   y>0.5)  |
    +-----------+-----------+
    |  rho=0.14 |  rho=0.53 |
    |  v=1.206  |  v=0      |
    |  u=1.206  |  u=1.206  |
    |  p=0.029  |  p=0.3    |
    |  (x<0.5,  |  (x>0.5,  |
    |   y<0.5)  |   y<0.5)  |
    +-----------+-----------+

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    Lax, P. D. & Liu, X.-D. (1998), SIAM J. Sci. Comput. 19, 319
    """
    print("2D Riemann problem (Lax & Liu 1998, Configuration 3)")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    fluid.vel3[:, :] = 0.0
    eos = EOSdata(7.0 / 5.0)
    par.timefin = 0.3
    par.timenow = 0.0

    cx1 = grid.cx1
    cx2 = grid.cx2

    # Quadrant I: x > 0.5, y > 0.5
    m1 = (cx1 >= 0.5) & (cx2 >= 0.5)
    # Quadrant II: x < 0.5, y > 0.5
    m2 = (cx1 < 0.5) & (cx2 >= 0.5)
    # Quadrant III: x < 0.5, y < 0.5
    m3 = (cx1 < 0.5) & (cx2 < 0.5)
    # Quadrant IV: x > 0.5, y < 0.5
    m4 = (cx1 >= 0.5) & (cx2 < 0.5)

    fluid.dens[:, :] = 1.5 * m1 + 0.5323 * m2 + 0.138 * m3 + 0.5323 * m4
    fluid.vel1[:, :] = 0.0 * m1 + 1.206 * m2 + 1.206 * m3 + 0.0 * m4
    fluid.vel2[:, :] = 0.0 * m1 + 0.0 * m2 + 1.206 * m3 + 1.206 * m4
    fluid.pres[:, :] = 1.5 * m1 + 0.3 * m2 + 0.029 * m3 + 0.3 * m4

    par.BC[:] = 'free'

    return grid, fluid, par, eos




def IC_hydro2D_implosion(grid, fluid, par):
    """
    Initialize the 2D implosion problem (Liska & Wendroff 2003).

    A diamond-shaped low-pressure, low-density region is placed inside
    a higher-pressure medium. The converging shock forms a jet along
    the diagonal. This problem is a stringent test for symmetry
    preservation of the numerical scheme.

    Domain: [0, 0.3] x [0, 0.3]
    Inside  (x+y < 0.15): rho=0.125, p=0.14
    Outside (x+y > 0.15): rho=1,     p=1

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    Liska, R. & Wendroff, B. (2003), SIAM J. Sci. Comput. 25, 995
    """
    print("2D implosion problem (Liska & Wendroff 2003)")

    x1ini, x1fin = 0.0, 0.3
    x2ini, x2fin = 0.0, 0.3
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    eos = EOSdata(7.0 / 5.0)
    par.timefin = 2.5
    par.timenow = 0.0

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.cx1[i, j] + grid.cx2[i, j] < 0.15:
                fluid.dens[i, j] = 0.125
                fluid.pres[i, j] = 0.14
            else:
                fluid.dens[i, j] = 1.0
                fluid.pres[i, j] = 1.0

    par.BC[:] = 'wall'

    return grid, fluid, par, eos




def IC_hydro1D_Einfeldt(grid, fluid, par):
    """
    Initialize the 1D Einfeldt rarefaction test (1-2-3 problem).

    Two symmetric rarefaction waves propagate outward from the centre,
    leaving a near-vacuum in between.  This problem is a severe test for
    positivity preservation and for the carbuncle/entropy-fix behaviour
    of Riemann solvers (especially Roe).

    Left  state: rho=1, v=-2, p=0.4
    Right state: rho=1, v= 2, p=0.4
    Gamma = 7/5, t_fin = 0.15

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    Einfeldt, B., Munz, C. D., Roe, P. L. & Sjogreen, B. (1991),
    J. Comput. Phys. 92, 273
    """
    print("1D Einfeldt rarefaction test (1-2-3 problem)")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    par.timefin = 0.15
    par.timenow = 0.0
    eos = EOSdata(7.0 / 5.0)

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.5:
                fluid.dens[i, j] = 1.0
                fluid.vel1[i, j] = -2.0
                fluid.pres[i, j] = 0.4
            else:
                fluid.dens[i, j] = 1.0
                fluid.vel1[i, j] = 2.0
                fluid.pres[i, j] = 0.4

    par.BC[:] = 'free'

    return grid, fluid, par, eos




def IC_hydro2D_DMR(grid, fluid, par):
    """
    Initialize the 2D double Mach reflection problem (Woodward & Colella 1984).

    A Mach-10 shock in air (Gamma=7/5) impinges on a reflecting wedge
    at a 60-degree angle. The problem produces a complex pattern of
    reflected shocks, Mach stems, contact discontinuities, and a jet
    along the wall. It is one of the most demanding standard
    benchmarks for compressible flow solvers.

    Domain: [0, 4] x [0, 1]
    The initial shock runs from (x1=1/6, x2=0) at 60 degrees.
    Post-shock (left of the shock):  rho=8, v1=8.25*cos(pi/6),
                                     v2=-8.25*sin(pi/6), p=116.5
    Pre-shock  (right of the shock): rho=1.4, v=0, p=1

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    Woodward, P. & Colella, P. (1984), J. Comput. Phys. 54, 115
    """
    print("2D double Mach reflection (Woodward & Colella 1984)")

    x1ini, x1fin = 0.0, 3.25
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timefin = 0.2
    par.timenow = 0.0
    eos = EOSdata(7.0 / 5.0)

    # Post-shock state (Mach 10 shock at 60 degrees)
    rho_post = 8.0
    v1_post = 8.25 * np.cos(np.pi / 6.0)
    v2_post = -8.25 * np.sin(np.pi / 6.0)
    p_post = 116.5

    # Pre-shock state
    rho_pre = 1.4
    p_pre = 1.0

    x_shock_origin = 1.0 / 6.0

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            x = grid.cx1[i, j]
            y = grid.cx2[i, j]
            # Shock position: x = x_shock_origin + y / tan(60deg)
            x_shock = x_shock_origin + y / np.tan(np.pi / 3.0)
            if x < x_shock:
                fluid.dens[i, j] = rho_post
                fluid.vel1[i, j] = v1_post
                fluid.vel2[i, j] = v2_post
                fluid.pres[i, j] = p_post
            else:
                fluid.dens[i, j] = rho_pre
                fluid.vel1[i, j] = 0.0
                fluid.vel2[i, j] = 0.0
                fluid.pres[i, j] = p_pre

    fluid.vel3[:, :] = 0.0

    # Bottom: wall (reflecting wedge), Top: free (inflow handled by IC)
    # Left: free, Right: free
    par.BC[0] = 'free'
    par.BC[1] = 'wall'
    par.BC[2] = 'free'
    par.BC[3] = 'free'

    return grid, fluid, par, eos




def IC_hydro2D_Gresho(grid, fluid, par):
    """
    Initialize the 2D Gresho vortex (Gresho & Chan 1990).

    A stationary isentropic vortex in exact rotational equilibrium.
    The azimuthal velocity increases linearly for r < 0.2, then
    decreases linearly for 0.2 < r < 0.4, and vanishes for r > 0.4.
    Pressure is set to balance centripetal acceleration exactly.

    This problem has an exact steady-state solution and tests the
    scheme's ability to preserve angular momentum and vortex
    structures over long integration times.

    Domain: [0, 1] x [0, 1], periodic
    Gamma = 7/5, t_fin = 3 (several turnover times)

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    Gresho, P. M. & Chan, S. T. (1990), Int. J. Numer. Methods Fluids 11, 621
    Liska, R. & Wendroff, B. (2003), SIAM J. Sci. Comput. 25, 995
    """
    print("2D Gresho vortex (Gresho & Chan 1990)")

    x1ini, x1fin = 0.0, 1.0
    x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timefin = 3.0
    par.timenow = 0.0
    eos = EOSdata(7.0 / 5.0)

    x0, y0 = 0.5, 0.5

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            dx = grid.cx1[i, j] - x0
            dy = grid.cx2[i, j] - y0
            r = np.sqrt(dx**2 + dy**2)

            if r < 1e-14:
                fluid.vel1[i, j] = 0.0
                fluid.vel2[i, j] = 0.0
                fluid.pres[i, j] = 5.0 + 12.5 * r**2
            elif r < 0.2:
                v_phi = 5.0 * r
                fluid.pres[i, j] = 5.0 + 12.5 * r**2
                fluid.vel1[i, j] = -v_phi * dy / r
                fluid.vel2[i, j] = v_phi * dx / r
            elif r < 0.4:
                v_phi = 2.0 - 5.0 * r
                fluid.pres[i, j] = (9.0 + 12.5 * r**2
                                    - 20.0 * r + 4.0 * np.log(r / 0.2))
                fluid.vel1[i, j] = -v_phi * dy / r
                fluid.vel2[i, j] = v_phi * dx / r
            else:
                fluid.vel1[i, j] = 0.0
                fluid.vel2[i, j] = 0.0
                fluid.pres[i, j] = 3.0 + 4.0 * np.log(2.0)

            fluid.dens[i, j] = 1.0

    fluid.vel3[:, :] = 0.0

    par.BC[:] = 'peri'

    return grid, fluid, par, eos




def IC_hydro2D_shock_cloud(grid, fluid, par):
    """
    Initialize the 2D shock-cloud interaction problem.

    A Mach-10 shock impacts a dense circular cloud (density ratio
    chi = 10). The cloud is disrupted by Richtmyer-Meshkov and
    Kelvin-Helmholtz instabilities, producing complex vortical
    structures. This is a classic astrophysical benchmark modelling
    e.g. supernova remnant / interstellar cloud interactions.

    Domain: [0, 1] x [0, 1]
    Cloud: centre (0.25, 0.5), radius 0.1, rho=10, p=1
    Pre-shock (ambient): rho=1, p=1, v=0
    Post-shock (x < 0.05): rho=3.86, v1=11.2, p=167
    Gamma = 5/3, t_fin = 0.06

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    Klein, R. I., McKee, C. F. & Colella, P. (1994), ApJ 420, 213
    """
    print("2D shock-cloud interaction (Klein et al. 1994)")

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

    # Post-shock state (Mach 10 shock in gamma=5/3 gas)
    rho_post = 3.857143
    v1_post = 11.2
    p_post = 167.0

    x_shock = 0.05  # initial shock position

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            x = grid.cx1[i, j]
            y = grid.cx2[i, j]
            r = np.sqrt((x - xc)**2 + (y - yc)**2)

            if x < x_shock:
                # Post-shock region
                fluid.dens[i, j] = rho_post
                fluid.vel1[i, j] = v1_post
                fluid.pres[i, j] = p_post
            elif r < rc:
                # Dense cloud
                fluid.dens[i, j] = rho_cloud
                fluid.vel1[i, j] = 0.0
                fluid.pres[i, j] = p_amb
            else:
                # Pre-shock ambient
                fluid.dens[i, j] = rho_amb
                fluid.vel1[i, j] = 0.0
                fluid.pres[i, j] = p_amb

    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0

    par.BC[0] = 'free'
    par.BC[1] = 'free'
    par.BC[2] = 'free'
    par.BC[3] = 'free'

    return grid, fluid, par, eos




def IC_hydro2D_gap_opening(grid, fluid, par):
    """
    Initialize a 2D gap-opening problem in a protoplanetary disk (polar coords).

    A Jupiter-mass planet orbits in a locally isothermal Keplerian disk.
    The planet's gravity carves an annular gap through tidal torques.
    This is the standard test for planet-disk interaction codes.

    Coordinate system: polar (R, phi)
    Domain: R in [0.4, 2.5], phi in [0, 2*pi]
    Stellar mass M_star = 1 (at origin)
    Planet mass  M_p = 1e-3 (at R=1, phi=pi)
    Disk: Sigma ~ R^(-0.5), Keplerian rotation, locally isothermal h/R=0.05
    Gamma = 7/5, t_fin = 10 (orbital periods at R=1)

    Source terms F1, F2 encode stellar + planetary gravity.

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    de Val-Borro, M. et al. (2006), MNRAS 370, 529
    """
    print("2D gap-opening problem in a protoplanetary disk (polar)")

    R_in, R_out = 0.4, 2.5
    phi_in, phi_out = 0.0, 2.0 * np.pi
    grid.PolarGrid(R_in, R_out, phi_in, phi_out)

    par.timenow = 0.0
    par.timefin = 10.0 * 2.0 * np.pi   # 10 orbits at R=1
    eos = EOSdata(7.0 / 5.0)

    # Disk parameters
    Sigma0 = 1.0        # reference surface density
    p_index = -0.5       # surface density power-law index
    h_over_r = 0.05      # disk aspect ratio

    # Planet parameters
    M_star = 1.0
    M_planet = 1.0e-3
    R_planet = 1.0
    phi_planet = np.pi
    eps = 0.6 * h_over_r * R_planet   # gravitational softening

    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            R = grid.cx1[i, j]
            phi = grid.cx2[i, j]

            # Surface density profile
            Sigma = Sigma0 * R**p_index

            # Keplerian velocity: v_phi = sqrt(GM/R)
            v_phi = np.sqrt(M_star / R)

            # Locally isothermal sound speed: c_s = h/R * v_K
            cs = h_over_r * v_phi

            # Pressure from isothermal EOS: P = Sigma * cs^2
            pres = Sigma * cs**2

            fluid.dens[i, j] = Sigma
            fluid.vel1[i, j] = 0.0         # v_R = 0
            fluid.vel2[i, j] = v_phi        # v_phi = Keplerian
            fluid.vel3[i, j] = 0.0
            fluid.pres[i, j] = pres

            # Gravitational source terms
            # Stellar gravity: g_R = -GM/R^2
            g_R_star = -M_star / R**2

            # Planet gravity (softened)
            dx = R * np.cos(phi) - R_planet * np.cos(phi_planet)
            dy = R * np.sin(phi) - R_planet * np.sin(phi_planet)
            d = np.sqrt(dx**2 + dy**2 + eps**2)
            g_x_planet = -M_planet * dx / d**3
            g_y_planet = -M_planet * dy / d**3

            # Convert planet gravity to polar components
            g_R_planet = g_x_planet * np.cos(phi) + g_y_planet * np.sin(phi)
            g_phi_planet = (-g_x_planet * np.sin(phi) + g_y_planet * np.cos(phi))

            # Centrifugal correction for rotating frame is handled by the solver
            fluid.F1[i - grid.Ngc, j - grid.Ngc] = Sigma * (g_R_star + g_R_planet)
            fluid.F2[i - grid.Ngc, j - grid.Ngc] = Sigma * g_phi_planet

    par.BC[0] = 'free'
    par.BC[1] = 'peri'
    par.BC[2] = 'free'
    par.BC[3] = 'peri'

    return grid, fluid, par, eos




def IC_hydro2D_jet_cyl(grid, fluid, par):
    """
    Initialize an axisymmetric non-relativistic jet in cylindrical (R,Z) coords.

    A supersonic jet (Mach 6) is injected from the bottom boundary
    along the symmetry axis into a uniform ambient medium. The jet
    develops a bow shock, cocoon, and Mach disk structure.

    Coordinate system: cylindrical (R, Z)
    Domain: R in [0, 5], Z in [0, 20]
    Jet nozzle: R < 1 at Z = 0, rho=1, v_z=6, p=1/(gamma*Mach^2)
    Ambient:    rho=10, v=0, p=1/(gamma*Mach^2)
    Density ratio eta = rho_jet/rho_amb = 0.1

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    Norman, M. L. et al. (1982), A&A 113, 285
    Bodo, G. et al. (1998), A&A 333, 1117
    """
    print("2D axisymmetric non-relativistic jet (cylindrical)")

    R_in, R_out = 0.0, 5.0
    Z_in, Z_out = 0.0, 20.0
    grid.CylindricalGrid(R_in, R_out, Z_in, Z_out)

    par.timenow = 0.0
    par.timefin = 15.0
    eos = EOSdata(5.0 / 3.0)

    # Jet parameters
    Mach = 6.0
    rho_jet = 1.0
    rho_amb = 10.0    # density ratio eta = 0.1
    v_jet = Mach      # normalised so cs_jet ~ 1
    p_amb = 1.0 / (eos.GAMMA * Mach**2) * rho_jet * v_jet**2
    p_amb = 1.0       # uniform pressure
    r_jet = 1.0       # jet radius

    # Ambient medium
    fluid.dens[:, :] = rho_amb
    fluid.pres[:, :] = p_amb
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0

    # Jet nozzle at Z = 0 (first few active cells and ghost cells)
    for i in range(0, grid.Ngc + 3):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.cx1[i, j] < r_jet:
                fluid.dens[i, j] = rho_jet
                fluid.vel2[i, j] = v_jet
                fluid.pres[i, j] = p_amb

    par.BC[0] = 'axis'
    par.BC[1] = 'wall'
    par.BC[2] = 'free'
    par.BC[3] = 'free'

    return grid, fluid, par, eos




