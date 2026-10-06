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
    - Shu-Osher problem (Shu & Osher 1989) – shock-entropy wave interaction
    - Einfeldt rarefaction (Einfeldt et al. 1991) – 1-2-3 problem

2D:
    - Kelvin-Helmholtz instability (KH)
    - Rayleigh-Taylor instability (RT)
    - Cylindrical Sod shock tube in Cartesian domain
    - Planar Sedov-Taylor explosion
    - Cylindrical Sedov-Taylor explosion
    - Four-quadrant 2D Riemann problem (Lax & Liu 1998)
    - Gresho vortex (Gresho & Chan 1990) – angular momentum preservation
    - Isentropic vortex - convergence 
    - Shock-cloud interaction (Klein et al. 1994) – astrophysical
    - Gap opening in protoplanetary disk (polar coordinates) – astrophysical
    - Axisymmetric non-relativistic jet (cylindrical coordinates) – astrophysical
    - Jeans instability in 1D 
    - Gravitational collapse of a uniform cloud (Colgate & White (1966))
    - Merger of two gravitating blobs in 2D cartesian geometry (setup is close to ULULA code)

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
from src.gravity import selfgravity_poisson, planet_gravity_polar


def IC_HD_user_defined(grid, fluid, par):
    """
    Initialize user-defined problem.

    Parameters
    ----------
    grid : object
        Grid object.
    fluid : object
        SimState object to be initialized.
    par : object
        Simulation parameters including BC, timefin, timenow.

    Returns
    -------
    grid, fluid, par, eos : objects
        Updated grid data, fluid state, parameters, and EOS object.

    """
    print("user-defined problem for hydrodynamics")
    
    #define the grid 
    x1ini, x1fin = 0.0, 0.5; x2ini, x2fin = 0.0, 0.5
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    #grid.CylindricalGrid(x1ini, x1fin, x2ini, x2fin)
    
    eos = EOSdata(7.0/5.0)
    par.timenow = 0.0; par.timefin = 0.2
    

    fluid.vel1[:, :] = 0.0; fluid.vel2[:, :] = 0.0; fluid.vel3[:, :] = 0.0
    fluid.dens[:, :] = 1.0
    fluid.pres[:, :] = 1.0
       
    #boundary conditions
    #all support walls, axis, periodic and free-outflow boundaries
    par.BC[0] = 'wall'; par.BC[1] = 'wall'
    par.BC[2] = 'free'; par.BC[3] = 'free'
    
    raise ValueError(
        "User-defined HD problem – see 'HD_init_cond.py', "
        "set your ICs and remove this line."
    )
    
    return grid, fluid, par, eos


# ============================================================================
#   1D problems
# ============================================================================

def IC_HD1D_Sod_cart(grid, fluid, par):
    """
    Initialize the 1D Sod shock tube test in Cartesian geometry.

    Parameters
    ----------
    grid : object
        Grid object with Nx1r, Nx2r, fx1, fx2, etc.
    fluid : object
        SimState object containing vel1, vel2, vel3, dens, pres.
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
    
    print("cartesian 1D Sod shock tube test (G.A. Sod (1978))")
    
    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
        
    par.timenow = 0.0; par.timefin = 0.2
        
    eos = EOSdata(7.0/5.0)    
    
    left = grid.cx1 < 0.5
    
    fluid.vel1[:, :] = fluid.vel2[:, :] = fluid.vel3[:, :] = 0.0
    fluid.dens[:, :] = np.where(left, 1.0, 0.125)
    fluid.pres[:, :] = np.where(left, 1.0, 0.1)

    # boundary conditions 
    par.BC[:] = 'wall'
    
    return grid, fluid, par, eos


def IC_HD1D_Sod_cyl(grid, fluid, par):
    """
    Initialize the 1D Sod shock tube test in cylindrical geometry.

    Parameters
    ----------
    grid : object
        Grid object with Nx1r, Nx2r, fx1, fx2, etc.
    fluid : object
        SimState object containing vel1, vel2, vel3, dens, pres.
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
    
    print("cylindrical 1D Sod shock tube test (G.A. Sod (1978))")
    
    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CylindricalGrid(x1ini, x1fin, x2ini, x2fin)
        
    par.timenow = 0.0; par.timefin = 0.2
        
    eos = EOSdata(7.0/5.0)    
    
    left = grid.cx1 < 0.5
    
    fluid.vel1[:, :] = fluid.vel2[:, :] = fluid.vel3[:, :] = 0.0
    fluid.dens[:, :] = np.where(left, 1.0, 0.125)
    fluid.pres[:, :] = np.where(left, 1.0, 0.1)

    # boundary conditions 
    par.BC[:] = 'wall'
    
    return grid, fluid, par, eos


def IC_HD1D_Sod_sph(grid, fluid, par):
    """
    Initialize the 1D Sod shock tube test in spherical geometry.

    Parameters
    ----------
    grid : object
        Grid object with Nx1r, Nx2r, fx1, fx2, etc.
    fluid : object
        SimState object containing vel1, vel2, vel3, dens, pres.
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
    
    print("spherical 1D Sod shock tube test (G.A. Sod (1978))")
    
    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.SphericalPolarGrid(x1ini, x1fin, x2ini, x2fin)
        
    par.timenow = 0.0; par.timefin = 0.2
        
    eos = EOSdata(7.0/5.0)    
    
    left = grid.cx1 < 0.5
    
    fluid.vel1[:, :] = fluid.vel2[:, :] = fluid.vel3[:, :] = 0.0
    fluid.dens[:, :] = np.where(left, 1.0, 0.125)
    fluid.pres[:, :] = np.where(left, 1.0, 0.1)

    # boundary conditions 
    par.BC[:] = 'wall'
    
    return grid, fluid, par, eos


def IC_HD1D_strong_shock(grid, fluid, par):
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
    
    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.008
    
    eos = EOSdata(7.0/5.0)
    
    fluid.dens[:, :] = 1.0
    fluid.vel1[:, :] = fluid.vel2[:, :] = fluid.vel3[:, :] = 0.0
    left = grid.cx1 < 0.5
    fluid.pres[:, :] = np.where(left, 1000.0, 0.01)

    # boundary conditions 
    par.BC[:] = 'wall'
    
    return grid, fluid, par, eos


def IC_HD1D_DBW(grid, fluid, par):
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
    
    #grid creation, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.038
        
    eos = EOSdata(7.0/5.0)
    
    fluid.dens[:, :] = 1.0
    fluid.vel1[:, :] = 0.0; fluid.vel2[:, :] = 0.0; fluid.vel3[:, :] = 0.0
    fluid.pres = np.where(grid.cx1 < 0.1, 1000.0,
        np.where(grid.cx1 < 0.9, 0.01,100.0))

    # boundary conditions 
    par.BC[:] = 'wall'
    
    return grid, fluid, par, eos


def IC_HD1D_ShuOsher(grid, fluid, par):
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

    # grid creation 
    x1ini, x1fin = -5.0, 5.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 1.8
    
    eos = EOSdata(7.0 / 5.0)

    fluid.vel2[:, :] = 0.0; fluid.vel3[:, :] = 0.0
    fluid.dens[:, :] = np.where(grid.cx1 < -4.0, 3.857143, \
        1.0 + 0.2 * np.sin(5.0 * grid.cx1))
    fluid.vel1[:, :] = np.where(grid.cx1 < -4.0, 2.629369, 0.0)
    fluid.pres[:, :] = np.where(grid.cx1 < -4.0, 10.33333, 1.0)
                
    #boundaries 
    par.BC[:] = 'free'

    return grid, fluid, par, eos


def IC_HD1D_Einfeldt(grid, fluid, par):
    """
    Initialize the 1D Einfeldt rarefaction test (1-2-3 problem).

    Two symmetric rarefaction waves propagate outward from the centre,
    leaving a near-vacuum in between.  This problem is a severe test for
    positivity preservation and for Riemann solvers (especially Roe).

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

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.15
    
    eos = EOSdata(7.0 / 5.0)    

    fluid.vel2[:, :] = 0.0; fluid.vel3[:, :] = 0.0
    fluid.dens[:, :] = 1.0; fluid.pres[:, :] = 0.4
    fluid.vel1[:, :] = np.where(grid.cx1 < 0.5, -2.0, 2.0)

    par.BC[:] = 'free'

    return grid, fluid, par, eos


# ============================================================================
#   2D problems
# ============================================================================

def IC_HD2D_KHI(grid, fluid, par):
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
    
    #grid creation, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    eos = EOSdata(5.0/3.0)
    
    par.timenow = 0.0; par.timefin = 2.0
    
    fluid.vel3[:,:] = 0.0
    fluid.pres[:,:] = 2.5
    
    sigma1 = 0.05/np.sqrt(2.0)
    # interior block
    sl = (slice(grid.Ngc, grid.Nx1r), slice(grid.Ngc, grid.Nx2r))   
    x  = grid.cx1[sl]; y  = grid.cx2[sl]

    # the |y - 0.5| > 0.25 branch
    outer = np.abs(y - 0.5) > 0.25         
    fluid.vel1[sl] = np.where(outer, -0.5, 0.5)
    fluid.dens[sl] = np.where(outer,  1.0, 2.0)

    fluid.vel2[sl] = 0.1 * np.sin(4.0 * np.pi * x) * (
        np.exp(-(y - 0.25)**2 / (2.0 * sigma1**2)) +
        np.exp(-(y - 0.75)**2 / (2.0 * sigma1**2)))

    # boundary conditions 
    par.BC[0] = 'peri'; par.BC[1] = 'wall'
    par.BC[2] = 'peri'; par.BC[3] = 'wall'
    
    return grid, fluid, par, eos


def IC_HD2D_RTI(grid, fluid, par):
    """
    Initialize the 2D Rayleigh-Taylor instability problem.

    Parameters
    ----------
    grid : object
        Grid object used to create the domain.
    fluid : object
        SimState object to be initialized.
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
    - Hydrostatic equilibrium is NOT satisfied in the vertical direction 
      (it is not a big deal for this test problem though -- the error is small).
    - Boundary conditions: wall-peri-wall-peri.
    """
    print("Rayleigh-Taylor instability in 2D")
    
    x1ini, x1fin = -0.5, 0.5; x2ini, x2fin = -1.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 5.0

    eos = EOSdata(7.0/5.0)

    rho_u, rho_d = 2.0, 1.0
    
    P0 = 10.0/7.0 + 0.25
    P1 = 10.0/7.0 - 0.25

    # gravity acceleration    
    g_ff = -0.5
    fluid.F1[:, :] = 0.0
    fluid.F2[:, :] = g_ff
    
    fluid.vel1[:, :] = 0.0; fluid.vel2[:, :] = 0.0; fluid.vel3[:, :] = 0.0

    h0 = 0.03; kappa = 2.0 * np.pi
    # interior block
    sl = (slice(grid.Ngc, grid.Nx1r), slice(grid.Ngc, grid.Nx2r))   
    x  = grid.cx1[sl]; y  = grid.cx2[sl]

    # the perturbed interface: upper region where cx2 > h0*cos(kappa*cx1 + pi)
    upper = y > h0 * np.cos(x * kappa + np.pi)
    fluid.dens[sl] = np.where(upper, rho_u, rho_d)
    fluid.pres[sl] = np.where(
        upper,
        P1 + y * g_ff * rho_u,            # heavy-on-top hydrostatic branch
        P0 + (y + 1.0) * g_ff * rho_d)    # lower branch

    # boundary conditions 
    par.BC[0] = 'peri'; par.BC[1] = 'wall'; 
    par.BC[2] = 'peri'; par.BC[3] = 'wall'
    
    return grid, fluid, par, eos


def IC_HD2D_Sod(grid, fluid, par):
    """
    Initialize the 2D cylindrical Sod shock tube problem (quadrant symmetry).
    in Cartesian coordinates 
    
    Parameters
    ----------
    grid : object
        Grid object.
    fluid : object
        SimState object to be initialized.
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
    
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.2
    
    eos = EOSdata(7.0/5.0)
    
    fluid.vel1[:, :] = 0.0; fluid.vel2[:, :] = 0.0; fluid.vel3[:, :] = 0.0
    
    rad = np.sqrt(grid.cx1**2 + grid.cx2**2)
    
    fluid.dens[:, :] = np.where(rad < 0.5, 1.0, 0.125)
    fluid.pres[:, :] = np.where(rad < 0.5, 1.0, 0.1)
    
    # quadrant symmetry BC 
    par.BC[0] = 'wall'; par.BC[1] = 'wall'
    par.BC[2] = 'free'; par.BC[3] = 'free'
    
    return grid, fluid, par, eos


def IC_HD2D_Sod_sph(grid, fluid, par):
    """
    Initialize the 1D cartesian Sod shock tube test in 2D spherical geometry.

    Parameters
    ----------
    grid : object
        Grid object with Nx1r, Nx2r, fx1, fx2, etc.
    fluid : object
        SimState object containing vel1, vel2, vel3, dens, pres.
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
    The domain is divided at rcos(theta)=0, 
    Left state: rho=1, p=1; right state: rho=0.125, p=0.1.
    Boundary conditions are set to 'axis, axis, free, axis'.
    """
    
    print("cartesian 1D Sod shock tube test (G.A. Sod (1978)) in a 2D spherical-polar geometry")
    
    #grid creation
    x1ini, x1fin = 0.0, 0.5; x2ini, x2fin = 0.0, np.pi
    grid.SphericalPolarGrid(x1ini, x1fin, x2ini, x2fin)
        
    par.timenow = 0.0; par.timefin = 0.2
        
    eos = EOSdata(7.0/5.0)    
    
    left = grid.cx1 * np.cos(grid.cx2) < 0.0
    
    fluid.vel1[:, :] = fluid.vel2[:, :] = fluid.vel3[:, :] = 0.0
    fluid.dens[:, :] = np.where(left, 1.0, 0.125)
    fluid.pres[:, :] = np.where(left, 1.0, 0.1)

    # boundary conditions 
    par.BC[0] = 'axis'; par.BC[1] = 'axis'
    par.BC[2] = 'free'; par.BC[3] = 'axis'
    
    return grid, fluid, par, eos


def IC_HD2D_Sod_polar(grid, fluid, par):
    """
    Initialize the 1D cartesian Sod shock tube test in 2D polar geometry.

    Parameters
    ----------
    grid : object
        Grid object with Nx1r, Nx2r, fx1, fx2, etc.
    fluid : object
        SimState object containing vel1, vel2, vel3, dens, pres.
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
    The domain is divided at x=0.
    Left state: rho=1, p=1; right state: rho=0.125, p=0.1.
    Boundary conditions are set to 'axis, peri, free, peri'.
    """
    
    print("cartesian 1D Sod shock tube test (G.A. Sod (1978)) in a 2D polar geometry")
    
    #grid creation
    x1ini, x1fin = 0.0, 0.5; x2ini, x2fin = 0.0, 2.0*np.pi
    grid.PolarGrid(x1ini, x1fin, x2ini, x2fin)
        
    par.timenow = 0.0; par.timefin = 0.2
        
    eos = EOSdata(7.0/5.0)    
    
    left = grid.cx1 * np.cos(grid.cx2) < 0.0
    
    fluid.vel1[:, :] = fluid.vel2[:, :] = fluid.vel3[:, :] = 0.0
    fluid.dens[:, :] = np.where(left, 1.0, 0.125)
    fluid.pres[:, :] = np.where(left, 1.0, 0.1)

    # boundary conditions 
    par.BC[0] = 'axis'; par.BC[1] = 'peri'
    par.BC[2] = 'free'; par.BC[3] = 'peri'
    
    return grid, fluid, par, eos


def IC_HD2D_Sedov_cart(grid, fluid, par):
    """
    Initialize the 2D Sedov-Taylor explosion test in Cartesian coordinates.

    Parameters
    ----------
    grid : object
        Grid object.
    fluid : object
        SimState object to be initialized.
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
    
    #grid creation
    x1ini, x1fin = 0.0, 0.5; x2ini, x2fin = 0.0, 0.5
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.2
    
    eos = EOSdata(7.0/5.0)

    rad0 = 0.02; energ = 0.25

    fluid.vel1[:, :] = fluid.vel2[:, :] = fluid.vel3[:, :] = 0.0
    fluid.dens[:, :] = 1.0

    rad   = np.sqrt(grid.cx1**2 + grid.cx2**2)
    inside = rad < rad0

    # total volume of the hot region -- grid.cVol is interior-only (no ghost
    # cells), so only the boolean mask (built from the ghost-inclusive cx1/cx2)
    # needs the ghost-offset slice; cVol itself must not be re-sliced.
    sl = (slice(grid.Ngc, grid.Nx1r), slice(grid.Ngc, grid.Nx2r))
    if not np.any(inside[sl]):
        # rad0 is smaller than half a cell at this resolution -- no cell
        # centre falls inside the analytic disk. Deposit into the single
        # interior cell nearest the origin instead, so the blast energy is
        # never silently lost at coarse resolution.
        inside[:, :] = False
        i0, j0 = np.unravel_index(np.argmin(rad[sl]), rad[sl].shape)
        inside[grid.Ngc + i0, grid.Ngc + j0] = True
    volume = np.sum(grid.cVol[inside[sl]])

    # pass 2: deposit the blast energy as pressure over exactly that volume
    fluid.pres[:, :] = np.where(inside, (eos.GAMMA - 1.0) * energ / volume, 1e-4)

    # quadrant symmetry BC
    par.BC[0] = 'wall'; par.BC[1] = 'wall'
    par.BC[2] = 'free'; par.BC[3] = 'free'
    
    return grid, fluid, par, eos


def IC_HD2D_Sedov_cyl(grid, fluid, par):
    """
    Initialize the 3D Sedov-Taylor explosion test in 2D Cylindrical coordinates.

    Parameters
    ----------
    grid : object
        Grid object.
    fluid : object
        SimState object to be initialized.
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
    - Boundary conditions: axis-wall-free-free.
    - Uses quadrant symmetry.
    """
    print("3D Sedov-Taylor explosion test in Cylindrical (R,Z) geometry")
    
    #grid creation
    x1ini, x1fin = 0.0, 0.5; x2ini, x2fin = 0.0, 0.5
    grid.CylindricalGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.15
    
    eos = EOSdata(7.0/5.0)

    rad0 = 0.02; energ = 0.5

    fluid.vel1[:, :] = fluid.vel2[:, :] = fluid.vel3[:, :] = 0.0
    fluid.dens[:, :] = 1.0

    rad   = np.sqrt(grid.cx1**2 + grid.cx2**2)
    inside = rad < rad0

    # total volume of the hot region -- grid.cVol is interior-only (no ghost
    # cells), so only the boolean mask (built from the ghost-inclusive cx1/cx2)
    # needs the ghost-offset slice; cVol itself must not be re-sliced.
    sl = (slice(grid.Ngc, grid.Nx1r), slice(grid.Ngc, grid.Nx2r))
    if not np.any(inside[sl]):
        # rad0 is smaller than half a cell at this resolution -- no cell
        # centre falls inside the analytic disk. Deposit into the single
        # interior cell nearest the origin instead, so the blast energy is
        # never silently lost at coarse resolution.
        inside[:, :] = False
        i0, j0 = np.unravel_index(np.argmin(rad[sl]), rad[sl].shape)
        inside[grid.Ngc + i0, grid.Ngc + j0] = True
    volume = np.sum(grid.cVol[inside[sl]])

    # pass 2: deposit the blast energy as pressure over exactly that volume
    fluid.pres[:, :] = np.where(inside, (eos.GAMMA - 1.0) * energ / volume, 1e-4)

    # equatorial symmetry BC
    par.BC[0] = 'axis'; par.BC[1] = 'wall'
    par.BC[2] = 'free'; par.BC[3] = 'free'
    
    return grid, fluid, par, eos


def IC_HD2D_RP2D(grid, fluid, par):
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

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.3
    
    eos = EOSdata(7.0 / 5.0)

    x = grid.cx1; y = grid.cx2

    m1 = (x >= 0.5) & (y >= 0.5) # Quadrant I: x > 0.5, y > 0.5
    m2 = (x < 0.5) & (y >= 0.5) # Quadrant II: x < 0.5, y > 0.5
    m3 = (x < 0.5) & (y < 0.5) # Quadrant III: x < 0.5, y < 0.5
    m4 = (x >= 0.5) & (y < 0.5) # Quadrant IV: x > 0.5, y < 0.5

    fluid.dens[:, :] = 1.5 * m1 + 0.5323 * m2 + 0.138 * m3 + 0.5323 * m4
    fluid.vel1[:, :] = 0.0 * m1 + 1.206 * m2 + 1.206 * m3 + 0.0 * m4
    fluid.vel2[:, :] = 0.0 * m1 + 0.0 * m2 + 1.206 * m3 + 1.206 * m4
    fluid.vel3[:, :] = 0.0
    fluid.pres[:, :] = 1.5 * m1 + 0.3 * m2 + 0.029 * m3 + 0.3 * m4

    par.BC[:] = 'free'

    return grid, fluid, par, eos


def IC_HD2D_Gresho(grid, fluid, par):
    """
    Initialize the 2D Gresho vortex (Gresho & Chan 1990).

    A stationary vortex in steady rotational equilibrium.
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

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 3.0
    
    eos = EOSdata(7.0 / 5.0)

    x0, y0 = 0.5, 0.5
    r = np.sqrt((grid.cx1 - x0)**2 + (grid.cx2 - y0)**2)
    
    fluid.dens[:, :] = 1.0
    fluid.vel1[:, :] = (grid.cx2 - y0) * np.where(
        r < 1e-14, 0.0, np.where(r < 0.2, -5.0,
        np.where(r < 0.4, -(2.0 - 5.0 * r) / r, 0.0)))
    fluid.vel2[:, :] = (grid.cx1 - x0) * np.where(
        r < 1e-14, 0.0, np.where(r < 0.2,  5.0,
        np.where(r < 0.4,  (2.0 - 5.0 * r) / r, 0.0)))
    fluid.vel3[:, :] = 0.0
    fluid.pres[:, :] = np.where(
        r < 0.2, 5.0 + 12.5 * r**2, np.where(r < 0.4, \
        9.0 + 12.5 * r**2 - 20.0 * r + 4.0 * np.log(r / 0.2),
        3.0 + 4.0 * np.log(2.0)))

    par.BC[:] = 'peri'

    return grid, fluid, par, eos
    
    
def IC_HD2D_vortex(grid, fluid, par):
    """
    Initialize the 2D isentropic vortex

    Domain: [0, 1] x [0, 1], periodic
    Gamma = 7/5, t_fin = 10

    Parameters
    ----------
    grid : object
    fluid : object
    par : object

    Returns
    -------
    grid, fluid, par, eos : objects
    """
    
    print("2D isentropic vortex problem")
    
    x1ini, x1fin = -5.0, 5.0; x2ini, x2fin = -5.0, 5.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 10.0
    
    eos = EOSdata(7.0/5.0)
    
    x, y = grid.cx1, grid.cx2
    beta = 5.0
    bump = np.exp(0.5 * (1.0 - x**2 - y**2))
    tmpr = 1.0 - (eos.GAMMA - 1.0) * beta**2 / (8.0 * eos.GAMMA * np.pi**2) * bump**2
    fluid.dens[:, :] = tmpr**(1.0 / (eos.GAMMA - 1.0))
    fluid.pres[:, :] = fluid.dens * tmpr
    fluid.vel1[:, :] = 1.0 - beta / (2.0 * np.pi) * bump * y
    fluid.vel2[:, :] = 1.0 + beta / (2.0 * np.pi) * bump * x
    fluid.vel3[:, :] = 0.0
    
    par.BC[:] = 'peri'

    return grid, fluid, par, eos


def IC_HD2D_shock_cloud(grid, fluid, par):
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
    Post-shock: Mach-10 Rankine-Hugoniot state, rho=3.8835, v1=9.5856, p=124.75
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

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.06
    
    eos = EOSdata(5.0/3.0)

    # Cloud parameters
    xc, yc = 0.25, 0.5
    rc = 0.1; rho_cloud = 10.0

    # Pre-shock (ambient)
    rho_amb = 1.0; p_amb = 1.0

    # Post-shock state: Rankine-Hugoniot jump of a Mach-M shock moving
    # into gas at rest (rho_amb, p_amb)
    gam    = eos.GAMMA
    M_shk  = 10.0
    D_shk  = M_shk * np.sqrt(gam * p_amb / rho_amb)          # shock speed
    rho_post = rho_amb * (gam + 1.0) * M_shk**2 / ((gam - 1.0) * M_shk**2 + 2.0)
    p_post   = p_amb * (2.0 * gam * M_shk**2 - (gam - 1.0)) / (gam + 1.0)
    v1_post  = D_shk * (1.0 - rho_amb / rho_post)            # mass conservation
    # gamma = 5/3, M = 10:  rho_post = 3.8835, v1_post = 9.5856, p_post = 124.75

    x_shock = 0.05  # initial shock position

    x = grid.cx1; y = grid.cx2
    r = np.sqrt((x - xc)**2 + (y - yc)**2)

    post  = x < x_shock; cloud = r < rc

    # nested where mirrors  if post ... elif cloud ... else ...
    fluid.dens[:, :] = np.where(post, rho_post, np.where(cloud, rho_cloud, rho_amb))
    fluid.vel1[:, :] = np.where(post, v1_post, 0.0)
    fluid.pres[:, :] = np.where(post, p_post, p_amb)
    fluid.vel2[:, :] = 0.0; fluid.vel3[:, :] = 0.0

    par.BC[0] = 'free'; par.BC[1] = 'free'
    par.BC[2] = 'free'; par.BC[3] = 'free'

    return grid, fluid, par, eos


def IC_HD2D_gap_opening(grid, fluid, par):
    """
    Simplified 2D gap-opening problem in a protoplanetary disk (polar coords).

    Educational version, built to be CONSISTENT with the adiabatic gamma-law
    EOS (not locally-isothermal, but gamma ~ 1). The disk is initialised in exact radial
    equilibrium for the chosen gamma, so it stays steady until the planet's
    gravity carves a gap. A low-mass planet sits at (R=1, phi=pi) on a fixed
    circular orbit; its softened gravity is applied as a source term.

    Design choices for clarity:
      * power-law Sigma(R) and P(R) chosen so that radial balance is EXACT,
        so v_phi includes the (sub-Keplerian) pressure correction;
      * F1, F2 are accelerations, calculated by before_step
      * planet on a fixed orbit (no migration, no disk back-reaction) -- the
        standard teaching simplification.

    Coordinate system : polar (R, phi)
    Domain            : R in [0.4, 2.5], phi in [0, 2*pi]
    Star              : M_star = 1 at the origin
    Planet            : M_p = 5e-4 at (R=1, phi=pi), fixed circular orbit

    References
    ----------
    de Val-Borro, M. et al. (2006), MNRAS 370, 529 
    """
    print("2D gap-opening in a protoplanetary disk (simplified, adiabatic)")

    R_in, R_out = 0.4, 2.5; phi_in, phi_out = 0.0, 2.0 * np.pi
    grid.PolarGrid(R_in, R_out, phi_in, phi_out)

    par.timenow = 0.0; par.timefin = 25.0 * 2.0 * np.pi # 25 orbits at R = 1

    eos = EOSdata(1.01)

    # --- disk parameters ---
    Sigma0 = 1.0
    M_star = 1.0
    h0 = 0.05 # aspect ratio (h/R) at R = 1; sets the disk "thickness"
    a = 0.5 # Sigma ~ R^(-a)

    # Temperature/pressure power law chosen so that c_s ~ v_K * h0 at R=1.
    # For a gamma-law gas, locally  c_s^2 = gamma * P / Sigma.  We want the disk
    # geometrically thin and in radial balance, so we set
    #     P(R) = P0 * R^(-(a + 1))          (so c_s^2 ~ 1/R, like a real disk)
    # and choose P0 so that  (c_s / v_K)|_{R=1} = h0.
    #   c_s^2(1) = gamma P0 / Sigma0 ,  v_K^2(1) = M_star  ->  P0 = Sigma0 h0^2 M_star / gamma
    P0 = Sigma0 * h0**2 * M_star / eos.GAMMA
    q  = a + 1.0 # pressure power-law index  P ~ R^(-q)

    # --- planet parameters ---
    M_planet = 5.0e-4  # ~0.5 Jupiter-ish in these units; small enough to be gentle
    R_planet = 1.0
    phi_planet = np.pi
    eps = 0.6 * h0 * R_planet # gravitational softening length

    # ------------------------------------------------------------------
    # Disk state (vectorised, full array including ghosts)
    # ------------------------------------------------------------------
    R = grid.cx1; Sigma = Sigma0 * R**(-a); P = P0 * R**(-q)

    # Exact radial equilibrium for gamma-law EOS:
    #   v_phi^2 / R = M_star / R^2 + (1/Sigma) dP/dR
    # dP/dR = -q P / R, so:
    #   v_phi^2 = M_star / R - q P / Sigma
    # the disk is slightly sub-Keplerian 
    vphi2 = M_star / R - q * P / Sigma
    vphi2 = np.maximum(vphi2, 0.0) # guard (only triggers if disk is too hot)
    v_phi = np.sqrt(vphi2)

    fluid.dens[:, :] = Sigma; fluid.pres[:, :] = P
    fluid.vel1[:, :] = 0.0 # v_R = 0
    fluid.vel2[:, :] = v_phi # balanced (sub-Keplerian) rotation
    fluid.vel3[:, :] = 0.0

    # periodic in phi, open in R
    par.BC[0] = 'free'; par.BC[1] = 'peri'
    par.BC[2] = 'free'; par.BC[3] = 'peri'
    
    # radial boundaries: ghost cells hold the initial (sub-Keplerian) equilibrium
    Ngc = grid.Ngc
    ghost_in  = np.s_[0:Ngc,  Ngc:-Ngc]          # R < R_in
    ghost_out = np.s_[-Ngc:, Ngc:-Ngc]           # R > R_out
    par.BC_fixed[0] = [(0, grid.Nx2, {'dens': Sigma[ghost_in],  'pres': P[ghost_in],
                                      'vel1': 0.0, 'vel2': v_phi[ghost_in],  'vel3': 0.0})]
    par.BC_fixed[2] = [(0, grid.Nx2, {'dens': Sigma[ghost_out], 'pres': P[ghost_out],
                                      'vel1': 0.0, 'vel2': v_phi[ghost_out], 'vel3': 0.0})]

    #set function for gravity potential evaluation
    par.before_step = lambda grid, state, par, dt: planet_gravity_polar(
        grid, state, par, M_star=M_star, M_planet=M_planet, r_planet=R_planet,
        phi0=phi_planet, indirect=True, soft=eps)
    par.before_step(grid, fluid, par, 0.0)   
    
    return grid, fluid, par, eos


def IC_HD2D_jet_cyl(grid, fluid, par):
    """
    Axisymmetric non-relativistic jet in cylindrical (R, Z) coordinates.

    A supersonic (Mach 6) light jet is injected through a nozzle at the BOTTOM
    boundary (Z = 0, the x2-inner face) over R < r_jet, and propagates in +Z
    into a uniform, pressure-matched ambient medium, developing a bow shock,
    cocoon and Mach disk.

    Coordinate system : cylindrical (R, Z) = (x1, x2)
    Domain            : R in [0, 10], Z in [0, 25]
    Inlet (face 1)    : R < 1, rho=1, v_z=6, p = rho cs^2 / gamma  (Mach 6)
    Ambient           : rho=10, v=0, same p     (eta = rho_jet/rho_amb = 0.1)

    The inlet is a fixed (Dirichlet) ghost-fill registered in par.BC_fixed[1];
    it requires boundCond_HD to apply apply_bc_fixed (face 1) AFTER the standard
    'wall' fill and to be passed par.BC_fixed.  The interior starts as pure
    ambient -- the jet turns on from the boundary, so there is no initial
    internal discontinuity.

    References
    ----------
    Norman, M. L. et al. (1982), A&A 113, 285
    Bodo, G. et al. (1998), A&A 333, 1117
    """
    print("2D axisymmetric non-relativistic jet (cylindrical, inlet BC)")

    # --- grid + time ---
    R_in, R_out = 0.0, 10.0
    Z_in, Z_out = 0.0, 25.0
    grid.CylindricalGrid(R_in, R_out, Z_in, Z_out)
    par.timenow = 0.0
    par.timefin = 15.0
    eos = EOSdata(5.0/3.0)

    # --- aliases ---
    Ngc  = grid.Ngc; Nx1  = grid.Nx1; Nx1r = grid.Nx1r

    # --- jet / ambient parameters (Mach 6, pressure equilibrium) ---
    Mach    = 6.0
    rho_jet = 1.0; rho_amb = 10.0 # eta = rho_jet/rho_amb = 0.1
    cs_jet  = 1.0 # normalise the jet sound speed
    v_jet   = Mach * cs_jet # => internal Mach number = 6
    p_jet   = rho_jet * cs_jet**2 / eos.GAMMA   # p = rho cs^2 / gamma  (= 0.6)
    p_amb   = p_jet # pressure-matched across the nozzle
    r_jet   = 1.0

    # --- uniform ambient everywhere (incl. ghosts), vectorised ---
    fluid.dens[:, :] = rho_amb; fluid.pres[:, :] = p_amb
    fluid.vel1[:, :] = fluid.vel2[:, :] = fluid.vel3[:, :] = 0.0 #v_R,Z,phi = 0

    # --- nozzle extent along R (tangential to the bottom face) ---
    Rc = grid.cx1[Ngc:Nx1r, Ngc]         # 1D interior R cell-centres
    in_jet = np.nonzero(Rc < r_jet)[0]   # contiguous from the axis
    start  = int(in_jet[0])              # 0
    end    = int(in_jet[-1]) + 1         # one-past-last interior R index

    # --- fixed (Dirichlet) inlet on the bottom face (x2-inner = face 1) ---
    jet_state = {'dens': rho_jet, 'pres': p_jet,
                 'vel1': 0.0, 'vel2': v_jet, 'vel3': 0.0}
    par.BC_fixed[1] = [(start, end, jet_state)]

    # --- seed the bottom ghost cells at t=0 so the state is self-consistent
    #     before the first BC fill (mirrors what apply_bc_fixed will maintain) ---
    i0, i1 = Ngc + start, Ngc + end
    fluid.dens[i0:i1, 0:Ngc] = rho_jet
    fluid.pres[i0:i1, 0:Ngc] = p_jet
    fluid.vel2[i0:i1, 0:Ngc] = v_jet     # +Z, into the domain

    # --- boundaries: axis at R=0, wall at Z=0 (overridden by the nozzle),
    #     outflow at R=5 and Z=20 ---
    par.BC[0] = 'axis'  # x1 inner  (R = 0)
    par.BC[1] = 'wall'  # x2 inner  (Z = 0 + nozzle via BC_fixed)
    par.BC[2] = 'free'  # x1 outer  (R = 5)
    par.BC[3] = 'free'  # x2 outer  (Z = 20)
    return grid, fluid, par, eos


def IC_HD1D_Jeans(grid, fluid, par):
    """
    Linear Jeans instability of a uniform self-gravitating gas (periodic).

    rho = rho0 (1 + eps cos kx),  p = p0 (rho/rho0)^gamma (adiabatic mode),  v = 0
    Linear theory:  omega^2 = cs^2 k^2 - 4 pi G rho0
        omega^2 < 0 :  delta(t) = eps cosh(sigma t)      (collapse)
        omega^2 > 0 :  delta(t) = eps cos(omega t)       (Jeans-modified sound wave)
    Choose the branch with  jeans_ratio = 4 pi G rho0 / (cs k)^2  (= (k_J/k)^2):
    > 1 unstable, < 1 stable.

    In a periodic box the Poisson solver subtracts the mean density (the
    "Jeans swindle") -- here this is exactly the intended physics.
    Gravity is recomputed once per step (par.before_step), so the coupling
    is first-order accurate in time.
    """
    print("1D Jeans instability (periodic)")

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    eos = EOSdata(5.0/3.0)

    rho0, cs, eps = 1.0, 1.0, 1.0e-4
    k = 2.0 * np.pi # one wavelength in the box
    jeans_ratio = 2.0 # 4 pi G rho0 / (cs k)^2 ; try 0.5 for the stable branch
    G = jeans_ratio * (cs * k)**2 / (4.0 * np.pi * rho0)
    w2 = (cs * k)**2 * (1.0 - jeans_ratio)

    p0 = rho0 * cs**2 / eos.GAMMA
    fluid.dens[:, :] = rho0 * (1.0 + eps * np.cos(k * grid.cx1))
    fluid.pres[:, :] = p0 * (fluid.dens / rho0)**eos.GAMMA
    fluid.vel1[:, :] = 0.0; fluid.vel2[:, :] = 0.0; fluid.vel3[:, :] = 0.0

    par.timenow = 0.0
    if w2 < 0.0:
        par.timefin = 3.0 / np.sqrt(-w2) # growth by cosh(3) ~ 10: still linear
    else:
        par.timefin = 2.0 * np.pi / np.sqrt(w2) # one full oscillation period

    par.BC[:] = 'peri'
    par.before_step = lambda grid, state, par, dt: selfgravity_poisson(
        grid, state, par, G=G, BC=['peri'] * 4)
    return grid, fluid, par, eos


def IC_HD2D_collapse_sph(grid, fluid, par):
    """
    Pressureless (cold) collapse of a uniform self-gravitating sphere,
    spherical-polar (r, theta) grid; Nx2 = 1 gives the purely radial problem.

    A sphere of radius R0 and density rho0 starts at rest in a tenuous
    ambient medium; pressure is dynamically negligible (sound-crossing time
    >> free-fall time). The sphere stays uniform and every shell follows the
    free-fall solution  (the same for all shells, since M(<r)/r^3 = const), 
    (see the analytical solution by Colgate and White, ApJ (1966))
    reaching the centre at  t_ff = sqrt(3 pi / (32 G rho0)).
    The ambient medium is hot (c_s = 2) and dilute.

    Gravity: Poisson solve with an isolated boundary condition -- the outer
    radial face gets the exact exterior (monopole) potential -G M/r_out of
    the total mass on the grid (exact for a spherical mass distribution);
    r = 0 and the theta axes are zero-flux ('free').
    """
    print("cold collapse of a uniform sphere (spherical-polar, self-gravity)")

    R0, r_out = 1.0, 2.0
    grid.SphericalPolarGrid(0.0, r_out, 0.0, np.pi)
    eos = EOSdata(5.0/3.0)

    G, rho0, rho_amb = 1.0, 1.0, 1.0e-3
    # The CFL condition sees only sound and flow speeds, not the gravitational
    # acceleration: gas that is cold AND at rest would allow a first step of a
    # sizeable fraction of t_ff.
    cs_amb = 2.0
    p0 = rho_amb * cs_amb**2 / eos.GAMMA
    t_ff = np.sqrt(3.0 * np.pi / (32.0 * G * rho0))
    par.timenow = 0.0; par.timefin = 0.95 * t_ff

    fluid.dens[:, :] = np.where(grid.cx1 < R0, rho0, rho_amb)
    fluid.pres[:, :] = p0
    fluid.vel1[:, :] = 0.0; fluid.vel2[:, :] = 0.0; fluid.vel3[:, :] = 0.0

    par.BC[0] = 'axis'; par.BC[1] = 'axis'
    par.BC[2] = 'free'; par.BC[3] = 'axis'

    def gravity(grid, state, par, dt = 0.0):
        Ngc = grid.Ngc
        M = np.sum(grid.cVol * state.dens[Ngc:-Ngc, Ngc:-Ngc])   # total mass on the grid
        return selfgravity_poisson(grid, state, par, G=G,
                                   BC=['free', 'free', 'dirichlet', 'free'],
                                   BC_value={2: -G * M / r_out})
    par.before_step = gravity
    return grid, fluid, par, eos


def IC_HD2D_merger(grid, fluid, par):
    """
    Merger of two self-gravitating Gaussian gas blobs in a periodic box
    (cf. the "merger" setup of the Ulula code, Diemer 2025, Fig. 5).

    Two identical blobs,  rho = rho_bg + rho_b [exp(-|x-x1|^2/2s^2) + exp(-|x-x2|^2/2s^2)],
    in a uniform-pressure background, fall together under their mutual gravity,
    collide, bounce and slowly settle into a single oscillating object.
    Optional counter-moving velocities (v_orb) give the pair angular momentum
    and an off-axis, spiralling merger.

    NB: in 2D Cartesian geometry the Poisson equation describes infinitely
    long "cylinders", so the force between the blobs falls off as 1/d, not
    1/d^2. In a periodic box the mean density is subtracted from the source ---
    only the density contrast gravitates.

    Useful checks: total mass is conserved to round-off; the mirror symmetry
    of the set-up (x -> 1-x, y -> 1-y; and y -> 1-y for v_orb = 0) is preserved;
    the total energy E_kin + E_th + (1/2) int rho Phi dV is conserved only to
    truncation error, since gravity enters as a non-conservative source.
    """
    print("merger of two self-gravitating blobs (periodic box)")

    grid.CartesianGrid(0.0, 1.0, 0.0, 1.0)
    eos = EOSdata(5.0 / 3.0)

    G      = 1.0
    rho_bg = 0.1          # background density
    rho_b  = 1.0          # blob peak density (above background)
    s      = 0.05         # blob Gaussian width
    d      = 0.2          # half-separation: centres at (0.5 -/+ d, 0.5)
    p0     = 0.01         # uniform pressure -> blob line mass 2 pi s^2 rho_b ~ 0.016
                          # is below the critical 2 c_s^2 / G ~ 0.033: blobs are
                          # pressure-supported and do not collapse on their own
    v_orb  = 0.0          # |v_y| of each blob (opposite signs); 0 = head-on

    par.timenow = 0.0; par.timefin = 10.0

    x, y = grid.cx1, grid.cx2
    g1 = np.exp(-((x - (0.5 - d))**2 + (y - 0.5)**2) / (2.0 * s**2))
    g2 = np.exp(-((x - (0.5 + d))**2 + (y - 0.5)**2) / (2.0 * s**2))
    fluid.dens[:, :] = rho_bg + rho_b * (g1 + g2)
    fluid.pres[:, :] = p0
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = v_orb * (g1 - g2) / (g1 + g2 + 1e-30) * (g1 + g2 > 1e-3)
    fluid.vel3[:, :] = 0.0

    par.BC[:] = 'peri'
    
    def gravity(grid, state, par, dt = 0.0):
        Ngc = grid.Ngc
        M = np.sum(grid.cVol * state.dens[Ngc:-Ngc, Ngc:-Ngc])   # total mass on the grid
        return selfgravity_poisson(grid, state, par, G=G, BC=['peri', 'peri', 'peri', 'peri'])
    par.before_step = gravity
        
    return grid, fluid, par, eos