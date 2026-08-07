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
from src.gravity import (
    corotating_planet_disk,
    corotating_planet_disk_hook,
    selfgravity_poisson,
    selfgravity_poisson_hook,
)



def IC_HD_user_defined(grid, fluid, par):
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
    
    print("cylindrical 1D Sod shock tube test (G.A. Sod (1978))")
    
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



def IC_HD1D_Sod_sph(grid, fluid, par):
    """
    Initialize the 1D Sod shock tube test in spherical geometry.

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
    
    print("spherical 1D Sod shock tube test (G.A. Sod (1978))")
    
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
    fluid.dens[:, :] = np.where(grid.cx1 < 4.0, 3.857143, \
        1.0 + 0.2 * np.sin(5.0 * grid.cx1))
    fluid.vel1[:, :] = np.where(grid.cx1 < 4.0, 2.629369, 0.0)
    fluid.pres[:, :] = np.where(grid.cx1 < 4.0, 10.33333, 1.0)
                
    #boundaries 
    par.BC[:] = 'free'

    return grid, fluid, par, eos



def IC_HD1D_Einfeldt(grid, fluid, par):
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
    
    #grid creation
    x1ini, x1fin = 0.0, 0.5; x2ini, x2fin = 0.0, 0.5
    grid.CylindricalGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.2
    
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

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.06
    
    eos = EOSdata(5.0 / 3.0)

    # Cloud parameters
    xc, yc = 0.25, 0.5
    rc = 0.1; rho_cloud = 10.0

    # Pre-shock (ambient)
    rho_amb = 1.0; p_amb = 1.0

    # Post-shock state (Mach 10 shock in gamma=5/3 gas)
    rho_post = 3.857143; v1_post = 11.2; p_post = 167.0

    x_shock = 0.05  # initial shock position

    x = grid.cx1; y = grid.cx2
    r = np.sqrt((x - xc)**2 + (y - yc)**2)

    post  = x < x_shock          # highest priority
    cloud = r < rc               # applies only where NOT post-shock (the elif)

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
    2D gap opening by a planet in a protoplanetary disk (polar coordinates),
    in the frame CO-ROTATING with the planet.

    This is the de Val-Borro et al. (2006) disk-planet comparison problem,
    reduced to its essentials: a thin, radially-balanced disk orbits a star,
    a low-mass planet on a fixed circular orbit at R = 1 is switched on
    smoothly over the first few orbits, and its torques open an annular gap
    around its orbit while launching the characteristic one-armed spiral
    wakes inside and outside it.

    Frame
    -----
    Everything is solved in the frame rotating at the planet's orbital
    frequency Omega_p = sqrt(G (M_star + M_p) / R_p^3), so the planet is
    STATIONARY at (R_p, phi_p) and only the frame forces move.  This is the
    standard choice for this problem: in the lab frame the disk streams past
    the grid at the local Keplerian speed, which shrinks the timestep and
    smears the slowly-growing gap by advection diffusion.  The price is a
    centrifugal and a Coriolis term, supplied together with the gravity by
    ``gravity.corotating_planet_disk``.

    Because the Coriolis force depends on the current velocity (and the
    planet mass on the current time, while it ramps), the source term CANNOT
    be written once into F1/F2 here -- it is installed as a per-stage
    ``fluid.body_force`` hook, which the solvers re-evaluate at every
    Runge-Kutta stage.

    Initial state
    -------------
    Power-law surface density and pressure,

        Sigma(R) = Sigma0 R^(-a),     P(R) = P0 R^(-(a+1)) ,

    with P0 fixed by the aspect ratio h0 = (c_s / v_K) at R = 1.  The
    azimuthal velocity is the EXACT radial-equilibrium solution for this
    gamma-law gas,

        v_phi,inertial^2 = G M_star / R - (a+1) P / Sigma ,

    i.e. slightly sub-Keplerian because pressure supports part of gravity.
    Dropping that correction is the classic way to get a disk that drifts
    radially from the first step.  The stored velocity is then transformed
    into the rotating frame,  v_phi = v_phi,inertial - Omega_p R.

    Coordinate system : polar (R, phi) = (x1, x2)
    Domain            : R in [0.4, 2.5], phi in [0, 2 pi]
    Star              : M_star = 1 at the origin
    Planet            : M_p = 1e-3 (~1 Jupiter at 1 M_sun), fixed at (1, pi)
    Aspect ratio      : h/R = 0.05 at R = 1
    Run length        : 30 planet orbits (a gap is clearly visible by ~20)

    Known simplifications
    ---------------------
    * the planet is on a FIXED orbit -- no migration, no disk back-reaction;
    * no wave-damping zones near the radial boundaries, so the spiral wakes
      partially reflect off the open R boundaries after a few orbits.  The
      benchmark damps the solution towards the initial state in
      R < 0.5 and R > 2.1 for exactly this reason;
    * the disk is 2D and vertically integrated, as in the benchmark.

    Parameters
    ----------
    grid : object   Grid; a PolarGrid is built here.
    fluid : object  HD SimState (dens, pres, vel1..3, F1, F2, body_force).
    par : object    Parameters (BC, timenow, timefin).

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    de Val-Borro, M. et al. (2006), MNRAS 370, 529   (the comparison problem)
    Lin, D. N. C. & Papaloizou, J. (1986), ApJ 309, 846   (gap-opening theory)
    """
    print("2D gap opening by a planet in a protoplanetary disk "
          "(polar, co-rotating frame)")

    # --- grid ---
    R_in, R_out = 0.4, 2.5; phi_in, phi_out = 0.0, 2.0 * np.pi
    grid.PolarGrid(R_in, R_out, phi_in, phi_out)

    eos = EOSdata(5.0 / 3.0)

    # --- star / planet / disk parameters ---
    G        = 1.0
    M_star   = 1.0
    M_planet = 1.0e-3          # ~1 M_Jupiter for a 1 M_sun star
    R_planet = 1.0
    phi_planet = np.pi         # mid-domain: furthest from the periodic seam
    Sigma0   = 1.0
    h0       = 0.05            # aspect ratio (h/R) at R = 1 -> thin disk
    a        = 0.5             # Sigma ~ R^(-a)
    q        = a + 1.0         # pressure power law  P ~ R^(-q)

    # gravitational softening: the standard 0.6 H at the planet's orbit.
    # Below this scale a 2D point-mass potential is not a meaningful model of
    # a 3D planet anyway, and an unsoftened one would be grid-dependent.
    soft = 0.6 * h0 * R_planet

    # P0 from the aspect ratio: c_s^2 = gamma P / Sigma and v_K^2 = G M_star
    # at R = 1, so (c_s/v_K)|_{R=1} = h0  <=>  P0 = Sigma0 h0^2 G M_star / gamma
    P0 = Sigma0 * h0**2 * G * M_star / eos.GAMMA

    # --- orbital frequency of the frame, run length, planet ramp ---
    Omega_p = np.sqrt(G * (M_star + M_planet) / R_planet**3)
    T_orbit = 2.0 * np.pi / Omega_p
    par.timenow = 0.0
    par.timefin = 30.0 * T_orbit         # 30 planet orbits
    t_ramp      = 5.0 * T_orbit          # switch the planet on over 5 orbits

    # --- disk state (vectorised, full arrays including ghosts) ---
    R = grid.cx1

    Sigma = Sigma0 * R**(-a)
    P     = P0 * R**(-q)

    # exact radial equilibrium in the INERTIAL frame:
    #   v_phi^2 / R = G M_star / R^2 + (1/Sigma) dP/dR ,   dP/dR = -q P / R
    vphi2 = G * M_star / R - q * P / Sigma
    vphi2 = np.maximum(vphi2, 0.0)       # guard (only trips if the disk is hot)
    v_phi_inertial = np.sqrt(vphi2)

    fluid.dens[:, :] = Sigma
    fluid.pres[:, :] = P
    fluid.vel1[:, :] = 0.0                                   # v_R = 0
    fluid.vel2[:, :] = v_phi_inertial - Omega_p * R          # into the co-rotating frame
    fluid.vel3[:, :] = 0.0

    # --- body force: star + planet gravity + centrifugal + Coriolis ---
    # Installed as a per-stage hook because Coriolis depends on the current
    # velocity and the planet mass on the current time while it ramps.
    fluid.body_force = corotating_planet_disk_hook(
        M_star=M_star, M_planet=M_planet, r_planet=R_planet,
        phi_planet=phi_planet, soft=soft, G=G,
        indirect=True, t_ramp=t_ramp)
    # evaluate once now so F1/F2 are consistent with the state at t = 0
    fluid = corotating_planet_disk(
        grid, fluid, par, M_star=M_star, M_planet=M_planet,
        r_planet=R_planet, phi_planet=phi_planet, soft=soft, G=G,
        indirect=True, t_ramp=t_ramp)

    # --- boundaries: periodic in phi, equilibrium-pinned in R ---
    # A zero-gradient radial boundary cannot represent a power-law disk: it
    # copies Sigma and v_phi from the last interior cell, which is NOT the
    # equilibrium value at the ghost radius, so a spurious boundary layer
    # forms and propagates inward (measured: ~10% surface-density error at
    # the edges after 2 orbits, versus ~0.2% mid-disk).  Instead pin both
    # radial ghost layers to the analytic equilibrium evaluated at the ghost
    # radii -- physically, the annulus is cut out of a much larger disk that
    # stays unperturbed.
    def _equilibrium_at(Rg):
        """Analytic (Sigma, P, v_phi_rot) at the ghost radii Rg, shape (Ngc,1)."""
        Sig = Sigma0 * Rg**(-a)
        Prs = P0 * Rg**(-q)
        vp2 = np.maximum(G * M_star / Rg - q * Prs / Sig, 0.0)
        return {'dens': Sig, 'pres': Prs,
                'vel1': np.zeros_like(Rg),
                'vel2': np.sqrt(vp2) - Omega_p * Rg,
                'vel3': np.zeros_like(Rg)}

    Ngc = grid.Ngc
    R_in_ghost  = grid.cx1[0:Ngc,      Ngc][:, None]     # (Ngc, 1) -> broadcasts
    R_out_ghost = grid.cx1[-Ngc:,      Ngc][:, None]     #            over phi
    par.BC[0] = 'free'; par.BC[2] = 'free'    # overwritten by BC_fixed below
    par.BC_fixed[0] = [(0, grid.Nx2, _equilibrium_at(R_in_ghost))]
    par.BC_fixed[2] = [(0, grid.Nx2, _equilibrium_at(R_out_ghost))]

    par.BC[1] = 'peri'; par.BC[3] = 'peri'    # periodic in azimuth

    return grid, fluid, par, eos



def IC_HD2D_jet_cyl(grid, fluid, par):
    """
    Axisymmetric non-relativistic jet in cylindrical (R, Z) coordinates.

    A supersonic, LIGHT jet is injected through a nozzle at the bottom
    boundary (Z = 0, the x2-inner face) over R < r_jet and propagates in +Z
    into a uniform, pressure-matched ambient medium.  It develops the
    textbook morphology of an astrophysical jet: a bow shock ahead of the
    beam, a Mach disk (terminal shock) where the beam decelerates, and a
    back-flowing cocoon of shocked jet material enveloping the beam.

    Coordinate system : cylindrical (R, Z) = (x1, x2)
    Domain            : R in [0, 5], Z in [0, 25]
    Beam radius       : r_jet = 1
    Inlet (face 1)    : R < r_jet, rho = 1, v_Z = 6, p = rho cs^2/gamma
    Ambient           : rho = 10, v = 0, same p
    Density contrast  : eta = rho_jet / rho_amb = 0.1   (a LIGHT jet)
    Internal Mach no. : v_jet / cs_jet = 6

    The 1D momentum balance between beam and ambient predicts a head
    advance speed

        v_head = v_jet / (1 + sqrt(rho_amb / rho_jet)) = 6 / (1 + sqrt(10))
               ~ 1.44 ,

    so the beam crosses the Z = 25 domain in t ~ 17; the run stops at
    t = 15, just before the head leaves the grid.  Checking the measured
    head position against this estimate is the standard sanity test for a
    jet setup, and is why the domain and final time are chosen together.

    The inlet is a fixed (Dirichlet) ghost-fill registered in
    ``par.BC_fixed[1]`` and applied by ``boundCond_HD`` after the standard
    'wall' fill.  Because the prescribed state lives in the ghost cells, the
    boundary-face Riemann solve produces the correct inlet flux with no
    change to the flux routine.  The interior starts as pure ambient -- the
    jet switches on from the boundary, so there is no initial internal
    discontinuity to relax.

    Parameters
    ----------
    grid : object   Grid; a CylindricalGrid is built here.
    fluid : object  HD SimState (dens, pres, vel1..3).
    par : object    Parameters (BC, BC_fixed, timenow, timefin).

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    Norman, M. L. et al. (1982), A&A 113, 285
    Bodo, G. et al. (1998), A&A 333, 1117
    """
    print("2D axisymmetric non-relativistic jet (cylindrical, inlet BC)")

    # --- grid + time ---
    R_in, R_out = 0.0, 5.0
    Z_in, Z_out = 0.0, 25.0
    grid.CylindricalGrid(R_in, R_out, Z_in, Z_out)
    par.timenow = 0.0
    par.timefin = 15.0
    eos = EOSdata(5.0 / 3.0)

    # --- aliases ---
    Ngc  = grid.Ngc; Nx1  = grid.Nx1; Nx1r = grid.Nx1r

    # --- jet / ambient parameters (Mach 6, pressure equilibrium) ---
    Mach    = 6.0
    rho_jet = 1.0
    rho_amb = 10.0                       # eta = rho_jet/rho_amb = 0.1
    cs_jet  = 1.0                        # normalise the jet sound speed
    v_jet   = Mach * cs_jet              # => internal Mach number = 6
    p_jet   = rho_jet * cs_jet**2 / eos.GAMMA   # p = rho cs^2 / gamma  (= 0.6)
    p_amb   = p_jet                      # pressure-matched across the nozzle
    r_jet   = 1.0

    # --- uniform ambient everywhere (incl. ghosts), vectorised ---
    fluid.dens[:, :] = rho_amb
    fluid.pres[:, :] = p_amb
    fluid.vel1[:, :] = 0.0               # v_R
    fluid.vel2[:, :] = 0.0               # v_Z
    fluid.vel3[:, :] = 0.0               # v_phi

    # --- nozzle extent along R (tangential to the bottom face) ---
    Rc = grid.cx1[Ngc:Nx1r, Ngc]         # 1D interior R cell-centres
    in_jet = np.nonzero(Rc < r_jet)[0]   # contiguous from the axis
    if in_jet.size == 0:
        raise ValueError(
            "jet2Dcyl: the nozzle (R < %g) is not resolved by a single cell at "
            "Nx1 = %d; use a finer radial grid." % (r_jet, Nx1))
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
    #     outflow at R=R_out and Z=Z_out ---
    par.BC[0] = 'axis'                   # x1 inner  (R = 0)
    par.BC[1] = 'wall'                   # x2 inner  (Z = 0, nozzle via BC_fixed)
    par.BC[2] = 'free'                   # x1 outer  (R = 5)
    par.BC[3] = 'free'                   # x2 outer  (Z = 25)
    return grid, fluid, par, eos


# ============================================================================
#   Self-gravitating problems  (see gravity.py / poisson_solver.py)
# ============================================================================

def IC_HD1D_dust_collapse(grid, fluid, par):
    """
    1D spherical collapse of a uniform, pressureless (dust) sphere under its
    own gravity -- the standard quantitative test of a self-gravity solver,
    because it has a closed-form exact solution.

    A uniform sphere of density rho0 and radius R0 released from rest
    collapses HOMOLOGOUSLY: it stays uniform, no shell ever overtakes
    another, and every shell follows the free-fall (cycloid) solution

        r(t) = r0 cos^2(xi) ,    t = t_ff * (2/pi) * (xi + sin(xi) cos(xi)) ,

    reaching r = 0 at the free-fall time

        t_ff = sqrt( 3 pi / (32 G rho0) ) .

    Because the collapse is homologous the interior density stays uniform
    and follows

        rho(t) = rho0 * (R0 / R(t))^3 ,

    so a code can be checked against an exact curve, not merely against
    "looks plausible".  With G = rho0 = R0 = 1 (the units used here),
    t_ff = sqrt(3 pi / 32) = 0.542700...

    Why "dust"
    ----------
    The exact solution assumes ZERO pressure.  A Godunov code cannot run at
    exactly p = 0 (the sound speed and the conservative-to-primitive
    inversion both degenerate), so the gas here is given a tiny pressure --
    a free-fall Mach number of order 10^3 -- which is dynamically negligible
    but keeps the solver well posed.  The run stops at 0.8 t_ff, before the
    central singularity forms and before pressure could matter.

    Gravity
    -------
    Solved with the finite-volume Poisson solver every Runge-Kutta stage
    (via ``selfgravity_poisson_hook``), NOT written once by this function:
    the density is what is collapsing, so a potential computed at t = 0 is
    stale immediately.

    The boundary condition matters here.  This is an ISOLATED object, so the
    outer boundary uses the exact exterior potential of a point mass,

        Phi(r_out) = -G M_tot / r_out    ('dirichlet') ,

    which is constant because no mass leaves the domain.  A pure-Neumann
    ('free' everywhere) setup would NOT do: with no Dirichlet face the
    solver must enforce the solvability condition by subtracting the mean of
    the source, which silently adds a uniform negative background density
    and changes the enclosed mass.  That is the right thing for a periodic
    box (see 'jeans2D') and the wrong thing for an isolated sphere.
    At r = 0, 'free' is exact: dPhi/dr = 0 there by spherical symmetry.

    A spherically symmetric configuration like this one is also handled
    exactly, and far more cheaply, by
    ``gravity.selfgravity_monopole_spherical`` (a radial cumulative sum
    rather than an elliptic solve).  The Poisson solver is used here on
    purpose, so the test measures the general machinery.

    Coordinate system : spherical-polar (r, theta), 1D (Nx2 = 1)
    Domain            : r in [0, 2], theta in [0, pi]  (a full sphere)
    Sphere            : rho = 1 for r < 1, a light ambient outside
    Final time        : 0.8 t_ff

    Parameters
    ----------
    grid : object   Grid; a SphericalPolarGrid is built here.  Use Nx2 = 1.
    fluid : object  HD SimState (dens, pres, vel1..3, F1, F2, body_force).
    par : object    Parameters (BC, timenow, timefin).

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    Truelove, J. K. et al. (1997), ApJ 489, L179   (self-gravity test suite)
    Binney, J. & Tremaine, S., Galactic Dynamics, 2nd ed., section 4.1
    """
    print("1D spherical pressureless (dust) collapse -- exact free-fall solution")

    G = 1.0; rho0 = 1.0; R0 = 1.0
    r_out = 2.0

    grid.SphericalPolarGrid(0.0, r_out, 0.0, np.pi)
    eos = EOSdata(5.0 / 3.0)

    t_ff = np.sqrt(3.0 * np.pi / (32.0 * G * rho0))
    par.timenow = 0.0
    par.timefin = 0.8 * t_ff          # stop before the central singularity

    r = grid.cx1

    # --- uniform sphere in a light ambient ---
    rho_amb = 1.0e-3 * rho0
    fluid.dens[:, :] = np.where(r < R0, rho0, rho_amb)

    # --- negligible pressure: fix the free-fall Mach number, not p itself,
    #     so the "dust" limit is explicit and resolution-independent.
    #     Mach 30 already makes pressure support ~1/Mach^2 ~ 0.1% of gravity
    #     -- indistinguishable from Mach 100 in practice (measured: the
    #     collapsed radius differs by 0.06%) -- while keeping the Riemann
    #     solver away from the genuinely pressureless limit, where the Euler
    #     equations lose strict hyperbolicity and a Godunov scheme fails in
    #     the near-vacuum states that collapse produces. ---
    Mach_ff = 30.0
    v_ff    = np.sqrt(2.0 * G * (4.0 / 3.0 * np.pi * R0**3 * rho0) / R0)
    cs      = v_ff / Mach_ff
    fluid.pres[:, :] = fluid.dens[:, :] * cs**2 / eos.GAMMA

    fluid.vel1[:, :] = 0.0            # released from rest
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0

    # --- self-gravity: exact exterior (point-mass) potential on the outer
    #     face, zero-gradient at the origin.  M_tot is constant: the domain
    #     is closed, so this Dirichlet value never needs updating. ---
    M_tot = (4.0 / 3.0) * np.pi * (R0**3 * rho0
                                    + (r_out**3 - R0**3) * rho_amb)
    BC_phi = ['free', 'free', 'dirichlet', 'free']
    BC_val = {2: -G * M_tot / r_out}
    # A spherical grid spans a huge range of cell volumes (cVol ~ r^2 dr), so
    # the Poisson operator is poorly conditioned near the origin and CG needs
    # more headroom than its default cap of Nx1*Nx2 iterations.  Leaving the
    # default here silently returns an unconverged potential, which at these
    # near-vacuum densities destroys the run outright.
    cg = dict(tol=1e-8, maxiter=max(500, 10 * (grid.Nx1 + grid.Nx2)))
    fluid.body_force = selfgravity_poisson_hook(G=G, BC=BC_phi,
                                                 BC_value=BC_val, **cg)
    # evaluate once so F1/F2 are consistent with the state at t = 0
    fluid = selfgravity_poisson(grid, fluid, par, G=G, BC=BC_phi,
                                 BC_value=BC_val, **cg)

    # --- boundaries: reflecting at the origin, outflow at r_out ---
    par.BC[0] = 'axis'; par.BC[2] = 'free'
    par.BC[1] = 'axis'; par.BC[3] = 'axis'      # unused: Nx2 = 1

    return grid, fluid, par, eos



def IC_HD2D_jeans(grid, fluid, par):
    """
    2D Jeans instability: gravitational growth of a small density
    perturbation in a uniform, self-gravitating medium.

    The textbook linear-theory problem for self-gravity, and the natural
    partner to the dust collapse: that one tests the NONLINEAR solution of an
    isolated body, this one tests the LINEAR growth rate against an exact
    dispersion relation,

        omega^2 = c_s^2 k^2 - 4 pi G rho0 .

    Perturbations with k below the Jeans wavenumber
    k_J = sqrt(4 pi G rho0) / c_s are unstable and grow as exp(sigma t) with
    sigma = sqrt(4 pi G rho0 - c_s^2 k^2); shorter wavelengths are stabilised
    by pressure and merely oscillate as sound waves.  Measuring sigma from a
    run and comparing with the formula is a sharp, quantitative check of the
    coupling between the Poisson solve and the momentum update.

    This setup seeds the PURE GROWING MODE.  A density perturbation alone
    would excite the decaying mode as well and give a contaminated growth
    rate; the matching velocity follows from linearised continuity,

        drho/rho0 = A cos(kx)   =>   dv_x = -(sigma A / k) sin(kx) .

    The Jeans swindle
    -----------------
    A uniform infinite medium is not actually an equilibrium: the unperturbed
    density sources a potential that has nowhere to point.  The standard
    resolution ("Jeans swindle") is to let only the PERTURBATION source the
    potential, i.e. to solve  div grad Phi = 4 pi G (rho - <rho>).  On this
    fully periodic domain the Poisson solver does exactly that on its own:
    with no Dirichlet face the problem is solvable only if the source has
    zero volume integral, so ``solve_poisson`` subtracts the volume-weighted
    mean of the right-hand side automatically.  The swindle is therefore not
    a fudge added here -- it is the solvability condition of the periodic
    Poisson problem.

    Domain            : Cartesian [0, 1] x [0, 1], periodic on all sides
    Background        : rho0 = 1, c_s = 0.4 (=> p0 = rho0 c_s^2 / gamma)
    Perturbation      : one wavelength in x, amplitude 1e-3
    With G = 1        : sigma = sqrt(4 pi - c_s^2 k^2) ~ 2.4999,
                        so the mode e-folds about 5 times by t = 2

    Parameters
    ----------
    grid : object   Grid; a CartesianGrid is built here.
    fluid : object  HD SimState (dens, pres, vel1..3, F1, F2, body_force).
    par : object    Parameters (BC, timenow, timefin).

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    Jeans, J. H. (1902), Phil. Trans. R. Soc. A 199, 1
    Binney, J. & Tremaine, S., Galactic Dynamics, 2nd ed., section 5.2
    """
    print("2D Jeans instability -- linear growth against the exact "
          "dispersion relation")

    grid.CartesianGrid(0.0, 1.0, 0.0, 1.0)
    eos = EOSdata(5.0 / 3.0)

    G     = 1.0
    rho0  = 1.0
    cs    = 0.4
    amp   = 1.0e-3
    L     = 1.0
    k     = 2.0 * np.pi / L            # one wavelength across the box

    p0 = rho0 * cs**2 / eos.GAMMA

    # growth rate of the unstable mode (positive => unstable)
    omega2 = cs**2 * k**2 - 4.0 * np.pi * G * rho0
    if omega2 >= 0.0:
        raise ValueError(
            "jeans2D: the seeded mode is STABLE (omega^2 = %g >= 0). Lower c_s "
            "or lengthen the box so that k < k_J = sqrt(4 pi G rho0)/c_s."
            % omega2)
    sigma = np.sqrt(-omega2)

    par.timenow = 0.0
    par.timefin = 2.0                  # ~5 e-foldings at sigma ~ 2.5

    x = grid.cx1

    # --- pure growing mode: density and velocity phased by linear theory ---
    fluid.dens[:, :] = rho0 * (1.0 + amp * np.cos(k * x))
    fluid.vel1[:, :] = -(sigma * amp / k) * np.sin(k * x)
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    # isentropic perturbation, consistent with the adiabatic sound speed used
    # in the dispersion relation
    fluid.pres[:, :] = p0 * (fluid.dens[:, :] / rho0)**eos.GAMMA

    # --- self-gravity on a periodic box: the mean-subtraction that makes the
    #     pure-Neumann problem solvable IS the Jeans swindle (see docstring) ---
    BC_phi = ['peri', 'peri', 'peri', 'peri']
    fluid.body_force = selfgravity_poisson_hook(G=G, BC=BC_phi)
    fluid = selfgravity_poisson(grid, fluid, par, G=G, BC=BC_phi)

    par.BC[:] = 'peri'

    return grid, fluid, par, eos



def IC_HD2D_cloud_collapse(grid, fluid, par):
    """
    2D axisymmetric collapse of a rotating, self-gravitating cloud in
    cylindrical (R, z) coordinates -- the flattening of a protostellar core.

    A uniform, marginally-bound sphere in solid-body rotation is released.
    Gravity wins over its thermal and rotational support, so it collapses --
    but angular momentum is conserved, so infall along the rotation axis is
    unopposed while infall in the midplane is increasingly centrifugally
    resisted.  The cloud therefore does not collapse to a point: it FLATTENS
    into a rotationally-supported disk, which is the reason protostars are
    born with disks.

    This is the qualitative, genuinely 2D companion to the quantitative 1D
    'collapse1D' and the linear 2D 'jeans2D'.

    Support parameters
    ------------------
    The initial state is fixed by the two standard dimensionless ratios of
    cloud-collapse theory, thermal and rotational energy over the magnitude
    of the gravitational energy of a uniform sphere:

        alpha = E_therm / |W| ,   beta = E_rot / |W| ,
        |W|   = (3/5) G M^2 / R0 ,   E_rot = (1/5) M R0^2 Omega^2 ,
        E_therm = (3/2) p0 V .

    Both well below 1/2 means the cloud is bound and must collapse.  Here
    alpha = 0.1 and beta = 0.15: thermally weak, rotationally significant --
    the combination that flattens rather than fragments.  Solid-body
    rotation gives v_phi = Omega R, carried in vel3, whose centrifugal
    support the cylindrical curvature source term supplies automatically.

    Gravity
    -------
    Poisson solve every Runge-Kutta stage (the density is collapsing).  The
    cloud is isolated, so the outer boundaries take the MONOPOLE potential

        Phi = -G M_tot / sqrt(R^2 + z^2)

    evaluated on each boundary face -- exact for a spherical mass
    distribution and a good approximation while the boundary stays far from
    the (initially spherical) cloud.  M_tot is constant because the domain
    is closed.  On the axis R = 0, 'free' is exact by symmetry.

    Note this is a genuine approximation: once the cloud flattens strongly,
    its exterior potential acquires a quadrupole the monopole boundary does
    not represent.  The boundary is placed at twice the cloud radius to keep
    that error small; a production code would use a multipole expansion.

    Coordinate system : cylindrical (R, z) = (x1, x2)
    Domain            : R in [0, 2], z in [-2, 2]
    Cloud             : uniform rho0 = 1 for sqrt(R^2+z^2) < 1, ambient 1e-2
    Final time        : 1.2 t_ff, by which the disk has formed

    Parameters
    ----------
    grid : object   Grid; a CylindricalGrid is built here.
    fluid : object  HD SimState (dens, pres, vel1..3, F1, F2, body_force).
    par : object    Parameters (BC, timenow, timefin).

    Returns
    -------
    grid, fluid, par, eos : objects

    References
    ----------
    Boss, A. P. & Bodenheimer, P. (1979), ApJ 234, 289
    Norman, M. L., Wilson, J. R. & Barton, R. T. (1980), ApJ 239, 968
    """
    print("2D rotating self-gravitating cloud collapse (cylindrical) -- "
          "disk formation")

    G = 1.0; rho0 = 1.0; R0 = 1.0
    R_out = 2.0; z_half = 2.0

    grid.CylindricalGrid(0.0, R_out, -z_half, z_half)
    eos = EOSdata(5.0 / 3.0)

    alpha = 0.10        # thermal / gravitational energy
    beta  = 0.15        # rotational / gravitational energy

    M_cloud = (4.0 / 3.0) * np.pi * R0**3 * rho0
    V_cloud = (4.0 / 3.0) * np.pi * R0**3
    W_abs   = 0.6 * G * M_cloud**2 / R0            # |W| of a uniform sphere

    p0    = alpha * W_abs / (1.5 * V_cloud)        # from E_therm = (3/2) p0 V
    Omega = np.sqrt(3.0 * G * M_cloud * beta / R0**3)   # from E_rot = beta |W|

    t_ff = np.sqrt(3.0 * np.pi / (32.0 * G * rho0))
    par.timenow = 0.0
    par.timefin = 1.2 * t_ff

    R = grid.cx1; Z = grid.cx2
    rsph = np.sqrt(R**2 + Z**2)
    inside = rsph < R0

    # --- cloud + ambient at the SAME temperature, so the ambient sound speed
    #     (hence the timestep) is not inflated by a hot, tenuous medium ---
    f_amb = 1.0e-2
    fluid.dens[:, :] = np.where(inside, rho0, f_amb * rho0)
    fluid.pres[:, :] = np.where(inside, p0,   f_amb * p0)

    fluid.vel1[:, :] = 0.0                                   # v_R
    fluid.vel2[:, :] = 0.0                                   # v_z
    fluid.vel3[:, :] = np.where(inside, Omega * R, 0.0)      # solid-body v_phi

    # --- self-gravity with monopole Dirichlet boundaries.
    #     apply_bc_scalar_Ngc1 writes a whole ghost row/column, so each face
    #     value must span the FULL tangential extent, ghost cells included. ---
    M_tot = float(np.sum(grid.cVol * fluid.dens[grid.Ngc:-grid.Ngc,
                                                 grid.Ngc:-grid.Ngc]))
    Ngc = grid.Ngc
    R_face = grid.fx1[grid.Nx1r, Ngc]        # outer R face position
    z_lo   = grid.fx2[Ngc, Ngc]              # bottom z face position
    z_hi   = grid.fx2[Ngc, grid.Nx2r]        # top z face position
    R_row  = grid.cx1[:, Ngc]                # R at every x1 index (with ghosts)
    z_col  = grid.cx2[Ngc, :]                # z at every x2 index (with ghosts)

    def _monopole(Rv, Zv):
        return -G * M_tot / np.maximum(np.sqrt(Rv**2 + Zv**2), 1e-30)

    BC_phi = ['free', 'dirichlet', 'dirichlet', 'dirichlet']
    BC_val = {1: _monopole(R_row, z_lo),     # z = z_min face, varies with R
              3: _monopole(R_row, z_hi),     # z = z_max face, varies with R
              2: _monopole(R_face, z_col)}   # R = R_out face, varies with z
    fluid.body_force = selfgravity_poisson_hook(G=G, BC=BC_phi,
                                                 BC_value=BC_val)
    fluid = selfgravity_poisson(grid, fluid, par, G=G, BC=BC_phi,
                                 BC_value=BC_val)

    # --- boundaries: axis at R = 0, outflow elsewhere ---
    par.BC[0] = 'axis'                       # R = 0
    par.BC[1] = 'free'; par.BC[2] = 'free'; par.BC[3] = 'free'

    return grid, fluid, par, eos
