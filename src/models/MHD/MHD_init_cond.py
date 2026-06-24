# -*- coding: utf-8 -*-
"""
Newtonian MHD Initial Conditions Module

This module provides functions to set up 1D and 2D MHD
test problems. It initializes the grid, fluid state, 
and simulation time parameters.

Author: mrkondratyev
Date: June 17, 2024
"""

import numpy as np
from src.common.eos_setup import EOSdata
from src.grid.grid_misc import (
    interp_face_to_cell)



def IC_MHD_user_defined(grid, MHD, par):
    """
    user-defined MHD problem.


    Parameters
    ----------
    grid : object
        Grid class object
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
    
    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.5
    
    eos = EOSdata(5.0/3.0)
    
    MHD.vel1[:, :] = MHD.vel2[:, :] = MHD.vel3[:, :] = 0.0   
    MHD.bfi2[:, :] = MHD.bfi3[:, :] = 0.0
    
    MHD.fb2[:, :] = 0.0
    
    MHD.dens[:, :] = 1.0
    MHD.pres[:, :] = 5.0/12.0/np.pi  
    
    MHD.bfi1[:, :] = 1.0
    MHD.fb1[:, :] = 1.0
    
    #boundary conditions
    #all support walls, periodic, axis and free-outflow boundaries
    par.BC[:] = 'peri'
    par.BCm[:] = par.BC[:]
    
    raise ValueError(
        "User-defined MHD problem – see 'MHD_init_cond.py', "
        "set your ICs and remove this line."
    )
    
    return grid, MHD, par, eos



# ============================================================================
#   1D problems
# ============================================================================
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

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 1.0

    eos = EOSdata(5.0 / 3.0)
    
    Ngc  = grid.Ngc
    Nx1  = grid.Nx1
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r

    rho0 = 1.0; p0   = 0.1
    B0   = 1.0; amp  = 0.1

    # --- uniform background ---
    MHD.dens[:, :] = rho0
    MHD.pres[:, :] = p0
    MHD.vel1[:, :] = 0.0
    MHD.bfi1[:, :] = B0
    MHD.fb1[:, :]  = B0

    # --- staggered fb2: indexed 0..Nx1-1, coordinate from the ghost-offset
    #     face array fx1[Ngc:Ngc+Nx1, Ngc]; sin varies along x, broadcast
    #     across columns (original wrote MHD.fb2[i, :]).
    x_face = grid.fx1[Ngc:Ngc + Nx1, Ngc]                 # shape (Nx1,)
    MHD.fb2[:Nx1, :] = amp * np.sin(2.0 * np.pi * x_face)[:, None]

    # --- cell-centered block: smooth sinusoids; velocity tied to B by the
    #     Alfven relation v_perp = -B_perp / sqrt(rho0). Nx2r+1 upper bound.
    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r + 1))
    x  = grid.cx1[sl]

    MHD.bfi2[sl] = amp * np.sin(2.0 * np.pi * x)
    MHD.bfi3[sl] = amp * np.cos(2.0 * np.pi * x)
    MHD.vel2[sl] = -MHD.bfi2[sl] / np.sqrt(rho0)
    MHD.vel3[sl] = -MHD.bfi3[sl] / np.sqrt(rho0)

    par.BC[:] = 'peri'
    par.BCm[:] = par.BC[:]

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

    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)

    par.timenow = 0.0; par.timefin = 0.1
    
    eos = EOSdata(10.0 / 5.0)

    Ngc  = grid.Ngc
    Nx1  = grid.Nx1
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r

    MHD.vel1[:, :] = MHD.vel2[:, :] = MHD.vel3[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0
    MHD.bfi1[:, :] = 0.75
    MHD.fb1[:, :]  = 0.75

    # --- staggered fb2:
    left_face = grid.fx1[Ngc:Ngc + Nx1, 1] < 0.5 
    MHD.fb2[:Nx1, :] = np.where(left_face[:, None], 1.0, -1.0)

    # --- cell-centered block
    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r + 1))
    left = grid.cx1[sl] < 0.5

    MHD.dens[sl] = np.where(left, 1.0,  0.125)
    MHD.pres[sl] = np.where(left, 1.0,  0.1)
    MHD.bfi2[sl] = np.where(left, 1.0, -1.0)

    par.BC[:] = 'free'
    par.BCm[:] = par.BC[:]

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
    
    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    par.timenow = 0.0; par.timefin = 0.08

    eos = EOSdata(5.0 / 3.0)
    Ngc  = grid.Ngc
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r

    b0 = 5.0 / np.sqrt(4.0 * np.pi)        # uniform field magnitude

    MHD.vel2[:, :] = MHD.vel3[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0
    MHD.bfi1[:, :] = b0
    MHD.fb1[:, :]  = b0
    MHD.fb2[:, :]  = b0
    MHD.bfi2[:, :] = b0
    MHD.dens[:, :] = 1.0

    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r + 1))
    left = grid.cx1[sl] < 0.5

    MHD.pres[sl] = np.where(left, 20.0, 1.0)
    MHD.vel1[sl] = np.where(left, 10.0, -10.0)

    par.BC[:] = 'free'
    par.BCm[:] = par.BC[:]
    
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

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.2

    eos = EOSdata(5.0 / 3.0)
    
    Ngc  = grid.Ngc
    Nx1  = grid.Nx1
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r

    s4pi = np.sqrt(4.0 * np.pi)
    # uniform normal field
    Bx   = 2.0 / s4pi                      

    MHD.bfi1[:, :] = Bx; MHD.fb1[:, :]  = Bx
    MHD.bfi3[:, :] = 2.0 / s4pi

    # --- staggered fb2 
    left_face = grid.fx1[Ngc:Ngc + Nx1, 1] < 0.5          # shape (Nx1,)
    MHD.fb2[:Nx1, :] = np.where(left_face[:, None], 3.6 / s4pi, 4.0 / s4pi)

    # --- cell-centered block
    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r + 1))
    left = grid.cx1[sl] < 0.5

    MHD.dens[sl] = np.where(left, 1.08, 1.0)
    MHD.vel1[sl] = np.where(left, 1.2, 0.0)
    MHD.vel2[sl] = np.where(left, 0.01, 0.0)
    MHD.vel3[sl] = np.where(left, 0.5, 0.0)
    MHD.pres[sl] = np.where(left, 0.95, 1.0)
    MHD.bfi2[sl] = np.where(left, 3.6 / s4pi, 4.0 / s4pi)

    par.BC[:] = 'free'
    par.BCm[:] = par.BC[:]

    return grid, MHD, par, eos



# ============================================================================
#   2D problems
# ============================================================================
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
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.2
    
    eos = EOSdata(7.0 / 5.0)
    
    Ngc  = grid.Ngc
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r
    
    # uniform diagonal field component
    b0 = 1.0 / np.sqrt(2.0)                 
    
    MHD.vel1[:, :] = MHD.vel2[:, :] = MHD.vel3[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0
    MHD.dens[:, :] = 1.0
    MHD.bfi1[:, :] = b0; MHD.bfi2[:, :] = b0
    MHD.fb1[:, :]  = b0; MHD.fb2[:, :]  = b0
    
    sl  = (slice(Ngc, Nx1r), slice(Ngc, Nx2r))
    rad = np.sqrt((grid.fx1[sl] - 0.5)**2 + (grid.fx2[sl] - 0.5)**2)
    MHD.pres[sl] = np.where(rad < 0.1, 10.0, 0.1)
    
    par.BC[:] = 'free'
    par.BCm[:] = par.BC[:]
    
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
    x1ini, x1fin = 0.0, 0.5; x2ini, x2fin = -0.5, 0.5
    grid.CylindricalGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.2

    eos = EOSdata(7.0 / 5.0)
    
    Ngc  = grid.Ngc
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r

    MHD.vel1[:, :] = MHD.vel2[:, :] = MHD.vel3[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0
    MHD.dens[:, :] = 1.0
    MHD.bfi1[:, :] = 0.0; MHD.bfi2[:, :] = 1.0
    MHD.fb1[:, :]  = 0.0; MHD.fb2[:, :]  = 1.0
    
    sl  = (slice(Ngc, Nx1r), slice(Ngc, Nx2r))
    rad = np.sqrt(grid.cx1[sl]**2 + grid.cx2[sl]**2)
    MHD.pres[sl] = np.where(rad < 0.1, 10.0, 0.1)

    par.BC[0] = 'axis'; par.BC[1] = 'free'
    par.BC[2] = 'free'; par.BC[3] = 'free'
    
    par.BCm[:] = par.BC[:]
    
    return grid, MHD, par, eos



def IC_MHD2D_blast_sph(grid, MHD, par):
    """
    2D magnetized explosion test (spherical axisymmetry).

    A standard blast wave problem in a magnetized medium. 
    A high-pressure circular region is initialized in the center 
    of a uniform low-pressure medium with a vertical background magnetic field.

    Parameters
    ----------
    grid : object
        spherical grid.
    MHD : object
        MHD state container.
    par : object
        Simulation parameters.

    Returns
    -------
    grid : object
        Updated Spherical grid.
    MHD : object
        Initialized density, pressure, velocity, and magnetic fields.
    par : object
        Parameters with timefin = 0.2 and BCs.
    eos : EOSdata
        Equation of state with γ = 7/5.
    """    
    print("magnetized explosion test in 2D planar Cartesian geometry")
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = 0.0, 0.5; x2ini, x2fin = 0.0, np.pi
    grid.SphericalPolarGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.2
    
    eos = EOSdata(7.0 / 5.0)
    
    Ngc  = grid.Ngc
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r
    
    # uniform vertical field component
    b0 = 1.0                
    
    MHD.vel1[:, :] = MHD.vel2[:, :] = MHD.vel3[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0
    MHD.dens[:, :] = 1.0
    MHD.bfi1[:, :] = b0*np.cos(grid.cx2)
    MHD.bfi2[:, :] = -b0*np.sin(grid.cx2)
    
    #corner coordinates 
    r_c = grid.fx1[Ngc:Nx1r + 1, Ngc:Nx2r + 1]  
    t_c = grid.fx2[Ngc:Nx1r + 1, Ngc:Nx2r + 1] 
    Aphi = 0.5 * b0 * r_c * np.sin(t_c)

    MHD.fb1 = (Aphi[:,1:]*grid.edg3[:,1:]  - Aphi[:,:-1]*grid.edg3[:,:-1])/(grid.fS1[:,:]+1e-30)
    MHD.fb2 = -(Aphi[1:,:]*grid.edg3[1:,:] - Aphi[:-1,:]*grid.edg3[:-1,:])/(grid.fS2[:,:]+1e-30)
    
    MHD.bfi1[Ngc:Nx1r, Ngc:Nx2r], MHD.bfi2[Ngc:Nx1r, Ngc:Nx2r] = \
        interp_face_to_cell(grid, MHD.fb1, MHD.fb2)
    
    MHD.pres[:, :] = np.where(grid.cx1 < 0.1, 10.0, 0.1)
    
    par.BC[0] = 'axis'; par.BC[1] = 'axis'
    par.BC[2] = 'free'; par.BC[3] = 'axis'
    
    par.BCm[:] = par.BC[:]
    
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
    
    #grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    par.timenow = 0.0; par.timefin = 0.5

    eos = EOSdata(5.0 / 3.0)
    Ngc  = grid.Ngc
    Nx1  = grid.Nx1; Nx2 = grid.Nx2
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r

    # field amplitude
    b0 = 1.0 / np.sqrt(4.0 * np.pi)        

    MHD.vel3[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0
    MHD.dens[:, :] = 25.0 / 36.0 / np.pi
    MHD.pres[:, :] =  5.0 / 12.0 / np.pi

    # --- staggered fb1: 
    MHD.fb1[:Nx1 + 1, :] = -np.sin(2.0 * np.pi * grid.cx2[:Nx1 + 1, Ngc:-Ngc]) * b0

    # --- staggered fb2: 
    MHD.fb2[:, :Nx2 + 1] =  np.sin(4.0 * np.pi * grid.cx1[Ngc:-Ngc, :Nx2 + 1]) * b0

    # --- cell-centered block
    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r))
    x  = grid.cx1[sl]; y  = grid.cx2[sl]

    MHD.bfi1[sl] = -np.sin(2.0 * np.pi * y) * b0
    MHD.bfi2[sl] =  np.sin(4.0 * np.pi * x) * b0
    MHD.vel1[sl] = -np.sin(2.0 * np.pi * y)
    MHD.vel2[sl] =  np.sin(2.0 * np.pi * x)

    par.BC[:] = 'peri'
    par.BCm[:] = par.BC[:]
    
    return grid, MHD, par, eos



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
    
    # grid creation
    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.15

    eos = EOSdata(7.0 / 5.0)
    Ngc  = grid.Ngc
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r
    
    # uniform Bx
    b0 = 5.0 / np.sqrt(4.0 * np.pi)       

    # ------------------------------------------------------------------
    # Initial uniform medium
    # ------------------------------------------------------------------
    MHD.vel1[:, :] = MHD.vel2[:, :] = MHD.vel3[:, :] = 0.0
    MHD.bfi1[:, :] = b0; MHD.bfi2[:, :] = 0.0; MHD.bfi3[:, :] = 0.0
    MHD.fb1[:, :]  = b0; MHD.fb2[:, :]  = 0.0
    MHD.pres[:, :] = 1.0

    # Rotor parameters
    x0, y0  = 0.5, 0.5
    r0, r1  = 0.1, 0.115
    rho_in, rho_out = 10.0, 1.0
    omega   = 20.0

    # ------------------------------------------------------------------
    # Fill domain: three regions (rotor / taper / ambient) collapse into one
    # set of formulas via a clipped taper f, where
    #   f = 1            (r <= r0,   solid rotor)
    #   f = (r1-r)/(r1-r0)  (r0 < r <= r1, linear taper)
    #   f = 0            (r > r1,    ambient)
    # ------------------------------------------------------------------
    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r))
    dx = grid.cx1[sl] - x0; dy = grid.cx2[sl] - y0
    r  = np.sqrt(dx*dx + dy*dy)

    f = np.clip((r1 - r) / (r1 - r0), 0.0, 1.0)    # 1 inside r0, 0 outside r1

    MHD.dens[sl] = rho_out + (rho_in - rho_out) * f
    MHD.vel1[sl] = -omega * dy * f; MHD.vel2[sl] =  omega * dx * f

    par.BC[:] = 'free'
    par.BCm[:] = par.BC[:]
    
    return grid, MHD, par, eos



def IC_MHD2D_current_sheet(grid, MHD, par):
    """
    2D current sheet / magnetic reconnection test (Gardiner & Stone 2005).

    Two anti-parallel current sheets are set up by a magnetic field that points
    along x and reverses sign twice across y. A small, domain-filling velocity
    perturbation PARALLEL to the field (v_x = amp * sin(pi y)) excites a standing
    shear Alfven wave; with only numerical resistivity the sheets are unstable to
    tearing, forming plasmoids that merge into the characteristic island pattern.

    This is a stringent divergence-cleaning / numerical-resistivity test: it
    continuously generates div(B) error at the reversals, so it should be run
    with CT or a cleaning scheme and the divB monitor watched.

    Coordinate system : Cartesian (x, y) = (x1, x2), periodic both directions.
    Domain            : [0, 2] x [0, 2]
    State             : rho = 1, p = 0.1, gamma = 5/3
    Field            : B_x = +B0 for 0.5 < y < 1.5, -B0 otherwise; B_y = B_z = 0
    Perturbation      : v_x = amp * sin(pi y)  (field-parallel, amp = 0.1);
                        v_y = v_z = 0

    NOTE on convention: the canonical statement uses B_y(x) with v_x perturbation;
    this routine uses the equivalent x<->y relabelling (B_x(y), v_x(y) seed). The
    physics (field-parallel shear across anti-parallel sheets) is identical.

    Parameters
    ----------
    grid : object   CartesianGrid, cx1, cx2, Ngc, Nx1, Nx2, Nx1r, Nx2r.
    MHD  : object    MHD SimState (dens, pres, vel1..3, bfi1..3, fb1, fb2).
    par  : object    Parameters (BC, timenow, timefin).

    Returns
    -------
    grid, MHD, par, eos : objects

    References
    ----------
    Gardiner, T. A. & Stone, J. M. (2005), J. Comput. Phys. 205, 509
    Fromang, S. et al. (2006), A&A 457, 371
    """
    print("2D MHD current sheet / reconnection test (Gardiner & Stone 2005)")

    # --- grid + time ---
    x1ini, x1fin = 0.0, 2.0
    x2ini, x2fin = 0.0, 2.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    par.timenow = 0.0
    par.timefin = 5.0
    eos = EOSdata(5.0 / 3.0)

    # --- aliases ---
    Ngc  = grid.Ngc
    Nx1  = grid.Nx1
    Nx2  = grid.Nx2
    Nx1r = grid.Nx1r
    Nx2r = grid.Nx2r

    B0  = 1.0
    amp = 0.1

    # --- uniform fields (incl. ghosts) ---
    MHD.dens[:, :] = 1.0
    MHD.pres[:, :] = 0.1
    MHD.vel1[:, :] = 0.0
    MHD.vel2[:, :] = 0.0
    MHD.vel3[:, :] = 0.0
    MHD.bfi2[:, :] = 0.0      # B_y = 0
    MHD.bfi3[:, :] = 0.0      # B_z = 0
    MHD.fb2[:, :]  = 0.0      # staggered B_y on x2-faces = 0

    # --- staggered B_x on x1-faces (fb1): B_x depends only on y, so each
    #     face column just takes the cell-centre y of that column. ---
    #     fb1 has shape (Nx1+1, Nx2): one extra point along x1 (the faces),
    #     Nx2 cells along x2.  Sample interior cell-centre y for the columns.
    yc_face = grid.cx2[Ngc, Ngc:Nx2r]                  # (Nx2,) interior y centres
    Bx_col  = np.where((yc_face > 0.5) & (yc_face < 1.5), B0, -B0)   # (Nx2,)
    MHD.fb1[:, :] = Bx_col[None, :]                    # broadcast over all x1-faces

    # --- cell-centred block ---
    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r))
    y  = grid.cx2[sl]
    in_sheet = (y > 0.5) & (y < 1.5)

    MHD.bfi1[sl] = np.where(in_sheet, B0, -B0)         # B_x(y)
    MHD.vel1[sl] = amp * np.sin(np.pi * y)             # field-parallel seed

    # --- boundaries: periodic in both directions ---
    par.BC[:] = 'peri'
    par.BCm[:] = par.BC[:]

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

    x1ini, x1fin = -1.0, 1.0; x2ini, x2fin = -0.5, 0.5
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 1.0

    eos = EOSdata(5.0 / 3.0)
    
    Ngc  = grid.Ngc
    Nx1  = grid.Nx1;  Nx2  = grid.Nx2
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r

    R0 = 0.3; A0 = 1.0e-3

    # --- background state (uniform) ---
    MHD.dens[:, :] = 1.0
    MHD.pres[:, :] = 1.0
    MHD.vel1[:, :] = 2.0; MHD.vel2[:, :] = 1.0; MHD.vel3[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0

    # --- vector potential A_z at cell corners: A_z = A0*(R0 - r), 0 outside --- 
    xc = grid.fx1[Ngc:Ngc + Nx1 + 1, Ngc:Ngc + Nx2 + 1]   # corner x  (Nx1+1, Nx2+1)
    yc = grid.fx2[Ngc:Ngc + Nx1 + 1, Ngc:Ngc + Nx2 + 1]   # corner y
    rc = np.sqrt(xc**2 + yc**2)
    Az = np.where(rc < R0, A0 * (R0 - rc), 0.0) # (Nx1+1, Nx2+1)

    dx = grid.dx1[Ngc, Ngc]; dy = grid.dx2[Ngc, Ngc]      # uniform spacing

    # --- face B from discrete curl of A_z (divergence-free by construction) 
    MHD.fb1[:Nx1 + 1, :Nx2] =  (Az[:, 1:] - Az[:, :-1]) / dy    # (Nx1+1, Nx2)
    MHD.fb2[:Nx1, :Nx2 + 1] = -(Az[1:, :] - Az[:-1, :]) / dx    # (Nx1,  Nx2+1)

    # --- cell-centred B by averaging faces (consistent with fb1/fb2) 
    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r))
    MHD.bfi1[sl] = 0.5 * (MHD.fb1[:Nx1, :Nx2] + MHD.fb1[1:Nx1 + 1, :Nx2])
    MHD.bfi2[sl] = 0.5 * (MHD.fb2[:Nx1, :Nx2] + MHD.fb2[:Nx1, 1:Nx2 + 1])

    par.BC[:] = 'peri'
    par.BCm[:] = par.BC[:]
    
    return grid, MHD, par, eos



def IC_MHD2D_disk(grid, MHD, par):
    """
    Magnetized accretion torus -- the 'magnetized accretion torus' test of
    Mignone et al. (2007), Sec. 5.5 (cylindrical cases c/d), after Hawley (2000).

    A constant-angular-momentum torus in a pseudo-Newtonian (Paczynski-Wiita)
    potential Phi = -1/(r-1), seeded with a weak poloidal field, becomes MRI-
    unstable and turbulent after a few orbits.  Cell-centred field treatment
    (GLM / divergence cleaning); the staggered-CT seed is a later extension.

    Setup (Mignone+2007 Sec. 5.5):
      potential   Phi = -1/(r_sph - 1)           (pseudo-Newtonian, r_g = 1)
      inner edge  r = 3   (sets the integration constant C)
      pressure max / l_kep evaluation at r = 4.7, where T_orbit = 50
      l_kep = r^{3/2} / (r - 1)   evaluated at r = 4.7  (constant over the torus)
      enthalpy integral:
          (gamma/(gamma-1)) p/rho = C - Phi - l_kep^2 / (2 R^2)
      polytrope   p = K rho^gamma ,  gamma = 5/3
      field       A_phi ∝ min[rho(R,z) - 5, 0] (with rho normalised so rho_max=...),
                  normalised so min(2p/|B|^2) = beta_min = 100
    Cylindrical box (case c/d): 0 <= R <= 20, -20 <= z <= 20 (uniform core
    1.5 <= R <= 11.5, -5 <= z <= 5 in the paper; here a single uniform grid).
    Region r_sph < 1.5 is excluded from the computation.

    DIVERGENCE CONTROL -- CLEANING ONLY (GLM).  Run with divb_tr='GLM'.

    Parameters
    ----------
    grid : object   CylindricalGrid, cx1, cx2, Ngc, Nx1, Nx2, Nx1r, Nx2r.
    MHD  : object    MHD SimState (dens, pres, vel1..3, bfi1..3, fb1, fb2, bglm,
                     F1, F2).
    par  : object    Parameters (BC, BCm, divb_tr, timenow, timefin).

    Returns
    -------
    grid, MHD, par, eos : objects

    References
    ----------
    Mignone, A. et al. (2007), ApJS 170, 228, Sec. 5.5
    Hawley, J. F. (2000), ApJ 528, 462
    Paczynski, B. & Wiita, P. J. (1980), A&A 88, 23   (pseudo-Newtonian potential)
    """
    print("2D magnetized accretion torus (Mignone et al. 2007, Sec. 5.5)")

    # --- grid + time ---
    R_in_g, R_out_g = 3.0, 20.0
    Z_bot, Z_top    = -20.0, 20.0
    grid.CylindricalGrid(R_in_g, R_out_g, Z_bot, Z_top)

    gamma   = 5.0 / 3.0
    R_inner = 3.0          # torus inner edge (sets C)
    R_max   = 4.7          # pressure maximum (l_kep evaluation point)
    r_excl  = 1.5          # excision radius (r_sph < r_excl excluded)
    beta_min = 100.0       # min(2 p / |B|^2)
    eos = EOSdata(gamma)

    # --- pseudo-Newtonian (Paczynski-Wiita) potential and constant l_kep ---
    def Phi(rsph):
        return -1.0 / (rsph - 1.0)

    l_kep = R_max**1.5 / (R_max - 1.0)          # specific ang. mom. at r=4.7
    T0    = 50.0                                  # orbital period at R_max (paper)
    par.timenow = 0.0
    par.timefin = 6.0 * T0                        # several orbits  <-- adjust

    # --- aliases ---
    Ngc  = grid.Ngc
    Nx1 = grid.Nx1; Nx2 = grid.Nx2
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r
    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r))

    R = grid.cx1[sl]; Z = grid.cx2[sl]
    rsph = np.sqrt(R**2 + Z**2)

    # --- enthalpy: (gamma/(gamma-1)) p/rho = C - Phi - l_kep^2/(2 R^2) ---
    # In cylindrical (R,z): r sin(theta) = R, so the rotation term is l^2/(2 R^2).
    # C fixes the inner edge (midplane, R = R_inner, z = 0): enthalpy = 0 there.
    C = Phi(R_inner) + l_kep * l_kep / (2.0 * R_inner * R_inner)
    Rsafe = np.maximum(R, 1e-30)
    Wenth = C - Phi(rsph) - l_kep * l_kep / (2.0 * Rsafe * Rsafe)   # = (g/(g-1)) p/rho
    inside = (Wenth > 0.0) & (rsph > r_excl)

    # --- polytropic density from the enthalpy: p/rho = (g-1)/g * Wenth,
    #     and p = K rho^g  =>  rho = [ (g-1)/(g K) * Wenth ]^{1/(g-1)}.
    #     K is fixed by choosing rho_max = 1 at the pressure maximum (R_max, 0). ---
    W_max = C - Phi(R_max) - l_kep * l_kep / (2.0 * R_max * R_max)
    if W_max <= 0.0:
        raise ValueError("torus does not close (W_max <= 0): check R_inner, R_max, l_kep.")
    rho_max = 1.0
    Kpoly = (gamma - 1.0) / gamma * W_max / rho_max**(gamma - 1.0)

    rho_t = np.where(inside,
        ((gamma - 1.0) / (gamma * Kpoly) * np.maximum(Wenth, 0.0))**(1.0 / (gamma - 1.0)),
        0.0)

    rho_atm = 1.0e-2 * rho_max
    p_atm   = Kpoly * rho_atm**gamma
    in_t    = rho_t > rho_atm
    rho     = np.where(in_t, rho_t, rho_atm)
    pres    = np.where(in_t, Kpoly * rho_t**gamma, p_atm)

    # --- uniform fields (full arrays, incl. ghosts) ---
    MHD.vel1[:, :] = 0.0
    MHD.vel2[:, :] = 0.0
    MHD.bfi1[:, :] = 0.0
    MHD.bfi2[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0
    MHD.fb1[:, :]  = 0.0
    MHD.fb2[:, :]  = 0.0
    if hasattr(MHD, 'bglm'):
        MHD.bglm[:, :] = 0.0

    # --- cell-centred profiles ---
    MHD.dens[sl] = rho
    MHD.pres[sl] = pres
    # constant specific angular momentum: v_phi = l_kep / R (in the torus)
    MHD.vel3[sl] = np.where(in_t, l_kep / Rsafe, 0.0)

    # --- gravity acceleration a = -dPhi/dr * r_hat,  Phi = -1/(r-1)
    #     a_r = -1/(r-1)^2 (inward).  a_R = a_r R/r , a_z = a_r z/r.
    #     Solver applies Res += -dens*F, so store F = -a (positive, inward pull).
    inv = 1.0 / (rsph - 1.0)**2
    MHD.F1[:, :] = inv * R / rsph        # F_R = +(R/r)/(r-1)^2
    MHD.F2[:, :] = inv * Z / rsph        # F_z = +(z/r)/(r-1)^2

    # --- weak poloidal seed from A_phi ∝ min[rho - rho_cut, 0] (paper: rho-5),
    #     scaled to our rho_max=1 normalisation as a fraction of rho_max. ---
    rho_cut = 0.2 * rho_max                       # paper uses 5 (on rho_max~25); ~0.2 here
    Az = np.maximum(rho_t - rho_cut, 0.0)         # >0 only well inside the torus

    R1d = grid.cx1[Ngc:Nx1r, Ngc]                 # 1D R (axis 0)
    Z1d = grid.cx2[Ngc, Ngc:Nx2r]                 # 1D z (axis 1)
    dAz_dZ  = np.gradient(Az,     Z1d, axis=1)
    dRAz_dR = np.gradient(R * Az, R1d, axis=0)
    B_R =  -dAz_dZ
    B_z =   dRAz_dR / Rsafe

    pmag  = 0.5 * (B_R * B_R + B_z * B_z)
    ratio = np.max(np.where(in_t & (pmag > 0.0), pmag / pres, 0.0))   # = 1/beta_cur
    fac   = 0.0#np.sqrt((1.0 / beta_min) / ratio) if ratio > 0.0 else 0.0
    MHD.bfi1[sl] = fac * B_R
    MHD.bfi2[sl] = fac * B_z

    # --- boundaries: axis at R=0, outflow elsewhere (paper Sec. 5.5) ---
    par.BC[0] = 'free'    # x1 inner (R = 0)
    par.BC[1] = 'free'    # x2 inner (z = -20)
    par.BC[2] = 'free'    # x1 outer (R = 20)
    par.BC[3] = 'free'    # x2 outer (z = +20)
    par.BCm[:] = par.BC[:]

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

    x1ini, x1fin = 0.0, 1.0; x2ini, x2fin = 0.0, 1.0
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    
    par.timenow = 0.0; par.timefin = 0.06

    eos = EOSdata(5.0 / 3.0)
    
    Ngc  = grid.Ngc
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r

    # Cloud / shock parameters
    xc, yc    = 0.25, 0.5
    rc        = 0.1
    rho_cloud = 10.0
    rho_amb   = 1.0;  p_amb = 1.0
    B0        = 2.0 / np.sqrt(4.0 * np.pi)
    rho_post  = 3.857143; v1_post = 11.2; p_post = 167.0
    x_shock   = 0.05

    MHD.vel2[:, :] = 0.0  ; MHD.vel3[:, :] = 0.0
    # field-aligned with shock normal
    MHD.bfi1[:, :] = B0; MHD.bfi2[:, :] = 0.0; MHD.bfi3[:, :] = 0.0
    MHD.fb1[:, :]  = B0; MHD.fb2[:, :]  = 0.0 

    # --- cell-centered block: priority chain  post-shock > cloud > ambient ---
    #     (the elif means cloud applies only where NOT post-shock)
    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r))
    x  = grid.cx1[sl]
    y  = grid.cx2[sl]
    r  = np.sqrt((x - xc)**2 + (y - yc)**2)

    post  = x < x_shock          # highest priority
    cloud = r < rc               # only where not post-shock

    # only dens, vel1, pres differ; nested where preserves elif precedence
    MHD.dens[sl] = np.where(post, rho_post, np.where(cloud, rho_cloud, rho_amb))
    MHD.vel1[sl] = np.where(post, v1_post, 0.0)
    MHD.pres[sl] = np.where(post, p_post,  p_amb)

    par.BC[:] = 'free'
    par.BCm[:] = par.BC[:]

    return grid, MHD, par, eos



def IC_MHD2D_jet_cyl(grid, MHD, par):
    """
    Axisymmetric magnetized non-relativistic jet in cylindrical (R, Z) coords.

    Simplified teaching version of the PLUTO MHD jet (Mignone et al. 2007):
    a supersonic light beam is injected through a nozzle on the BOTTOM boundary
    (x2-inner, face 1) over R < r_jet, carrying a CONSTANT axial field B_z (and,
    optionally, a constant toroidal B_phi).  The ambient is uniform and threaded
    by the same axial B_z so the poloidal flux does not terminate in vacuum.

    Simplifications (vs. the full Tesileanu/PLUTO setup):
      * fields are CONSTANT across the nozzle (no radial B_z, B_phi profiles),
        so the inlet values are scalars and the nozzle gas pressure is uniform;
      * a pure axial field (B_phi = 0) is in exact radial equilibrium.  A
        constant B_phi != 0 is NOT in radial balance near the axis (its hoop
        stress ~ B_phi^2 / R diverges as R -> 0); it is offered only as a crude
        option and the beam will readjust near the axis if it is used.

    Coordinate system : cylindrical (R, Z) = (x1, x2)
    Domain            : R in [0, 5], Z in [0, 20]
    Inlet (face 1)    : R < r_jet, rho=rho_jet, v_Z=Mach*cs, B_z=B0 (const),
                        B_phi=Bphi0 (const, default 0), p=p_jet
    Ambient           : rho=rho_amb, v=0, B_z=B0 (axial), total-pressure matched
    Density ratio     : eta = rho_jet / rho_amb = 0.1
    Magnetization     : beta_jet = 2 p_jet / B0^2

    DIVERGENCE CONTROL -- CLEANING ONLY (GLM preferred):
      Field imposed as a cell-centred Dirichlet ghost-fill (bfi1=0, bfi2=B0,
      bfi3=Bphi0) via par.BC_fixed[1], plus pinning GLM psi (bglm)=0.  NOT
      CT-compatible.  Run with divb_tr='GLM' (preferred) or '8wave'; watch
      max|divB| near the nozzle.  The poloidal jump at R=r_jet is a physical
      current sheet.

    Requires: Parameters with BC_fixed = {0:[],1:[],2:[],3:[]}; boundCond_MHD
    applying apply_bc_fixed with bfi1/2/3 and bglm in state_fields;
    par.divb_tr in ('GLM','8wave').

    Parameters
    ----------
    grid : object   CylindricalGrid, cx1, Ngc, Nx1, Nx1r, Nx2.
    MHD  : object    MHD SimState (dens, pres, vel1..3, bfi1..3, fb1, fb2, bglm).
    par  : object    Parameters (BC, BC_fixed, divb_tr, timenow, timefin).

    Returns
    -------
    grid, MHD, par, eos : objects

    References
    ----------
    Mignone, A. et al. (2007), ApJS 170, 228   (PLUTO; MHD/Jet test, simplified)
    """
    print("2D axisymmetric magnetized jet (cylindrical, constant inlet field, GLM)")

    # --- grid + time ---
    R_in, R_out = 0.0, 5.0
    Z_in, Z_out = 0.0, 20.0
    grid.CylindricalGrid(R_in, R_out, Z_in, Z_out)
    par.timenow = 0.0
    par.timefin = 15.0
    eos = EOSdata(5.0 / 3.0)

    # --- aliases ---
    Ngc  = grid.Ngc
    Nx1  = grid.Nx1
    Nx1r = grid.Nx1r

    # --- jet / ambient parameters ---
    Mach     = 6.0
    rho_jet  = 1.0
    rho_amb  = 10.0                       # eta = 0.1
    cs_jet   = 1.0
    v_jet    = Mach * cs_jet              # internal Mach = 6
    p_jet    = rho_jet * cs_jet**2 / eos.GAMMA   # gas pressure (= 0.6)
    r_jet    = 1.0

    beta_jet = 100.0                      # 2 p_jet / B0^2  -> axial field strength
    B0       = np.sqrt(2.0 * p_jet / beta_jet)   # constant axial field B_z
    Bphi0    = 0.0                        # constant toroidal field (0 = pure axial)

    # --- ambient: uniform, threaded by axial B_z = B0; total-pressure matched ---
    p_amb = p_jet + 0.5 * B0**2           # gas+magnetic balance with the beam

    MHD.dens[:, :] = rho_amb
    MHD.pres[:, :] = p_amb
    MHD.vel1[:, :] = 0.0
    MHD.vel2[:, :] = 0.0
    MHD.vel3[:, :] = 0.0
    MHD.bfi1[:, :] = 0.0                  # B_R = 0
    MHD.bfi2[:, :] = B0                   # B_Z = axial ambient field
    MHD.bfi3[:, :] = 0.0                  # B_phi = 0 in ambient
    MHD.fb1[:, :]  = 0.0                  # staggered faces unused (cleaning run)
    MHD.fb2[:, :]  = B0
    MHD.bglm[:, :] = 0.0

    # --- nozzle extent along R (tangential to the bottom face) ---
    Rc = grid.cx1[Ngc:Nx1r, Ngc]          # 1D interior R cell-centres
    in_jet = np.nonzero(Rc < r_jet)[0]    # contiguous from the axis
    start  = int(in_jet[0])               # 0
    end    = int(in_jet[-1]) + 1

    # --- fixed (Dirichlet) inlet on the bottom face (face 1): all scalars ---
    jet_state = {'dens': rho_jet, 'pres': p_jet,
                 'vel1': 0.0, 'vel2': v_jet, 'vel3': 0.0,
                 'bfi1': 0.0, 'bfi2': B0, 'bfi3': Bphi0, 'bglm' : 0}
    par.BC_fixed[1] = [(start, end, jet_state)]

    # --- boundaries ---
    par.BC[0] = 'axis'    # x1 inner (R = 0)
    par.BC[1] = 'wall'    # x2 inner (Z = 0, nozzle via BC_fixed[1])
    par.BC[2] = 'free'    # x1 outer (R = 5)
    par.BC[3] = 'free'    # x2 outer (R = 20)

    return grid, MHD, par, eos

