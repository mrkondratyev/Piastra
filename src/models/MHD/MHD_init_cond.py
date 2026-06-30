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
    interp_face_to_cell,
    edge_to_face_curl)



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
    print("magnetized explosion test in 2D spherical-polar geometry")

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
    
    #corner coordinates 
    r_c = grid.fx1[Ngc:Nx1r + 1, Ngc:Nx2r + 1]  
    t_c = grid.fx2[Ngc:Nx1r + 1, Ngc:Nx2r + 1] 
    Aphi = 0.5 * b0 * r_c * np.sin(t_c)
    
    MHD.fb1, MHD.fb2 = edge_to_face_curl(grid, Aphi)
    
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



def IC_MHD2D_Alfven(grid, MHD, par):
    """
    2D circularly polarized Alfven wave (Toth 2000; Gardiner & Stone 2005).
 
    An exact nonlinear MHD eigenmode and a standard convergence / divergence-
    control benchmark.  With rho = 1, P = 1 and a parallel (background) field
    B_par = 1, the Alfven speed is 1, so the wave returns to its initial state
    after each unit of time; |B| and P stay exactly constant in the continuum.
 
    Propagation-frame setup, with  xprop = x1 cos(theta) + x2 sin(theta):
        B_par  = 1
        B_perp = 0.1 cos(2 pi xprop)        (in-plane transverse)
        B_z    = 0.1 sin(2 pi xprop)        (out-of-plane)
        v_par  = 0
        v_perp = -B_perp ,  v_z = -B_z      (Alfven relation, sqrt(rho) = 1)
    rotated into the grid frame exactly as in the supplied Fortran IC:
        B1 = B_par cos t - B_perp sin t ,  B2 = B_par sin t + B_perp cos t
        v1 = B_perp sin t , v2 = -B_perp cos t , v3 = -B_z .
 
    The in-plane field (B1, B2) is built from a corner vector potential A_z and a
    discrete curl, so div B = 0 to MACHINE PRECISION for any theta (CT-consistent;
    also valid for GLM / 8-wave, which simply use the cell-centred field):
        A_z = B_par * yprop - (0.1 / 2pi) sin(2 pi xprop),
        yprop = -x1 sin t + x2 cos t,   B1 = dAz/dx2,  B2 = -dAz/dx1 .
    Cell-centred B1, B2 are then the average of the two bracketing faces.
 
    theta = 0 (default) reproduces the Fortran IC exactly: the wave runs along x1,
    is uniform in x2, with one wavelength in x1 (period 1, Alfven speed 1).  For
    OBLIQUE theta the domain must be chosen so xprop is periodic across the box
    (Gardiner & Stone use Lx = 1, Ly = 1/2, tan(theta) = Lx/Ly, one wavelength
    along the diagonal); set theta and the grid extents accordingly.
 
    Domain [0,1] x [0,1], periodic in both directions; gamma = 5/3,
    t_fin = 5 (five wave crossings).
 
    Parameters
    ----------
    grid : object
    MHD  : object   MHD SimState (dens, pres, vel1..3, bfi1..3, fb1, fb2, bglm)
    par  : object   Parameters (BC, BCm, divb_tr, timenow, timefin)
 
    Returns
    -------
    grid, MHD, par, eos : objects
 
    References
    ----------
    Toth, G. (2000), J. Comput. Phys. 161, 605
    Gardiner, T. A. & Stone, J. M. (2005), J. Comput. Phys. 205, 509
    """
    print("2D circularly polarized Alfven wave (Toth 2000; Gardiner & Stone 2005)")
 
    # --- grid + time ---
    x1ini, x1fin = 0.0, np.sqrt(2.0); x2ini, x2fin = 0.0, np.sqrt(2.0)
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    par.timenow = 0.0; par.timefin = 5.0
 
    eos = EOSdata(5.0 / 3.0)
 
    Ngc  = grid.Ngc
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r
 
    # --- wave parameters ---
    theta = np.pi/4.0
    ct, st = np.cos(theta), np.sin(theta)
    Bpar = 1.0                 # parallel / background field
    dB   = 0.1                 # transverse amplitude
    k    = 2.0 * np.pi         # one wavelength along the propagation direction
 
    # --- uniform background ---
    MHD.dens[:, :] = 1.0
    MHD.pres[:, :] = 1.0
    MHD.bglm[:, :] = 0.0
 
    # --- cell-centred transverse fields and velocities (full array, incl. ghosts) ---
    xprop_c = grid.cx1 * ct + grid.cx2 * st
    Bperp_c = dB * np.cos(k * xprop_c)
    Bz_c    = dB * np.sin(k * xprop_c)
 
    MHD.bfi3[:, :] =  Bz_c
    MHD.vel1[:, :] =  Bperp_c * st
    MHD.vel2[:, :] = -Bperp_c * ct
    MHD.vel3[:, :] = -Bz_c
 
    # --- in-plane field from corner vector potential A_z (divergence-free curl) ---
    xc = grid.fx1[Ngc:Nx1r + 1, Ngc:Nx2r + 1]      # corner x  (Nx1+1, Nx2+1)
    yc = grid.fx2[Ngc:Nx1r + 1, Ngc:Nx2r + 1]      # corner y
    xprop_n = xc * ct + yc * st                    # propagation coord at corners
    yprop_n = -xc * st + yc * ct                   # perpendicular coord at corners
    Az = Bpar * yprop_n - (dB / k) * np.sin(k * xprop_n)
    
    #face-centered magnetic fields
    MHD.fb1, MHD.fb2 = edge_to_face_curl(grid, Az)
 
    # --- cell-centred in-plane field by averaging the bracketing faces ---
    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r))
    MHD.bfi1[sl], MHD.bfi2[sl] = \
        interp_face_to_cell(grid, MHD.fb1, MHD.fb2)
        
    # --- periodic boundaries ---
    par.BC[:] = 'peri'
    par.BCm[:] = par.BC[:]
 
    return grid, MHD, par, eos



def IC_MHD2D_current_sheet(grid, MHD, par):
    """
    2D current sheet / magnetic reconnection test (Athena code, Stone et al (2008)).

    Two anti-parallel current sheets are set up by a magnetic field that points
    along y and reverses sign twice across x. A small, domain-filling velocity
    perturbation perpendicular to the field (v_x = A * sin(2pi y)) excites a standing
    shear Alfven wave; with only numerical resistivity the sheets are unstable to
    tearing, forming plasmoids that merge into the characteristic island pattern.

    This is a stringent divergence-cleaning / numerical-resistivity test: it
    continuously generates div(B) error at the reversals, so it should be run
    with CT or a cleaning scheme and the divB monitor watched.

    Coordinate system : Cartesian (x, y) = (x1, x2), periodic both directions.
    Domain            : [-0.5, 0.5] x [-0.5, 0.5]
    State             : rho = 1, p = beta/2, gamma = 5/3
    Field            : B_y = -B0 for -0.25 < x < 0.25, B0 otherwise; B_x = B_z = 0
    Perturbation      : v_x = A * sin(2pi y);
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
    x1ini, x1fin = -0.5, 0.5; x2ini, x2fin = -0.5, 0.5
    grid.CartesianGrid(x1ini, x1fin, x2ini, x2fin)
    par.timenow = 0.0; par.timefin = 50.0
    eos = EOSdata(5.0 / 3.0)

    # --- aliases ---
    Ngc  = grid.Ngc
    Nx1 = grid.Nx1
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r

    #current sheet parameters 
    B0  = 1.0 # * np.sqrt(4.0*np.pi)
    A = 0.1; beta = 0.01

    # --- uniform fields (incl. ghosts) ---
    MHD.dens[:, :] = 1.0
    MHD.pres[:, :] = beta/2.0
    MHD.vel2[:, :] = 0.0
    MHD.vel3[:, :] = 0.0
    MHD.bfi3[:, :] = 0.0   
    MHD.fb1[:, :] = 0.0 
    
    # --- cell-centred block ---
    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r))

    # --- staggered fb2:
    center = np.abs(grid.fx1[Ngc:Ngc + Nx1, 1]) < 0.25 
    MHD.fb2[:Nx1, :] = np.where(center[:, None], -B0, B0)
    
    MHD.vel1[sl] = A * np.sin(2.0 * np.pi * grid.cx2[sl]) 

    MHD.bfi1[sl], MHD.bfi2[sl] = \
        interp_face_to_cell(grid, MHD.fb1, MHD.fb2)

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
    
    # --- face B from discrete curl of A_z (divergence-free by construction) 
    MHD.fb1, MHD.fb2 = edge_to_face_curl(grid, Az)

    # --- cell-centred B by averaging faces (consistent with fb1/fb2) 
    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r))
    MHD.bfi1[sl], MHD.bfi2[sl] = \
        interp_face_to_cell(grid, MHD.fb1, MHD.fb2)
    par.BC[:] = 'peri'
    par.BCm[:] = par.BC[:]
    
    return grid, MHD, par, eos



def IC_MHD2D_disk(grid, MHD, par):
    """
    Newtonian constant-angular-momentum torus (Papaloizou & Pringle 1984;
    Hawley 1991/2000), seeded with a weak poloidal field for the MRI.

    Point-mass gravity Phi = -GM/r (no pseudo-Newtonian throat, no excision).
    A torus of CONSTANT specific angular momentum l = sqrt(GM*R_max) is in
    hydrostatic + centrifugal equilibrium; with a weak field it goes MRI-
    unstable after a few orbits.  Cell-centred field + divergence cleaning (GLM).

    Construction
    ------------
        Keplerian l at the pressure maximum:   l = sqrt(GM * R_max)     (constant)
        enthalpy integral (polytrope p = K rho^gamma):
            (gamma/(gamma-1)) p/rho = W = C + GM/r - l^2 / (2 R^2)
        inner edge sets C (W = 0 at R = R_inner, z = 0):
            C = -GM/R_inner + l^2 / (2 R_inner^2)
        K fixed by rho_max = 1 at the pressure maximum (R_max, 0):
            K = (gamma-1)/gamma * W_max ,   W_max = W(R_max, 0)
        rotation:   v_phi = l / R          (the SAME l as the enthalpy term)
        gravity:    a = -grad Phi = -GM r_vec / r^3   (stored in F as the
                    actual acceleration; the solver applies Res += -dens*F)

    Bound-torus condition
    ---------------------
        W -> C as r -> infinity, so the torus closes only if C < 0, i.e.
            R_inner > R_max / 2 .
        (With R_max = 1, R_inner = 0.65 the torus spans R in [0.65, 2.17],
         peaks at R = 1, and has half-thickness z ~ 0.6 at the pressure max.)

    Coordinate system : cylindrical (R, z) = (x1, x2).
    Geometry          : CylindricalGrid, R in [0.4, 2.6], z in [-1, 1].
    Divergence control: cleaning only (run with divb_tr = 'GLM').

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
    Papaloizou, J. C. B. & Pringle, J. E. (1984), MNRAS 208, 721
    Hawley, J. F. (2000), ApJ 528, 462
    """
    print("2D Newtonian constant-l accretion torus (Papaloizou-Pringle / Hawley)")

    # --- grid ---
    R_in_g, R_out_g = 0.4, 2.6
    Z_bot,  Z_top   = -1.0, 1.0
    grid.CylindricalGrid(R_in_g, R_out_g, Z_bot, Z_top)

    eos   = EOSdata(5.0 / 3.0)
    gamma = eos.GAMMA

    # --- torus parameters ---
    GM       = 1.0
    R_max    = 1.0          # pressure maximum
    R_inner  = 0.65         # inner edge   (must exceed R_max/2 for a bound torus)
    beta_min = 100.0        # min plasma beta of the seed field (large -> weak)
    rho_max  = 1.0
    n_orbit  = 20.0         # run length in orbital periods at R_max

    if R_inner <= 0.5 * R_max:
        raise ValueError("unbound torus: need R_inner > R_max/2 (C must be < 0).")

    l_kep = np.sqrt(GM * R_max)                       # constant specific ang. mom.
    C     = -GM / R_inner + l_kep**2 / (2.0 * R_inner**2)
    W_max = C + GM / R_max - l_kep**2 / (2.0 * R_max**2)
    if W_max <= 0.0:
        raise ValueError("torus does not close (W_max <= 0): check R_inner, R_max.")
    Kpoly = (gamma - 1.0) / gamma * W_max / rho_max**(gamma - 1.0)

    T_orbit = 2.0 * np.pi * R_max / np.sqrt(GM / R_max)
    par.timenow = 0.0
    par.timefin = n_orbit * T_orbit

    # --- aliases / coordinates ---
    Ngc  = grid.Ngc
    Nx1r = grid.Nx1r; Nx2r = grid.Nx2r
    sl = (slice(Ngc, Nx1r), slice(Ngc, Nx2r))
    R = grid.cx1[sl]; Z = grid.cx2[sl]
    rsph  = np.sqrt(R**2 + Z**2)
    Rsafe = np.maximum(R, 1e-30)

    # --- enthalpy, density, pressure ---
    W      = C + GM / rsph - l_kep**2 / (2.0 * Rsafe**2)
    inside = W > 0.0
    rho_t  = np.where(inside,
                      (np.maximum(W, 0.0) / W_max)**(1.0 / (gamma - 1.0)) * rho_max,
                      0.0)

    rho_atm = 1.0e-3 * rho_max          # ambient floor (see note in chat: not HSE)
    p_atm   = Kpoly * rho_atm**gamma
    in_t    = rho_t > rho_atm
    rho     = np.where(in_t, rho_t, rho_atm)
    pres    = np.where(in_t, Kpoly * rho_t**gamma, p_atm)

    # --- zero fields (full arrays incl. ghosts) ---
    MHD.vel1[:, :] = MHD.vel2[:, :] = 0.0
    MHD.bfi1[:, :] = MHD.bfi2[:, :] = MHD.bfi3[:, :] = 0.0
    MHD.fb1[:, :]  = MHD.fb2[:, :]  = 0.0
    MHD.bglm[:, :] = 0.0

    # --- cell-centred profiles ---
    MHD.dens[sl] = rho
    MHD.pres[sl] = pres
    MHD.vel3[sl] = np.where(in_t, l_kep / Rsafe, 0.0)     # v_phi = l / R

    # --- gravity: store the ACTUAL acceleration  a = -GM r_vec / r^3 (inward).
    #     solver applies Res += -dens*F with U -= dt*Res, so F holds a itself
    #     (negative components), matching IC_HD2D_gap_opening and the RTI ICs. ---
    inv = GM / rsph**3
    MHD.F1[:, :] = -inv * R         # a_R = -GM R / r^3   (inward, NEGATIVE)
    MHD.F2[:, :] = -inv * Z         # a_z = -GM z / r^3   (inward, NEGATIVE)

    # --- weak poloidal seed: A_phi ~ max(rho - rho_cut, 0), scaled to beta_min ---
    #     B_R = -dA_phi/dz ,  B_z = (1/R) d(R A_phi)/dR     (curl of A_phi e_phi)
    rho_cut = 0.2 * rho_max
    Aphi    = np.maximum(rho_t - rho_cut, 0.0)
    R1d = grid.cx1[Ngc:Nx1r, Ngc]
    Z1d = grid.cx2[Ngc, Ngc:Nx2r]
    B_R = -np.gradient(Aphi,        Z1d, axis=1)
    B_z =  np.gradient(Rsafe * Aphi, R1d, axis=0) / Rsafe

    pmag  = 0.5 * (B_R**2 + B_z**2)
    ratio = np.max(np.where(in_t & (pmag > 0.0), pmag / pres, 0.0))   # = 1/beta
    fac   = np.sqrt((1.0 / beta_min) / ratio) if ratio > 0.0 else 0.0
    MHD.bfi1[sl] = fac * B_R
    MHD.bfi2[sl] = fac * B_z

    # --- boundaries ---
    par.BC[0]  = 'wall'; par.BC[1:3]  = 'wall' 
    par.BCm[:] = 'free'

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
    
    par.BCm[0] = 'axis'; par.BCm[1:3] = 'free'

    return grid, MHD, par, eos

