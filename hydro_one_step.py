# -*- coding: utf-8 -*-
"""
Hydro2D - Container class for 2D compressible hydrodynamics routines.

Hybrid approach:
- Functions remain modular and pedagogically simple
- Lightweight container class provides a clean interface for timestepping
  and groups related routines for compressible 2D hydrodynamics

This class handles:
- CFL-limited timestep calculation
- Single-step Runge-Kutta updates (RK1, RK2, RK3)
- Residual computation via Godunov-type finite-volume method
- Approximate Riemann solver HDux evaluation in 2D
- Primitive-to-conservative and conservative-to-primitive transformations
- Boundary condition handling

The underlying methods are suitable for explicit, finite-volume hydrodynamics
simulations with compressible HDows. The code assumes that the following objects
are provided:

Attributes
----------
g : object
    Grid object containing domain size, spacing, volumes, and face areas.
HD : object
    Fluid state object containing primitive (density, velocities, pressure) and
    conservative (mass, momentum, total energy) variables.
eos : object
    Equation of state object providing methods such as sound_speed, etc.
params : object
    Simulation parameters including:
        - CFL : Courant number
        - RK_order : 'RK1', 'RK2', or 'RK3'
        - HDux_type : type of approximate Riemann solver
        - rec_type : reconstruction type
        - BC : boundary condition type
        - phystime, phystimefin : current and final physical time

Example usage
-------------
>>> hydro = Hydro2D(g, HD, eos, par)
>>> HD = hydro.step_RK()  # advances the solution by one RK timestep
"""


import numpy as np
import copy
from hydro_phys import *
from reconstruction import VarReconstruct


class Hydro2D:
    """
    Container class for 2D compressible hydrodynamics routines.

    Attributes
    ----------
    g : object
        Grid object with domain sizes, spacing, volumes, and face areas.
    HD : object
        FluidState object containing primitive and conservative variables.
    par : object
        Simulation parameters including CFL, RK_order, HDux_type, rec_type, phystime, phystimefin.
    eos : object
        Equation of state object.
    """

    def __init__(self, g, HD, eos, par):
        """
        Initialize the Hydro2D container.

        Parameters
        ----------
        g : object
            Grid object.
        HD : object
            FluidState object.
        eos : object
            Equation of state object.
        par : object
            Simulation parameters object.
        """
        self.g = g
        self.HD = HD
        self.eos = eos
        self.par = par

    def step_RK(self):
        """
        Perform a single Runge-Kutta timestep.

        Returns
        -------
        HD : object
            Updated FluidState object.
        """
        dt = min(CFLcondition_hydro(self.g, self.HD, self.eos, self.par.CFL),
                 self.par.timefin - self.par.timenow)
        self.HD = oneStep_hydro_RK(self.g, self.HD, self.eos, self.par, dt)
        self.par.timenow += dt
        return self.HD



# -------------------------
# Function definitions
# -------------------------
def CFLcondition_hydro(g, HD, eos, CFL):
    """
    Compute the maximum stable timestep for 2D compressible hydrodynamics
    according to the CFL (Courant-Friedrichs-Lewy) condition.
    
    The CFL condition ensures that during a timestep, the fastest wave in the system
    does not propagate more than one cell.
    
    Notes
    -----
    - This function accounts for both advection velocities and local sound speed.
    - Based on the local cell size in each direction.
    - For compressible HDows, sound speed is computed using the EOS.
    
    Parameters
    ----------
    g : object
        Grid object with attributes dx1, dx2 (cell spacings) and Ngc (ghost cells).
    HD : object
        Fluid state object with attributes dens, vel1, vel2 (density and velocities).
    eos : object
        Equation of state object providing sound_speed(density, pressure).
    CFL : HDoat
        CFL number (0 < CFL <= 1) controlling timestep size.
    
    Returns
    -------
    dt : HDoat
        Maximum stable timestep according to CFL condition.
    """
    #make copy of ghost cells number to simplify indexing below 
    Ngc = g.Ngc
    
    #sound speed calculation for whole domain
    sound = eos.sound_speed(HD.dens[Ngc:-Ngc, Ngc:-Ngc], HD.pres[Ngc:-Ngc, Ngc:-Ngc])
    
    #FIRST APPROACH
    #maximal possible timestep in each direction
    #dt1 = np.min( g.dx1[Ngc:-Ngc, Ngc:-Ngc] / (np.abs(HD.vel1[Ngc:-Ngc, Ngc:-Ngc]) + sound) )
    #dt2 = np.min( g.dx2[Ngc:-Ngc, Ngc:-Ngc] / (np.abs(HD.vel2[Ngc:-Ngc, Ngc:-Ngc]) + sound) )
    #return CFL * min(dt1, dt2)
    
    #SECOND APPROACH 
    dt_inv = np.max((np.abs(HD.vel1[Ngc:-Ngc, Ngc:-Ngc]) + sound)/g.dx1[Ngc:-Ngc, Ngc:-Ngc] + \
        (np.abs(HD.vel2[Ngc:-Ngc, Ngc:-Ngc]) + sound)/g.dx2[Ngc:-Ngc, Ngc:-Ngc])
    return CFL/dt_inv




def oneStep_hydro_RK(g, HD, eos, par, dt):
    """
    Perform a single Runge-Kutta timestep for 2D compressible hydrodynamics.

    This function implements first-, second-, and third-order explicit Runge-Kutta
    schemes for updating the conservative HDuid variables, including
    primitive variable recovery using the EOS.

    Notes
    -----
    - RK1: simple forward Euler update (1st order)
    - RK2: predictor-corrector scheme (2nd order)
    - RK3: 3-stage scheme (3rd order)
    - Fluxes are calculated using approximate Riemann solvers (Godunov-type)
    - Residuals are computed via finite-volume integral form:
        U_t + RES = 0
        RES = (1/Volume) * sum(HDux * face_area)
    - At each stage:
        1. Fill ghost zones according to boundary conditions.
        2. Reconstruct primitive variables at cell faces.
        3. Compute HDuxes via Riemann solver.
        4. Compute residuals.
        5. Update conservative variables.
        6. Recover primitive variables for next stage.
    - Stages 1-4 are done for 2 directions in 2D 
        
    Parameters
    ----------
    g : object
        Grid object with attributes Nx1, Nx2, Ngc, dx1, dx2, fS1, fS2, cVol.
    HD : object
        Fluid state object containing:
            - dens, vel1, vel2, vel3, pres : primitive variables
            - mass, mom1, mom2, mom3, etot : conservative variables
    eos : object
        Equation of state object.
    par : object
        Simulation parameters including:
            - CFL : CFL number
            - RK_order : 'RK1', 'RK2', or 'RK3'
            - phystime, phystimefin : current and final simulation time
    dt : HDoat
        Suggested timestep (bounded by CFL condition).

    Returns
    -------
    HD : object
        Updated FluidState object after one Runge-Kutta timestep.
    """
    
    #define local copy of ghost cells number to simplify array indexing
    Ngc = g.Ngc
    
    #here we define the copy for the auxilary HDuid state
    HD_h = copy.deepcopy(HD)
    
    #conservative variables at the beginning of timestep
    HD.mass, HD.mom1, HD.mom2, HD.mom3, HD.etot = \
        prim2cons_nr_hydro(HD.dens[Ngc:-Ngc,Ngc:-Ngc], 
        HD.vel1[Ngc:-Ngc,Ngc:-Ngc], HD.vel2[Ngc:-Ngc,Ngc:-Ngc], 
        HD.vel3[Ngc:-Ngc,Ngc:-Ngc], HD.pres[Ngc:-Ngc,Ngc:-Ngc], eos)
    
    #residuals for conservative variables calculation
    #1st Runge-Kutta iteration - predictor stage
    ResM, Res1, Res2, Res3, ResE = flux_calc_hydro(g, HD, par, eos)
    
    # Conservative update - 1st RK iteration (predictor stage)
    HD_h.mass = HD.mass - dt * ResM 
    HD_h.mom1 = HD.mom1 - dt * Res1 
    HD_h.mom2 = HD.mom2 - dt * Res2 
    HD_h.mom3 = HD.mom3 - dt * Res3 
    HD_h.etot = HD.etot - dt * ResE 
    
    #first order Runge-Kutta scheme
    if (par.RK_order == 'RK1'): 
        
        #simply rewrite the conservative state here for clarity
        HD.mass = HD_h.mass
        HD.mom1 = HD_h.mom1
        HD.mom2 = HD_h.mom2
        HD.mom3 = HD_h.mom3
        HD.etot = HD_h.etot
    
    #second-order Runge-Kutta scheme
    if (par.RK_order == 'RK2'):
        
        #Primitive variables recovery after predictor stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        HD_h.dens[Ngc:-Ngc, Ngc:-Ngc], HD_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.vel2[Ngc:-Ngc, Ngc:-Ngc], HD_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.pres[Ngc:-Ngc, Ngc:-Ngc] = cons2prim_nr_hydro(HD_h.mass, 
            HD_h.mom1, HD_h.mom2, HD_h.mom3, HD_h.etot, eos) 
            
        #2nd Runge-Kutta iteration - corrector stage
        ResM, Res1, Res2, Res3, ResE = flux_calc_hydro(g, HD_h, par, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy
        HD.mass = (HD_h.mass + HD.mass) / 2.0 - dt * ResM / 2.0
        HD.mom1 = (HD_h.mom1 + HD.mom1) / 2.0 - dt * Res1 / 2.0 
        HD.mom2 = (HD_h.mom2 + HD.mom2) / 2.0 - dt * Res2 / 2.0 
        HD.mom3 = (HD_h.mom3 + HD.mom3) / 2.0 - dt * Res3 / 2.0  
        HD.etot = (HD_h.etot + HD.etot) / 2.0 - dt * ResE / 2.0 
    
    if (par.RK_order == 'RK3'):
        
        #Primitive variables recovery after 1st RK stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        HD_h.dens[Ngc:-Ngc, Ngc:-Ngc], HD_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.vel2[Ngc:-Ngc, Ngc:-Ngc], HD_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.pres[Ngc:-Ngc, Ngc:-Ngc] = cons2prim_nr_hydro(HD_h.mass, 
            HD_h.mom1, HD_h.mom2, HD_h.mom3, HD_h.etot, eos) 
        
        #residuals for conservative variables calculation
        #2nd Runge-Kutta iteration 
        ResM, Res1, Res2, Res3, ResE = flux_calc_hydro(g, HD_h, par, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy
        HD_h.mass = (HD_h.mass + 3.0 * HD.mass) / 4.0 - dt * ResM / 4.0
        HD_h.mom1 = (HD_h.mom1 + 3.0 * HD.mom1) / 4.0 - dt * Res1 / 4.0 
        HD_h.mom2 = (HD_h.mom2 + 3.0 * HD.mom2) / 4.0 - dt * Res2 / 4.0 
        HD_h.mom3 = (HD_h.mom3 + 3.0 * HD.mom3) / 4.0 - dt * Res3 / 4.0  
        HD_h.etot = (HD_h.etot + 3.0 * HD.etot) / 4.0 - dt * ResE / 4.0 
    
        # Primitive variables recovery after second stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        HD_h.dens[Ngc:-Ngc, Ngc:-Ngc], HD_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.vel2[Ngc:-Ngc, Ngc:-Ngc], HD_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            HD_h.pres[Ngc:-Ngc, Ngc:-Ngc] = cons2prim_nr_hydro(HD_h.mass, 
            HD_h.mom1, HD_h.mom2, HD_h.mom3, HD_h.etot, eos)
        
        ResM, Res1, Res2, Res3, ResE = flux_calc_hydro(g, HD_h, par, eos)
        
        # Conservative update - final 3rd RK iteration
        # update mass, three components of momentum and total energy
        HD.mass = (2.0 * HD_h.mass + HD.mass) / 3.0 - 2.0 * dt * ResM / 3.0
        HD.mom1 = (2.0 * HD_h.mom1 + HD.mom1) / 3.0 - 2.0 * dt * Res1 / 3.0 
        HD.mom2 = (2.0 * HD_h.mom2 + HD.mom2) / 3.0 - 2.0 * dt * Res2 / 3.0 
        HD.mom3 = (2.0 * HD_h.mom3 + HD.mom3) / 3.0 - 2.0 * dt * Res3 / 3.0  
        HD.etot = (2.0 * HD_h.etot + HD.etot) / 3.0 - 2.0 * dt * ResE / 3.0 
        
    # Primitive variables recovery at the end of the timestep
    #density, 3 components of velocity and pressure are evaluated 
    HD.dens[Ngc:-Ngc, Ngc:-Ngc], HD.vel1[Ngc:-Ngc, Ngc:-Ngc], \
        HD.vel2[Ngc:-Ngc, Ngc:-Ngc], HD.vel3[Ngc:-Ngc, Ngc:-Ngc], \
        HD.pres[Ngc:-Ngc, Ngc:-Ngc] = cons2prim_nr_hydro(HD.mass, 
        HD.mom1, HD.mom2, HD.mom3, HD.etot, eos)
    
    #return the updated class object of the HDuid state on the next timestep 
    return HD




def flux_calc_hydro(g, HD, par, eos):
    """
    Compute residuals for conservative variables in 2D compressible hydrodynamics.
    
    Notes
    ----------
    Residuals are calculated using a Godunov-type method:
    - boundary conditions are taken into account via ghost cells,
    - primitive variables are reconstructed to cell faces,
    - HDuxes are computed via approximate Riemann solvers,
    - source terms are calculated, if needed, 
    - residuals are obtained via finite-volume integral form.    
    
    Parameters
    ----------
    g : object
        Grid object with attributes Nx1, Nx2, Ngc, fS1, fS2, cVol.
    HD : object
        Fluid state object at current time step.
    params : object
        Simulation parameters including reconstruction type (rec_type) and flux_type.
    eos : object
        Equation of state object.

    Returns
    -------
    ResM : np.ndarray
        Residual array for mass density.
    Res1 : np.ndarray
        Residual array for x-momentum.
    Res2 : np.ndarray
        Residual array for y-momentum.
    Res3 : np.ndarray
        Residual array for z-momentum.
    ResE : np.ndarray
        Residual array for total energy.
    """
    #fill the ghost cells
    HD = boundCond_HD(g, par.BC, HD)
    
    #make copies of ghost cell numbers to simplify indexing below 
    Ngc = g.Ngc 
    
    #residuals initialization (only for real cells)
    ResM = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res1 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res2 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    Res3 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    ResE = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    
    #HDuxes in 1-dimension 
    if (g.Nx1 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 1-dim
        #here we reconstruct density, 3 components of velocity and pressure
        dens_L, dens_R = VarReconstruct(HD.dens, g, par.rec_type, 1)
        vel1_L, vel1_R = VarReconstruct(HD.vel1, g, par.rec_type, 1)
        vel2_L, vel2_R = VarReconstruct(HD.vel2, g, par.rec_type, 1)
        vel3_L, vel3_R = VarReconstruct(HD.vel3, g, par.rec_type, 1)
        pres_L, pres_R = VarReconstruct(HD.pres, g, par.rec_type, 1)

        #HDuxes calculation with approximate Riemann solver (see HDux_type) in 1-dim
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            Riemann_nr_hydro(dens_L, dens_R, \
                vel1_L, vel1_R, vel2_L, vel2_R, vel3_L, vel3_R, \
                pres_L, pres_R, eos, par.flux_type, 1)
        
        #residuals calculation for mass, 3 components of momentum and total energy in 1-dim
        ResM = ( Fmass[1:,:]*g.fS1[1:,:] - Fmass[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        Res1 = ( Fmomx[1:,:]*g.fS1[1:,:] - Fmomx[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        Res2 = ( Fmomy[1:,:]*g.fS1[1:,:] - Fmomy[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        Res3 = ( Fmomz[1:,:]*g.fS1[1:,:] - Fmomz[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        ResE = ( Fetot[1:,:]*g.fS1[1:,:] - Fetot[:-1,:]*g.fS1[:-1,:] ) / g.cVol[:,:]
        
        
    #HDuxes in 2-dimension
    if (g.Nx2 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 2-dim
        #here we reconstruct density, 3 components of velocity and pressure
        dens_L, dens_R = VarReconstruct(HD.dens, g, par.rec_type, 2)
        pres_L, pres_R = VarReconstruct(HD.pres, g, par.rec_type, 2)
        vel1_L, vel1_R = VarReconstruct(HD.vel1, g, par.rec_type, 2)
        vel2_L, vel2_R = VarReconstruct(HD.vel2, g, par.rec_type, 2)
        vel3_L, vel3_R = VarReconstruct(HD.vel3, g, par.rec_type, 2)
     
        #HDuxes calculation with approximate Riemann solver (see HDux_type) in 2-dim
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            Riemann_nr_hydro(dens_L, dens_R, \
                vel1_L, vel1_R, vel2_L, vel2_R, vel3_L, vel3_R, \
                pres_L, pres_R, eos, par.flux_type, 2)
        
        #residuals calculation for mass, 3 components of momentum and total energy in 2-dim
        #here we add the HDuxes differences to the residuals after 1-dim calculation
        ResM += ( Fmass[:,1:]*g.fS2[:,1:] - Fmass[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        Res1 += ( Fmomx[:,1:]*g.fS2[:,1:] - Fmomx[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        Res2 += ( Fmomy[:,1:]*g.fS2[:,1:] - Fmomy[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        Res3 += ( Fmomz[:,1:]*g.fS2[:,1:] - Fmomz[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        ResE += ( Fetot[:,1:]*g.fS2[:,1:] - Fetot[:,:-1]*g.fS2[:,:-1] ) / g.cVol[:,:]
        
    #curvature source terms for different curvilinear coordinates
    ST1, ST2, ST3 = hydro_curv_sources(g, HD)
    
    #finally, here we add the external force + curvature source terms
    #source term for momentum residual 
    Res1 += - HD.dens[Ngc:-Ngc, Ngc:-Ngc] * HD.F1[:,:] - ST1
    Res2 += - HD.dens[Ngc:-Ngc, Ngc:-Ngc] * HD.F2[:,:] - ST2
    Res3 += - ST3
    #source term for energy residual 
    ResE += - HD.dens[Ngc:-Ngc, Ngc:-Ngc]*(HD.F1[:,:] * HD.vel1[Ngc:-Ngc, Ngc:-Ngc] + HD.F2[:,:] * HD.vel2[Ngc:-Ngc, Ngc:-Ngc])
             
    #return the residuals for mass, 3 components of momentum and total energy
    return ResM, Res1, Res2, Res3, ResE



    
def hydro_curv_sources(g, HD):
    """
    Compute geometric source terms for the hydrodynamic equations 
    in curvilinear coordinates (finite-volume formulation).

    In Cartesian coordinates, the Euler equations are source-free, but in 
    curvilinear geometries (e.g., cylindrical, spherical) additional terms 
    appear due to the divergence operator expressed in non-Cartesian bases.
    This function evaluates those terms for momentum equations.

    Currently implemented for cylindrical ('cyl') and polar ('pol') geometries

    Parameters
    ----------
    g : object
        Grid object containing:
        - ``geom`` : str, geometry type ('cyl' supported).
        - ``cx1`` : ndarray, radial cell-center positions.
        - ``Ngc`` : int, number of ghost cells.
        - ``Nx1, Nx2`` : int, number of grid points.
    HD : object
        Fluid state containing:
        - ``dens`` : ndarray, density field.
        - ``pres`` : ndarray, pressure field.
        - ``vel1`` : ndarray, radial velocity.
        - ``vel3`` : ndarray, azimuthal velocity.

    Returns
    -------
    ST1 : ndarray
        Radial momentum source term.
    ST2 : ndarray
        Axial momentum source term (zero in cylindrical geometry).
    ST3 : ndarray
        Azimuthal momentum source term.

    Notes
    -----
    - Arrays are allocated with the full grid size (including ghost cells).
    - Source terms are nonzero only inside the physical domain
      (ghost zones excluded).
    - Extension to spherical coordinates would require additional terms.
    """
    Ngc = g.Ngc 
    ST1 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    ST2 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    ST3 = np.zeros((g.Nx1, g.Nx2), dtype=np.double)
    
    #cylindrical (R,Z) geometry
    if (g.geom == 'cyl'):
        ST1 = (HD.pres[Ngc:-Ngc,Ngc:-Ngc] +
            HD.dens[Ngc:-Ngc,Ngc:-Ngc] * HD.vel3[Ngc:-Ngc,Ngc:-Ngc]**2) / \
            g.cx1[Ngc:-Ngc,Ngc:-Ngc]

        ST3 = -HD.dens[Ngc:-Ngc,Ngc:-Ngc] * HD.vel3[Ngc:-Ngc,Ngc:-Ngc] * \
            HD.vel1[Ngc:-Ngc,Ngc:-Ngc] / g.cx1[Ngc:-Ngc,Ngc:-Ngc]
    
    #polar (R,phi) geometry
    if (g.geom == 'pol'):
        ST1 = (HD.pres[Ngc:-Ngc,Ngc:-Ngc] +
            HD.dens[Ngc:-Ngc,Ngc:-Ngc] * HD.vel2[Ngc:-Ngc,Ngc:-Ngc]**2) / \
            g.cx1[Ngc:-Ngc,Ngc:-Ngc]
        ST3 = -HD.dens[Ngc:-Ngc,Ngc:-Ngc] * HD.vel2[Ngc:-Ngc,Ngc:-Ngc] * \
            HD.vel1[Ngc:-Ngc,Ngc:-Ngc] / g.cx1[Ngc:-Ngc,Ngc:-Ngc]
            
    return ST1, ST2, ST3


