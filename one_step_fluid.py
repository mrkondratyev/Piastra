# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:41:18 2023

@author: mrkondratyev
"""
from ideal_hydro_functions import prim2cons_idealHydro, cons2prim_idealHydro
from boundaries import boundCond_fluid 
from reconstruction import VarReconstruct 
import numpy as np 
import copy 
from flux_numpy import Riemann_flux_nr_fluid 




"""

in the function "oneStep_hydro_RK" we call all the key ingredients of our compressible fluid simulations.
the high order in space can be adjusted by rec_type = 'PCM' (1st order), 'PLM' (2nd order) or 'WENO' (3rd or 5th order CWENO or WENO5 methods, see reconstuction.py)
the Riemann problem approximate solution can be switched bewteen Local Lax-Friedrichs (Rusanov) flux ('flux_type = 'LLF'), HLL flux ('HLL') or HLLC flux ('HLLC') 
here we use multistage Total Variation Diminishing Runge-Kutta timestepping with "RK_order" = 'RK1', 'RK2' or 'RK3'. 
input: GRID class object (grid), FLUIDSTATE class object (fluid) at time t, EOSdata class object (eos),
timestep dt, "rec_type" -- type of reconstruction, "flux_type" -- Riemann problem solution approximation method and "RK_order" -- order of temporal integration
output: FLUIDSTATE class object (fluid) at time t + dt

For RK timestepping one can see (Shu and Osher (1988))

for a given timestep dt and a primitive fluid state, we calculate conservative state and the residuals for them 
if RK method is beyond the first order, we additionally introduce the intermediate conservative and primitive states
on the predictor stage, we update the initial fluid state to the intermediate one on each stage, 
and after the final stage, we update the fluid state itself, using the information from the intermediate stages 

"""

#this function provides one hydro time step
def oneStep_fluid_RK(grid, fluid, eos, dt, rec_type, flux_type, RK_order):
    
    
    #define local copy of ghost cells number to simplify array indexing
    Ngc = grid.Ngc
    
    #here we define the copy for the auxilary fluid state
    fluid_h = copy.deepcopy(fluid)
    
    #conservative variables at the beginning of timestep
    fluid.mass, fluid.mom1, fluid.mom2, fluid.mom3, fluid.etot = \
        prim2cons_idealHydro(fluid.dens[Ngc:-Ngc,Ngc:-Ngc], 
        fluid.vel1[Ngc:-Ngc,Ngc:-Ngc], fluid.vel2[Ngc:-Ngc,Ngc:-Ngc], 
        fluid.vel3[Ngc:-Ngc,Ngc:-Ngc], fluid.pres[Ngc:-Ngc,Ngc:-Ngc], eos)
    
    #residuals for conservative variables calculation
    #1st Runge-Kutta iteration - predictor stage
    ResM, Res1, Res2, Res3, ResE = flux_calc_hydro(grid, fluid, rec_type, flux_type, eos)
    
    # Conservative update - 1st RK iteration (predictor stage)
    fluid_h.mass = fluid.mass - dt * ResM 
    fluid_h.mom1 = fluid.mom1 - dt * Res1 
    fluid_h.mom2 = fluid.mom2 - dt * Res2 
    fluid_h.mom3 = fluid.mom3 - dt * Res3 
    fluid_h.etot = fluid.etot - dt * ResE 
    
    
    #first order Runge-Kutta scheme
    if (RK_order == 'RK1'): 
        
        #simply rewrite the conservative state here for clarity
        fluid.mass = fluid_h.mass
        fluid.mom1 = fluid_h.mom1
        fluid.mom2 = fluid_h.mom2
        fluid.mom3 = fluid_h.mom3
        fluid.etot = fluid_h.etot
    
    
    #second-order Runge-Kutta scheme
    if (RK_order == 'RK2'):
        
        #Primitive variables recovery after predictor stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        fluid_h.dens[Ngc:-Ngc, Ngc:-Ngc], fluid_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            fluid_h.vel2[Ngc:-Ngc, Ngc:-Ngc], fluid_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            fluid_h.pres[Ngc:-Ngc, Ngc:-Ngc] = cons2prim_idealHydro(fluid_h.mass, 
            fluid_h.mom1, fluid_h.mom2, fluid_h.mom3, fluid_h.etot, eos) 
            
        #2nd Runge-Kutta iteration - corrector stage
        ResM, Res1, Res2, Res3, ResE = flux_calc_hydro(grid, fluid_h, rec_type, flux_type, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy
        fluid.mass = (fluid_h.mass + fluid.mass) / 2.0 - dt * ResM / 2.0
        fluid.mom1 = (fluid_h.mom1 + fluid.mom1) / 2.0 - dt * Res1 / 2.0 
        fluid.mom2 = (fluid_h.mom2 + fluid.mom2) / 2.0 - dt * Res2 / 2.0 
        fluid.mom3 = (fluid_h.mom3 + fluid.mom3) / 2.0 - dt * Res3 / 2.0  
        fluid.etot = (fluid_h.etot + fluid.etot) / 2.0 - dt * ResE / 2.0 
    
    if (RK_order == 'RK3'):
        
        #Primitive variables recovery after 1st RK stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        fluid_h.dens[Ngc:-Ngc, Ngc:-Ngc], fluid_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            fluid_h.vel2[Ngc:-Ngc, Ngc:-Ngc], fluid_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            fluid_h.pres[Ngc:-Ngc, Ngc:-Ngc] = cons2prim_idealHydro(fluid_h.mass, 
            fluid_h.mom1, fluid_h.mom2, fluid_h.mom3, fluid_h.etot, eos) 
        
        #residuals for conservative variables calculation
        #2nd Runge-Kutta iteration 
        ResM, Res1, Res2, Res3, ResE = flux_calc_hydro(grid, fluid_h, rec_type, flux_type, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy
        fluid_h.mass = (fluid_h.mass + 3.0 * fluid.mass) / 4.0 - dt * ResM / 4.0
        fluid_h.mom1 = (fluid_h.mom1 + 3.0 * fluid.mom1) / 4.0 - dt * Res1 / 4.0 
        fluid_h.mom2 = (fluid_h.mom2 + 3.0 * fluid.mom2) / 4.0 - dt * Res2 / 4.0 
        fluid_h.mom3 = (fluid_h.mom3 + 3.0 * fluid.mom3) / 4.0 - dt * Res3 / 4.0  
        fluid_h.etot = (fluid_h.etot + 3.0 * fluid.etot) / 4.0 - dt * ResE / 4.0 
    
        # Primitive variables recovery after second stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        fluid_h.dens[Ngc:-Ngc, Ngc:-Ngc], fluid_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            fluid_h.vel2[Ngc:-Ngc, Ngc:-Ngc], fluid_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            fluid_h.pres[Ngc:-Ngc, Ngc:-Ngc] = cons2prim_idealHydro(fluid_h.mass, 
            fluid_h.mom1, fluid_h.mom2, fluid_h.mom3, fluid_h.etot, eos)
        
        ResM, Res1, Res2, Res3, ResE = flux_calc_hydro(grid, fluid_h, rec_type, flux_type, eos)
        
        # Conservative update - final 3rd RK iteration
        # update mass, three components of momentum and total energy
        fluid.mass = (2.0 * fluid_h.mass + fluid.mass) / 3.0 - 2.0 * dt * ResM / 3.0
        fluid.mom1 = (2.0 * fluid_h.mom1 + fluid.mom1) / 3.0 - 2.0 * dt * Res1 / 3.0 
        fluid.mom2 = (2.0 * fluid_h.mom2 + fluid.mom2) / 3.0 - 2.0 * dt * Res2 / 3.0 
        fluid.mom3 = (2.0 * fluid_h.mom3 + fluid.mom3) / 3.0 - 2.0 * dt * Res3 / 3.0  
        fluid.etot = (2.0 * fluid_h.etot + fluid.etot) / 3.0 - 2.0 * dt * ResE / 3.0 
        
        
    # Primitive variables recovery at the end of the timestep
    #density, 3 components of velocity and pressure are evaluated 
    fluid.dens[Ngc:-Ngc, Ngc:-Ngc], fluid.vel1[Ngc:-Ngc, Ngc:-Ngc], \
        fluid.vel2[Ngc:-Ngc, Ngc:-Ngc], fluid.vel3[Ngc:-Ngc, Ngc:-Ngc], \
        fluid.pres[Ngc:-Ngc, Ngc:-Ngc] = cons2prim_idealHydro(fluid.mass, 
        fluid.mom1, fluid.mom2, fluid.mom3, fluid.etot, eos)
    
    
    #return the updated class object of the fluid state on the next timestep 
    return fluid




"""
the function "flux_calc_hydro" is the key ingredient of the code, where we calculate the residuals 
for the conservative fluid state variable at each Runge-Kutta stage.
input: GRID class object (grid), FLUIDSTATE class object (fluid) at time t, EOSdata class object (eos)
output: residuals to update conservative state ResM, Res1, Res2, Res3, ResE (for mass, 3 components of momentum and total energy)

to obtain them, we should:
    1) fill the ghost zones according to boundary conditions, see init_cond.py and boundaries.py
    2) reconstruct the fluid state from the cells to the faces of the cells in dim1 to increase the order of accuracy
    3) calculate the fluxes of conservative variables through the faces of the cells in dim1 using the approximate Riemann solver 
        The latter is the key ingredient of Godunov-type methods (Godunov 1959), where
        the conservative states in neighbouring cells shares the fluxes between each other.
        The general idea here, that the flux can be calculated, using the solution of the Riemann problem, 
        because the states in adjusement cells represent the arbitrary discontinuity of the fluid. 
        #################################################################
        (see E.F. Toro "Riemann Solvers and Numerical Methods for Fluid Dynamics: A practical introduction" (2009))
        #################################################################
    4) calculate the residuals for each cell, using the Gauss law for the integral form of the fluid equations
        U_t + RES = 0, where
        RES = (1/VOL)*SUM(Surf*norm_vect*flux). 
    
    5) repeat steps 2-3 for the dimension 2, and update the residuals (step 4), by adding the impact of 2nd dim fluxes to them 

"""
#flux calculation on the 2D grid for compresible hydrodynamics
def flux_calc_hydro(grid, fluid, rec_type, flux_type, eos):
    
    #fill the ghost cells
    fluid = boundCond_fluid(grid, fluid)
    
    #make copies of ghost cell and real cell numbers in each direction
    #to simplify indexing below 
    Ngc = grid.Ngc 
    Nx1r = grid.Nx1r
    Nx2r = grid.Nx2r
    
    #residuals initialization (only for real cells)
    ResM = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    Res1 = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    Res2 = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    Res3 = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    ResE = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    
    #fluxes in 1-dimension 
    if (grid.Nx1 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 1-dim
        #here we reconstruct density, 3 components of velocity and pressure
        dens_rec_L, dens_rec_R = VarReconstruct(fluid.dens, grid, rec_type, 1)
        vel1_rec_L, vel1_rec_R = VarReconstruct(fluid.vel1, grid, rec_type, 1)
        vel2_rec_L, vel2_rec_R = VarReconstruct(fluid.vel2, grid, rec_type, 1)
        vel3_rec_L, vel3_rec_R = VarReconstruct(fluid.vel3, grid, rec_type, 1)
        pres_rec_L, pres_rec_R = VarReconstruct(fluid.pres, grid, rec_type, 1)

        #fluxes calculation with approximate Riemann solver (see flux_type) in 1-dim
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            Riemann_flux_nr_fluid(dens_rec_L, dens_rec_R, vel1_rec_L, vel1_rec_R, \
                vel2_rec_L, vel2_rec_R, vel3_rec_L, vel3_rec_R, \
                    pres_rec_L, pres_rec_R, eos, flux_type, 1)
        
        #residuals calculation for mass, 3 components of momentum and total energy in 1-dim
        ResM = ( Fmass[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - Fmass[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        Res1 = ( Fmomx[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - Fmomx[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        Res2 = ( Fmomy[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - Fmomy[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        Res3 = ( Fmomz[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - Fmomz[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResE = ( Fetot[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - Fetot[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        
        
    #fluxes in 2-dimension
    if (grid.Nx2 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 2-dim
        #here we reconstruct density, 3 components of velocity and pressure
        dens_rec_L, dens_rec_R = VarReconstruct(fluid.dens, grid, rec_type, 2)
        pres_rec_L, pres_rec_R = VarReconstruct(fluid.pres, grid, rec_type, 2)
        vel1_rec_L, vel1_rec_R = VarReconstruct(fluid.vel1, grid, rec_type, 2)
        vel2_rec_L, vel2_rec_R = VarReconstruct(fluid.vel2, grid, rec_type, 2)
        vel3_rec_L, vel3_rec_R = VarReconstruct(fluid.vel3, grid, rec_type, 2)
     
        #fluxes calculation with approximate Riemann solver (see flux_type) in 2-dim
        Fmass, Fmomx, Fmomy, Fmomz, Fetot = \
            Riemann_flux_nr_fluid(dens_rec_L, dens_rec_R, vel1_rec_L, vel1_rec_R, \
                vel2_rec_L, vel2_rec_R, vel3_rec_L, vel3_rec_R, \
                    pres_rec_L, pres_rec_R, eos, flux_type, 2)
        
        #residuals calculation for mass, 3 components of momentum and total energy in 2-dim
        #here we add the fluxes differences to the residuals after 1-dim calculation
        ResM = ResM + ( Fmass[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            Fmass[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        Res1 = Res1 + ( Fmomx[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            Fmomx[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        Res2 = Res2 + ( Fmomy[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            Fmomy[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        Res3 = Res3 + ( Fmomz[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            Fmomz[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResE = ResE + ( Fetot[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            Fetot[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        
        #finally, here we add the external force source terms
        #we add forces in momentum res, while in energy we add Power = Force*Vel 
        Res1 = Res1 - fluid.dens[Ngc:-Ngc, Ngc:-Ngc] * fluid.F1
        Res2 = Res2 - fluid.dens[Ngc:-Ngc, Ngc:-Ngc] * fluid.F2
        ResE = ResE - fluid.dens[Ngc:-Ngc, Ngc:-Ngc] * \
            (fluid.F1 * fluid.vel1[Ngc:-Ngc, Ngc:-Ngc] + \
            fluid.F2 * fluid.vel2[Ngc:-Ngc, Ngc:-Ngc])
             
    #return the residuals for mass, 3 components of momentum and total energy
    return ResM, Res1, Res2, Res3, ResE




