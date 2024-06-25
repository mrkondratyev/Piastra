# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 22:32:41 2024

@author: mrkondratyev
"""

from prim_cons_fluid_MHD import prim2cons_idealMHD, cons2prim_idealMHD
from boundaries import boundCond_mhd 
from reconstruction import VarReconstruct 
import numpy as np 
import copy 
from Riemann_fluxes import Riemann_flux_nr_mhd 




"""

in the function "oneStep_MHD_RK" we call all the key ingredients of our MHD simulations.
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
def oneStep_MHD_RK(grid, mhd, eos, dt, rec_type, flux_type, RK_order):
    
    
    #define local copy of ghost cells number to simplify array indexing
    Ngc = grid.Ngc
    
    #here we define the copy for the auxilary fluid state
    mhd_h = copy.deepcopy(mhd)
    
    #conservative variables at the beginning of timestep
    mhd.mass, mhd.mom1, mhd.mom2, mhd.mom3, mhd.etot,  mhd.bcon1, mhd.bcon2, mhd.bcon3 = \
        prim2cons_idealMHD(mhd.dens[Ngc:-Ngc,Ngc:-Ngc], 
        mhd.vel1[Ngc:-Ngc,Ngc:-Ngc], mhd.vel2[Ngc:-Ngc,Ngc:-Ngc],  
        mhd.vel3[Ngc:-Ngc,Ngc:-Ngc], mhd.pres[Ngc:-Ngc,Ngc:-Ngc], 
        mhd.bfi1[Ngc:-Ngc,Ngc:-Ngc], mhd.bfi2[Ngc:-Ngc,Ngc:-Ngc], 
        mhd.bfi3[Ngc:-Ngc,Ngc:-Ngc], eos)
    
    #residuals for conservative variables calculation
    #1st Runge-Kutta iteration - predictor stage
    ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = flux_calc_mhd(grid, mhd, rec_type, flux_type, eos)
    
    # Conservative update - 1st RK stage (predictor)
    mhd_h.mass = mhd.mass - dt * ResM 
    mhd_h.mom1 = mhd.mom1 - dt * ResV1 
    mhd_h.mom2 = mhd.mom2 - dt * ResV2 
    mhd_h.mom3 = mhd.mom3 - dt * ResV3 
    mhd_h.etot = mhd.etot - dt * ResE 
    mhd_h.bcon1 = mhd.bcon1 - dt * ResB1
    mhd_h.bcon2 = mhd.bcon2 - dt * ResB2
    mhd_h.bcon3 = mhd.bcon3 - dt * ResB3
    
    #first order Runge-Kutta scheme
    if (RK_order == 'RK1'): 
        
        #simply rewrite the conservative state here for clarity
        mhd.mass = mhd_h.mass
        mhd.mom1 = mhd_h.mom1
        mhd.mom2 = mhd_h.mom2
        mhd.mom3 = mhd_h.mom3
        mhd.etot = mhd_h.etot
        mhd.bcon1 = mhd_h.bcon1
        mhd.bcon2 = mhd_h.bcon2
        mhd.bcon3 = mhd_h.bcon3
    
    
    #second-order Runge-Kutta scheme
    if (RK_order == 'RK2'):
        
        #Primitive variables recovery after predictor stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        mhd_h.dens[Ngc:-Ngc, Ngc:-Ngc], mhd_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            mhd_h.vel2[Ngc:-Ngc, Ngc:-Ngc], mhd_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            mhd_h.pres[Ngc:-Ngc, Ngc:-Ngc], mhd_h.bfi1[Ngc:-Ngc, Ngc:-Ngc],  \
            mhd_h.bfi2[Ngc:-Ngc, Ngc:-Ngc], mhd_h.bfi3[Ngc:-Ngc, Ngc:-Ngc] = \
            cons2prim_idealMHD(mhd_h.mass, mhd_h.mom1, mhd_h.mom2, mhd_h.mom3, mhd_h.etot, \
            mhd_h.bcon1, mhd_h.bcon2, mhd_h.bcon3, eos) 
            
        #2nd Runge-Kutta stage - corrector
        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = flux_calc_mhd(grid, mhd_h, rec_type, flux_type, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy
        mhd.mass = (mhd_h.mass + mhd.mass) / 2.0 - dt * ResM / 2.0
        mhd.mom1 = (mhd_h.mom1 + mhd.mom1) / 2.0 - dt * ResV1 / 2.0 
        mhd.mom2 = (mhd_h.mom2 + mhd.mom2) / 2.0 - dt * ResV2 / 2.0 
        mhd.mom3 = (mhd_h.mom3 + mhd.mom3) / 2.0 - dt * ResV3 / 2.0  
        mhd.etot = (mhd_h.etot + mhd.etot) / 2.0 - dt * ResE / 2.0 
        mhd.bcon1 = (mhd_h.bcon1 + mhd.bcon1) / 2.0 - dt * ResB1 / 2.0 
        mhd.bcon2 = (mhd_h.bcon2 + mhd.bcon2) / 2.0 - dt * ResB2 / 2.0 
        mhd.bcon3 = (mhd_h.bcon3 + mhd.bcon3) / 2.0 - dt * ResB3 / 2.0  
        
    if (RK_order == 'RK3'):
        
        #Primitive variables recovery after predictor stage
        #auxilary density, 3 components of velocity and pressure are evaluated 
        mhd_h.dens[Ngc:-Ngc, Ngc:-Ngc], mhd_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            mhd_h.vel2[Ngc:-Ngc, Ngc:-Ngc], mhd_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            mhd_h.pres[Ngc:-Ngc, Ngc:-Ngc], mhd_h.bfi1[Ngc:-Ngc, Ngc:-Ngc],  \
            mhd_h.bfi2[Ngc:-Ngc, Ngc:-Ngc], mhd_h.bfi3[Ngc:-Ngc, Ngc:-Ngc] = \
            cons2prim_idealMHD(mhd_h.mass, mhd_h.mom1, mhd_h.mom2, mhd_h.mom3, mhd_h.etot, \
            mhd_h.bcon1, mhd_h.bcon2, mhd_h.bcon3, eos) 
            
        #2nd Runge-Kutta stage
        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = flux_calc_mhd(grid, mhd_h, rec_type, flux_type, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy
        mhd_h.mass = (mhd_h.mass + 3.0 * mhd.mass) / 4.0 - dt * ResM / 4.0
        mhd_h.mom1 = (mhd_h.mom1 + 3.0 * mhd.mom1) / 4.0 - dt * ResV1 / 4.0 
        mhd_h.mom2 = (mhd_h.mom2 + 3.0 * mhd.mom2) / 4.0 - dt * ResV2 / 4.0 
        mhd_h.mom3 = (mhd_h.mom3 + 3.0 * mhd.mom3) / 4.0 - dt * ResV3 / 4.0  
        mhd_h.etot = (mhd_h.etot + 3.0 * mhd.etot) / 4.0 - dt * ResE / 4.0 
        mhd_h.bcon1 = (mhd_h.bcon1 + 3.0 * mhd.bcon1) / 4.0 - dt * ResB1 / 4.0 
        mhd_h.bcon2 = (mhd_h.bcon2 + 3.0 * mhd.bcon2) / 4.0 - dt * ResB2 / 4.0 
        mhd_h.bcon3 = (mhd_h.bcon3 + 3.0 * mhd.bcon3) / 4.0 - dt * ResB3 / 4.0  
        
        # Primitive variables recovery after the second stage
        #density, 3 components of velocity and pressure are evaluated 
        mhd_h.dens[Ngc:-Ngc, Ngc:-Ngc], mhd_h.vel1[Ngc:-Ngc, Ngc:-Ngc], \
            mhd_h.vel2[Ngc:-Ngc, Ngc:-Ngc], mhd_h.vel3[Ngc:-Ngc, Ngc:-Ngc], \
            mhd_h.pres[Ngc:-Ngc, Ngc:-Ngc], mhd_h.bfi1[Ngc:-Ngc, Ngc:-Ngc],  \
            mhd_h.bfi2[Ngc:-Ngc, Ngc:-Ngc], mhd_h.bfi3[Ngc:-Ngc, Ngc:-Ngc] = \
            cons2prim_idealMHD(mhd_h.mass, mhd_h.mom1, mhd_h.mom2, mhd_h.mom3, mhd_h.etot, \
            mhd_h.bcon1, mhd_h.bcon2, mhd_h.bcon3, eos) 
        
        #3rd Runge-Kutta stage
        ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3 = flux_calc_mhd(grid, mhd_h, rec_type, flux_type, eos)
        
        # Conservative update - 2nd RK iteration
        # update mass, 3 components of momentum, total energy and 3 comps of magnetic field
        mhd.mass = (2.0 * mhd_h.mass + mhd.mass) / 3.0 - 2.0 * dt * ResM / 3.0
        mhd.mom1 = (2.0 * mhd_h.mom1 + mhd.mom1) / 3.0 - 2.0 * dt * ResV1 / 3.0 
        mhd.mom2 = (2.0 * mhd_h.mom2 + mhd.mom2) / 3.0 - 2.0 * dt * ResV2 / 3.0 
        mhd.mom3 = (2.0 * mhd_h.mom3 + mhd.mom3) / 3.0 - 2.0 * dt * ResV3 / 3.0  
        mhd.etot = (2.0 * mhd_h.etot + mhd.etot) / 3.0 - 2.0 * dt * ResE / 3.0 
        mhd.bcon1 = (2.0 * mhd_h.bcon1 + mhd.bcon1) / 3.0 - 2.0 * dt * ResB1 / 3.0 
        mhd.bcon2 = (2.0 * mhd_h.bcon2 + mhd.bcon2) / 3.0 - 2.0 * dt * ResB2 / 3.0 
        mhd.bcon3 = (2.0 * mhd_h.bcon3 + mhd.bcon3) / 3.0 - 2.0 * dt * ResB3 / 3.0  
        
    # Primitive variables recovery at the end of the timestep
    #density, 3 components of velocity and pressure are evaluated 
    mhd.dens[Ngc:-Ngc, Ngc:-Ngc], mhd.vel1[Ngc:-Ngc, Ngc:-Ngc], \
        mhd.vel2[Ngc:-Ngc, Ngc:-Ngc], mhd.vel3[Ngc:-Ngc, Ngc:-Ngc], \
        mhd.pres[Ngc:-Ngc, Ngc:-Ngc], mhd.bfi1[Ngc:-Ngc, Ngc:-Ngc],  \
        mhd.bfi2[Ngc:-Ngc, Ngc:-Ngc], mhd.bfi3[Ngc:-Ngc, Ngc:-Ngc] = \
        cons2prim_idealMHD(mhd.mass, mhd.mom1, mhd.mom2, mhd.mom3, mhd.etot, \
        mhd.bcon1, mhd.bcon2, mhd.bcon3, eos) 
    
    
    #return the updated class object of the fluid state on the next timestep 
    return mhd




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
def flux_calc_mhd(grid, mhd, rec_type, flux_type, eos):
    
    #fill the ghost cells
    mhd = boundCond_mhd(grid, mhd)
    
    #make copies of ghost cell and real cell numbers in each direction
    #to simplify indexing below 
    Ngc = grid.Ngc 
    Nx1r = grid.Nx1r
    Nx2r = grid.Nx2r
    
    #nulifying the divergence of the magnetic field 
    mhd.divB[:,:] = 0.0
    
    #residuals initialization (only for real cells)
    ResM = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    ResV1 = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    ResV2 = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    ResV3 = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    ResE = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    ResB1 = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    ResB2 = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    ResB3 = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    
    #fluxes in 1-dimension 
    if (grid.Nx1 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 1-dim
        #here we reconstruct density, 3 components of velocity and pressure
        dens_rec_L, dens_rec_R = VarReconstruct(mhd.dens, grid, rec_type, 1)
        vel1_rec_L, vel1_rec_R = VarReconstruct(mhd.vel1, grid, rec_type, 1)
        vel2_rec_L, vel2_rec_R = VarReconstruct(mhd.vel2, grid, rec_type, 1)
        vel3_rec_L, vel3_rec_R = VarReconstruct(mhd.vel3, grid, rec_type, 1)
        pres_rec_L, pres_rec_R = VarReconstruct(mhd.pres, grid, rec_type, 1)
        bfi1_rec_L, bfi1_rec_R = VarReconstruct(mhd.bfi1, grid, rec_type, 1)
        bfi2_rec_L, bfi2_rec_R = VarReconstruct(mhd.bfi2, grid, rec_type, 1)
        bfi3_rec_L, bfi3_rec_R = VarReconstruct(mhd.bfi3, grid, rec_type, 1)
        
        #fluxes calculation with approximate Riemann solver (see flux_type) in 1-dim
        Fmass, Fmom1, Fmom2, Fmom3, Fetot, Fbfi1, Fbfi2, Fbfi3 = \
            Riemann_flux_nr_mhd(dens_rec_L, dens_rec_R, vel1_rec_L, vel1_rec_R, \
            vel2_rec_L, vel2_rec_R, vel3_rec_L, vel3_rec_R, \
            pres_rec_L, pres_rec_R, bfi1_rec_L, bfi1_rec_R, \
            bfi2_rec_L, bfi2_rec_R, bfi3_rec_L, bfi3_rec_R,  eos, flux_type, 1)
        
        #residuals calculation for mass, 3 components of momentum, 
        #total energy and 3 component of magnetic field in 1-dim
        ResM = ( Fmass[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - \
            Fmass[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResV1 = ( Fmom1[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - \
            Fmom1[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResV2 = ( Fmom2[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - \
            Fmom2[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResV3 = ( Fmom3[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - \
            Fmom3[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResE = ( Fetot[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - \
            Fetot[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        #ResB1 = ( Fbfi1[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - \
            #Fbfi1[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResB2 = ( Fbfi2[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - \
            Fbfi2[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResB3 = ( Fbfi3[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - \
            Fbfi3[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        
        #calculation of magnetic field divergence for Powell (1999) 8-wave approach
        mhd.divB = ( (bfi1_rec_L[1:,:] + bfi1_rec_R[1:,:]) / 2.0 * grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - \
            ( bfi1_rec_L[:-1,:] + bfi1_rec_R[:-1,:]) / 2.0 * grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        
    #fluxes in 2-dimension
    if (grid.Nx2 > 1): #check if we even need to consider this dimension
        
        #primitive variables reconstruction in 2-dim
        #here we reconstruct density, 3 components of velocity and pressure
        dens_rec_L, dens_rec_R = VarReconstruct(mhd.dens, grid, rec_type, 2)
        pres_rec_L, pres_rec_R = VarReconstruct(mhd.pres, grid, rec_type, 2)
        vel1_rec_L, vel1_rec_R = VarReconstruct(mhd.vel1, grid, rec_type, 2)
        vel2_rec_L, vel2_rec_R = VarReconstruct(mhd.vel2, grid, rec_type, 2)
        vel3_rec_L, vel3_rec_R = VarReconstruct(mhd.vel3, grid, rec_type, 2)
        bfi1_rec_L, bfi1_rec_R = VarReconstruct(mhd.bfi1, grid, rec_type, 2)
        bfi2_rec_L, bfi2_rec_R = VarReconstruct(mhd.bfi2, grid, rec_type, 2)
        bfi3_rec_L, bfi3_rec_R = VarReconstruct(mhd.bfi3, grid, rec_type, 2)
     
        #fluxes calculation with approximate Riemann solver (see flux_type) in 2-dim
        Fmass, Fmom1, Fmom2, Fmom3, Fetot, Fbfi1, Fbfi2, Fbfi3 = \
            Riemann_flux_nr_mhd(dens_rec_L, dens_rec_R, vel1_rec_L, vel1_rec_R, \
            vel2_rec_L, vel2_rec_R, vel3_rec_L, vel3_rec_R, \
            pres_rec_L, pres_rec_R, bfi1_rec_L, bfi1_rec_R, \
            bfi2_rec_L, bfi2_rec_R, bfi3_rec_L, bfi3_rec_R,  eos, flux_type, 2)
        
        #residuals calculation for mass, 3 components of momentum, 
        #total energy and 3 components of magnetic field in 2-dim
        #here we add the fluxes differences to the residuals after 1-dim calculation
        ResM = ResM + ( Fmass[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            Fmass[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResV1 = ResV1 + ( Fmom1[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            Fmom1[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResV2 = ResV2 + ( Fmom2[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            Fmom2[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResV3 = ResV3 + ( Fmom3[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            Fmom3[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResE = ResE + ( Fetot[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            Fetot[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResB1 = ResB1 + ( Fbfi1[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            Fbfi1[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        #ResB2 = ResB2 + ( Fbfi2[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            #Fbfi2[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        ResB3 = ResB3 + ( Fbfi3[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            Fbfi3[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
         
        #calculation of magnetic field divergence for Powell (1999) 8-wave approach
        mhd.divB = mhd.divB + ( (bfi2_rec_L[:,1:] + bfi2_rec_R[:,1:]) / 2.0 * grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - \
            ( bfi2_rec_L[:,:-1] + bfi2_rec_R[:,:-1]) / 2.0 * grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]     
            
        #finally, here we add the external force source terms
        #we add forces in momentum res, while in energy we add Power = Force*Vel 
        ResV1 = ResV1 - mhd.dens[Ngc:-Ngc, Ngc:-Ngc] * mhd.F1
        ResV2 = ResV2 - mhd.dens[Ngc:-Ngc, Ngc:-Ngc] * mhd.F2
        ResE = ResE - mhd.dens[Ngc:-Ngc, Ngc:-Ngc] * \
            (mhd.F1 * mhd.vel1[Ngc:-Ngc, Ngc:-Ngc] + \
            mhd.F2 * mhd.vel2[Ngc:-Ngc, Ngc:-Ngc])
                
        #Powell 8-wave cleaning method
        ResV1 = ResV1 + mhd.bfi1[Ngc:-Ngc, Ngc:-Ngc] * mhd.divB
        ResV2 = ResV2 + mhd.bfi2[Ngc:-Ngc, Ngc:-Ngc] * mhd.divB
        ResE = ResE + mhd.divB * \
            (mhd.vel1[Ngc:-Ngc, Ngc:-Ngc] * mhd.bfi1[Ngc:-Ngc, Ngc:-Ngc] + \
            mhd.vel2[Ngc:-Ngc, Ngc:-Ngc] * mhd.bfi2[Ngc:-Ngc, Ngc:-Ngc])
        ResB1 = ResB1 + mhd.vel1[Ngc:-Ngc, Ngc:-Ngc] * mhd.divB
        ResB2 = ResB2 + mhd.vel2[Ngc:-Ngc, Ngc:-Ngc] * mhd.divB
        
    #return the residuals for mass, 3 components of momentum, total energy and magnetic field
    return ResM, ResV1, ResV2, ResV3, ResE, ResB1, ResB2, ResB3




