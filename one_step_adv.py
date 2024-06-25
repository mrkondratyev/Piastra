# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:30:04 2024

@author: mrkondratyev
"""
from reconstruction import VarReconstruct 
import numpy as np 
import copy 
from flux_numpy import Riemann_flux_adv 

#this function provides one hydro time step
def oneStep_advection_RK(grid, adv, dt, rec_type, flux_type, RK_order):
    
    
    #define local copy of ghost cells number to simplify array indexing
    Ngc = grid.Ngc
    
    #here we define the copy for the auxilary fluid state
    adv_h = copy.deepcopy(adv)
    
    #residuals for conservative variables calculation
    #1st Runge-Kutta iteration - predictor stage
    Res = flux_adv(grid, adv, flux_type, rec_type, dt)
    
    # Conservative update - 1st RK iteration (predictor stage)
    adv_h.adv[Ngc:-Ngc, Ngc:-Ngc] = adv.adv[Ngc:-Ngc, Ngc:-Ngc] - dt * Res
    
    if (flux_type == 'LW'):
        #simply rewrite the conservative state here for clarity
        adv.adv = adv_h.adv
        return adv 
        
    
    #first order Runge-Kutta scheme
    if (RK_order == 'RK1'): 
        
        #simply rewrite the conservative state here for clarity
        adv.adv = adv_h.adv
    
    #second-order Runge-Kutta scheme
    elif (RK_order == 'RK2'):
        
        #2nd Runge-Kutta iteration - corrector stage
        Res = flux_adv(grid, adv_h, flux_type, rec_type, dt)
        # Update - 2nd RK iteration
        adv.adv[Ngc:-Ngc, Ngc:-Ngc] = (adv_h.adv[Ngc:-Ngc, Ngc:-Ngc] + adv.adv[Ngc:-Ngc, Ngc:-Ngc]) / 2.0 - dt * Res / 2.0
    
    elif (RK_order == 'RK3'):
        
        #2nd Runge-Kutta iteration 
        Res = flux_adv(grid, adv_h, flux_type, rec_type, dt)
        
        # Conservative update - 2nd RK iteration
        # update mass, three components of momentum and total energy
        adv_h.adv[Ngc:-Ngc, Ngc:-Ngc] = (adv_h.adv[Ngc:-Ngc, Ngc:-Ngc] + 3.0 * adv.adv[Ngc:-Ngc, Ngc:-Ngc]) / 4.0 - dt * Res / 4.0

        #2nd Runge-Kutta iteration 
        Res = flux_adv(grid, adv_h, flux_type, rec_type, dt)
        
        # Conservative update - final 3rd RK iteration
        # update mass, three components of momentum and total energy
        adv.adv[Ngc:-Ngc, Ngc:-Ngc] = (2.0 * adv_h.adv[Ngc:-Ngc, Ngc:-Ngc] + adv.adv[Ngc:-Ngc, Ngc:-Ngc]) / 3.0 - 2.0 * dt * Res / 3.0
    
    else: 
        print('WRONG RK parameter, use RK1, RK2 or RK3')
    
    #return the updated class object of the fluid state on the next timestep 
    return adv





def flux_adv(grid, adv, flux_type, rec_type, dt):
    
    
    
    #make copies of ghost cell and real cell numbers in each direction
    #to simplify indexing below 
    Ngc = grid.Ngc 
    Nx1r = grid.Nx1r
    Nx2r = grid.Nx2r
    
    #boundary conditions are always periodic in both dimensions
    #inner boundary in 1-direction
    for i in range(0,Ngc):
        #inner boundary in 1-direction
        adv.adv[i, :] = adv.adv[Nx1r - Ngc + i, :]
        #outer boundary in 1-direction
        adv.adv[Nx1r + i, :] = adv.adv[Ngc + i, :]
            
    for i in range(0,grid.Ngc):
        #inner boundary in 2-direction
        adv.adv[:, i] = adv.adv[:, Nx2r - Ngc + i]
        #outer boundary in 2-direction
        adv.adv[:, Nx2r + i] = adv.adv[:, Ngc + i]
    
    
    
    #residuals initialization (only for real cells)
    Res = np.zeros((Nx1r - Ngc, Nx2r - Ngc))
    
    if (flux_type == 'adv'):
        
        #fluxes in 1-dimension 
        if (grid.Nx1 > 1): #check if we even need to consider this dimension
            
            #primitive variables reconstruction in 1-dim
            #here we reconstruct density, 3 components of velocity and pressure
            adv_rec_L, adv_rec_R = VarReconstruct(adv.adv, grid, rec_type, 1)
        
            #fluxes calculation with approximate Riemann solver (see flux_type) in 1-dim
            flux = Riemann_flux_adv(adv_rec_L, adv_rec_R, adv.vel1, 1)
            
            #residual calculation for advected variable according to finite volume method
            Res = ( flux[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - flux[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
        
        
        #fluxes in 2-dimension
        if (grid.Nx2 > 1): #check if we even need to consider this dimension
            
            #primitive variables reconstruction in 2-dim
            #here we reconstruct density, 3 components of velocity and pressure
            adv_rec_L, adv_rec_R = VarReconstruct(adv.adv, grid, rec_type, 2)
             
            #fluxes calculation with approximate Riemann solver (see flux_type) in 2-dim
            flux = Riemann_flux_adv(adv_rec_L, adv_rec_R, adv.vel2, 2)
            
            #residual calculation for advected variable according to finite volume method
            #here we add the fluxes differences to the residuals after 1-dim calculation
            Res = Res + ( flux[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - flux[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
                
        
    elif (flux_type == 'LW'):
            
        if (grid.Nx1 > 1): #check if we even need to consider this dimension
                
            #Lax-Wendroff flux
            flux = adv.vel1 * (adv.adv[Ngc-1:Nx1r, Ngc:-Ngc] + adv.adv[Ngc:Nx1r+1, Ngc:-Ngc])/2.0 + \
            adv.vel1 * (adv.vel1 * dt / grid.dx1[1,1]) * (adv.adv[Ngc-1:Nx1r, Ngc:-Ngc] - adv.adv[Ngc:Nx1r+1, Ngc:-Ngc])/2.0
            #residual calculation for advected variable according to finite volume method
            Res = ( flux[1:,:]*grid.fS1[Ngc+1:Nx1r + 1, Ngc:-Ngc] - flux[:-1,:]*grid.fS1[Ngc:Nx1r, Ngc:-Ngc] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
            
        if (grid.Nx2 > 1):
                
            #Lax-Wendroff flux
            flux = adv.vel2 * (adv.adv[Ngc:-Ngc, Ngc-1:Nx2r] + adv.adv[Ngc:-Ngc,Ngc:Nx2r+1])/2.0 + \
            adv.vel2 * (adv.vel2 * dt / grid.dx2[1,1]) * (adv.adv[Ngc:-Ngc, Ngc-1:Nx2r] - adv.adv[Ngc:-Ngc,Ngc:Nx2r+1])/2.0
            #here we add the fluxes differences to the residuals after 1-dim calculation
            Res = Res + ( flux[:,1:]*grid.fS2[Ngc:-Ngc, Ngc+1:Nx2r + 1] - flux[:,:-1]*grid.fS2[Ngc:-Ngc, Ngc:Nx2r] ) / grid.cVol[Ngc:-Ngc, Ngc:-Ngc]
                 
        
        #additional antidiffusion to improve stability in multidimensions
        #if ((flux_type == 'LW') & (grid.Nx1 != 1) & (grid.Nx2 != 1)):
        Res = Res - dt * adv.vel1 *  adv.vel2 * (adv.adv[Ngc-1:Nx1r-1, Ngc-1:Nx2r-1] - adv.adv[Ngc-1:Nx1r-1, Ngc+1:Nx2r+1] - \
        adv.adv[Ngc+1:Nx1r+1, Ngc-1:Nx2r-1] + adv.adv[Ngc+1:Nx1r+1, Ngc+1:Nx2r+1]) / 4.0 / grid.dx1[1,1] / grid.dx2[1,1]
             
    #return the residuals for advected value
    return Res