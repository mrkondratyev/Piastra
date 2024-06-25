# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:07:25 2024

@author: mrkondratyev
"""


import numpy as np
from eos_setup import EOSdata

#Sod shock tube problem in 1D (along desired direction)
def init_cond_brio_wu_cart_1D(grid,mhd,aux):
    
    
    print("Brio-Wu (1988) shock tube test - is one of the most popular benchmark for 1D mhd codes")
    
    mhd.vel1[:, :] = 0.0
    mhd.vel2[:, :] = 0.0
    mhd.vel3[:, :] = 0.0    
    mhd.bfi3[:, :] = 0.0
    
    mhd.bfi1[:, :] = 0.0 + 0.75    
        
    aux.Tfin = 0.1
    aux.time = 0.0
    
    
    eos = EOSdata(10.0/5.0)
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.5:
                mhd.dens[i, j] = 1.0
                mhd.pres[i, j] = 1.0
                mhd.bfi2[i, j] = 0.0 + 1.0
            else:
                mhd.dens[i, j] = 0.125
                mhd.pres[i, j] = 0.1
                mhd.bfi2[i, j] = 0.0 - 1.0
            
    mhd.boundMark[:] = 100
    #return initial conditions for fluid state
    return mhd, aux, eos


def init_cond_mhd_expl_cart_2D(grid,mhd,aux):
    
    
    print("magnetized explosion test")
    
    mhd.vel1[:, :] = 0.0
    mhd.vel2[:, :] = 0.0
    mhd.vel3[:, :] = 0.0    
    mhd.bfi3[:, :] = 0.0
    
    mhd.dens[:, :] = 1.0
    mhd.bfi1[:, :] = 1.0/np.sqrt(2.0)
    mhd.bfi2[:, :] = 1.0/np.sqrt(2.0)    
        
    aux.Tfin = 0.2
    aux.time = 0.0
    
    
    eos = EOSdata(7.0/5.0)
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            rad = np.sqrt(np.abs(grid.fx1[i, j] - 0.5)**2 + np.abs(grid.fx2[i, j] - 0.5)**2) 
            if rad < 0.1:
                mhd.pres[i, j] = 10.0
            else:
                mhd.pres[i, j] = 0.1
            
    mhd.boundMark[:] = 100
    #return initial conditions for fluid state
    return mhd, aux, eos



def init_cond_orszag_tang_cart_2D(grid,mhd,aux):
    
    
    print("2D Orszag-Tang vortex problem")
    
    mhd.vel3[:, :] = 0.0    
    mhd.bfi3[:, :] = 0.0
    
    mhd.dens[:, :] = 25.0/36.0/np.pi
    mhd.pres[:, :] = 5.0/12.0/np.pi
      
        
    aux.Tfin = 0.5
    aux.time = 0.0
    
    
    eos = EOSdata(5.0/3.0)
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            mhd.bfi1[i, j] = np.sin(4.0 * np.pi * grid.cx2[i,j])/np.sqrt(4.0 * np.pi)
            mhd.bfi2[i, j] = -np.sin(2.0 * np.pi * grid.cx1[i,j])/np.sqrt(4.0 * np.pi)
            mhd.vel1[i, j] = np.sin(2.0 * np.pi * grid.cx2[i,j])
            mhd.vel2[i, j] = -np.sin(2.0 * np.pi * grid.cx1[i,j])
            
            
    mhd.boundMark[:] = 300
    #return initial conditions for fluid state
    return mhd, aux, eos

