# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:34:25 2024

@author: mrkon
"""

import numpy as np

#Sod shock tube problem in 1D (along desired direction)
def init_cond_advection_1D(grid,adv,aux):
    
    
    print("linear advection test problem")
    
    adv.adv[:, :] = 0.0
    
    aux.Tfin = 1.0
    aux.time = 0.0
    
    adv.vel1 = 1.0
    adv.vel2 = 0.0
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.2:
                adv.adv[i, j] = 1.0
            elif grid.fx1[i, j] < 0.4:
                adv.adv[i, j] = 2.0
            else:
                
                adv.adv[i, j] = 1.0
            
    #return initial conditions for fluid state
    return adv, aux 


#Sod shock tube problem in 1D (along desired direction)
def init_cond_advection_2D(grid,adv,aux):
    
    
    print("linear advection test problem")
    
    adv.adv[:, :] = 0.0
    
    aux.Tfin = 1.0
    aux.time = 0.0
    
    rad0 = 0.1
    x0 = 0.5
    y0 = 0.5
    
    adv.vel1 = 0.0 + 1.0
    adv.vel2 = 0.0 + 1.0
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            
            rad = np.sqrt((grid.fx1[i, j] - x0)**2 + (grid.fx2[i, j] - y0)**2) 
            
            if rad < rad0:
                adv.adv[i, j] = 1.0
            else:
                adv.adv[i, j] = 0.0
            
                #adv.adv[i,j] = np.exp(-rad**2)
            
    #return initial conditions for fluid state
    return adv, aux 