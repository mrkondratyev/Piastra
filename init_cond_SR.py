# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:14:24 2024

@author: mrkon
"""
import numpy as np
from eos_setup import EOSdata


def init_cond_RP1_cart_1D_SR(grid,fluid,aux):
    
    
    print("Special relativistic shock tube from Mignone & Bodo (2005) - Riemann Problem 1")
    
    fluid.vel3[:, :] = 0.0
    
    aux.Tfin = 0.4
    aux.time = 0.0
    
    aux.rel = 'SR'
    
    eos = EOSdata(5.0/3.0)
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.5:
                fluid.dens[i, j] = 1.0
                fluid.vel1[i, j] = 0.9
                fluid.vel2[i, j] = 0.0
                fluid.pres[i, j] = 1.0
            else:
                fluid.dens[i, j] = 1.0
                fluid.vel1[i, j] = 0.0
                fluid.vel2[i, j] = 0.0
                fluid.pres[i, j] = 10.0
            
    fluid.boundMark[:] = 100
    
    #return initial conditions for fluid state
    return fluid, aux, eos






def init_cond_RP3_cart_1D_SR(grid,fluid,aux):
    
    
    print("Special relativistic shock tube from Mignone & Bodo (2005) - Riemann Problem 3")
    
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    
    aux.Tfin = 0.4
    aux.time = 0.0
    
    aux.rel = 'SR'
    
    eos = EOSdata(5.0/3.0)
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.5:
                fluid.dens[i, j] = 10.0
                fluid.vel1[i, j] = 0.0
                fluid.pres[i, j] = 40.0 / 3.0
            else:
                fluid.dens[i, j] = 1.0
                fluid.vel1[i, j] = 0.0
                fluid.pres[i, j] = 2.0 / 3.0 * 1e-6
            
    fluid.boundMark[:] = 100
    
    #return initial conditions for fluid state
    return fluid, aux, eos



def init_cond_RP4_cart_1D_SR(grid,fluid,aux):
    
    
    print("Special relativistic shock tube from Mignone & Bodo (2005) - Riemann Problem 4")
    
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    
    aux.Tfin = 0.4
    aux.time = 0.0
    
    aux.rel = 'SR'
    
    eos = EOSdata(5.0/3.0)
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.5:
                fluid.dens[i, j] = 1.0
                fluid.vel1[i, j] = 0.0
                fluid.pres[i, j] = 1000.0
            else:
                fluid.dens[i, j] = 1.0
                fluid.vel1[i, j] = 0.0
                fluid.pres[i, j] = 0.01
            
    fluid.boundMark[:] = 100
    
    #return initial conditions for fluid state
    return fluid, aux, eos




def init_cond_RP5_cart_1D_SR(grid,fluid,aux):
    
    
    print("Special relativistic shock tube with tangential velocity")
    
    fluid.vel3[:, :] = 0.0
    
    aux.Tfin = 0.4
    aux.time = 0.0
    
    aux.rel = 'SR'
    
    eos = EOSdata(5.0/3.0)
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.5:
                fluid.dens[i, j] = 1.0
                fluid.vel1[i, j] = 0.0
                fluid.vel2[i, j] = 0.0
                fluid.pres[i, j] = 1000.0
            else:
                fluid.dens[i, j] = 1.0
                fluid.vel1[i, j] = 0.0
                fluid.vel2[i, j] = 0.99
                fluid.pres[i, j] = 0.01
            
    fluid.boundMark[:] = 100
    
    #return initial conditions for fluid state
    return fluid, aux, eos





def init_cond_RP_cart_2D_SR(grid,fluid,aux):
    
    
    print("Special relativistic 2D Riemann problem from Mignone & Bodo (2005)")
    

    fluid.vel3[:, :] = 0.0
    
    aux.Tfin = 0.8
    aux.time = 0.0
    
    #coordinate range in each direction, by default x and y are in range [0..1]
    x1ini, x1fin = -1.0, 1.0
    x2ini, x2fin = -1.0, 1.0

    #filling the grid arrays with grid data (by now it is only uniform Cartesian grid)
    grid.uniCartGrid(x1ini, x1fin, x2ini, x2fin)
    
    aux.CFL = 0.8
    aux.rel = 'SR'
    
    eos = EOSdata(5.0/3.0)
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            
            if grid.cx1[i, j] > 0.0 and grid.cx2[i, j] > 0.0:
                fluid.dens[i, j] = 0.1
                fluid.vel1[i, j] = 0.0
                fluid.vel2[i, j] = 0.0
                fluid.pres[i, j] = 0.01
                
            if grid.cx1[i, j] < 0.0 and grid.cx2[i, j] > 0.0:
                fluid.dens[i, j] = 0.1
                fluid.vel1[i, j] = 0.99
                fluid.vel2[i, j] = 0.0
                fluid.pres[i, j] = 1.0
                
            if grid.cx1[i, j] < 0.0 and grid.cx2[i, j] < 0.0:
                fluid.dens[i, j] = 0.5
                fluid.vel1[i, j] = 0.0
                fluid.vel2[i, j] = 0.0
                fluid.pres[i, j] = 1.0
                
            if grid.cx1[i, j] > 0.0 and grid.cx2[i, j] < 0.0:
                fluid.dens[i, j] = 0.1
                fluid.vel1[i, j] = 0.0
                fluid.vel2[i, j] = 0.99
                fluid.pres[i, j] = 1.0
            
    fluid.boundMark[:] = 100
    
    #return initial conditions for fluid state
    return fluid, aux, eos




#Rayleigh-Taylor instability in 2D 
def init_cond_RT_inst_2D_SR(grid,fluid,aux):
    
    
    print("relativistic Rayleigh-Taylor instability in 2D")
    
    x1ini, x1fin = -1.0, 1.0
    x2ini, x2fin = 0.0, 1.0

    #filling the grid arrays with grid data (by now it is only uniform Cartesian grid)
    grid.uniCartGrid(x1ini, x1fin, x2ini, x2fin)
    
    fluid.vel1[:,:] = 0.0
    fluid.vel2[:,:] = 0.0
    fluid.vel3[:,:] = 0.0
    
    
    
    #adiabatic gamma index 
    eos = EOSdata(5.0/3.0)
    
    #densities
    rho_u = 2.0
    rho_d = 1.0 
    
    aux.CFL = 0.8
    aux.rel = 'SR'
    
    #free-fall acceleration value
    g_ff = -1.0 / 2.0
    
    
    P0 = 10.0 / 7.0 + 1.0 / 4.0
    P1 = 10.0 / 7.0 - 1.0 / 4.0 
    
    #forces calculation
    fluid.F1[:,:] = g_ff
    fluid.F2[:,:] = 0.0
    
    aux.Tfin = 10.0
    aux.time = 0.0
    
    #parameters for the interface perturbation
    h0 = 0.03
    kappa = 4.0 * np.pi
            
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j]  > h0 * np.cos(grid.fx2[i, j] * kappa):
                fluid.dens[i, j] = rho_u
                fluid.pres[i, j] = P1 + (grid.cx1[i,j]) * g_ff * rho_u
            else:
                fluid.dens[i, j] = rho_d
                fluid.pres[i, j] = P0 + (grid.cx1[i,j] + 1.0) * g_ff * rho_d
            #pressure should satisfy the hydrostatic equilibrium
            
            #fluid.vel2[i,j] = 0.03 * np.sin(grid.fx2[i, j] * 2.0 * np.pi + np.pi) * np.exp(-(grid.cx1[i,j])**2 / 0.02)
            
            #here we perturb the contact surface
            #if np.abs(grid.fx1[i, j])  < 0.1:
                #fluid.dens[i, j] = fluid.dens[i, j] + h0 * np.cos(grid.fx1[i, j] * kappa)
                
                
    fluid.boundMark[0] = 101
    fluid.boundMark[1] = 300
    fluid.boundMark[2] = 101
    fluid.boundMark[3] = 300
    
    #return initial conditions for fluid state
    return fluid, aux, eos
