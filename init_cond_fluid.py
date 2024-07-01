# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:27:10 2023

Initial conditions for different flows 

The "boundMark" variable is a boundary marker. it marks each boundary (by default the boundary marker is set to 100)
    100 - non-reflecting boundary condition (zero gradient)
    101 - wall bounary condition (normal velocity is set to zero)
    102 - semi-transparent wall (i.e. the fluid can flow away, but nothing flows backward)
    300 - periodic boundaries (even or odd indexes have to the same in this case)

@author: mrkondratyev
"""

import numpy as np
from eos_setup import EOSdata

#Sod shock tube problem in 1D (along desired direction)
def init_cond_Sod_cart_1D(grid,fluid,aux):
    
    
    print("Sod shock tube test (G.A. Sod (1978)) - is one of the most popular benchmark for hydro codes")
    
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    
    aux.Tfin = 0.2
    aux.time = 0.0
    
    
    eos = EOSdata(7.0/5.0)
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.5:
                fluid.dens[i, j] = 1.0
                fluid.pres[i, j] = 1.0
            else:
                fluid.dens[i, j] = 0.125
                fluid.pres[i, j] = 0.1
            
    fluid.boundMark[:] = 100
    #return initial conditions for fluid state
    return fluid, aux, eos



#strong shock tube problem in 1D (along desired direction)
def init_cond_strong_cart_1D(grid,fluid,aux):
    
    
    print("Shock tube test with a strong shock")
    
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    
    aux.Tfin = 0.008
    aux.time = 0.0
    
    
    eos = EOSdata(7.0/5.0)
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.5:
                fluid.dens[i, j] = 1.0
                fluid.pres[i, j] = 1000.0
            else:
                fluid.dens[i, j] = 1.0
                fluid.pres[i, j] = 0.01
            
    fluid.boundMark[:] = 100
    #return initial conditions for fluid state
    return fluid, aux, eos






#double blast wave problem in 1D (along desired direction)
def init_cond_DBW_cart_1D(grid,fluid,aux):
    
    
    print("Double blast wave test by Woodward and Collela (1984)")
    
    fluid.dens[:, :] = 1.0
    fluid.vel1[:, :] = 0.0
    fluid.vel2[:, :] = 0.0
    fluid.vel3[:, :] = 0.0
    
    aux.Tfin = 0.038
    aux.time = 0.0
    
    
    eos = EOSdata(7.0/5.0)
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j] < 0.1:
                fluid.pres[i, j] = 1000.0
            elif grid.fx1[i, j] < 0.9:
                fluid.pres[i, j] = 0.01
            else:
                fluid.pres[i, j] = 100.0
            
    fluid.boundMark[:] = 101
    #return initial conditions for fluid state
    return fluid, aux, eos



#Kelvin-Helmholtz instability in 2D 
def init_cond_KH_inst_2D(grid,fluid,aux):
    
    
    print("Kelvin-Helmholtz instability in 2D")
    
    fluid.vel3[:,:] = 0.0
    fluid.pres[:,:] = 2.5
    
    eos = EOSdata(5.0/3.0)
    
    aux.Tfin = 2.0
    aux.time = 0.0
    
    sigma1 = 0.05/np.sqrt(2.0)
            
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if np.abs(grid.fx1[i, j] - 0.5) > 0.25:
                fluid.vel2[i, j] = -0.5
                fluid.dens[i, j] = 1.0
            else:
                fluid.vel2[i, j] = 0.5
                fluid.dens[i, j] = 2.0 
            fluid.vel1[i,j] = 0.1*np.sin(4.0*3.1415926*grid.cx2[i, j])*(np.exp(-(grid.cx1[i, j] - 
                0.25)**2/2.0/sigma1**2)+np.exp(-(grid.cx1[i, j] - 0.75)**2/2.0/sigma1**2))
            
    fluid.boundMark[0] = 101
    fluid.boundMark[1] = 300
    fluid.boundMark[2] = 101
    fluid.boundMark[3] = 300
    
    #return initial conditions for fluid state
    return fluid, aux, eos





#Rayleigh-Taylor instability in 2D 
def init_cond_RT_inst_2D(grid,fluid,aux):
    
    
    print("Rayleigh-Taylor instability in 2D")
    
    x1ini, x1fin = -1.0, 1.0
    x2ini, x2fin = -0.5, 0.5

    #filling the grid arrays with grid data (by now it is only uniform Cartesian grid)
    grid.uniCartGrid(x1ini, x1fin, x2ini, x2fin)
    
    fluid.vel1[:,:] = 0.0
    fluid.vel2[:,:] = 0.0
    fluid.vel3[:,:] = 0.0
    
    
    
    #adiabatic gamma index 
    eos = EOSdata(7.0/5.0)
    
    #densities
    rho_u = 2.0
    rho_d = 1.0 
    
    #forces calculation
    
    #free-fall acceleration value
    g_ff = -1.0 / 2.0
    
    
    P0 = 10.0 / 7.0 + 1.0 / 4.0
    P1 = 10.0 / 7.0 - 1.0 / 4.0 
    
    #forces calculation
    fluid.F1[:,:] = g_ff
    fluid.F2[:,:] = 0.0
    
    aux.Tfin = 5.0
    aux.time = 0.0
    
    #parameters for the interface perturbation
    h0 = 0.03
    kappa = 2.0 * np.pi
            
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            if grid.fx1[i, j]  > h0 * np.cos(grid.cx2[i, j] * kappa + np.pi):
                fluid.dens[i, j] = rho_u
                fluid.pres[i, j] = P1 + (grid.cx1[i,j]) * g_ff * rho_u
            else:
                fluid.dens[i, j] = rho_d
                fluid.pres[i, j] = P0 + (grid.cx1[i,j] + 1.0) * g_ff * rho_d
            #pressure should satisfy the hydrostatic equilibrium
            
            #fluid.vel2[i,j] = 0.02 * np.sin(grid.fx2[i, j] * 2.0 * np.pi + np.pi) * np.exp(-(grid.cx1[i,j])**2 / 0.02)
            
            #here we perturb the contact surface
            #if np.abs(grid.fx1[i, j])  < 0.1:
                #fluid.dens[i, j] = fluid.dens[i, j] + h0 * np.cos(grid.fx1[i, j] * kappa)
                
                
    fluid.boundMark[0] = 101
    fluid.boundMark[1] = 300
    fluid.boundMark[2] = 101
    fluid.boundMark[3] = 300
    
    #return initial conditions for fluid state
    return fluid, aux, eos





#Cylindrical Sod problem (in quadrant symmetry)
def init_cond_Sod_cyl_2D(grid,fluid,aux):
    
    
    print("cylindrical 2D Sod shock tube test")
    #velocity is zero everywhere
    fluid.vel1[:,:] = 0.0
    fluid.vel2[:,:] = 0.0
    fluid.vel3[:,:] = 0.0
    
    eos = EOSdata(7.0/5.0)
    
    aux.Tfin = 0.2
    aux.time = 0.0
    
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            #rad = np.sqrt(np.abs(grid.fx1[i, j] - 0.5)**2 + np.abs(grid.fx2[i, j] - 0.5)**2) 
            rad = np.sqrt(grid.fx1[i, j]**2 + grid.fx2[i, j]**2) 
            
            if rad < 0.5:
                fluid.dens[i, j] = 1.0
                fluid.pres[i, j] = 1.0
            else:
                fluid.dens[i, j] = 0.125
                fluid.pres[i, j] = 0.1
    
    #set the boundary conditions for the cylindrical Sod shock tube problem
    fluid.boundMark[0] = 101
    fluid.boundMark[1] = 101
    fluid.boundMark[2] = 100
    fluid.boundMark[3] = 100
    
    #return initial conditions for fluid state
    return fluid, aux, eos



#Cylindrical Sedov-Taylor explosion test problem (in quadrant symmetry)
def init_cond_Sedov_blast_2D(grid,fluid,aux):
    
    
    print("flat 2D Sedov-Taylor explosion test in Cartesian geometry")
    #velocity is zero everywhere
    fluid.vel1[:,:] = 0.0
    fluid.vel2[:,:] = 0.0
    fluid.vel3[:,:] = 0.0
    
    #density is set to zero
    fluid.dens[:,:] = 1.0
    
    eos = EOSdata(7.0/5.0)
    
    aux.Tfin = 0.2
    aux.time = 0.0
    
    #calculate the volume where explosios is set
    volume = 0.0
    rad0 = 0.02
    energ = 0.25 #one forth because of symmetry 
    
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            
            rad = np.sqrt(grid.fx1[i, j]**2 + grid.fx2[i, j]**2) 
            if rad < rad0:
                volume = volume + grid.cVol[i,j]
    
    #set the initial conditions
    for i in range(grid.Ngc, grid.Nx1r):
        for j in range(grid.Ngc, grid.Nx2r):
            #rad = np.sqrt(np.abs(grid.fx1[i, j] - 0.5)**2 + np.abs(grid.fx2[i, j] - 0.5)**2) 
            rad = np.sqrt(grid.fx1[i, j]**2 + grid.fx2[i, j]**2) 
            
            if rad < rad0:
                fluid.pres[i, j] = (eos.GAMMA - 1.0) * energ/volume
            else:
                fluid.pres[i, j] = 0.0001
    
    #set the boundary conditions for the Sedov blast wave problem
    fluid.boundMark[0] = 101
    fluid.boundMark[1] = 101
    fluid.boundMark[2] = 100
    fluid.boundMark[3] = 100
    
    #return initial conditions for fluid state
    return fluid, aux, eos