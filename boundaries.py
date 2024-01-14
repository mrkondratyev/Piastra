# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:00:11 2023


"boundCond" function fills the ghost zones with data of the 2D computational domain.
by now, periodic, reflective (wall/symmetry and so on) and 
non-reflective (zero force/free boundary) boundaries are supported

The "boundMark" variable is a boundary marker. it marks each boundary (by default the boundary marker is set to 100)
    100 - non-reflecting boundary condition (zero gradient)
    101 - wall bounary condition (normal velocity is set to zero)
    102 - semi-transparent wall (i.e. the fluid can flow away, but nothing flows backward)
    300 - periodic boundaries (even or odd indexes have to the same in this case)
 

@author: mrkondratyev
"""
import numpy as np

def boundCond(grid, fluid):
    
    #local variables -- numbers of cells in each direction + number of ghost cells
    Nx1 = grid.Nx1
    Nx2 = grid.Nx2
    Ngc = grid.Ngc
    
    
    #inner boundary in 1-direction
    for i in range(0,Ngc):
        if fluid.boundMark[0] == 100: #non-reflective boundary
            fluid.dens[i, :] = fluid.dens[2 * Ngc - 1 - i, :]
            fluid.pres[i, :] = fluid.pres[2 * Ngc - 1 - i, :]
            fluid.vel1[i, :] = fluid.vel1[2 * Ngc - 1 - i, :]
            fluid.vel2[i, :] = fluid.vel2[2 * Ngc - 1 - i, :]
            fluid.vel3[i, :] = fluid.vel3[2 * Ngc - 1 - i, :]
            
        elif fluid.boundMark[0] == 101: #reflective (wall or symmetry) boundary
            fluid.dens[i, :] = fluid.dens[2 * Ngc - 1 - i, :]
            fluid.pres[i, :] = fluid.pres[2 * Ngc - 1 - i, :]
            fluid.vel1[i, :] = - fluid.vel1[2 * Ngc - 1 - i, :]
            fluid.vel2[i, :] = fluid.vel2[2 * Ngc - 1 - i, :]
            fluid.vel3[i, :] = fluid.vel3[2 * Ngc - 1 - i, :]
            
        elif fluid.boundMark[0] == 300: #periodic boundary
            fluid.dens[i, :] = fluid.dens[Nx1 - 1 + i, :]
            fluid.pres[i, :] = fluid.pres[Nx1 - 1 + i, :]
            fluid.vel1[i, :] = fluid.vel1[Nx1 - 1 + i, :]
            fluid.vel2[i, :] = fluid.vel2[Nx1 - 1 + i, :]
            fluid.vel3[i, :] = fluid.vel3[Nx1 - 1 + i, :]
                
            
        #outer boundary in 1-direction
        if fluid.boundMark[2] == 100: #non-reflective boundary
            fluid.dens[Nx1 + Ngc + i, :] = fluid.dens[Nx1 + Ngc - 1 - i, :]
            fluid.pres[Nx1 + Ngc + i, :] = fluid.pres[Nx1 + Ngc - 1 - i, :]
            fluid.vel1[Nx1 + Ngc + i, :] = fluid.vel1[Nx1 + Ngc - 1 - i, :]
            fluid.vel2[Nx1 + Ngc + i, :] = fluid.vel2[Nx1 + Ngc - 1 - i, :]
            fluid.vel3[Nx1 + Ngc + i, :] = fluid.vel3[Nx1 + Ngc - 1 - i, :]
            
        elif fluid.boundMark[2] == 101: #reflective (wall or symmetry) boundary
            fluid.dens[Nx1 + Ngc + i, :] = fluid.dens[Nx1 + Ngc - 1 - i, :]
            fluid.pres[Nx1 + Ngc + i, :] = fluid.pres[Nx1 + Ngc - 1 - i, :]
            fluid.vel1[Nx1 + Ngc + i, :] = - fluid.vel1[Nx1 + Ngc - 1 - i, :]
            fluid.vel2[Nx1 + Ngc + i, :] = fluid.vel2[Nx1 + Ngc - 1 - i, :]
            fluid.vel3[Nx1 + Ngc + i, :] = fluid.vel3[Nx1 + Ngc - 1 - i, :]
            
        elif fluid.boundMark[2] == 300: #periodic boundary
            fluid.dens[Nx1 + Ngc + i, :] = fluid.dens[Ngc + i, :]
            fluid.pres[Nx1 + Ngc + i, :] = fluid.pres[Ngc + i, :]
            fluid.vel1[Nx1 + Ngc + i, :] = fluid.vel1[Ngc + i, :]
            fluid.vel2[Nx1 + Ngc + i, :] = fluid.vel2[Ngc + i, :]
            fluid.vel3[Nx1 + Ngc + i, :] = fluid.vel3[Ngc + i, :]
            
            
    for i in range(0,Ngc):
        #inner boundary in 2-direction
        if fluid.boundMark[1] == 100: #non-reflective boundary
            fluid.dens[:, i] = fluid.dens[:, 2 * Ngc - 1 - i]
            fluid.pres[:, i] = fluid.pres[:, 2 * Ngc - 1 - i]
            fluid.vel1[:, i] = fluid.vel1[:, 2 * Ngc - 1 - i]
            fluid.vel2[:, i] = fluid.vel2[:, 2 * Ngc - 1 - i]
            fluid.vel3[:, i] = fluid.vel3[:, 2 * Ngc - 1 - i]
            
        elif fluid.boundMark[1] == 101: #reflective (wall or symmetry) boundary
            fluid.dens[:, i] = fluid.dens[:, 2 * Ngc - 1 - i]
            fluid.pres[:, i] = fluid.pres[:, 2 * Ngc - 1 - i]
            fluid.vel1[:, i] = fluid.vel1[:, 2 * Ngc - 1 - i]
            fluid.vel2[:, i] = - fluid.vel2[:, 2 * Ngc - 1 - i]
            fluid.vel3[:, i] = fluid.vel3[:, 2 * Ngc - 1 - i]
            
        elif fluid.boundMark[1] == 300: #periodic boundary
            fluid.dens[:, i] = fluid.dens[:, Nx2 - 1 + i]
            fluid.pres[:, i] = fluid.pres[:, Nx2 - 1 + i]
            fluid.vel1[:, i] = fluid.vel1[:, Nx2 - 1 + i]
            fluid.vel2[:, i] = fluid.vel2[:, Nx2 - 1 + i]
            fluid.vel3[:, i] = fluid.vel3[:, Nx2 - 1 + i]
               
            
        #outer boundary in 2-direction
        if fluid.boundMark[3] == 100: #non-reflective boundary
            fluid.dens[:, Nx2 + Ngc + i] = fluid.dens[:, Nx2 + Ngc - 1 - i]
            fluid.pres[:, Nx2 + Ngc + i] = fluid.pres[:, Nx2 + Ngc - 1 - i]
            fluid.vel1[:, Nx2 + Ngc + i] = fluid.vel1[:, Nx2 + Ngc - 1 - i]
            fluid.vel2[:, Nx2 + Ngc + i] = fluid.vel2[:, Nx2 + Ngc - 1 - i]
            fluid.vel3[:, Nx2 + Ngc + i] = fluid.vel3[:, Nx2 + Ngc - 1 - i]
            
        elif fluid.boundMark[3] == 101: #reflective (wall or symmetry) boundary
            fluid.dens[:, Nx2 + Ngc + i] = fluid.dens[:, Nx2 + Ngc - 1 - i]
            fluid.pres[:, Nx2 + Ngc + i] = fluid.pres[:, Nx2 + Ngc - 1 - i]
            fluid.vel1[:, Nx2 + Ngc + i] = fluid.vel1[:, Nx2 + Ngc - 1 - i]
            fluid.vel2[:, Nx2 + Ngc + i] = - fluid.vel2[:, Nx2 + Ngc - 1 - i]
            fluid.vel3[:, Nx2 + Ngc + i] = fluid.vel3[:, Nx2 + Ngc - 1 - i]
        
        elif fluid.boundMark[3] == 300: #periodic boundary
            fluid.dens[:, Nx2 + Ngc + i] = fluid.dens[:, Ngc + i]
            fluid.pres[:, Nx2 + Ngc + i] = fluid.pres[:, Ngc + i]
            fluid.vel1[:, Nx2 + Ngc + i] = fluid.vel1[:, Ngc + i]
            fluid.vel2[:, Nx2 + Ngc + i] = fluid.vel2[:, Ngc + i]
            fluid.vel3[:, Nx2 + Ngc + i] = fluid.vel3[:, Ngc + i]
            
            
    return fluid