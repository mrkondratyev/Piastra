# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:00:11 2023


"boundCond" function fills the ghost zones with data of the 2D computational domain.
by now, periodic, reflective (wall/symmetry and so on) and 
non-reflective (zero force/free boundary) boundaries are supported fro hydrodynamics and MHD equations

The "boundMark" variable is a boundary marker. it marks each boundary (by default the boundary marker is set to 100)
    100 - non-reflecting boundary condition (zero gradient)
    101 - wall bounary condition (normal velocity and B-field arer set to zero)
    300 - periodic boundaries (even or odd indexes have to the same in this case)
 

@author: mrkondratyev
"""
import numpy as np

def boundCond_fluid(grid, fluid):
    
    #local variables -- numbers of cells in each direction + number of ghost cells
    Nx1 = grid.Nx1
    Nx2 = grid.Nx2
    Ngc = grid.Ngc
    
    
    for i in range(0,Ngc):
        #inner boundary in 1-direction
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
            fluid.dens[i, :] = fluid.dens[Nx1 + i, :]
            fluid.pres[i, :] = fluid.pres[Nx1 + i, :]
            fluid.vel1[i, :] = fluid.vel1[Nx1 + i, :]
            fluid.vel2[i, :] = fluid.vel2[Nx1 + i, :]
            fluid.vel3[i, :] = fluid.vel3[Nx1 + i, :]
                
            
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
            fluid.dens[:, i] = fluid.dens[:, Nx2 + i]
            fluid.pres[:, i] = fluid.pres[:, Nx2 + i]
            fluid.vel1[:, i] = fluid.vel1[:, Nx2 + i]
            fluid.vel2[:, i] = fluid.vel2[:, Nx2 + i]
            fluid.vel3[:, i] = fluid.vel3[:, Nx2 + i]
               
            
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






def boundCond_mhd(grid, mhd):
    
    #local variables -- numbers of cells in each direction + number of ghost cells
    Nx1 = grid.Nx1
    Nx2 = grid.Nx2
    Ngc = grid.Ngc
    
    
    
    for i in range(0,Ngc):
        #inner boundary in 1-direction
        if mhd.boundMark[0] == 100: #non-reflective boundary
            mhd.dens[i, :] = mhd.dens[2 * Ngc - 1 - i, :]
            mhd.pres[i, :] = mhd.pres[2 * Ngc - 1 - i, :]
            mhd.vel1[i, :] = mhd.vel1[2 * Ngc - 1 - i, :]
            mhd.vel2[i, :] = mhd.vel2[2 * Ngc - 1 - i, :]
            mhd.vel3[i, :] = mhd.vel3[2 * Ngc - 1 - i, :]
            mhd.bfi1[i, :] = mhd.bfi1[2 * Ngc - 1 - i, :]
            mhd.bfi2[i, :] = mhd.bfi2[2 * Ngc - 1 - i, :]
            mhd.bfi3[i, :] = mhd.bfi3[2 * Ngc - 1 - i, :]
            
        elif mhd.boundMark[0] == 101: #reflective (wall or symmetry) boundary
            mhd.dens[i, :] = mhd.dens[2 * Ngc - 1 - i, :]
            mhd.pres[i, :] = mhd.pres[2 * Ngc - 1 - i, :]
            mhd.vel1[i, :] = - mhd.vel1[2 * Ngc - 1 - i, :]
            mhd.vel2[i, :] = mhd.vel2[2 * Ngc - 1 - i, :]
            mhd.vel3[i, :] = mhd.vel3[2 * Ngc - 1 - i, :]
            mhd.bfi1[i, :] = - mhd.bfi1[2 * Ngc - 1 - i, :]
            mhd.bfi2[i, :] = mhd.bfi2[2 * Ngc - 1 - i, :]
            mhd.bfi3[i, :] = mhd.bfi3[2 * Ngc - 1 - i, :]
            
        elif mhd.boundMark[0] == 300: #periodic boundary
            mhd.dens[i, :] = mhd.dens[Nx1 + i, :]
            mhd.pres[i, :] = mhd.pres[Nx1 + i, :]
            mhd.vel1[i, :] = mhd.vel1[Nx1 + i, :]
            mhd.vel2[i, :] = mhd.vel2[Nx1 + i, :]
            mhd.vel3[i, :] = mhd.vel3[Nx1 + i, :]
            mhd.bfi1[i, :] = mhd.bfi1[Nx1 + i, :]
            mhd.bfi2[i, :] = mhd.bfi2[Nx1 + i, :]
            mhd.bfi3[i, :] = mhd.bfi3[Nx1 + i, :]
                
            
        #outer boundary in 1-direction
        if mhd.boundMark[2] == 100: #non-reflective boundary
            mhd.dens[Nx1 + Ngc + i, :] = mhd.dens[Nx1 + Ngc - 1 - i, :]
            mhd.pres[Nx1 + Ngc + i, :] = mhd.pres[Nx1 + Ngc - 1 - i, :]
            mhd.vel1[Nx1 + Ngc + i, :] = mhd.vel1[Nx1 + Ngc - 1 - i, :]
            mhd.vel2[Nx1 + Ngc + i, :] = mhd.vel2[Nx1 + Ngc - 1 - i, :]
            mhd.vel3[Nx1 + Ngc + i, :] = mhd.vel3[Nx1 + Ngc - 1 - i, :]
            mhd.bfi1[Nx1 + Ngc + i, :] = mhd.bfi1[Nx1 + Ngc - 1 - i, :]
            mhd.bfi2[Nx1 + Ngc + i, :] = mhd.bfi2[Nx1 + Ngc - 1 - i, :]
            mhd.bfi3[Nx1 + Ngc + i, :] = mhd.bfi3[Nx1 + Ngc - 1 - i, :]
            
        elif mhd.boundMark[2] == 101: #reflective (wall or symmetry) boundary
            mhd.dens[Nx1 + Ngc + i, :] = mhd.dens[Nx1 + Ngc - 1 - i, :]
            mhd.pres[Nx1 + Ngc + i, :] = mhd.pres[Nx1 + Ngc - 1 - i, :]
            mhd.vel1[Nx1 + Ngc + i, :] = - mhd.vel1[Nx1 + Ngc - 1 - i, :]
            mhd.vel2[Nx1 + Ngc + i, :] = mhd.vel2[Nx1 + Ngc - 1 - i, :]
            mhd.vel3[Nx1 + Ngc + i, :] = mhd.vel3[Nx1 + Ngc - 1 - i, :]
            mhd.bfi1[Nx1 + Ngc + i, :] = -mhd.bfi1[Nx1 + Ngc - 1 - i, :]
            mhd.bfi2[Nx1 + Ngc + i, :] = mhd.bfi2[Nx1 + Ngc - 1 - i, :]
            mhd.bfi3[Nx1 + Ngc + i, :] = mhd.bfi3[Nx1 + Ngc - 1 - i, :]
            
        elif mhd.boundMark[2] == 300: #periodic boundary
            mhd.dens[Nx1 + Ngc + i, :] = mhd.dens[Ngc + i, :]
            mhd.pres[Nx1 + Ngc + i, :] = mhd.pres[Ngc + i, :]
            mhd.vel1[Nx1 + Ngc + i, :] = mhd.vel1[Ngc + i, :]
            mhd.vel2[Nx1 + Ngc + i, :] = mhd.vel2[Ngc + i, :]
            mhd.vel3[Nx1 + Ngc + i, :] = mhd.vel3[Ngc + i, :]
            mhd.bfi1[Nx1 + Ngc + i, :] = mhd.bfi1[Ngc + i, :]
            mhd.bfi2[Nx1 + Ngc + i, :] = mhd.bfi2[Ngc + i, :]
            mhd.bfi3[Nx1 + Ngc + i, :] = mhd.bfi3[Ngc + i, :]
            
            
    for i in range(0,Ngc):
        #inner boundary in 2-direction
        if mhd.boundMark[1] == 100: #non-reflective boundary
            mhd.dens[:, i] = mhd.dens[:, 2 * Ngc - 1 - i]
            mhd.pres[:, i] = mhd.pres[:, 2 * Ngc - 1 - i]
            mhd.vel1[:, i] = mhd.vel1[:, 2 * Ngc - 1 - i]
            mhd.vel2[:, i] = mhd.vel2[:, 2 * Ngc - 1 - i]
            mhd.vel3[:, i] = mhd.vel3[:, 2 * Ngc - 1 - i]
            mhd.bfi1[:, i] = mhd.bfi1[:, 2 * Ngc - 1 - i]
            mhd.bfi2[:, i] = mhd.bfi2[:, 2 * Ngc - 1 - i]
            mhd.bfi3[:, i] = mhd.bfi3[:, 2 * Ngc - 1 - i]
            
        elif mhd.boundMark[1] == 101: #reflective (wall or symmetry) boundary
            mhd.dens[:, i] = mhd.dens[:, 2 * Ngc - 1 - i]
            mhd.pres[:, i] = mhd.pres[:, 2 * Ngc - 1 - i]
            mhd.vel1[:, i] = mhd.vel1[:, 2 * Ngc - 1 - i]
            mhd.vel2[:, i] = - mhd.vel2[:, 2 * Ngc - 1 - i]
            mhd.vel3[:, i] = mhd.vel3[:, 2 * Ngc - 1 - i]
            mhd.bfi1[:, i] = mhd.bfi1[:, 2 * Ngc - 1 - i]
            mhd.bfi2[:, i] = - mhd.bfi2[:, 2 * Ngc - 1 - i]
            mhd.bfi3[:, i] = mhd.bfi3[:, 2 * Ngc - 1 - i]
            
        elif mhd.boundMark[1] == 300: #periodic boundary
            mhd.dens[:, i] = mhd.dens[:, Nx2 + i]
            mhd.pres[:, i] = mhd.pres[:, Nx2 + i]
            mhd.vel1[:, i] = mhd.vel1[:, Nx2 + i]
            mhd.vel2[:, i] = mhd.vel2[:, Nx2 + i]
            mhd.vel3[:, i] = mhd.vel3[:, Nx2 + i]
            mhd.bfi1[:, i] = mhd.bfi1[:, Nx2 + i]
            mhd.bfi2[:, i] = mhd.bfi2[:, Nx2 + i]
            mhd.bfi3[:, i] = mhd.bfi3[:, Nx2 + i]
               
            
        #outer boundary in 2-direction
        if mhd.boundMark[3] == 100: #non-reflective boundary
            mhd.dens[:, Nx2 + Ngc + i] = mhd.dens[:, Nx2 + Ngc - 1 - i]
            mhd.pres[:, Nx2 + Ngc + i] = mhd.pres[:, Nx2 + Ngc - 1 - i]
            mhd.vel1[:, Nx2 + Ngc + i] = mhd.vel1[:, Nx2 + Ngc - 1 - i]
            mhd.vel2[:, Nx2 + Ngc + i] = mhd.vel2[:, Nx2 + Ngc - 1 - i]
            mhd.vel3[:, Nx2 + Ngc + i] = mhd.vel3[:, Nx2 + Ngc - 1 - i]
            mhd.bfi1[:, Nx2 + Ngc + i] = mhd.bfi1[:, Nx2 + Ngc - 1 - i]
            mhd.bfi2[:, Nx2 + Ngc + i] = mhd.bfi2[:, Nx2 + Ngc - 1 - i]
            mhd.bfi3[:, Nx2 + Ngc + i] = mhd.bfi3[:, Nx2 + Ngc - 1 - i]
            
        elif mhd.boundMark[3] == 101: #reflective (wall or symmetry) boundary
            mhd.dens[:, Nx2 + Ngc + i] = mhd.dens[:, Nx2 + Ngc - 1 - i]
            mhd.pres[:, Nx2 + Ngc + i] = mhd.pres[:, Nx2 + Ngc - 1 - i]
            mhd.vel1[:, Nx2 + Ngc + i] = mhd.vel1[:, Nx2 + Ngc - 1 - i]
            mhd.vel2[:, Nx2 + Ngc + i] = - mhd.vel2[:, Nx2 + Ngc - 1 - i]
            mhd.vel3[:, Nx2 + Ngc + i] = mhd.vel3[:, Nx2 + Ngc - 1 - i]
            mhd.bfi1[:, Nx2 + Ngc + i] = mhd.bfi1[:, Nx2 + Ngc - 1 - i]
            mhd.bfi2[:, Nx2 + Ngc + i] = - mhd.bfi2[:, Nx2 + Ngc - 1 - i]
            mhd.bfi3[:, Nx2 + Ngc + i] = mhd.bfi3[:, Nx2 + Ngc - 1 - i]
        
        elif mhd.boundMark[3] == 300: #periodic boundary
            mhd.dens[:, Nx2 + Ngc + i] = mhd.dens[:, Ngc + i]
            mhd.pres[:, Nx2 + Ngc + i] = mhd.pres[:, Ngc + i]
            mhd.vel1[:, Nx2 + Ngc + i] = mhd.vel1[:, Ngc + i]
            mhd.vel2[:, Nx2 + Ngc + i] = mhd.vel2[:, Ngc + i]
            mhd.vel3[:, Nx2 + Ngc + i] = mhd.vel3[:, Ngc + i]
            mhd.bfi1[:, Nx2 + Ngc + i] = mhd.bfi1[:, Ngc + i]
            mhd.bfi2[:, Nx2 + Ngc + i] = mhd.bfi2[:, Ngc + i]
            mhd.bfi3[:, Nx2 + Ngc + i] = mhd.bfi3[:, Ngc + i]
            
            
    return mhd








def boundCond_Efld_x(grid, Efld, mhd):
    
    #local variables -- numbers of cells in each direction + number of ghost cells
    Nx1 = grid.Nx1
    Nx2 = grid.Nx2
    Ngc = grid.Ngc
    
    
    
    for i in range(0,Ngc):
        #inner boundary in 1-direction
        if mhd.boundMark[0] == 100: #non-reflective boundary
            Efld[i, :] = Efld[2 * Ngc - 1 - i, :]
            
            
        elif mhd.boundMark[0] == 101: #reflective (wall or symmetry) boundary
            Efld[i, :] = -Efld[2 * Ngc - 1 - i, :]
            
        elif mhd.boundMark[0] == 300: #periodic boundary
            Efld[i, :] = Efld[Nx1 + i, :]
            
                
            
        #outer boundary in 1-direction
        if mhd.boundMark[2] == 100: #non-reflective boundary
            Efld[Nx1 + Ngc + i, :] = Efld[Nx1 + Ngc - 1 - i, :]
            
        elif mhd.boundMark[2] == 101: #reflective (wall or symmetry) boundary
            Efld[Nx1 + Ngc + i, :] = -Efld[Nx1 + Ngc - 1 - i, :]
            
            
        elif mhd.boundMark[2] == 300: #periodic boundary
            Efld[Nx1 + Ngc + i, :] = Efld[Ngc + i, :]
            
        
    return Efld




def boundCond_Efld_y(grid, Efld, mhd):
    
    #local variables -- numbers of cells in each direction + number of ghost cells
    Nx1 = grid.Nx1
    Nx2 = grid.Nx2
    Ngc = grid.Ngc

    for i in range(0,Ngc):
        #inner boundary in 2-direction
        if mhd.boundMark[1] == 100: #non-reflective boundary
            Efld[:, i] = Efld[:, 2 * Ngc - 1 - i]
            
            
        elif mhd.boundMark[1] == 101: #reflective (wall or symmetry) boundary
            Efld[:, i] = -Efld[:, 2 * Ngc - 1 - i]
            
        elif mhd.boundMark[1] == 300: #periodic boundary
            Efld[:, i] = Efld[:, Nx2 + i]
               
            
        #outer boundary in 2-direction
        if mhd.boundMark[3] == 100: #non-reflective boundary
            Efld[:, Nx2 + Ngc + i] = Efld[:, Nx2 + Ngc - 1 - i]
            
            
        elif mhd.boundMark[3] == 101: #reflective (wall or symmetry) boundary
            Efld[:, Nx2 + Ngc + i] = Efld[:, Nx2 + Ngc - 1 - i]
        
        elif mhd.boundMark[3] == 300: #periodic boundary
            Efld[:, Nx2 + Ngc + i] = Efld[:, Ngc + i]
            
            
    return Efld