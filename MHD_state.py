# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 09:50:36 2024

@author: mrkondratyev

notes on the fluid state variables:
    here we suppose, that compressible MHD is represented by density field (dens), velocity field (vel1,vel2,vel3) 
    and pressure field (pres), as well as magnetic field components Bfld1, Bfld2, Bfld3. additionally we have a conservative fluid state with 
    mass, momentum(1,2,3) and total energy, all of them are given for the unit volume
    
The "boundMark" variable is a boundary marker. it marks each boundary (by default the boundary marker is set to 100)
    100 - non-reflecting boundary condition (zero gradient)
    101 - wall bounary condition (normal velocity is set to zero)
    300 - periodic boundaries (even or odd indexes have to the same in this case)
    
"""


import numpy as np

class MHDState:
    def __init__(self, grid):

        self.dc8wave = 0
        self.MHDwCT = 1        

        # Fluid state at each grid cell including ghost cells
        self.dens = np.zeros(grid.grid_shape, dtype=np.double)
        self.vel1 = np.zeros(grid.grid_shape, dtype=np.double)
        self.vel2 = np.zeros(grid.grid_shape, dtype=np.double)
        self.vel3 = np.zeros(grid.grid_shape, dtype=np.double)
        self.pres = np.zeros(grid.grid_shape, dtype=np.double)
        
        #magnetic fields 
        self.bfi1 = np.zeros(grid.grid_shape, dtype=np.double)
        self.bfi2 = np.zeros(grid.grid_shape, dtype=np.double)
        self.bfi3 = np.zeros(grid.grid_shape, dtype=np.double)
        
        #staggered fields 
        self.fb1 = np.zeros((grid.Nx1r - grid.Ngc + 1, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.fb2 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc + 1), dtype=np.double)
        
        
        # Conservative variables inside the real cells only (!)
        self.mass = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.mom1 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.mom2 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.mom3 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.etot = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.bcon1 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.bcon2 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.bcon3 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        
        
        #divergence of magnetic field 
        self.divB = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        
        #boundary marker - by default we set transmissive boundaries
        self.boundMark = np.zeros(4, dtype=np.int32)
        self.boundMark[:] = 100
        
        #user-defined source terms (e.g. gravity acceleration and so on)
        self.F1 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)
        self.F2 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc), dtype=np.double)

        
        