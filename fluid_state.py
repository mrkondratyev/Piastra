from grid_setup import *
import numpy as np


"""
notes on the fluid state variables:
    here we suppose, that compressible fluid is represented by density field (dens), velocity field (vel1,vel2,vel3) 
    and pressure field (pres). additionally we have a conservative fluid state with 
    mass, momentum(1,2,3) and total energy, all of them are given for the unit volume
    
The "boundMark" variable is a boundary marker. it marks each boundary (by default the boundary marker is set to 100)
    100 - non-reflecting boundary condition (zero gradient)
    101 - wall bounary condition (normal velocity is set to zero)
    102 - semi-transparent wall (i.e. the fluid can flow away, but nothing flows backward)
    300 - periodic boundaries (even or odd indexes have to the same in this case)
    
@author: mrkondratyev
"""

class FluidState:
    def __init__(self, grid):

        # Fluid state at each grid cell including ghost cells
        self.dens = np.zeros(grid.grid_shape, dtype=np.double)
        self.vel1 = np.zeros(grid.grid_shape, dtype=np.double)
        self.vel2 = np.zeros(grid.grid_shape, dtype=np.double)
        self.vel3 = np.zeros(grid.grid_shape, dtype=np.double)
        self.pres = np.zeros(grid.grid_shape, dtype=np.double)
        
        # Conservative variables inside the real cells only (!)
        self.mass = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc))
        self.mom1 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc))
        self.mom2 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc))
        self.mom3 = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc))
        self.etot = np.zeros((grid.Nx1r - grid.Ngc, grid.Nx2r - grid.Ngc))
        
        #boundary marker 
        self.boundMark = np.zeros(4, dtype=np.int32)
        self.boundMark[:] = 100
        
        #user-defined source terms (e.g. gravity acceleration and so on)
        #TBD 
        
        