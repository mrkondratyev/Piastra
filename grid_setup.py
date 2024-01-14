"""
Grid class has all the data about our 2D grid, 
inluding the ghost cells to satisfy the boundary conditions


it contains coordinates (cells and cell faces) along dimension 1 and 2, 
cell volumes and faces, mesh (grid) resolution. 

in "__init__" we initialize important data using only the numbers of zones
in "Grid.uniCartGrid" we fill the grid arrays with data for uniform cartesian grid 
after this stage, the finite volume scheme in this code does not use the specific 
geometry and operates with volumes and surfaces in order to extend this code for 
diffrent geometries/grid spacing in future

@author: mrkondratyev
"""


import numpy as np

#by now we consider only simple Cartesian uniform grid in two dimensions
class Grid:
    
    #here we initialize all important data for our grid 
    def __init__(self, Nx1, Nx2, Ngc):
        
        #numbers of cells along 1- and 2-direction
        self.Nx1 = Nx1
        self.Nx2 = Nx2
        #number of ghost cells at each boundary
        self.Ngc = Ngc
        
        #starting index for real cell loops
        self.i0r = Ngc
        #final index for real cell loops in 1-direction 
        self.Nx1r = Nx1 + Ngc
        #final index for real cell loops in 2-direction
        self.Nx2r = Nx2 + Ngc   
        
        #Grid (and hydrodynamical state variables) shape including ghost cells
        self.grid_shape = (Nx1 + Ngc * 2, Nx2 + Ngc * 2)
        
        #face surfaces initialization in each direction
        self.fS1 = np.zeros((Nx1 + Ngc * 2 + 1, Nx2 + Ngc * 2), dtype=np.double) 
        self.fS2 = np.zeros((Nx1 + Ngc * 2, Nx2 + Ngc * 2 + 1), dtype=np.double)
        
        #cell volumes initializtion
        self.cVol = np.zeros((Nx1 + Ngc * 2, Nx2 + Ngc * 2), dtype=np.double)
        
        #face coordinates initialization in each direction
        self.fx1 = np.zeros((Nx1 + Ngc * 2 + 1, Nx2 + Ngc * 2), dtype=np.double)
        self.fx2 = np.zeros((Nx2 + Ngc * 2 + 1, Nx2 + Ngc * 2), dtype=np.double)
        
        #cell center coordinates in each direction initialization
        self.cx1 = np.zeros((Nx1 + Ngc * 2, Nx2 + Ngc * 2), dtype=np.double)
        self.cx2 = np.zeros((Nx1 + Ngc * 2, Nx2 + Ngc * 2), dtype=np.double)
        
        #grid resolution in each direction initialization
        self.dx1 = np.zeros(Nx1 + Ngc * 2, dtype=np.double)
        self.dx2 = np.zeros(Nx2 + Ngc * 2, dtype=np.double)
        
        
        
        
    #uniform 2D Cartesian grid setup
    def uniCartGrid(self, x1ini, x1fin, x2ini,x2fin):
        
        #local variables (cell number in each direction + number of ghost cells)
        Nx1 = self.Nx1
        Nx2 = self.Nx2
        Ngc = self.Ngc
        
        #uniform cartesian grid resolution (simply 1 number for each direction)
        dx1uc = ( x1fin - x1ini )/Nx1
        dx2uc = ( x2fin - x2ini )/Nx2
        
        #grid resolution (array for local cell size in each direction)
        self.dx1[:] = dx1uc
        self.dx2[:] = dx2uc
        
        #face coordinates in each direction
        fx1 = np.linspace(x1ini - Ngc * dx1uc, x1fin + Ngc * dx1uc, Nx1 + Ngc * 2 + 1)
        fx2 = np.linspace(x2ini - Ngc * dx2uc, x2fin + Ngc * dx2uc, Nx2 + Ngc * 2 + 1)
        
        #2D arrays of face coordinates in each direction
        self.fx1 = np.tile(fx1, (Nx2 + Ngc*2, 1)).T
        self.fx2 = np.tile(fx2, (Nx1 + Ngc*2, 1))
        
        #cell center coordinates in each direction
        cx1 = np.linspace(x1ini - (Ngc - 0.5) * dx1uc, x1fin + (Ngc - 0.5) * dx1uc, Nx1 + Ngc * 2)
        cx2 = np.linspace(x2ini - (Ngc - 0.5) * dx2uc, x2fin + (Ngc - 0.5) * dx2uc, Nx2 + Ngc * 2)
        
        #2D arrays of cell center coordinates in each direction
        self.cx1 = np.tile(cx1, (Nx2 + Ngc*2, 1)).T
        self.cx2 = np.tile(cx2, (Nx1 + Ngc*2, 1))
        
        #face surfaces in each direction including ghost cells
        self.fS1[:, :] = dx2uc * 1.0
        self.fS2[:, :] = dx1uc * 1.0
        
        #cell volumes including ghost cells
        self.cVol[:, :] = dx1uc * dx2uc * 1.0
        
        