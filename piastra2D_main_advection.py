# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:31:26 2024

@author: mrkondratyev
"""

from grid_setup import Grid
from init_cond_advection import *
from advection_state import AdvState
from CFL_condition import CFLcondition_adv
from aux_routines import auxData
from one_step_advection import oneStep_advection_RK
import matplotlib.pyplot as plt
import numpy as np
import time
from visualization import visual

#here we introduce the grid cell numbers in each direction + the number of ghost cells
Nx1 = 256
Nx2 = 1
Ngc = 3



#here we initialize the grid
grid = Grid(Nx1, Nx2, Ngc)

#coordinate range in each direction, by default x and y are in range [0..1]
x1ini, x1fin = 0.0, 1.0
x2ini, x2fin = 0.0, 1.0

#filling the grid arrays with grid data (by now it is only uniform Cartesian grid)
grid.uniCartGrid(x1ini, x1fin, x2ini, x2fin)

#obtaining auxilary data (timestep, final time and so on)
aux = auxData(grid)

#initialize fluid state
adv = AdvState(grid)

#fill the fluid state arrays with initial data
#see "init_cond.py" for different examples/tests
###############################################################
adv, aux = init_cond_advection_1D(grid,adv,aux)
###############################################################

print("grid resolution = ", grid.Nx1, grid.Nx2)

#here we adjust the solver parameters and print them
aux.rec_type = 'PPMorig'
aux.flux_type = 'adv'
aux.RK_order = 'RK3'
print("reconstruction type = ", aux.rec_type)
print("Temporal integration = ", aux.RK_order)

#print final phys time 
print("final phys time = ", aux.Tfin)    


#set the start timer to check the elapsed time 
start_time1 = time.time() 

print("START OF SIMULATION")

#cycle over time 
i_time = 0
while aux.time < aux.Tfin:
       
    #current timestep
    i_time = i_time + 1
    
    #calculate the avaliable timestep according to Courant-Friedrichs-Lewy condition
    dt = CFLcondition_adv(grid, adv, aux.CFL)
    dt = min(dt, aux.Tfin - aux.time)
    
    #advected variable update 
    fluid = oneStep_advection_RK(grid, adv, dt, aux.rec_type, aux.flux_type, aux.RK_order)
    
    #"real time" output (animated)
    aux.time = aux.time + dt
    if (i_time % 6 == 0) or (aux.Tfin - aux.time) < 1e-12:
        
        print("phys time = ", aux.time)
        print('num of timesteps = ', i_time)
        
        visual(grid, adv.adv)
        
 
#print final physical time
print("final phys time = ", aux.time)    

print("END OF SIMULATION")
##calculate and the elapsed time of the simulation
end_time1 = time.time()
print("time of simulation = ", end_time1 - start_time1, " secs")
  
# Show the plot
plt.show()
  
plt.plot(grid.cx1[Ngc:-Ngc,Ngc], adv.adv[Ngc:-Ngc,Ngc])
plt.plot(grid.cx2[Ngc,Ngc:-Ngc], adv.adv[Ngc,Ngc:-Ngc])

#plt.imshow(fluid.dens, extent=(grid.cx1.min(), grid.cx1.max(), grid.cx2.min(), grid.cx2.max()), origin='lower', cmap='viridis', interpolation='nearest', aspect='auto')

# Add labels and a colorbar
#plt.colorbar(label='Colorbar Label')
#plt.xlabel('X Label')
#plt.ylabel('Y Label')
#plt.title('2D Plot of Data')

