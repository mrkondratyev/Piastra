# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:04:01 2024


The main code -- 
grid construction, 
initial data filling, 
simulation control and visualization live here 


@author: mrkondratyev
"""

from grid_setup import Grid
from init_cond_mhd import *
from MHD_state import MHDState
from CFL_condition import CFLcondition_mhd
from aux_routines import auxData
from one_step_mhd import oneStep_MHD_RK_8wave
import matplotlib.pyplot as plt
import numpy as np
import time
from one_step_mhd_CT import oneStep_MHD_RK_CT 
from visualization import visual

#here we introduce the grid cell numbers in each direction + the number of ghost cells
Nx1 = 512
Nx2 = 2
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
mhd = MHDState(grid)

#fill the fluid state arrays with initial data
#see "init_cond.py" for different examples/tests
###############################################################
#mhd, aux, eos =  init_cond_mhd_expl_cart_2D(grid, mhd, aux)
#mhd, aux, eos =  init_cond_orszag_tang_cart_2D(grid, mhd, aux)
mhd, aux, eos =  init_cond_brio_wu_cart_1D(grid, mhd, aux)
#mhd, aux, eos =  init_cond_toth_cart_1D(grid, mhd, aux)

###############################################################

print("grid resolution = ", grid.Nx1, grid.Nx2)

#here we adjust the solver parameters and print them
aux.rec_type = 'PLM'
aux.flux_type = 'LLF'
aux.RK_order = 'RK2'
print("reconstruction type = ", aux.rec_type)
print("Riemann flux = ", aux.flux_type)
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
    dt = CFLcondition_mhd(grid, mhd, eos, aux.CFL)
    dt = min(dt, aux.Tfin - aux.time)
    #fluid state variables update 
    
    #mhd = oneStep_MHD_RK_8wave(grid, mhd, eos, dt, aux.rec_type, aux.flux_type, aux.RK_order)
    mhd = oneStep_MHD_RK_CT(grid, mhd, eos, dt, aux.rec_type, aux.flux_type, aux.RK_order)
    
    #"real time" output (animated)
    aux.time = aux.time + dt
    if (i_time%25 == 0) or (aux.Tfin - aux.time) < 1e-12:
     
        if grid.Nx2 <= 2:
            # 1D plot along x1 axis
            plt.plot(grid.cx1[Ngc:-Ngc, Ngc], mhd.dens[Ngc:-Ngc, Ngc])
            plt.xlabel('x1')
            plt.ylabel('adv')
        elif grid.Nx1 <= 2:
            # 1D plot along x2 axis
            plt.plot(grid.cx2[Ngc, Ngc:-Ngc], mhd.dens[Ngc, Ngc:-Ngc])
            plt.xlabel('x2')
            plt.ylabel('adv')
        else:
            # 2D plot
            rhomin = np.min(mhd.dens[Ngc:-Ngc, Ngc:-Ngc])
            rhomax = np.max(mhd.dens[Ngc:-Ngc, Ngc:-Ngc])
            plt.imshow(mhd.dens[Ngc:-Ngc, Ngc:-Ngc], cmap='jet')
            plt.clim(rhomin, rhomax)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            
        plt.pause(0.03)
    
        #plt.clim(0.0, 1.0)
 
#print final physical time
print("final phys time = ", aux.time)    

print("END OF SIMULATION")
##calculate and the elapsed time of the simulation
end_time1 = time.time()
print("time of simulation = ", end_time1 - start_time1, " secs")
  


rhomin = np.min(mhd.divB)
rhomax = np.max(mhd.divB)
plt.imshow(mhd.divB[Ngc:-Ngc, Ngc:-Ngc], cmap='jet')
plt.clim(rhomin, rhomax)
print("divB max = ", rhomax)
# Show the plot
plt.show()
  
plt.plot(grid.cx1[Ngc:-Ngc,Ngc], mhd.dens[Ngc:-Ngc,Ngc])
#plt.plot(grid.cx2[Ngc,Ngc:-Ngc], mhd.dens[Ngc,Ngc:-Ngc])

#plt.imshow(fluid.dens, extent=(grid.cx1.min(), grid.cx1.max(), grid.cx2.min(), grid.cx2.max()), origin='lower', cmap='viridis', interpolation='nearest', aspect='auto')

# Add labels and a colorbar
#plt.colorbar(label='Colorbar Label')
#plt.xlabel('X Label')
#plt.ylabel('Y Label')
#plt.title('2D Plot of Data')

