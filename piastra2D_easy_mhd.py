# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:39:44 2024

@author: mrkon
"""

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
from one_step_mhd import oneStep_MHD_RK
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import time
from visualization import visual



def easy_mhd_solver_call(Nx1, Nx2, setup, CFL, flux_type, rec_type, RK_integr):

    #ghost cells
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
    if (setup == 'BW1D'):
        mhd, aux, eos = init_cond_brio_wu_cart_1D(grid, mhd, aux)
        rhomin = np.min(mhd.dens[Ngc:-Ngc, Ngc:-Ngc])
        rhomax = np.max(mhd.dens[Ngc:-Ngc, Ngc:-Ngc])
    elif setup == 'toth1D':
        mhd, aux, eos = init_cond_toth_cart_1D(grid, mhd, aux)
        rhomin = np.min(mhd.dens[Ngc:-Ngc, Ngc:-Ngc])
        rhomax = np.max(mhd.dens[Ngc:-Ngc, Ngc:-Ngc])
    elif (setup == 'BW1D_0'):
        mhd, aux, eos = init_cond_brio_wu_cart_1D(grid, mhd, aux)
        rhomin = np.min(mhd.dens[Ngc:-Ngc, Ngc:-Ngc])
        rhomax = np.max(mhd.dens[Ngc:-Ngc, Ngc:-Ngc])
        mhd.bfi1[:,:] = 0.0
        mhd.bfi2[:,:] = 0.0
    elif setup == 'toth1D_0':
        mhd, aux, eos = init_cond_toth_cart_1D(grid, mhd, aux)
        rhomin = np.min(mhd.dens[Ngc:-Ngc, Ngc:-Ngc])
        rhomax = np.max(mhd.dens[Ngc:-Ngc, Ngc:-Ngc])
        mhd.bfi1[:,:] = 0.0
        mhd.bfi2[:,:] = 0.0
    elif setup == 'OT2D':
        mhd, aux, eos = init_cond_orszag_tang_cart_2D(grid, mhd, aux)
        rhomax = 0.5
        rhomin = 0.1
    elif setup == 'expl2D':
        mhd, aux, eos = init_cond_mhd_expl_cart_2D(grid, mhd, aux)
        rhomax = 2.0
        rhomin = 0.1
    else:
        print('choose a problem from the list')
        aux.Tfin = -1.0
    ###############################################################
    
    print("grid resolution = ", grid.Nx1, grid.Nx2)
    
    #here we adjust the solver parameters and print them
    aux.rec_type = rec_type
    aux.flux_type = flux_type
    aux.RK_order = RK_integr
    aux.CFL = CFL
    print("reconstruction type = ", aux.rec_type)
    print("Riemann flux = ", aux.flux_type)
    print("Temporal integration = ", aux.RK_order)
    
    #print final phys time 
    print("final phys time = ", aux.Tfin)    
    
    
    #set plotting
    if (Nx2 == 1):
        fig, ax = plt.subplots()
        line, = ax.plot(grid.cx1[Ngc:-Ngc,Ngc], mhd.dens[Ngc:-Ngc,Ngc])
        ax.set_title('sol at time = ' + str(np.round(aux.time, 4)))
        ax.set_xlabel('x1')
        ax.set_ylabel('solution')
        plt.close()  
    
    elif (Nx1 == 1): 
        fig, ax = plt.subplots()
        line, = ax.plot(grid.cx2[Ngc,Ngc:-Ngc], mhd.dens[Ngc,Ngc:-Ngc])
        ax.set_title('sol at time = ' + str(np.round(aux.time, 4)))
        ax.set_xlabel('x2')
        ax.set_ylabel('solution')
        plt.close()  
        
    else:
        # figures and axes
        fig, ax = plt.subplots()
        
        im = ax.imshow(mhd.dens[Ngc:-Ngc, Ngc:-Ngc], origin='lower', \
        extent=[grid.cx2[Ngc,Ngc], grid.cx2[Ngc,Nx2+Ngc], grid.cx1[Ngc,Ngc], grid.cx1[Nx1+Ngc,Ngc]], vmin=rhomin, vmax=rhomax)
        ax.set_title('density at time = ' + str(np.round(aux.time, 2)))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        cbar = plt.colorbar(im, ax=ax) 
        plt.ion()
        plt.show()
    
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
        mhd = oneStep_MHD_RK(grid, mhd, eos, dt, aux.rec_type, aux.flux_type, aux.RK_order)
        
        #time update
        aux.time = aux.time + dt
        
        #output 
        if (i_time%30 == 0) or (aux.Tfin - aux.time) < 1e-12:
            
            print("phys time = ", aux.time)
            print('num of timesteps = ', i_time)
            
            
            if (grid.Nx2 == 1): 
                line.set_data(grid.cx1[Ngc:-Ngc,Ngc], mhd.dens[Ngc:-Ngc,Ngc])
                ax.set_title('density at time = '+ str(np.round(aux.time, 4)))
                
                ax.relim()
                ax.autoscale_view()
                
                clear_output(wait=True)
                plt.pause(0.1)
                display(fig)
                
            if (grid.Nx1 == 1):
                line.set_data(grid.cx2[Ngc,Ngc:-Ngc], mhd.dens[Ngc,Ngc:-Ngc])
                ax.set_title('density at time = '+ str(np.round(aux.time, 4)))
                
                ax.relim()
                ax.autoscale_view()
                
                clear_output(wait=True)
                plt.pause(0.1)
                display(fig)
            
            if (grid.Nx1 != 1 & grid.Nx2 != 1):
                im.set_data(mhd.dens[Ngc:-Ngc, Ngc:-Ngc]) 
                ax.set_title('density at time = '+ str(np.round(aux.time, 4)))
                
                
                clear_output(wait=True)
                display(fig)
                plt.pause(0.1)
     
        
     
    #print final physical time
    print("final phys time = ", aux.time)    
    
    print("END OF SIMULATION")
    ##calculate and the elapsed time of the simulation
    end_time1 = time.time()
    print("time of simulation = ", end_time1 - start_time1, " secs")
      
    
