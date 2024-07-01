# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:46:02 2024

@author: mrkondratyev
"""


from grid_setup import Grid
from init_cond_advection import *
from advection_state import AdvState
from CFL_condition import CFLcondition_adv
from aux_routines import auxData
from one_step_advection import oneStep_advection_RK
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import time
    

def easy_advection_solver_call(Nx1, Nx2, setup, CFL, flux_type, rec_type, RK_integr):
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
    adv = AdvState(grid)
    
    #fill the fluid state arrays with initial data
    #see "init_cond.py" for different examples/tests
    ###############################################################
    if setup == '1D':
        adv, aux = init_cond_advection_1D(grid,adv,aux)
    elif setup == '2D':
        adv, aux = init_cond_advection_2D(grid,adv,aux)
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
    print("Temporal integration = ", aux.RK_order)
    
    #print final phys time 
    print("final phys time = ", aux.Tfin)    
    
    #plotting
    if (Nx2 == 1):
        fig, ax = plt.subplots()
        line, = ax.plot(grid.cx1[Ngc:-Ngc,Ngc], adv.adv[Ngc:-Ngc,Ngc])
        ax.set_title('sol at time = ' + str(np.round(aux.time, 4)))
        ax.set_xlabel('x1')
        ax.set_ylabel('solution')
        plt.close()  
    
    elif (Nx1 == 1): 
        fig, ax = plt.subplots()
        line, = ax.plot(grid.cx2[Ngc,Ngc:-Ngc], adv.adv[Ngc,Ngc:-Ngc])
        ax.set_title('sol at time = ' + str(np.round(aux.time, 4)))
        ax.set_xlabel('x2')
        ax.set_ylabel('solution')
        plt.close()  
        
    else:
        # figures and axes
        fig, ax = plt.subplots()
        rhomin = np.min(adv.adv[Ngc:-Ngc, Ngc:-Ngc])
        rhomax = np.max(adv.adv[Ngc:-Ngc, Ngc:-Ngc])
        im = ax.imshow(adv.adv[Ngc:-Ngc, Ngc:-Ngc], origin='lower', \
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
        dt = CFLcondition_adv(grid, adv, aux.CFL)
        dt = min(dt, aux.Tfin - aux.time)
        
        #advected variable update 
        fluid = oneStep_advection_RK(grid, adv, dt, aux.rec_type, aux.flux_type, aux.RK_order)
        
        #"real time" output (animated)
        aux.time = aux.time + dt
        if (i_time % 6 == 0) or (aux.Tfin - aux.time) < 1e-12:
            
            print("phys time = ", aux.time)
            print('num of timesteps = ', i_time)
            
            
            if (grid.Nx2 == 1): 
                line.set_data(grid.cx1[Ngc:-Ngc,Ngc], adv.adv[Ngc:-Ngc,Ngc])
                ax.set_title('sol at time = '+ str(np.round(aux.time, 4)))
                
                ax.relim()
                ax.autoscale_view()
                
                clear_output(wait=True)
                plt.pause(0.1)
                display(fig)
                
            if (grid.Nx1 == 1):
                line.set_data(grid.cx2[Ngc,Ngc:-Ngc], adv.adv[Ngc,Ngc:-Ngc])
                ax.set_title('sol at time = '+ str(np.round(aux.time, 4)))
                
                ax.relim()
                ax.autoscale_view()
                
                clear_output(wait=True)
                plt.pause(0.1)
                display(fig)
            
            if (grid.Nx1 != 1 & grid.Nx2 != 1):
                im.set_data(adv.adv[Ngc:-Ngc, Ngc:-Ngc]) 
                ax.set_title('sol at time = '+ str(np.round(aux.time, 4)))
                
                
                clear_output(wait=True)
                display(fig)
                plt.pause(0.1)
            
     
    #print final physical time
    print("final phys time = ", aux.time)    
    
    print("END OF SIMULATION")
    ##calculate and the elapsed time of the simulation
    end_time1 = time.time()
    print("time of simulation = ", end_time1 - start_time1, " secs")
   
    
#Nx1 = 128
#Nx2 = 1
#setup = '1D'
#CFL = 0.4
#flux_type = 'adv'
#rec_type = 'PCM'
#RK_integr = 'RK1'
#easy_advection_solver_call(Nx1, Nx2, setup, CFL, flux_type, rec_type, RK_integr)