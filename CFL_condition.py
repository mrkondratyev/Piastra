# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:57:02 2024

This function calculates an avaliable timestep for compressible hydrodynamics simulation in 2D 
with accordance to well-known CFL condition (Courant, Friedrichs and Lewy (1928))
this conditions states, that during the simulation, the fastest wave in the system should not
propagate further than one cell during the timestep

input:
    1) "grid" -- GRID class object
    2) fluid -- fluid state class object
    3) eos -- equation of state class object 
output: 
    dt -- avaliable timestep for 2D simulation according CFL stability condition

@author: mrkondratyev
"""

import numpy as np
from ideal_hydro_functions import soundSpeed


def CFLcondition(grid,fluid,eos):
    
    #local variables - total number of cells in each direction incl ghost cells
    Ntot1 = grid.Nx1 + grid.Ngc * 2
    Ntot2 = grid.Nx2 + grid.Ngc * 2
    
    #grid resolution -- 2D arrays
    dx1dx1 = np.tile(grid.dx1, (Ntot2, 1)).T
    dx2dx2 = np.tile(grid.dx2, (Ntot1, 1))
    dx2dx2 = dx2dx2
    
    #sound speed calculation for whole domain
    sound = soundSpeed(fluid.dens, fluid.pres, eos)
    
    #maximal possible timestep in each direction
    dt1 = np.min( dx1dx1 / (1e-14 + np.abs(fluid.vel1) + sound) )
    dt2 = np.min( dx2dx2 / (1e-14 + np.abs(fluid.vel2) + sound) )
    
    #final timestep (it is already divided by two for 2D calculations for stability reasons)
    dt = 0.5 * min(dt1, dt2)
    
    return dt