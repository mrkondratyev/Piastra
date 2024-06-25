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
from prim_cons_fluid_MHD import soundSpeed

def CFLcondition_fluid(grid,fluid,eos,CFL):
      
    #sound speed calculation for whole domain
    sound = soundSpeed(fluid.dens, fluid.pres, eos)
    
    #maximal possible timestep in each direction
    dt1 = np.min( grid.dx1 / (1e-14 + np.abs(fluid.vel1) + sound) )
    dt2 = np.min( grid.dx2 / (1e-14 + np.abs(fluid.vel2) + sound) )
    
    #final timestep (it is already divided by two for 2D calculations for stability reasons)
    dt = CFL * min(dt1, dt2)
    
    return dt



"""
This function calculates an avaliable timestep for magnetohydrodynamics simulation in 2D 
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

def CFLcondition_mhd(grid,mhd,eos,CFL):
      
    Ngc = grid.Ngc
    #sound speed calculation for whole domain
    sound = soundSpeed(mhd.dens[Ngc:-Ngc, Ngc:-Ngc], mhd.pres[Ngc:-Ngc, Ngc:-Ngc], eos)
    
    #squared fast magnetosonic speed calculation for whole domain 
    cf2 = sound**2 + (mhd.bfi1[Ngc:-Ngc, Ngc:-Ngc]**2 + mhd.bfi2[Ngc:-Ngc, Ngc:-Ngc]**2 + mhd.bfi3[Ngc:-Ngc, Ngc:-Ngc]**2)/(mhd.dens[Ngc:-Ngc, Ngc:-Ngc] + 1e-14)
    
    #maximal possible timestep in each direction
    dt1 = np.min( grid.dx1[Ngc:-Ngc, Ngc:-Ngc] / (1e-14 + np.abs(mhd.vel1[Ngc:-Ngc, Ngc:-Ngc]) + np.sqrt(cf2)) )
    dt2 = np.min( grid.dx2[Ngc:-Ngc, Ngc:-Ngc] / (1e-14 + np.abs(mhd.vel2[Ngc:-Ngc, Ngc:-Ngc]) + np.sqrt(cf2)) )
    
    #final timestep (it is already divided by two for 2D calculations for stability reasons)
    dt = CFL * min(dt1, dt2)
    
    return dt


"""
This function calculates an avaliable timestep for linear advection simulation in 2D 
with accordance to well-known CFL condition (Courant, Friedrichs and Lewy (1928))
this conditions states, that during the simulation, the fastest wave in the system should not
propagate further than one cell during the timestep

input:
    1) "grid" -- GRID class object
    2) adv -- advected state class object
output: 
    dt -- avaliable timestep for 2D simulation according CFL stability condition

@author: mrkondratyev
"""


def CFLcondition_adv(grid,adv,CFL):

    #maximal possible timestep in each direction
    dt1 = np.min( grid.dx1 / (1e-14 + np.abs(adv.vel1)) )
    dt2 = np.min( grid.dx2 / (1e-14 + np.abs(adv.vel2)) )
    
    #final timestep (it is already divided by two for 2D calculations for stability reasons)
    dt = CFL * min(dt1, dt2)
    
    return dt
